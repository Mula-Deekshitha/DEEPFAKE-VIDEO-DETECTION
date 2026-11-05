"""Streamlit demo: upload a video, get a REAL / FAKE verdict from the fine-tuned model.
Run:  streamlit run app.py
Needs models/finetune.pt (from finetune_final.py)."""
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from crops import video_crops
from gradcam import explain, faithfulness, overlay
from finetune_cv import CROP, make_net, predict

MODEL_DIR = Path("models")
THRESHOLD = 0.5
UNSURE = (0.35, 0.65)

st.set_page_config(page_title="Deepfake Video Detector", page_icon="🎭", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    ck = torch.load(path, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = []
    for st in ck.get("states") or [ck["state"]]:
        n = make_net()
        n.load_state_dict(st)
        nets.append(n.to(dev).eval())
    return nets, ck["step"], dev, ck.get("datasets", []), ck.get("tta", False)


def analyse(path, nets, step, dev, tta):
    crops, faces = video_crops(path, 32, 160, return_faces=True)
    crops, faces = crops[::step], faces[::step]
    probs = np.mean([predict(n, crops[None], dev, tta)[0] for n in nets], axis=0)
    return crops, faces, probs


st.title("🎭 Deepfake Video Detector")
st.caption("Face crops from 32 sampled frames are scored by a fine-tuned EfficientNet-B0; "
           "the video score is the average frame score.")

MODEL_FILES = sorted(MODEL_DIR.glob("*.pt"))
if not MODEL_FILES:
    st.error("No model found in the `models` folder. Run `python finetune_final.py --dataset uadfv sdfvd "
             "--epochs 4 --fpv 4 --step 2` first.")
    st.stop()
model_file = st.sidebar.selectbox("Model", MODEL_FILES, format_func=lambda p: p.name)
nets, step, dev, trained_on, tta = load_model(str(model_file))

with st.sidebar:
    st.header("About this model")
    st.write(f"Trained on: **{', '.join(trained_on) or 'unknown'}**")
    st.write(f"Running on: **{dev}**")
    st.markdown(
        "**Measured accuracy** (videos never seen in training, sources kept apart):\n"
        "- UADFV: about 98%\n- SDFVD 2.0: about 95%\n\n"
        "**Blurred, shrunk or noisy videos:** the standard model drops to about 66% on average; a model trained "
        "with `--aug strong` (e.g. `finetune_robust.pt`) keeps about 87% but is about 4 points lower on clean video.\n\n"
        "**Videos unlike these two datasets:** only 51-58% in cross-dataset tests. "
        "Treat those results as a weak hint, not proof.")

up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
if up is None:
    st.info("Upload a short video of a person's face to analyse it.")
    st.stop()

show_cam = st.sidebar.checkbox("Show heatmaps", value=True)
cam_method = st.sidebar.radio("Heatmap method", ["Grad-CAM++", "Grad-CAM"])
frac = st.sidebar.slider("Face area hidden in the check (%)", 10, 40, 20) / 100

left, right = st.columns([1, 1])
with left:
    st.video(up)

suffix = Path(up.name).suffix or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(up.getbuffer())
    tmp_path = tmp.name
try:
    with st.spinner("Detecting faces and scoring frames..."):
        crops, faces, probs = analyse(tmp_path, nets, step, dev, tta)
except Exception as e:
    st.error(f"Could not analyse this video: {e}")
    st.stop()
finally:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

score = float(probs.mean())
n_face = int(faces.sum())
with right:
    if n_face < max(3, len(faces) // 2):
        st.warning(f"A face was found in only {n_face} of {len(faces)} sampled frames, so the result "
                   f"below is based mostly on non-face crops and is not reliable.")
    if UNSURE[0] < score < UNSURE[1]:
        st.warning(f"### Uncertain  \nP(fake) = {score:.1%}, too close to call.")
    elif score > THRESHOLD:
        st.error(f"### Likely FAKE  \nP(fake) = {score:.1%}")
    else:
        st.success(f"### Likely REAL  \nP(fake) = {score:.1%}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Video P(fake)", f"{score:.1%}")
    c2.metric("Frames flagged fake", f"{int((probs > THRESHOLD).sum())} / {len(probs)}")
    c3.metric("Frames with a face", f"{n_face} / {len(faces)}")
    st.progress(min(max(score, 0.0), 1.0))

st.subheader("Per-frame scores")
st.line_chart({"P(fake)": probs, "threshold": np.full_like(probs, THRESHOLD)}, height=220)

st.subheader("Face crops the model saw")
o = (crops.shape[1] - CROP) // 2
show = np.linspace(0, len(crops) - 1, min(8, len(crops))).round().astype(int)
cols = st.columns(len(show))
for c, i in zip(cols, show):
    c.image(crops[i, o:o + CROP, o:o + CROP], caption=f"{probs[i]:.0%} fake" + ("" if faces[i] else " (no face)"),
            use_container_width=True)

if show_cam:
    st.subheader(f"Where the model looks ({cam_method})")
    st.caption("Red = face regions that push the score towards FAKE (first network). Under each image: the fake score "
               "before -> after hiding the highlighted area of the face, and (random) after hiding an equally large "
               "random area. If the highlight is meaningful, hiding it lowers the score more than hiding random area.")
    try:
        method = "gradcam++" if cam_method == "Grad-CAM++" else "gradcam"
        score_fn = lambda imgs: predict(nets[0], imgs[:, None], dev)[:, 0]
        tested = wins = 0
        cols = st.columns(len(show))
        for c, i in zip(cols, show):
            cam, p = explain(nets[0], crops[i], dev, method)
            center = crops[i, o:o + CROP, o:o + CROP]
            fa = faithfulness(score_fn, center, cam, frac)
            if fa is None:
                cap = f"{p:.0%} fake. No region pushes towards fake."
            else:
                tested += 1
                wins += fa["drop"] > fa["random_drop"]
                cap = f"{fa['original']:.0%} -> {fa['removed']:.0%} (random {fa['random']:.0%})"
            c.image(overlay(center, cam), caption=cap, use_container_width=True)
        if tested:
            st.info(f"Hiding the highlighted {frac:.0%} of the face lowered the fake score more than hiding a random area "
                    f"of the same size in {wins} of {tested} frames.")
    except Exception as e:
        st.info(f"Heatmap could not be computed: {e}")

with st.expander("How to read this"):
    st.markdown(
        "- The decision threshold is fixed at 0.5; scores between 35% and 65% are shown as uncertain.\n"
        "- The model learned UADFV (face swaps) and SDFVD 2.0. A video made by a different method, or with "
        "very different quality, can be misjudged in either direction.\n"
        "- Clips from the training sets were seen during training, so a correct answer on them does not "
        "measure accuracy.")
