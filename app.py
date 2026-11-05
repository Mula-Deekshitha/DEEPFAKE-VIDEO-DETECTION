```python
"""
Streamlit demo: Upload a video and get a REAL / FAKE prediction.
Run: streamlit run app.py
Requires: models/finetune.pt
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from crops import video_crops
from gradcam import explain, faithfulness, overlay
from finetune_cv import CROP, make_net, predict


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_DIR = Path("models")
THRESHOLD = 0.5
UNSURE = (0.35, 0.65)

st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🎭",
    layout="wide"
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(path):

    ck = torch.load(path, map_location="cpu")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    nets = []

    for state in ck.get("states") or [ck["state"]]:

        net = make_net()
        net.load_state_dict(state)
        net.to(dev)
        net.eval()

        nets.append(net)

    return (
        nets,
        ck["step"],
        dev,
        ck.get("datasets", []),
        ck.get("tta", False)
    )


# --------------------------------------------------
# VIDEO ANALYSIS
# --------------------------------------------------

def analyse(path, nets, step, dev, tta):

    crops, faces = video_crops(
        path,
        32,
        160,
        return_faces=True
    )

    crops = crops[::step]
    faces = faces[::step]

    probs = np.mean(
        [
            predict(net, crops[None], dev, tta)[0]
            for net in nets
        ],
        axis=0
    )

    return crops, faces, probs


# --------------------------------------------------
# IMAGE CONVERSION
# --------------------------------------------------

def prepare_image(image):

    image = np.asarray(image)

    if image.dtype != np.uint8:

        if image.size > 0 and image.max() <= 1.0:
            image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🎭 Deepfake Video Detector")

st.caption(
    "Face crops from sampled video frames are analysed using "
    "a fine-tuned EfficientNet-B0 model."
)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

MODEL_FILES = sorted(MODEL_DIR.glob("*.pt"))

if not MODEL_FILES:

    st.error(
        "No model found in the models folder. "
        "Please add your trained .pt model."
    )

    st.stop()


model_file = st.sidebar.selectbox(
    "Select Model",
    MODEL_FILES,
    format_func=lambda p: p.name
)

nets, step, dev, trained_on, tta = load_model(
    str(model_file)
)


# --------------------------------------------------
# SIDEBAR INFORMATION
# --------------------------------------------------

with st.sidebar:

    st.header("About this model")

    st.write(
        f"**Trained on:** "
        f"{', '.join(trained_on) or 'Unknown'}"
    )

    st.write(f"**Running on:** {dev}")

    st.markdown(
        """
        **Reported evaluation results**

        - UADFV: about 98%
        - SDFVD 2.0: about 95%

        **Robustness**

        - Blurred, shrunk or noisy videos may reduce accuracy.
        - Videos from unfamiliar datasets may produce unreliable results.

        Predictions should be treated as model estimates, not proof.
        """
    )


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

up = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv", "webm"]
)

if up is None:

    st.info(
        "Upload a short video containing a person's face to analyse it."
    )

    st.stop()


# --------------------------------------------------
# HEATMAP SETTINGS
# --------------------------------------------------

with st.sidebar:

    st.header("Explainability Settings")

    show_cam = st.checkbox(
        "Show heatmaps",
        value=True
    )

    cam_method = st.radio(
        "Heatmap Method",
        ["Grad-CAM++", "Grad-CAM"]
    )

    frac = st.slider(
        "Face area hidden in faithfulness check (%)",
        10,
        40,
        20
    ) / 100


# --------------------------------------------------
# VIDEO PREVIEW
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("Uploaded Video")

    st.video(up)


# --------------------------------------------------
# TEMPORARY VIDEO PROCESSING
# --------------------------------------------------

suffix = Path(up.name).suffix or ".mp4"

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
) as tmp:

    tmp.write(up.getbuffer())
    tmp_path = tmp.name


try:

    with st.spinner("Detecting faces and analysing video..."):

        crops, faces, probs = analyse(
            tmp_path,
            nets,
            step,
            dev,
            tta
        )

except Exception as e:

    st.error(
        f"Could not analyse this video: "
        f"{type(e).__name__}: {e}"
    )

    st.stop()

finally:

    try:
        os.unlink(tmp_path)

    except OSError:
        pass


# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------

score = float(probs.mean())

n_face = int(faces.sum())

with right:

    st.subheader("Detection Result")

    if n_face < max(3, len(faces) // 2):

        st.warning(
            f"A face was detected in only {n_face} "
            f"of {len(faces)} sampled frames. "
            "The prediction may not be reliable."
        )

    if UNSURE[0] < score < UNSURE[1]:

        st.warning(
            f"### Uncertain\nP(fake) = {score:.1%}"
        )

    elif score > THRESHOLD:

        st.error(
            f"### Likely FAKE\nP(fake) = {score:.1%}"
        )

    else:

        st.success(
            f"### Likely REAL\nP(fake) = {score:.1%}"
        )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Video P(fake)",
        f"{score:.1%}"
    )

    c2.metric(
        "Frames flagged fake",
        f"{int((probs > THRESHOLD).sum())} / {len(probs)}"
    )

    c3.metric(
        "Frames with a face",
        f"{n_face} / {len(faces)}"
    )

    st.progress(
        min(max(score, 0.0), 1.0)
    )


# --------------------------------------------------
# PER-FRAME SCORES
# --------------------------------------------------

st.subheader("Per-frame Prediction Scores")

st.line_chart(
    {
        "P(fake)": probs,
        "Threshold": np.full_like(probs, THRESHOLD)
    },
    height=220
)


# --------------------------------------------------
# FACE CROPS
# --------------------------------------------------

st.subheader("Face Crops the Model Saw")

o = (crops.shape[1] - CROP) // 2

show = np.linspace(
    0,
    len(crops) - 1,
    min(8, len(crops))
).round().astype(int)


crop_cols = st.columns(4)

for j, i in enumerate(show):

    center = crops[i, o:o + CROP, o:o + CROP]

    center = prepare_image(center)

    caption = (
        f"Frame {i + 1}: {probs[i]:.0%} fake"
        if faces[i]
        else f"Frame {i + 1}: {probs[i]:.0%} fake (no face)"
    )

    with crop_cols[j % 4]:

        st.image(
            center,
            caption=caption,
            use_container_width=True
        )


# --------------------------------------------------
# GRAD-CAM / GRAD-CAM++
# --------------------------------------------------

if show_cam:

    st.divider()

    st.subheader(
        f"Where the Model Looks ({cam_method})"
    )

    st.caption(
        "Red regions indicate areas contributing to the FAKE prediction. "
        "The scores below compare the original prediction, the prediction "
        "after hiding the highlighted area, and the prediction after "
        "hiding an equally sized random area."
    )

    try:

        method = (
            "gradcam++"
            if cam_method == "Grad-CAM++"
            else "gradcam"
        )

        score_fn = lambda imgs: predict(
            nets[0],
            imgs[:, None],
            dev
        )[:, 0]

        tested = 0
        wins = 0

        cam_cols = st.columns(2)

        for j, i in enumerate(show):

            with st.spinner(
                f"Generating heatmap {j + 1}/{len(show)}..."
            ):

                # Generate Grad-CAM heatmap
                cam, p = explain(
                    nets[0],
                    crops[i],
                    dev,
                    method
                )

                # Extract centre face crop
                center = crops[
                    i,
                    o:o + CROP,
                    o:o + CROP
                ]

                # Evaluate faithfulness
                fa = faithfulness(
                    score_fn,
                    center,
                    cam,
                    frac
                )

                if fa is None:

                    caption = (
                        f"Frame {i + 1} | {p:.0%} fake\n"
                        "No region pushes towards fake."
                    )

                else:

                    tested += 1

                    if fa["drop"] > fa["random_drop"]:
                        wins += 1

                    caption = (
                        f"Frame {i + 1} | "
                        f"{fa['original']:.0%} → "
                        f"{fa['removed']:.0%} "
                        f"(Random: {fa['random']:.0%})"
                    )

                # Generate overlay
                heatmap = overlay(
                    center,
                    cam
                )

                # Convert to display-ready image
                heatmap = prepare_image(heatmap)

                # Display heatmap
                with cam_cols[j % 2]:

                    st.image(
                        heatmap,
                        caption=caption,
                        use_container_width=True
                    )

        # Faithfulness summary
        if tested:

            st.info(
                f"Hiding the highlighted {frac:.0%} of the face "
                f"lowered the fake score more than hiding a random "
                f"area of the same size in {wins} of {tested} "
                f"tested frames."
            )

    except Exception as e:

        st.error(
            f"Heatmap generation failed: "
            f"{type(e).__name__}: {e}"
        )


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

with st.expander("How to Read This Result"):

    st.markdown(
        """
        - The decision threshold is 0.5.
        - Scores between 35% and 65% are marked uncertain.
        - Face crops show the frames analysed by the model.
        - Grad-CAM++ highlights image regions that influence
          the model's prediction.
        - Faithfulness compares the score change after hiding
          highlighted regions against hiding random regions.
        - Videos from unfamiliar datasets or with different
          compression and quality may be misclassified.

        **Important:** A model prediction is an indication,
        not definitive proof that a video is real or fake.
        """
    )
```
