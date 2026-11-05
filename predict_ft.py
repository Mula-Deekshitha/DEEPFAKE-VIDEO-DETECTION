"""Step E: classify new videos with the fine-tuned model.
Example: python predict_ft.py clip1.mp4 clip2.mp4

Score = mean P(fake) over the sampled frames; threshold 0.5. The model was trained on UADFV and SDFVD 2.0
only. On videos unlike those (other sources, quality or forgery methods) it is much less reliable:
cross-dataset tests gave only 51-58 % accuracy, so treat such results as a weak hint."""
import argparse
import os

import numpy as np
import torch

from crops import video_crops
from finetune_cv import make_net, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="models/finetune.pt")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=160)
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = []
    for st in ck.get("states") or [ck["state"]]:
        n = make_net()
        n.load_state_dict(st)
        nets.append(n.to(dev))
    tta = ck.get("tta", False)
    for v in a.videos:
        if not os.path.isfile(v):
            print(f"{v}: file not found (check the path)")
            continue
        try:
            crops = video_crops(v, a.frames, a.size)[::ck["step"]]
        except Exception as e:
            print(f"{v}: could not be read ({e})")
            continue
        p = np.mean([predict(n, crops[None], dev, tta)[0] for n in nets], axis=0)
        s = float(p.mean())
        print(f"{v}: {'FAKE' if s > 0.5 else 'REAL'}  (P(fake)={s:.3f}; {int((p > 0.5).sum())}/{len(p)} frames look fake)")


if __name__ == "__main__":
    main()
