"""How trustworthy are the heatmaps? For frames the model calls FAKE, hide the hottest region and compare the drop in
P(fake) with hiding a random region of the same size, for Grad-CAM and Grad-CAM++.
Uses the saved model on saved crops (the model has seen these videos: this measures how faithful the explanations
are to the model, not how accurate the model is).
Example: python cam_faithfulness.py --model models/finetune.pt --dataset uadfv sdfvd --videos 30 --frames 4"""
import argparse

import numpy as np
import torch

from finetune_cv import CROP, load, make_net, predict
from gradcam import explain, faithfulness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/finetune.pt")
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--videos", type=int, default=30, help="fake videos sampled per dataset")
    ap.add_argument("--frames", type=int, default=4, help="frames sampled per video")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu")
    net = make_net()
    net.load_state_dict((ck.get("states") or [ck["state"]])[0])
    net.to(dev).eval()
    score_fn = lambda imgs: predict(net, imgs[:, None], dev)[:, 0]
    rng = np.random.RandomState(a.seed)
    rows = {"gradcam": [], "gradcam++": []}
    for d in a.dataset:
        X, y, _ = load(d, a.crops, a.step)
        fakes = np.where(y == 1)[0]
        for v in rng.choice(fakes, min(a.videos, len(fakes)), replace=False):
            o = (X.shape[2] - CROP) // 2
            for t in rng.choice(X.shape[1], min(a.frames, X.shape[1]), replace=False):
                crop = X[v, t]
                for m in rows:
                    cam, p = explain(net, crop, dev, m)
                    if p < 0.5:
                        continue  # only frames the model calls fake
                    fa = faithfulness(score_fn, crop[o:o + CROP, o:o + CROP], cam, a.frac, seed=int(v * 100 + t))
                    if fa:
                        rows[m].append(fa)
    print(f"\nHiding the hottest {a.frac:.0%} of the face, frames the model calls FAKE")
    print(f"{'method':10s} {'frames':>6s} {'P(fake) before':>15s} {'after hiding':>13s} {'random hiding':>14s} {'beats random':>13s}")
    for m, r in rows.items():
        if not r:
            print(f"{m:10s} no usable frames")
            continue
        print(f"{m:10s} {len(r):6d} {np.mean([x['original'] for x in r]):15.3f} {np.mean([x['removed'] for x in r]):13.3f} "
              f"{np.mean([x['random'] for x in r]):14.3f} {np.mean([x['drop'] > x['random_drop'] for x in r]):13.1%}")


if __name__ == "__main__":
    main()
