"""Step D: train the fine-tuned EfficientNet-B0 on ALL videos of the given datasets and save it.
Uses the same settings that were cross-validated.
Example: python finetune_final.py --dataset uadfv sdfvd --epochs 4 --fpv 4 --step 2"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from finetune_cv import load, train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--fpv", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", choices=["none", "strong"], default="none", help="training augmentation")
    ap.add_argument("--nets", type=int, default=1, help="number of networks to train and average")
    ap.add_argument("--tta", action="store_true", help="store flag: average frame and mirror image when predicting")
    ap.add_argument("--out", default="models/finetune.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xs, ys = [], []
    for d in a.dataset:
        X, y, _ = load(d, a.crops, a.step)
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs
    print(f"Training on {len(y)} videos ({int((y == 0).sum())} real / {int((y == 1).sum())} fake), device={dev}")
    cfg = SimpleNamespace(epochs=a.epochs, fpv=a.fpv, batch=a.batch, lr=a.lr, aug=a.aug)
    states = []
    for k in range(a.nets):
        print(f"  network {k + 1}/{a.nets}")
        states.append(train_fold(X, y, cfg, dev, a.seed + 100 * k).cpu().state_dict())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": states[0], "states": states, "tta": a.tta, "step": a.step, "datasets": a.dataset}, a.out)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
