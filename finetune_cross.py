"""Step C: cross-dataset test. Fine-tune on every video of one dataset, test on every video of the other.
No video of the test dataset is ever seen in training, and the datasets differ in source, quality and
forgery method, so this is the real generalisation figure.

Example: python finetune_cross.py --train uadfv --test sdfvd --epochs 6 --fpv 8
         python finetune_cross.py --train sdfvd --test uadfv --epochs 4 --fpv 4 --step 2"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import metrics, show
from finetune_cv import load, predict, train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fpv", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--aug", choices=["none", "strong"], default="none", help="training augmentation")
    ap.add_argument("--out", default="results_ft_cross")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, _ = load(a.train, a.crops, a.step)
    print(f"train {a.train}: {len(ytr)} videos | device={dev}")
    net = train_fold(Xtr, ytr, a, dev, a.seed)
    del Xtr
    Xte, yte, _ = load(a.test, a.crops, a.step)
    fr = predict(net, Xte, dev)
    res = {"video": metrics(yte, fr.mean(1)), "frames": metrics(np.repeat(yte, fr.shape[1]), fr.ravel())}
    show(f"train on {a.train} -> test on {a.test}", res)
    Path(a.out).mkdir(exist_ok=True)
    json.dump(res, open(Path(a.out) / f"{a.train}_to_{a.test}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
