"""Robustness test: how does accuracy change when the face crops of UNSEEN videos are blurred, shrunk,
JPEG-compressed or noisy?  For each cross-validation fold a model is trained on the training videos only and
then scored on the held-out videos under 13 conditions (clean + 4 degradations x 3 strengths).

Run it twice to see whether training augmentation helps:
  python finetune... robustness_cv.py --dataset uadfv sdfvd --group-regex "^(\\d+)|vs?(\\d+)_aug" --aug none   --out rob_none
  python robustness_cv.py --dataset uadfv sdfvd --group-regex "^(\\d+)|vs?(\\d+)_aug" --aug strong --out rob_strong --compare rob_none/robustness.json
Degradations are applied to the saved 160 px crops, an approximation of real video compression."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from degrade import CONDITIONS, KINDS, LEVELS, degrade_array
from dfd.data import groups_from_paths
from evaluate import metrics
from finetune_cv import load, predict, train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--fpv", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--aug", choices=["none", "strong"], default="none")
    ap.add_argument("--out", default="results_robustness")
    ap.add_argument("--compare", default=None, help="robustness.json of another run to compare against")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xs, ys, gs, off = [], [], [], 0
    for d in a.dataset:
        Xd, yd, pd_ = load(d, a.crops, a.step)
        gd = groups_from_paths(pd_, a.group_regex) + off
        off = gd.max() + 1
        Xs.append(Xd)
        ys.append(yd)
        gs.append(gd)
    X, y, groups = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    del Xs
    print(f"{'+'.join(a.dataset)}: {len(y)} videos, aug={a.aug}, device={dev}")

    fr = {name: np.zeros(X.shape[:2]) for name, _, _ in CONDITIONS}
    skf = StratifiedGroupKFold(a.folds, shuffle=True, random_state=a.seed)
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        print(f"  fold {f + 1}/{a.folds}: training")
        net = train_fold(X[tr], y[tr], a, dev, a.seed + f)
        for name, kind, level in CONDITIONS:
            Xt = X[te] if kind is None else degrade_array(X[te], kind, level, seed=f)
            fr[name][te] = predict(net, Xt, dev)

    res = {name: metrics(y, fr[name].mean(1)) for name, _, _ in CONDITIONS}
    other = json.load(open(a.compare)) if a.compare else None
    print(f"\n=== robustness ({'+'.join(a.dataset)}, training augmentation: {a.aug}) ===")
    print(f"{'condition':16s} {'acc':>7s} {'prec':>7s} {'rec':>7s} {'auc':>7s}" + ("   change vs compare" if other else ""))
    for name, m in res.items():
        line = f"{name:16s} {m['accuracy']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['auc']:7.4f}"
        if other and name in other:
            line += f"   {m['accuracy'] - other[name]['accuracy']:+.4f}"
        print(line)

    Path(a.out).mkdir(exist_ok=True)
    json.dump(res, open(Path(a.out) / "robustness.json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for k in KINDS:
            plt.plot(["clean"] + [str(v) for v in LEVELS[k]],
                     [res["clean"]["accuracy"]] + [res[f"{k}-{v}"]["accuracy"] for v in LEVELS[k]], marker="o", label=k)
        plt.ylim(0.4, 1.02)
        plt.ylabel("video accuracy")
        plt.xlabel("severity (left = mild, right = strong)")
        plt.title(f"Robustness, training augmentation: {a.aug}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(a.out) / "robustness.png", dpi=150)
        print(f"Saved {a.out}/robustness.json and robustness.png")
    except ImportError:
        print(f"Saved {a.out}/robustness.json (install matplotlib for the chart)")


if __name__ == "__main__":
    main()
