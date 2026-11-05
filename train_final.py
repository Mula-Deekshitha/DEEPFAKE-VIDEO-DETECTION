"""Step 3: fit the ensemble on ALL cached videos and save it for inference.
Example: python train_final.py --datasets sdfvd uadfv"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from dfd.data import load_cached
from dfd.models import Ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--models", nargs="+", default=["hgb", "lr", "fhgb", "tnet"], choices=["hgb", "lr", "fhgb", "tnet"])
    ap.add_argument("--out", default="models/ensemble.joblib")
    a = ap.parse_args()
    recs, ys = [], []
    for d in a.datasets:
        r, y, _ = load_cached(d, a.cache)
        recs += r
        ys.append(y)
    y = np.concatenate(ys)
    print(f"Training on {len(y)} videos")
    ens = Ensemble(a.seeds, a.epochs, a.models).fit(recs, y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ens, a.out)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
