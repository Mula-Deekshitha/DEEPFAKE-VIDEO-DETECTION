"""Writes SYNTHETIC feature caches (random numbers with a weak class signal) so the training and
evaluation code can be smoke-tested without any videos. The numbers it produces mean nothing."""
import argparse
from pathlib import Path

import numpy as np

from dfd.features import N_BLEND, N_FREQ, N_LM, N_QUAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_synthetic")
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    for name, n in (("synA", 60), ("synB", 40)):
        d = Path(a.cache) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            lab = i % 2
            rec = dict(emb=(rng.randn(32, 1280) + 0.15 * lab).astype(np.float16),
                       lm=(rng.randn(32, N_LM) + 0.2 * lab).astype(np.float32),
                       freq=(rng.randn(32, N_FREQ) + 0.2 * lab).astype(np.float32),
                       qual=rng.randn(32, N_QUAL).astype(np.float32),
                       blend=(rng.randn(32, N_BLEND) + 0.2 * lab).astype(np.float32), valid=np.ones(32, np.float32))
            np.savez_compressed(d / f"v{i:03d}.npz", label=lab, path=f"{name}/v{i:03d}.mp4", **rec)
    print("written to", a.cache)


if __name__ == "__main__":
    main()
