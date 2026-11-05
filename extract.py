"""Step 1: extract and cache features for every video.
Example: python extract.py --data sdfvd=/path/to/SDFVD2.0 uadfv=/path/to/UADFV
Folders must contain 'real' / 'fake' (or original / deepfake ...) in their path or file name."""
import argparse

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x

from dfd.data import cache_path, list_videos
from dfd.features import extract_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="name=folder pairs")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--frames", type=int, default=32, help="frames sampled per video (max 64)")
    a = ap.parse_args()
    for spec in a.data:
        name, root = spec.split("=", 1)
        for p, lab in tqdm(list_videos(root), desc=name):
            cp = cache_path(a.cache, name, p)
            if cp.exists() and "blend" in np.load(cp, allow_pickle=False).files:
                continue
            try:
                rec = extract_video(p, a.frames)
            except Exception as e:  # keep going on a corrupt file
                print("SKIP", p, e)
                continue
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cp, label=lab, path=str(p), **rec)


if __name__ == "__main__":
    main()
