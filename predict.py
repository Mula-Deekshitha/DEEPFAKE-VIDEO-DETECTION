"""Step 4: classify new videos.
Example: python predict.py clip1.mp4 clip2.mp4"""
import argparse

import joblib

from dfd.features import extract_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="models/ensemble.joblib")
    a = ap.parse_args()
    ens = joblib.load(a.model)
    for v in a.videos:
        rec = extract_video(v)
        each = ens.predict_each([rec])
        p = float(sum(x[0] for x in each.values()) / len(each))
        detail = "  ".join(f"{k}={x[0]:.3f}" for k, x in each.items())
        print(f"{v}: {'FAKE' if p > 0.5 else 'REAL'}  (P(fake)={p:.3f})  [{detail}]")


if __name__ == "__main__":
    main()
