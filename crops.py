"""Step A (for fine-tuning): save 32 aligned face crops per video as small uint8 arrays.
Example: python crops.py --data uadfv=C:\\path\\UADFV"""
import argparse

import cv2
import numpy as np

from dfd.data import cache_path, list_videos
from dfd.features import _crop_face, _get_mesh, sample_frames

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x


def video_crops(path, n, size, return_faces=False):
    """(T,size,size,3) uint8 RGB crops; with return_faces also a bool array: face found in that frame."""
    mesh = _get_mesh()
    out, found = [], []
    for fr in sample_frames(path, n):
        h0, w0 = fr.shape[:2]
        sc = 640.0 / max(h0, w0)
        if sc < 1:
            fr = cv2.resize(fr, (int(w0 * sc), int(h0 * sc)))
        h, w = fr.shape[:2]
        res = mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        pts = None
        if res.multi_face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark], np.float32)
        crop, _ = _crop_face(fr, pts)
        found.append(pts is not None and crop.size > 0)
        if crop.size == 0:
            crop = fr
        out.append(cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB))
    arr = np.stack(out).astype(np.uint8)
    return (arr, np.array(found)) if return_faces else arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="name=folder pairs")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=160)
    a = ap.parse_args()
    for spec in a.data:
        name, root = spec.split("=", 1)
        for p, lab in tqdm(list_videos(root), desc=name):
            cp = cache_path(a.out, name, p)
            if cp.exists():
                continue
            try:
                c = video_crops(p, a.frames, a.size)
            except Exception as e:
                print("SKIP", p, e)
                continue
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cp, label=lab, path=str(p), crops=c)


if __name__ == "__main__":
    main()
