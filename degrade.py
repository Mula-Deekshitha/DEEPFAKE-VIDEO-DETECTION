"""Image degradations used for (a) the robustness test and (b) optional training augmentation.
All functions take and return (H, W, 3) uint8 arrays. Pure numpy/OpenCV, no PyTorch needed."""
import cv2
import numpy as np

KINDS = ("blur", "downscale", "jpeg", "noise")
LEVELS = {"blur": (1.0, 2.0, 3.0),            # Gaussian sigma in pixels (crops are 160 px)
          "downscale": (0.5, 0.35, 0.25),     # shrink to this fraction, then scale back up
          "jpeg": (70, 40, 20),               # JPEG quality (lower = worse)
          "noise": (5, 10, 20)}               # Gaussian noise sigma on the 0-255 scale
RANGES = {"blur": (0.5, 3.0), "downscale": (0.25, 0.8), "jpeg": (20, 90), "noise": (2, 20)}
CONDITIONS = [("clean", None, None)] + [(f"{k}-{lv}", k, lv) for k in KINDS for lv in LEVELS[k]]


def apply(im, kind, level, rng=None):
    if kind == "blur":
        return cv2.GaussianBlur(im, (0, 0), float(level))
    if kind == "downscale":
        h, w = im.shape[:2]
        s = (max(8, int(w * level)), max(8, int(h * level)))
        small = cv2.resize(im, s, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if kind == "noise":
        rng = rng if rng is not None else np.random.RandomState(0)
        return np.clip(im.astype(np.float32) + rng.randn(*im.shape) * float(level), 0, 255).astype(np.uint8)
    raise ValueError(f"unknown degradation: {kind}")


def random_degrade(im, rng, p=0.7):
    """Training augmentation: with probability p apply one random degradation of random strength."""
    if rng.rand() >= p:
        return im
    kind = KINDS[rng.randint(len(KINDS))]
    lo, hi = RANGES[kind]
    return apply(im, kind, rng.uniform(lo, hi), rng)


def degrade_array(X, kind, level, seed=0):
    """X: (N, T, S, S, 3) uint8 -> degraded copy of the same shape."""
    rng = np.random.RandomState(seed)
    out = np.empty_like(X)
    N, T = X.shape[:2]
    for i in range(N):
        for t in range(T):
            out[i, t] = apply(X[i, t], kind, level, rng)
    return out
