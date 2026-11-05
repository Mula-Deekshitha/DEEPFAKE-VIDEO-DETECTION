"""Dataset listing, label inference and cached-feature loading."""
import hashlib
import re
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FAKE_WORDS = ("fake", "forged", "manipulated", "synthe", "swap", "generated")
REAL_WORDS = ("real", "original", "authentic", "pristine", "genuine")


def _label_from_text(text):
    t = text.lower()
    if any(w in t for w in FAKE_WORDS):
        return 1
    if any(w in t for w in REAL_WORDS):
        return 0
    return None


def infer_label(path, root):
    """Label from folder names (nearest folder first), then file name. 1 = fake, 0 = real."""
    parts = Path(path).relative_to(root).parts
    for part in reversed(parts[:-1]):
        lab = _label_from_text(part)
        if lab is not None:
            return lab
    return _label_from_text(parts[-1])


def list_videos(root):
    root = Path(root)
    out, unknown = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXT:
            lab = infer_label(p, root)
            (out if lab is not None else unknown).append((p, lab) if lab is not None else p)
    if unknown:
        print(f"WARNING: {len(unknown)} videos under {root} have no 'real'/'fake' in their folder or "
              f"file name and were skipped, e.g. {unknown[0]}")
    n_fake = sum(l for _, l in out)
    print(f"{root}: {len(out)} videos ({len(out) - n_fake} real, {n_fake} fake)")
    return out


def cache_path(cache_dir, dataset, video_path):
    h = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    return Path(cache_dir) / dataset / f"{Path(video_path).stem}_{h}.npz"


def load_cached(dataset, cache_dir):
    """Returns (records, labels, paths) from cache/<dataset>/*.npz."""
    files = sorted((Path(cache_dir) / dataset).glob("*.npz"))
    if not files:
        raise SystemExit(f"No cached features for '{dataset}' in {cache_dir}/{dataset}. Run extract.py first.")
    recs, ys, paths = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if "blend" not in z.files:
            raise SystemExit(f"{f} was made by an older version (no 'blend' features). "
                             f"Delete the cache folder or run extract.py again.")
        recs.append({k: z[k] for k in ("emb", "lm", "freq", "qual", "blend", "valid")})
        ys.append(int(z["label"]))
        paths.append(str(z["path"]))
    return recs, np.array(ys), paths


def groups_from_paths(paths, regex=None):
    """Group ids so that videos sharing a source never straddle train/test.
    regex: pattern whose first capture group is the source id taken from the file stem
    (e.g. '^(\\d+)' for names like 12_34.mp4). Without it every video is its own group."""
    if not regex:
        return np.arange(len(paths))
    rx = re.compile(regex)
    ids = {}
    out = []
    for p in paths:
        m = rx.search(Path(p).stem)
        key = next((g for g in m.groups() if g), p) if m else p  # first capture group that matched
        out.append(ids.setdefault(key, len(ids)))
    return np.array(out)
