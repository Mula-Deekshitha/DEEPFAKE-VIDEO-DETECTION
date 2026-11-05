"""Checks that need no PyTorch and no data: python selftest.py"""
import numpy as np

from degrade import CONDITIONS, KINDS, LEVELS, apply, degrade_array, random_degrade
from dfd.data import groups_from_paths
from gradcam import apply_mask, campp_from, cam_from, faithfulness, mask_from_cam, overlay

rng = np.random.RandomState(0)
img = (rng.rand(160, 160, 3) * 255).astype(np.uint8)
lap = lambda a: float(np.var(np.diff(a.astype(np.float32), axis=0)))

# degradations keep shape/dtype, and do what they say
for kind in KINDS:
    for lv in LEVELS[kind]:
        out = apply(img, kind, lv, np.random.RandomState(1))
        assert out.shape == img.shape and out.dtype == np.uint8, (kind, lv)
assert lap(apply(img, "blur", 3.0)) < lap(img) * 0.2, "blur should remove detail"
assert lap(apply(img, "downscale", 0.25)) < lap(img) * 0.5, "downscale should remove detail"
assert not np.array_equal(apply(img, "jpeg", 20), img), "jpeg should change pixels"
assert abs(float(apply(img, "noise", 20, np.random.RandomState(1)).astype(float).std() - img.astype(float).std())) >= 0
smooth = np.full((160, 160, 3), 128, np.uint8)
assert lap(apply(smooth, "noise", 20, np.random.RandomState(1))) > 100, "noise should add variance"
assert [c[0] for c in CONDITIONS][0] == "clean" and len(CONDITIONS) == 13
X = np.stack([[img, img]] * 3)
D = degrade_array(X, "blur", 2.0, seed=0)
assert D.shape == X.shape and D.dtype == np.uint8 and not np.array_equal(D, X)
assert np.array_equal(degrade_array(X, "noise", 10, seed=3), degrade_array(X, "noise", 10, seed=3)), "seeded = repeatable"
changed = sum(not np.array_equal(random_degrade(img, np.random.RandomState(s)), img) for s in range(200))
assert 100 < changed < 180, f"about 70% of images should be degraded, got {changed}/200"

# Grad-CAM maths
acts = np.zeros((4, 5, 5), np.float32)
acts[0, 2, 2] = 1.0
grads = np.zeros_like(acts)
grads[0] = 1.0
cam = cam_from(acts, grads)
assert cam.shape == (5, 5) and cam.max() == 1.0 and cam[2, 2] == 1.0 and cam.sum() == 1.0
assert cam_from(acts, -grads).max() == 0.0, "negative evidence must be clipped"
ov = overlay(img[:144, :144], cam)
assert ov.shape == (144, 144, 3) and ov.dtype == np.uint8

# Grad-CAM++ : peaks where positive evidence is, ignores negative evidence, stays in [0, 1]
acts2 = np.zeros((2, 5, 5), np.float32); acts2[0, 2, 2] = 1.0; acts2[1, 0, 0] = 0.2
grads2 = np.zeros_like(acts2); grads2[0] = 0.5; grads2[1] = 0.1
cpp = campp_from(acts2, grads2)
assert cpp.shape == (5, 5) and cpp.min() >= 0 and abs(cpp.max() - 1) < 1e-6 and np.unravel_index(cpp.argmax(), cpp.shape) == (2, 2)
assert campp_from(acts2, -grads2).max() == 0.0 and np.isfinite(campp_from(np.zeros((2, 5, 5)), np.zeros((2, 5, 5)))).all()

# deletion test: masks have exactly the requested area; a heatmap on the evidence beats a heatmap elsewhere
m = mask_from_cam(rng.rand(9, 9), 144, 0.2)
assert m.sum() == round(0.2 * 144 * 144)
face = np.full((144, 144, 3), 100, np.uint8)
face[:72, :72, 0] = 220                                   # the "fake evidence" sits in the top-left quadrant
score_fn = lambda imgs: imgs[:, :72, :72, 0].astype(float).mean(axis=(1, 2)) / 255.0
good = np.zeros((9, 9)); good[:4, :4] = 1.0               # heatmap on the evidence
bad = np.zeros((9, 9)); bad[5:, 5:] = 1.0                 # heatmap far from it
fg, fb = faithfulness(score_fn, face, good, 0.15), faithfulness(score_fn, face, bad, 0.15)
assert fg["drop"] > fg["random_drop"] + 0.05, fg
assert abs(fb["drop"]) < 1e-9 and fb["random_drop"] > 0, fb
assert faithfulness(score_fn, face, np.zeros((9, 9))) is None
assert apply_mask(face, m).shape == face.shape and (apply_mask(face, m)[m] == face.reshape(-1, 3).mean(0).astype(np.uint8)).all()

# grouping patterns used in the README
paths = ["u/0000.mp4", "u/0000_fake.mp4", "u/0001.mp4", "s/real_v10_aug_0.mp4", "s/fake_vs10_aug_1-Ab.mp4", "s/real_v11_aug_0.mp4"]
g = groups_from_paths(paths, r"^(\d+)|vs?(\d+)_aug")
assert g[0] == g[1] != g[2] and g[3] == g[4] != g[5]
print("selftest passed")
