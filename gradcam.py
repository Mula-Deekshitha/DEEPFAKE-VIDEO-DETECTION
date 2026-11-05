"""Heatmaps for the fine-tuned EfficientNet-B0 (which face regions push a frame towards FAKE) plus a check of
whether the highlighted region really drives the prediction.

* cam_from / campp_from : Grad-CAM and Grad-CAM++ maths (numpy)
* mask_from_cam, apply_mask, faithfulness : deletion test (numpy; the scoring function is passed in)
* explain : runs the network and returns a heatmap (needs PyTorch)

Heatmap = evidence the network used. The deletion test measures cause and effect on the score: hiding the
highlighted pixels should lower P(fake) more than hiding the same amount of random area."""
import cv2
import numpy as np


def _norm(c):
    m = c.max()
    return c / m if m > 0 else c


def cam_from(acts, grads):
    """Grad-CAM. acts, grads: (C, h, w) feature maps and d(fake-logit)/d(maps) -> (h, w) in [0, 1]."""
    weights = grads.mean(axis=(1, 2))
    return _norm(np.maximum((weights[:, None, None] * acts).sum(0), 0))


def campp_from(acts, grads):
    """Grad-CAM++: pixel-wise weights from higher-order gradients, so several separate regions and small
    regions are highlighted better than with plain Grad-CAM."""
    g2, g3 = grads ** 2, grads ** 3
    denom = 2 * g2 + acts.sum(axis=(1, 2), keepdims=True) * g3
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    weights = ((g2 / denom) * np.maximum(grads, 0)).sum(axis=(1, 2))
    return _norm(np.maximum((weights[:, None, None] * acts).sum(0), 0))


def resize_cam(cam, size):
    return np.clip(cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC), 0, 1)


def overlay(img, cam, alpha=0.45):
    """Blend a [0, 1] heatmap (any size) over an RGB uint8 image."""
    c = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (img.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)


def mask_from_cam(cam, size, frac):
    """Boolean (size, size) mask of exactly the hottest `frac` of the pixels."""
    c = resize_cam(cam, size).ravel()
    k = max(1, int(round(frac * size * size)))
    m = np.zeros(size * size, bool)
    m[np.argsort(-c, kind="stable")[:k]] = True
    return m.reshape(size, size)


def apply_mask(img, mask, fill=None):
    """Replace the masked pixels by the image's mean colour."""
    out = img.copy()
    fill = img.reshape(-1, 3).mean(0) if fill is None else fill
    out[mask] = np.asarray(fill).astype(img.dtype)
    return out


def faithfulness(score_fn, img, cam, frac=0.2, n_random=8, seed=0):
    """Deletion test on one frame. score_fn: (B, S, S, 3) uint8 -> (B,) P(fake).
    Returns None if the heatmap is empty (nothing pushed towards FAKE), else the score before, after hiding the
    hottest `frac` of the face, and after hiding equally large random blobs (average of n_random)."""
    if cam.max() <= 0:
        return None
    size = img.shape[0]
    rng = np.random.RandomState(seed)
    batch = [img, apply_mask(img, mask_from_cam(cam, size, frac))]
    for _ in range(n_random):
        blob = cv2.resize(rng.rand(9, 9).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        batch.append(apply_mask(img, mask_from_cam(blob, size, frac)))
    s = np.asarray(score_fn(np.stack(batch)), float)
    rnd = float(s[2:].mean())
    return dict(original=float(s[0]), removed=float(s[1]), random=rnd,
                drop=float(s[0] - s[1]), random_drop=float(s[0] - rnd))


def explain(net, crop, dev, method="gradcam++", layers=(5, 8)):
    """crop: (S, S, 3) uint8 RGB face crop as stored by crops.py. Heatmap over the centre CROP x CROP region,
    averaged over a mid-level (finer, 9x9) and the last (coarser, 5x5) feature map. Returns (cam, P(fake))."""
    import torch
    from finetune_cv import CROP, to_tensor
    net.eval()
    x = to_tensor(crop[None], False, None).to(dev)
    acts = {}
    with torch.enable_grad():
        h = x
        for i, block in enumerate(net.features):
            h = block(h)
            if i in layers:
                h.retain_grad()
                acts[i] = h
        logit = net.classifier(torch.flatten(net.avgpool(h), 1)).squeeze()
        net.zero_grad()
        logit.backward()
    f = campp_from if method == "gradcam++" else cam_from
    cams = [resize_cam(f(a[0].detach().cpu().numpy(), a.grad[0].cpu().numpy()), CROP) for a in acts.values()]
    return _norm(np.mean(cams, axis=0)), float(torch.sigmoid(logit.detach()))
