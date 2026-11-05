"""Per-video feature extraction.

Three complementary views of every sampled frame:
  * emb  : frozen ImageNet EfficientNet-B0 embedding of the face crop (spatial artefacts)
  * lm   : MediaPipe Face Mesh (468 pts) normalised geometry (distances, eye/mouth ratios, pose)
  * freq : radial log-power spectrum of the face crop (up-sampling / blending artefacts)
plus simple quality statistics.
"""
import cv2
import numpy as np

T_FRAMES = 32
CROP = 224
PTS = [33, 133, 362, 263, 1, 168, 61, 291, 13, 14, 152, 10, 234, 454, 105, 334, 159, 145, 386, 374]
IU = np.triu_indices(len(PTS), 1)
N_LM = len(IU[0]) + 9
N_FREQ = 32
N_QUAL = 4
N_BLEND = 10

_mesh = None
_net = None
_dev = None


def _get_mesh():
    global _mesh
    if _mesh is None:
        import mediapipe as mp
        _mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)
    return _mesh


def _get_net():
    global _net, _dev
    if _net is None:
        import torch
        import torchvision
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = torch.nn.Identity()
        _net = m.eval().to(_dev)
    return _net, _dev


def sample_frames(path, n=T_FRAMES):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        want = np.linspace(0, total - 1, n).round().astype(int)
        need = set(want.tolist())
        store, i = {}, 0
        last = max(need)
        while i <= last:
            if not cap.grab():
                break
            if i in need:
                ok, fr = cap.retrieve()
                if ok:
                    store[i] = fr
            i += 1
        frames = [store[i] for i in want if i in store]
    cap.release()
    if not frames:  # frame count unreliable: read everything
        cap = cv2.VideoCapture(str(path))
        allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr)
        cap.release()
        if not allf:
            raise RuntimeError("could not read any frame")
        frames = [allf[i] for i in np.linspace(0, len(allf) - 1, n).round().astype(int)]
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


def _geo(pts):
    P = pts[PTS]
    iod = np.linalg.norm(pts[33] - pts[263]) + 1e-6

    def d(a, b):
        return np.linalg.norm(pts[a] - pts[b])

    dist = np.linalg.norm(P[:, None] - P[None], axis=-1)[IU] / iod
    extra = [
        d(159, 145) / (d(33, 133) + 1e-6), d(386, 374) / (d(362, 263) + 1e-6),
        d(13, 14) / (d(61, 291) + 1e-6), d(61, 291) / iod, d(152, 10) / (d(234, 454) + 1e-6),
        (pts[1, 0] - pts[234, 0]) / (pts[454, 0] - pts[234, 0] + 1e-6),
        np.arctan2(pts[263, 1] - pts[33, 1], pts[263, 0] - pts[33, 0]),
        d(105, 159) / iod, d(334, 386) / iod,
    ]
    return np.concatenate([dist, extra]).astype(np.float32)


def _freq(crop):
    g = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = (g - g.mean()) * np.outer(np.hanning(128), np.hanning(128))
    ps = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    yy, xx = np.indices(ps.shape)
    r = np.minimum(np.hypot(yy - 64, xx - 64).astype(int), 63)
    rad = np.bincount(r.ravel(), ps.ravel(), minlength=64) / np.maximum(np.bincount(r.ravel(), minlength=64), 1)
    return rad.reshape(32, 2).mean(1).astype(np.float32)


def _crop_face(frame, pts):
    """Returns (crop, (x0, y0))."""
    h, w = frame.shape[:2]
    if pts is None:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return frame[y0:y0 + s, x0:x0 + s], (x0, y0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    side = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.3
    x0, x1 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, w))
    y0, y1 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, h))
    return frame[y0:y1, x0:x1], (x0, y0)


def _blend(crop, pts, off):
    """Face-swap boundary statistics: compares the face interior, a band along the face outline and the
    surrounding ring (sharpness, colour, edge density). Swapped faces rarely match their surroundings."""
    if pts is None or crop.size == 0:
        return np.zeros(N_BLEND, np.float32)
    S = 128
    ch, cw = crop.shape[:2]
    p = (pts - np.array(off, np.float32)) * np.array([S / cw, S / ch], np.float32)
    mask = np.zeros((S, S), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(p.astype(np.int32)), 1)
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(mask, k, iterations=5)
    band = cv2.dilate(mask, k, iterations=5) - cv2.erode(mask, k, iterations=5)
    outer = cv2.dilate(mask, k, iterations=20) - cv2.dilate(mask, k, iterations=5)
    if inner.sum() < 50 or band.sum() < 50 or outer.sum() < 50:
        return np.zeros(N_BLEND, np.float32)
    c = cv2.resize(crop, (S, S))
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F) ** 2
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = cv2.Canny(g, 60, 150).astype(np.float32) / 255.0

    def m(a, msk):
        return float(a[msk > 0].mean())

    li, lb, lo = (np.log1p(m(lap, x)) for x in (inner, band, outer))
    ci = [lab[..., j][inner > 0].mean() for j in range(3)]
    co = [lab[..., j][outer > 0].mean() for j in range(3)]
    si, so = lab[..., 0][inner > 0].std(), lab[..., 0][outer > 0].std()
    return np.array([li, lb, lo, lb - li, lo - li, ci[0] - co[0], ci[1] - co[1], ci[2] - co[2],
                     np.log((si + 1) / (so + 1)), m(edge, band) - m(edge, inner)], np.float32)


def extract_video(path, n=T_FRAMES):
    import torch
    frames = sample_frames(path, n)
    mesh = _get_mesh()
    lm = np.zeros((n, N_LM), np.float32)
    freq = np.zeros((n, N_FREQ), np.float32)
    qual = np.zeros((n, N_QUAL), np.float32)
    blend = np.zeros((n, N_BLEND), np.float32)
    valid = np.zeros(n, np.float32)
    crops = []
    for i, fr in enumerate(frames):
        h0, w0 = fr.shape[:2]
        sc = 640.0 / max(h0, w0)
        if sc < 1:
            fr = cv2.resize(fr, (int(w0 * sc), int(h0 * sc)))
        h, w = fr.shape[:2]
        res = mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        pts = None
        if res.multi_face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark], np.float32)
            lm[i] = _geo(pts)
            valid[i] = 1
        crop, off = _crop_face(fr, pts)
        if crop.size == 0:
            crop, off, pts_b = fr, (0, 0), None
        else:
            pts_b = pts
        blend[i] = _blend(crop, pts_b, off)
        freq[i] = _freq(crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        qual[i] = [np.log1p(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
                   hsv[..., 2].mean() / 255, hsv[..., 1].mean() / 255, valid[i]]
        crops.append(cv2.resize(crop, (CROP, CROP)))
    # fill landmark rows of frames with no face using the mean of detected frames
    if valid.sum() > 0:
        lm[valid == 0] = lm[valid == 1].mean(0)
    net, dev = _get_net()
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        emb = torch.cat([net(x[i:i + 16]) for i in range(0, n, 16)]).cpu().numpy()
    return dict(emb=emb.astype(np.float16), lm=lm, freq=freq, qual=qual, blend=blend, valid=valid)
