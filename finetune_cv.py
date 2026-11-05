"""Step B: fine-tune an ImageNet EfficientNet-B0 end-to-end on face crops, evaluated with the same
video-grouped cross-validation as evaluate.py (every score comes from videos unseen in training).
Video score = mean of the per-frame probabilities.

Example: python finetune_cv.py --dataset uadfv --epochs 6
Optional: --group-regex "^(\\d+)" keeps a real video and the fake made from it in the same fold (stricter)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import StratifiedGroupKFold

from degrade import random_degrade
from dfd.data import groups_from_paths
from evaluate import metrics, show

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CROP = 144


def load(dataset, folder, step=1):
    files = sorted((Path(folder) / dataset).glob("*.npz"))
    if not files:
        raise SystemExit(f"No crops in {folder}/{dataset}. Run crops.py first.")
    first = np.load(files[0], allow_pickle=False)["crops"][::step]
    X = np.empty((len(files),) + first.shape, np.uint8)  # preallocated: avoids a 2x memory peak
    y, p = [], []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        X[i] = z["crops"][::step][:first.shape[0]]
        y.append(int(z["label"]))
        p.append(str(z["path"]))
    return X, np.array(y), p


def to_tensor(b, train, rng, flip=False, aug=False):
    """b: (B,S,S,3) uint8 -> normalised (B,3,CROP,CROP)."""
    S = b.shape[1]
    out = []
    for im in b:
        if train:
            if aug:
                im = random_degrade(im, rng)
            y0, x0 = rng.randint(0, S - CROP + 1, 2)
            im = im[y0:y0 + CROP, x0:x0 + CROP]
            if rng.rand() < 0.5:
                im = im[:, ::-1]
        else:
            o = (S - CROP) // 2
            im = im[o:o + CROP, o:o + CROP]
            if flip:
                im = im[:, ::-1]
        out.append(np.ascontiguousarray(im))
    t = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    if train:  # brightness / contrast jitter
        t = t * (1 + 0.2 * (torch.rand(len(t), 1, 1, 1) - 0.5)) + 0.1 * (torch.rand(len(t), 1, 1, 1) - 0.5)
        t = t.clamp(0, 1)
    return (t - MEAN) / STD


def make_net():
    m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 1))
    return m


def train_fold(X, y, args, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(y) * args.fpv / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    T = X.shape[1]
    aug = getattr(args, "aug", "none") == "strong"
    for ep in range(args.epochs):
        net.train()
        idx = np.repeat(np.arange(len(y)), args.fpv)
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            fr = rng.randint(0, T, len(b))
            xb = to_tensor(X[b, fr], True, rng, False, aug).to(dev)
            yb = torch.tensor(y[b], dtype=torch.float32, device=dev) * 0.9 + 0.05
            loss = F.binary_cross_entropy_with_logits(net(xb).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += loss.item() * len(b)
        print(f"    epoch {ep + 1}/{args.epochs} train loss {tot / len(idx):.4f}")
    return net


@torch.no_grad()
def predict(net, X, dev, tta=False):
    """(N,T) frame probabilities; tta=True averages the frame and its mirror image."""
    net.eval()
    N, T = X.shape[:2]
    flat = X.reshape(N * T, *X.shape[2:])
    out = 0
    flips = (False, True) if tta else (False,)
    for fl in flips:
        ps = []
        for i in range(0, len(flat), 64):
            ps.append(torch.sigmoid(net(to_tensor(flat[i:i + 64], False, None, fl).to(dev)).squeeze(1)).cpu().numpy())
        out = out + np.concatenate(ps)
    return (out / len(flips)).reshape(N, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, help="one or more datasets (several = pooled)")
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fpv", type=int, default=8, help="random frames per video per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--nets", type=int, default=1, help="networks trained per fold (their predictions are averaged)")
    ap.add_argument("--aug", choices=["none", "strong"], default="none", help="training augmentation with blur/downscale/JPEG/noise")
    ap.add_argument("--tta", action="store_true", help="average each frame with its mirror image at test time")
    ap.add_argument("--step", type=int, default=1, help="use every N-th stored frame (2 halves memory)")
    ap.add_argument("--out", default="results_ft")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xs, ys, gs, tags, off = [], [], [], [], 0
    for d in a.dataset:
        Xd, yd, pd_ = load(d, a.crops, a.step)
        gd = groups_from_paths(pd_, a.group_regex) + off  # groups never mix across datasets
        off = gd.max() + 1
        Xs.append(Xd)
        ys.append(yd)
        gs.append(gd)
        tags += [d] * len(yd)
    X, y, groups, tags = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.array(tags)
    del Xs
    print(f"{'+'.join(a.dataset)}: {len(y)} videos, device={dev}")
    skf = StratifiedGroupKFold(a.folds, shuffle=True, random_state=a.seed)
    fr = np.zeros(X.shape[:2])
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        print(f"  fold {f + 1}/{a.folds}")
        pr = []
        for k in range(a.nets):
            net = train_fold(X[tr], y[tr], a, dev, a.seed + f + 100 * k)
            pr.append(predict(net, X[te], dev, a.tta))
        fr[te] = np.mean(pr, axis=0)
        pv = fr[te].mean(1)
        print(f"    fold video accuracy {((pv > 0.5) == y[te]).mean():.3f}")
    res = {"finetune-video": metrics(y, fr.mean(1)),
           "finetune-frames": metrics(np.repeat(y, X.shape[1]), fr.ravel())}
    name = "+".join(a.dataset)
    show(f"{name} - fine-tuned EfficientNet-B0, {a.folds}-fold video-grouped CV", res)
    if len(a.dataset) > 1:
        for d in a.dataset:
            m = tags == d
            sub = {"video": metrics(y[m], fr[m].mean(1)), "frames": metrics(np.repeat(y[m], fr.shape[1]), fr[m].ravel())}
            res[f"only:{d}"] = sub
            show(f"  of which {d} videos", sub)
    Path(a.out).mkdir(exist_ok=True)
    np.save(Path(a.out) / f"oof_{name}.npy", fr)
    json.dump(res, open(Path(a.out) / f"metrics_{name}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
