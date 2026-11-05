"""Step 2: honest evaluation.

Every reported number comes from videos the model never saw in training:
  * cv    : stratified, group-aware 5-fold cross-validation at VIDEO level (out-of-fold predictions)
  * cross : train on one dataset, test on the other (true cross-dataset generalisation)
  * --shuffle-check : repeats CV with shuffled labels; accuracy must collapse to ~50 % (leakage test)
Decision threshold is fixed at 0.5 (never tuned on test data).

Example: python evaluate.py --datasets sdfvd uadfv
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from dfd.data import groups_from_paths, load_cached
from dfd.models import Ensemble


def metrics(y, p):
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def run_cv(recs, y, groups, args, shuffle=False, seed=None):
    seed = args.seed if seed is None else seed
    rng = np.random.RandomState(0)
    yy = rng.permutation(y) if shuffle else y
    k = int(min(args.folds, np.bincount(y).min()))
    skf = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    oof, fr = {}, None
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), yy, groups)):
        ens = Ensemble(args.seeds, args.epochs, args.models).fit([recs[i] for i in tr], yy[tr])
        tr_recs = [recs[i] for i in te]
        for name, p in {**ens.predict_each(tr_recs), "ensemble": ens.predict_proba(tr_recs)}.items():
            oof.setdefault(name, np.zeros(len(y)))[te] = p
        if "fhgb" in args.models:
            pf = ens.predict_frames(tr_recs)
            fr = np.zeros((len(y), pf.shape[1])) if fr is None else fr
            fr[te] = pf
        print(f"  fold {f + 1}/{k} done")
    vid = {n: metrics(yy, p) for n, p in oof.items()}
    frm = None
    if fr is not None:
        frm = {"fhgb-frames": metrics(np.repeat(yy, fr.shape[1]), fr.ravel())}
    return vid, frm


def avg(runs):
    """Mean over repeated runs; adds acc_std."""
    out = {}
    for name in runs[0]:
        keys = runs[0][name].keys()
        out[name] = {k: float(np.nanmean([r[name][k] for r in runs])) for k in keys}
        out[name]["acc_std"] = float(np.std([r[name]["accuracy"] for r in runs]))
    return out


def cv_repeated(recs, y, groups, args, shuffle=False):
    v, f = [], []
    for i in range(args.repeats):
        if args.repeats > 1:
            print(f" repeat {i + 1}/{args.repeats}")
        a, b = run_cv(recs, y, groups, args, shuffle, args.seed + i)
        v.append(a)
        if b:
            f.append(b)
    return avg(v), (avg(f) if f else None)


def show(title, res):
    print(f"\n=== {title} ===")
    print(f"{'model':12s} {'acc':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'auc':>7s}   TN FP FN TP")
    for n, m in res.items():
        sd = f" (+-{m['acc_std']:.3f})" if m.get("acc_std") else ""
        print(f"{n:12s} {m['accuracy']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} "
              f"{m['auc']:7.4f}   {m['tn']:.4g} {m['fp']:.4g} {m['fn']:.4g} {m['tp']:.4g}{sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-regex", default=None,
                    help="regex with one capture group giving the source-video id from a file stem")
    ap.add_argument("--models", nargs="+", default=["hgb", "lr", "fhgb", "tnet"], choices=["hgb", "lr", "fhgb", "tnet"])
    ap.add_argument("--repeats", type=int, default=1, help="repeat CV with different fold splits and average")
    ap.add_argument("--shuffle-check", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    data = {d: load_cached(d, a.cache) for d in a.datasets}
    results = {}
    for d, (r, y, p) in data.items():
        print(f"\n{d}: {len(y)} videos ({int((y == 0).sum())} real / {int((y == 1).sum())} fake)")
        results[f"cv:{d}"], frm = cv_repeated(r, y, groups_from_paths(p, a.group_regex), a)
        show(f"{d} - {a.folds}-fold video-level CV (x{a.repeats})", results[f"cv:{d}"])
        if frm:
            results[f"cv-frame:{d}"] = frm
            show(f"{d} - frame-level score (frames of unseen videos)", frm)
        if a.shuffle_check:
            res, _ = cv_repeated(r, y, groups_from_paths(p, a.group_regex), a, shuffle=True)
            results[f"shuffled:{d}"] = res
            show(f"{d} - SHUFFLED-LABEL sanity check (should be ~0.5)", res)
    if len(data) > 1:
        rs = sum((data[d][0] for d in data), [])
        ys = np.concatenate([data[d][1] for d in data])
        ps = sum((data[d][2] for d in data), [])
        results["cv:combined"], _ = cv_repeated(rs, ys, groups_from_paths(ps, a.group_regex), a)
        show("combined - video-level CV", results["cv:combined"])
        for src in data:
            for dst in data:
                if src == dst:
                    continue
                ens = Ensemble(a.seeds, a.epochs, a.models).fit(data[src][0], data[src][1])
                te = data[dst][0]
                res = {n: metrics(data[dst][1], p) for n, p in {**ens.predict_each(te),
                                                                "ensemble": ens.predict_proba(te)}.items()}
                results[f"cross:{src}->{dst}"] = res
                show(f"train on {src} -> test on {dst}", res)
    Path(a.out).mkdir(exist_ok=True)
    with open(Path(a.out) / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {a.out}/metrics.json")


if __name__ == "__main__":
    main()
