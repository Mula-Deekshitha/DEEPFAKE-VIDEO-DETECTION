"""Video-level classifiers on the cached features and their ensemble.

Four diverse learners, probabilities averaged:
  fhgb : gradient boosting trained on individual FRAMES (32x more samples), video score = mean frame score
  hgb  : gradient boosting on hand-crafted aggregate statistics (+ PCA of embeddings)
  lr   : regularised logistic regression on the same features
  tnet : small Transformer over the per-frame sequence [embedding | geometry | geometry-motion | spectrum]
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # sklearn-only models still work
    HAS_TORCH = False
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def aggregate(r):
    lm, fr, q, bl = r["lm"], r["freq"], r["qual"], r["blend"]
    dlm = np.abs(np.diff(lm, axis=0))
    dfr = np.abs(np.diff(fr, axis=0))
    hand = np.concatenate([lm.mean(0), lm.std(0), dlm.mean(0), fr.mean(0), fr.std(0), dfr.mean(0),
                           q.mean(0), q.std(0), bl.mean(0), bl.std(0), np.median(bl, 0), [r["valid"].mean()]])
    e = r["emb"].astype(np.float32)
    return np.nan_to_num(np.concatenate([hand, e.mean(0), e.std(0)])), len(hand)


def sequence(r):
    lm, fr, q, bl = r["lm"], r["freq"], r["qual"], r["blend"]
    dlm = np.vstack([np.zeros((1, lm.shape[1]), np.float32), np.diff(lm, axis=0)])
    dfr = np.vstack([np.zeros((1, fr.shape[1]), np.float32), np.diff(fr, axis=0)])
    return r["emb"].astype(np.float32), np.nan_to_num(np.concatenate([lm, dlm, fr, dfr, q, bl], axis=1))


if HAS_TORCH:
    class TemporalNet(nn.Module):
        def __init__(self, d_emb, d_geo, d=96, heads=4, layers=2, drop=0.3):
            super().__init__()

            def proj(i):
                return nn.Sequential(nn.Dropout(drop), nn.Linear(i, d), nn.LayerNorm(d), nn.GELU())

            self.pe, self.pg = proj(d_emb), proj(d_geo)
            self.fuse = nn.Linear(2 * d, d)
            self.pos = nn.Parameter(torch.zeros(1, 64, d))
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, 1))

        def forward(self, e, g):
            h = self.fuse(torch.cat([self.pe(e), self.pg(g)], -1)) + self.pos[:, :e.shape[1]]
            h = self.enc(h)
            return self.head(torch.cat([h.mean(1), h.max(1).values], -1)).squeeze(-1)


def frame_matrix(r):
    """(T, D) per-frame matrix: embedding first (EMB columns), then everything else."""
    return np.nan_to_num(np.concatenate([r["emb"].astype(np.float32), r["lm"], r["freq"], r["qual"], r["blend"]], 1))


class Ensemble:
    def __init__(self, seeds=3, epochs=80, use=("hgb", "lr", "fhgb", "tnet")):
        self.seeds, self.epochs, self.use = seeds, epochs, tuple(use)
        if "tnet" in self.use and not HAS_TORCH:
            raise SystemExit("PyTorch is not installed; install it or pass --models hgb lr")
        self.dev = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    # ---------- training ----------
    def fit(self, recs, y):
        y = np.asarray(y)
        if "fhgb" in self.use:
            F0 = np.concatenate([frame_matrix(r) for r in recs])
            yf = np.repeat(y, [len(r["valid"]) for r in recs])
            de = recs[0]["emb"].shape[1]
            self.de = de
            self.fm = Pipeline([
                ("pre", ColumnTransformer([
                    ("emb", Pipeline([("s", StandardScaler()), ("p", PCA(32, random_state=0))]), slice(0, de)),
                    ("rest", "passthrough", slice(de, None))])),
                ("clf", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4,
                                                       l2_regularization=2.0, early_stopping=False,
                                                       random_state=0))]).fit(F0, yf)
        if "hgb" in self.use or "lr" in self.use:
            rows = [aggregate(r) for r in recs]
            X, H = np.stack([a for a, _ in rows]), rows[0][1]
            ncomp = int(min(24, len(y) - 2, X.shape[1] - H))

            def pre():
                return ColumnTransformer([
                    ("hand", "passthrough", slice(0, H)),
                    ("emb", Pipeline([("s", StandardScaler()), ("p", PCA(ncomp, random_state=0))]), slice(H, None))])

            self.m = {}
            if "hgb" in self.use:
                self.m["hgb"] = Pipeline([("pre", pre()), ("clf", HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=0))]).fit(X, y)
            if "lr" in self.use:
                self.m["lr"] = Pipeline([("pre", pre()), ("s", StandardScaler()), ("clf", LogisticRegression(
                    C=0.05, max_iter=3000))]).fit(X, y)
        if "tnet" in self.use:
            seqs = [sequence(r) for r in recs]
            E = np.stack([s[0] for s in seqs])
            G = np.stack([s[1] for s in seqs])
            self.em, self.es = E.reshape(-1, E.shape[2]).mean(0), E.reshape(-1, E.shape[2]).std(0) + 1e-6
            self.gm, self.gs = G.reshape(-1, G.shape[2]).mean(0), G.reshape(-1, G.shape[2]).std(0) + 1e-6
            E, G = (E - self.em) / self.es, (G - self.gm) / self.gs
            self.nets = [self._train_net(E, G, y, s) for s in range(self.seeds)]
            for n in self.nets:
                n.cpu()
        return self

    def _train_net(self, E, G, y, seed):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        T = E.shape[1]
        net = TemporalNet(E.shape[2], G.shape[2]).to(self.dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Et = torch.tensor(E, dtype=torch.float32, device=self.dev)
        Gt = torch.tensor(G, dtype=torch.float32, device=self.dev)
        yt = torch.tensor(y, dtype=torch.float32, device=self.dev)
        pw = torch.tensor(float((1 - y).sum() / max(y.sum(), 1)), device=self.dev)
        for _ in range(self.epochs):
            net.train()
            perm = rng.permutation(len(y))
            for i in range(0, len(y), 16):
                b = torch.tensor(perm[i:i + 16], device=self.dev)
                L = rng.randint(T // 2, T + 1)
                s = rng.randint(0, T - L + 1)
                e = Et[b, s:s + L]
                g = Gt[b, s:s + L]
                e = e + 0.1 * torch.randn_like(e)
                g = g + 0.1 * torch.randn_like(g)
                loss = F.binary_cross_entropy_with_logits(net(e, g), yt[b] * 0.9 + 0.05, pos_weight=pw)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return net

    # ---------- inference ----------
    def predict_each(self, recs):
        out = {}
        if "fhgb" in self.use:
            out["fhgb"] = self.predict_frames(recs).mean(1)
        if "hgb" in self.use or "lr" in self.use:
            X = np.stack([aggregate(r)[0] for r in recs])
            for k, m in self.m.items():
                out[k] = m.predict_proba(X)[:, 1]
        if "tnet" in self.use:
            seqs = [sequence(r) for r in recs]
            E = (np.stack([s[0] for s in seqs]) - self.em) / self.es
            G = (np.stack([s[1] for s in seqs]) - self.gm) / self.gs
            Et = torch.tensor(E, dtype=torch.float32, device=self.dev)
            Gt = torch.tensor(G, dtype=torch.float32, device=self.dev)
            ps = []
            for n in self.nets:
                n.to(self.dev).eval()
                with torch.no_grad():
                    ps.append(torch.sigmoid(n(Et, Gt)).cpu().numpy())
                n.cpu()
            out["tnet"] = np.mean(ps, axis=0)
        return out

    def predict_frames(self, recs):
        """(N, T) per-frame P(fake) from the frame-level model."""
        T = len(recs[0]["valid"])
        p = self.fm.predict_proba(np.concatenate([frame_matrix(r) for r in recs]))[:, 1]
        return p.reshape(len(recs), T)

    def predict_proba(self, recs):
        each = self.predict_each(recs)
        return np.mean(list(each.values()), axis=0)
