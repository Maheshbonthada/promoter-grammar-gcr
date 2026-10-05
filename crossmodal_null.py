"""Is the inflated positional null a fact about DNA, or about convolutional sequence encoders?

The architecture-matched nulls (review2_study.py) show that a convolutional promoter classifier scores far
above 0.5 on grammar discrimination after every canonical-TATA promoter is removed from its training set,
while a transformer of the same accuracy scores at chance. If that gap is a property of the encoder rather
than of DNA, it must reappear on a sequence task that shares the structure but none of the biology.

This script rebuilds the experiment on a synthetic 20-letter alphabet with no genomic content at all.
Two features of the promoter task have to be reproduced, or there is nothing for the effect to arise from:

  1. The composition shortcut must be strong but not sufficient. On the real data a composition-only model
     reaches AUROC 0.898 (composition_gd.py), so the two backgrounds here are calibrated to the same value.
     An earlier version drew them far apart, composition alone reached 1.000, and no model learned any
     position-dependent structure at all - the run is kept in git history as the negative control it is.
  2. The element under test must not be the only positionally constrained thing in the window. A promoter
     stripped of its TATA box still has an Inr, and a downstream co-element is what the convolutional trunk
     keeps responding to. So a second element sits downstream and is never removed.

The null condition then removes every positive carrying the element under test, exactly as the DNA nulls
remove every canonical-TATA promoter. Architectures, edits and the GD measure are unchanged; only the
alphabet and the generating grammar differ.

    python crossmodal_null.py           # -> results/crossmodal_null.json

Runs on CPU so it never competes with the GPU queue.
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DEV = torch.device("cpu")                 # the GPU is reserved for the DNA queue

A = 20                                    # alphabet size (protein-like, no genomic meaning)
L = 200
BLOCK = 8
TARGET_COMP_AUROC = 0.898                 # measured for GC content on the promoter task; fixed in advance

ELEM = np.array([5, 12, 3, 12, 5, 5, 12])           # the element under test, removed in the null condition
ELEM_POS, JITTER = 60, 2
CO_ELEM = np.array([17, 2, 9, 2, 17, 9, 2])         # downstream co-element, never removed
CO_POS = 100
FRAC_ELEM, FRAC_CO = 0.25, 0.80                     # 25% mirrors the share of promoters with a canonical TATA

FAR_TARGETS = np.arange(130, L - BLOCK)             # clear of both elements
CTRL_STARTS = np.r_[44:52, 72:80]                   # element-free blocks bracketing the element under test
N_TRAIN, N_TEST, N_PROBE = 12000, 4000, 600
EPOCHS, BS, SEEDS = 8, 128, [0, 1, 2]


# ------------------------------------------------------------------ backgrounds, matched to the real task
def _bayes_auroc(p, q, rng, n=4000):
    """AUROC of the best possible composition-only classifier: the log-likelihood ratio of the counts."""
    w = np.log(p) - np.log(q)
    cp = rng.multinomial(L, p, size=n) @ w
    cq = rng.multinomial(L, q, size=n) @ w
    return roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[cp, cq])


def build_backgrounds(target=TARGET_COMP_AUROC, seed=12345):
    """Two compositions whose optimal separability matches the promoter task's composition shortcut."""
    rng = np.random.default_rng(seed)
    p0 = rng.dirichlet(np.full(A, 50.0))
    d = rng.normal(size=A)
    d -= d.mean()
    mk = lambda e: ((p0 * np.exp(e * d)) / (p0 * np.exp(e * d)).sum(),
                    (p0 * np.exp(-e * d)) / (p0 * np.exp(-e * d)).sum())
    lo, hi = 0.0, 2.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if _bayes_auroc(*mk(mid), rng) < target:
            lo = mid
        else:
            hi = mid
    return mk((lo + hi) / 2)


# ------------------------------------------------------------------ data
def sample(n, p, rng, put_elem, put_co):
    X = rng.choice(A, size=(n, L), p=p)
    pos = np.full(n, -1)
    for i in np.where(put_co)[0]:
        s = CO_POS + rng.integers(-JITTER, JITTER + 1)
        X[i, s:s + len(CO_ELEM)] = CO_ELEM
    for i in np.where(put_elem)[0]:
        s = ELEM_POS + rng.integers(-JITTER, JITTER + 1)
        X[i, s:s + len(ELEM)] = ELEM
        pos[i] = s
    return X, pos


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    p, q = build_backgrounds()
    n = N_TRAIN + N_TEST + N_PROBE
    Xp, pos = sample(n, p, rng, rng.random(n) < FRAC_ELEM, rng.random(n) < FRAC_CO)
    Xn, _ = sample(n, q, rng, np.zeros(n, bool), np.zeros(n, bool))
    X = np.concatenate([Xp, Xn])
    y = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
    m = np.r_[pos, np.full(n, -1)]
    idx = rng.permutation(len(X))
    X, y, m = X[idx], y[idx], m[idx]
    a, b = 2 * N_TRAIN, 2 * (N_TRAIN + N_TEST)
    return (X[:a], y[:a], m[:a]), (X[a:b], y[a:b], m[a:b]), (X[b:], y[b:], m[b:]), (p, q)


# ------------------------------------------------------------------ edits (composition-preserving)
def swap(x, a, b, w=BLOCK):
    y = x.copy()
    y[a:a + w], y[b:b + w] = x[b:b + w], x[a:a + w]
    return y


def displace(X, pos, rng):
    t = rng.choice(FAR_TARGETS, len(X))
    return np.stack([swap(x, p, q) for x, p, q in zip(X, pos, t)])


def control(X, rng):
    s, t = rng.choice(CTRL_STARTS, len(X)), rng.choice(FAR_TARGETS, len(X))
    return np.stack([swap(x, p, q) for x, p, q in zip(X, s, t)])


# ------------------------------------------------------------------ models (as in models.py, alphabet A)
def one_hot(X):
    return F.one_hot(X, A).permute(0, 2, 1).float()


class ConvTrunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(A, c, 15, padding=7), nn.BatchNorm1d(c), nn.ReLU())
        self.res = nn.ModuleList([
            nn.Sequential(nn.Conv1d(c, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c), nn.ReLU())
            for d in (2, 4, 8)])

    def forward(self, x):
        h = self.stem(one_hot(x))
        for blk in self.res:
            h = h + blk(h)
        return h


class CNNGlobalPool(nn.Module):
    name = "CNN-GlobalPool"

    def __init__(self, c=64):
        super().__init__()
        self.trunk = ConvTrunk(c)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.trunk(x)
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], 1)).squeeze(-1)


class SmallTransformer(nn.Module):
    name = "Transformer"

    def __init__(self, d=64, layers=3, heads=4, stride=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(A, d, 9, padding=4), nn.GELU(), nn.Conv1d(d, d, stride, stride=stride))
        self.pos = nn.Parameter(torch.randn(L // stride, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        h = self.stem(one_hot(x)).transpose(1, 2) + self.pos
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        return self.head(self.enc(h)[:, 0]).squeeze(-1)


ARCHS = {m.name: m for m in (CNNGlobalPool, SmallTransformer)}


# ------------------------------------------------------------------ train / evaluate
def predict(model, X, bs=256):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(X[i:i + bs], dtype=torch.long, device=DEV)).cpu().numpy()
                               for i in range(0, len(X), bs)])


def fit(model, X, y, seed, lr=1e-3):
    torch.manual_seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        for i in range(0, len(X), BS):
            b = rng.choice(len(X), BS)
            z = model(torch.as_tensor(X[b], dtype=torch.long, device=DEV))
            loss = F.binary_cross_entropy_with_logits(z, torch.as_tensor(y[b], device=DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def grammar_disc(model, X, pos, seed, R=8):
    rng = np.random.default_rng(1000 + seed)
    z0 = predict(model, X)
    d_el = np.mean([z0 - predict(model, displace(X, pos, rng)) for _ in range(R)], 0)
    d_ct = np.mean([z0 - predict(model, control(X, rng)) for _ in range(R)], 0)
    lab = np.r_[np.ones(len(X)), np.zeros(len(X))]
    return float(roc_auc_score(lab, np.r_[d_el, d_ct]))


def main():
    (Xtr, ytr, mtr), (Xte, yte, _), (Xpr, ypr, mpr), (p, q) = make_data(0)
    probe = (mpr >= 0) & (ypr == 1)
    Xp, pp = Xpr[probe], mpr[probe]

    freq = lambda X: np.stack([np.bincount(x, minlength=A) for x in X]).astype(np.float32) / L
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(freq(Xtr), ytr)
    comp = float(roc_auc_score(yte, clf.decision_function(freq(Xte))))
    print(f"train {len(ytr)} | test {len(yte)} | probe {probe.sum()} element-bearing positives", flush=True)
    print(f"composition-only AUROC {comp:.3f}  (target {TARGET_COMP_AUROC:.3f}, the promoter task's value)",
          flush=True)

    out = {"composition_only_auroc": comp, "target_comp_auroc": TARGET_COMP_AUROC, "runs": {}}
    for cond in ("full", "noelem"):
        keep = np.ones(len(ytr), bool) if cond == "full" else ~((mtr >= 0) & (ytr == 1))
        for name, cls in ARCHS.items():
            for seed in SEEDS:
                t0 = time.time()
                m = fit(cls(), Xtr[keep], ytr[keep], seed)
                r = dict(auroc=float(roc_auc_score(yte, predict(m, Xte))),
                         grammar_disc=grammar_disc(m, Xp, pp, seed), n_train=int(keep.sum()))
                out["runs"][f"{cond}|{name}|{seed}"] = r
                print(f"{cond:8s} {name:16s} seed {seed}  AUROC {r['auroc']:.3f}  "
                      f"GD {r['grammar_disc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(RES, "crossmodal_null.json"), "w"), indent=1)

    print("\nmean GD (the DNA result is CNN null 0.78, Transformer null 0.48):", flush=True)
    for name in ARCHS:
        g = lambda c: np.mean([v["grammar_disc"] for k, v in out["runs"].items() if k.startswith(f"{c}|{name}|")])
        print(f"  {name:16s} trained with element {g('full'):.3f} | null (element removed) {g('noelem'):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
