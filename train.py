"""Training methods and the Core-Promoter Grammar Test (CPGT).

Methods
  baseline   BCE only
  randaug    + EvoAug-style random block translocations (label preserving)
  hardneg    + TATA-displaced counterfactuals labelled as negatives (naive)
  gcr_rank   + ranking margin  z(x) - z(x_displaced) >= m          (ours, ablation)
  gcr        + ranking margin  + invariance to position-matched control swaps (ours)
  gcr_decoy  identical to gcr, but the protected block is a meaningless downstream position
             (falsification control: does the gain come from the rule or from the regularizer?)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from counterfactuals import (BLOCK, FAR_TARGETS, CTRL_STARTS, DECOY_POS, DECOY_TARGETS, swap, move,
                             displace_tata, control_swap, random_translocation, control_double_sub)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["baseline", "randaug", "hardneg", "gcr_rank", "gcr", "gcr_decoy"]


# ---------------------------------------------------------------- GPU edits
def _swap_t(x, a, b, w=BLOCK):
    B, L = x.shape
    idx = torch.arange(L, device=x.device).repeat(B, 1)
    off = torch.arange(w, device=x.device)
    rows = torch.arange(B, device=x.device)[:, None]
    idx[rows, a[:, None] + off] = b[:, None] + off
    idx[rows, b[:, None] + off] = a[:, None] + off
    return x.gather(1, idx)


def _rand_from(choices, n, g):
    c = torch.as_tensor(choices, device=DEV)
    return c[torch.randint(len(c), (n,), device=DEV, generator=g)]


# ---------------------------------------------------------------- training
def train(model, X, y, tata, method="baseline", epochs=20, bs=128, lr=1e-3, seed=0,
          lam_rank=1.0, lam_inv=1.0, margin=1.0, n_aux=32, val_frac=0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = len(X)
    perm = rng.permutation(n)
    nv = int(val_frac * n)
    vi, ti = perm[:nv], perm[nv:]
    Xt = torch.as_tensor(X[ti], dtype=torch.long, device=DEV)
    yt = torch.as_tensor(y[ti], dtype=torch.float32, device=DEV)
    Xv, yv = X[vi], y[vi]
    canon = np.where((tata[ti] >= 0) & (y[ti] == 1))[0]
    Xc = Xt[canon]
    Pc = torch.as_tensor(tata[ti][canon], device=DEV)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), device=DEV, generator=g)
        for i in range(0, len(Xt), bs):
            b = order[i:i + bs]
            xb, yb = Xt[b], yt[b]
            if method == "randaug":
                xb = torch.as_tensor(random_translocation(xb.cpu().numpy(), rng), device=DEV)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            if method in ("hardneg", "gcr_rank", "gcr", "gcr_decoy") and len(Xc):
                k = torch.randint(len(Xc), (n_aux,), device=DEV, generator=g)
                xc, pc = Xc[k], Pc[k]
                if method == "gcr_decoy":       # same loss, deliberately wrong rule
                    pc = torch.full_like(pc, DECOY_POS)
                    x_cf = _swap_t(xc, pc, _rand_from(DECOY_TARGETS, n_aux, g))
                else:
                    x_cf = _swap_t(xc, pc, _rand_from(FAR_TARGETS, n_aux, g))
                if method == "hardneg":
                    loss = loss + F.binary_cross_entropy_with_logits(model(x_cf), torch.zeros(n_aux, device=DEV))
                elif os.environ.get("GCR_JOINT") and method == "gcr":
                    # One forward pass for the sequence and both counterfactuals, so that batch normalisation
                    # over dense features (as in DeepSTARR) sees them together and cannot re-centre each
                    # separately. Off by default: every reported model except the DeepSTARR check used the
                    # separate passes below. The random draws are the same in both paths.
                    x_ct = _swap_t(xc, _rand_from(CTRL_STARTS, n_aux, g), _rand_from(FAR_TARGETS, n_aux, g))
                    z, z_cf, z_ct = model(torch.cat([xc, x_cf, x_ct])).split(n_aux)
                    loss = loss + lam_rank * F.relu(margin - (z - z_cf)).mean() + lam_inv * (z - z_ct).pow(2).mean()
                else:
                    z, z_cf = model(xc), model(x_cf)
                    loss = loss + lam_rank * F.relu(margin - (z - z_cf)).mean()
                    if method in ("gcr", "gcr_decoy"):
                        x_ct = _swap_t(xc, _rand_from(CTRL_STARTS, n_aux, g), _rand_from(FAR_TARGETS, n_aux, g))
                        loss = loss + lam_inv * (z - model(x_ct)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        auc = roc_auc_score(yv, predict(model, Xv))
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # decision threshold chosen on validation data (same rule for every method)
    zv = predict(model, Xv)
    grid = np.quantile(zv, np.linspace(0.05, 0.95, 91))
    model.thr = float(grid[np.argmax([matthews_corrcoef(yv, zv > t) for t in grid])])
    return model


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    out = [model(torch.as_tensor(X[i:i + bs], dtype=torch.long, device=DEV)).float().cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


# ---------------------------------------------------------------- evaluation
SHIFTS = np.arange(-14, 27, 2)


def evaluate(model, X, y, tata, seed=0, R=8):
    rng = np.random.default_rng(1000 + seed)
    z = predict(model, X)
    res = dict(auroc=float(roc_auc_score(y, z)),
               mcc=float(matthews_corrcoef(y, z > getattr(model, "thr", 0.0))))
    idx = np.where((tata >= 0) & (y == 1))[0]
    Xc, pc, zc = X[idx], tata[idx], z[idx]
    d_tata = np.mean([zc - predict(model, displace_tata(Xc, pc, rng)) for _ in range(R)], 0)
    d_ctrl = np.mean([zc - predict(model, control_swap(Xc, rng)) for _ in range(R)], 0)
    # point mutations that destroy the TBP site: TATAAAA -> TGTGAAA (never used in training)
    Xm = Xc.copy()
    for k, p in enumerate(pc):
        Xm[k, p + 1] = 2
        Xm[k, p + 3] = 2
    d_mut = zc - predict(model, Xm)
    # positional tuning curve: move the TATA block by s bp (never used in training)
    curve = []
    for s in SHIFTS:
        Xs = np.stack([move(x, p, p + s) if s else x for x, p in zip(Xc, pc)])
        curve.append(float(np.mean(predict(model, Xs) - zc)))
    # held-out control for the double substitution: two A/T -> G/C changes in a non-TATA block
    d_mut_ctrl = np.mean([zc - predict(model, control_double_sub(Xc, rng)) for _ in range(R)], 0)
    lab = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
    ok = len(idx) > 1
    res.update(
        n_grammar=int(len(idx)),
        delta_tata=float(d_tata.mean()), delta_ctrl=float(d_ctrl.mean()),
        delta_ctrl_abs=float(np.abs(d_ctrl).mean()),
        delta_mut=float(d_mut.mean()), delta_mut_ctrl=float(d_mut_ctrl.mean()),
        # grammar discrimination: AUROC of TATA-displacement vs control-swap effects, both signed (null = 0.5)
        grammar_disc=float(roc_auc_score(lab, np.r_[d_tata, d_ctrl])) if ok else float("nan"),
        # the original asymmetric version (signed vs absolute; null ~ 0.25), kept for comparison
        grammar_disc_asym=float(roc_auc_score(lab, np.r_[d_tata, np.abs(d_ctrl)])) if ok else float("nan"),
        # held-out grammar: TATA double substitution vs matched control substitution (no training edit)
        heldout_gd=float(roc_auc_score(lab, np.r_[d_mut, d_mut_ctrl])) if ok else float("nan"),
        tuning_curve=curve,
    )
    return res
