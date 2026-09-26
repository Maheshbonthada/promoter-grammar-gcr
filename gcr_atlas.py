"""GCR-Atlas: Grammar-Consistency Regularization driven by the JASPAR positional atlas.

Every motif that the atlas finds to be positionally constrained (training
chromosomes only) contributes ranking constraints: a promoter should score
higher than the same promoter with that motif instance swapped out of its
preferred window (composition exactly preserved). Position-matched control
swaps of motif-free blocks provide the invariance term.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from atlas import parse_jaspar, log_odds, onehot, _kernels, _scores, FPR, L
from train import predict, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TSS = 49
PAD_WIN = 3          # tolerance around the atlas peak window (bp)
FAR_GAP = 20         # displaced copy must land at least this far from the window


def select_rules(max_peaks=2, min_enrich=1.5, min_hits=100, min_delta_null=5.0):
    """Positional rules from the atlas: up to `max_peaks` non-overlapping 10-bp windows per motif that pass
    (i) Bonferroni on the Poisson window test (1019 motifs x ~290 windows), (ii) enrichment >= min_enrich
    over the composition-matched expectation, (iii) >= min_delta_null orders of magnitude beyond the
    motif's own column-permuted controls, (iv) >= min_hits hits."""
    import json
    from scipy.stats import poisson
    A = pd.read_csv(os.path.join(RES, "atlas.csv"))
    P = json.load(open(os.path.join(RES, "atlas_profiles.json")))
    bonf = np.log10(0.05 / (290 * len(A)))
    A = A[(A.hits >= min_hits) & (A.log10p - A.null_log10p_mean <= -min_delta_null)].sort_values("log10p")
    mot = {m["id"]: m for m in parse_jaspar()}
    rules = []
    for _, r in A.iterrows():
        o = np.convolve(P[r.id]["obs"], np.ones(10), "valid")
        e = np.convolve(P[r.id]["exp"], np.ones(10), "valid")
        lp = poisson.logsf(o - 1, e) / np.log(10)
        w = mot[r.id]["pfm"].shape[1]
        for _ in range(max_peaks):
            k = int(np.argmin(lp))
            if lp[k] > bonf or (o[k] + 1) / (e[k] + 1) < min_enrich:
                break
            rules.append(dict(id=r.id, name=r["name"], pwm=log_odds(mot[r.id]["pfm"]), w=w,
                              lo=max(0, k - PAD_WIN), hi=min(L - w, k + 9 + PAD_WIN),
                              enrichment=float((o[k] + 1) / (e[k] + 1)), log10p=float(lp[k])))
            lp[max(0, k - 10):k + 10] = 0.0
    return rules


def dedupe_rules(rules, inst, max_rules=60, max_overlap=0.5):
    """Greedy (most significant first): drop a rule if most of its instances coincide (same sequence,
    start within +-4 bp) with instances of rules already kept -- removes TF-family redundancy."""
    kept, taken = [], set()
    for k in np.argsort([r["log10p"] for r in rules]):
        ins = inst[inst[:, 1] == k]
        if len(ins) == 0:
            continue
        keys = {(int(s), int(a) // 4) for s, a in zip(ins[:, 0], ins[:, 2])}
        if len(keys & taken) / len(keys) <= max_overlap:
            kept.append(int(k))
            taken |= keys
        if len(kept) >= max_rules:
            break
    return kept


def thresholds(rules, Xneg, bs=2048):
    kern, invalid, W = _kernels([r["pwm"] for r in rules])
    M = len(rules)
    kth = max(1, int(FPR * len(Xneg) * 2 * (L - 10)))
    top = None
    for i in range(0, len(Xneg), bs):
        s = _scores(Xneg[i:i + bs], kern, invalid, W).permute(2, 0, 1, 3).reshape(M, -1)
        s = torch.cat([s, top], 1) if top is not None else s
        top = torch.topk(s, kth, dim=1).values
    return top[:, -1].numpy()


def find_instances(X, y, rules, thr, bs=2048):
    """(seq index, rule index, start, width) for every rule hit inside its window, positives only."""
    kern, invalid, W = _kernels([r["pwm"] for r in rules])
    out = []
    pos_idx = np.where(y == 1)[0]
    for i in range(0, len(pos_idx), bs):
        ii = pos_idx[i:i + bs]
        s = _scores(X[ii], kern, invalid, W).numpy()          # [n, 2, M, L]
        for k, r in enumerate(rules):
            fw = s[:, 0, k, :] >= thr[k]
            rv = np.zeros_like(fw)
            rv[:, :L - r["w"] + 1] = (s[:, 1, k, :L - r["w"] + 1] >= thr[k])[:, ::-1]
            hit = (fw | rv)[:, r["lo"]:r["hi"] + 1]
            n_, p_ = np.nonzero(hit)
            for a, b in zip(n_, p_):
                out.append((ii[a], k, r["lo"] + b, r["w"]))
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def far_targets(lo, hi, w, n, rng, avoid=None):
    """Random start positions at least FAR_GAP away from [lo, hi+w), clear of the TSS core,
    and (optionally) not overlapping the control block starting at avoid[i]."""
    t = []
    for i, (a, b, ww) in enumerate(zip(lo, hi, w)):
        cand = np.r_[0:max(0, a - FAR_GAP - ww), min(L, b + ww + FAR_GAP):L - ww + 1]
        cand = cand[(cand + ww <= TSS - 8) | (cand >= TSS + 12)]
        if avoid is not None:
            cand = cand[(cand + ww <= avoid[i]) | (cand >= avoid[i] + ww)]
        t.append(rng.choice(cand))
    return np.array(t)


def control_starts(a, w, rng):
    """Same-width block in the flank of the motif (never overlapping it)."""
    return np.clip(a + rng.choice([-1, 1], len(a)) * (w + rng.integers(4, 12, len(a))), 0, L - w)


def swap_var(x, a, b, w, wmax):
    B, Lx = x.shape
    idx = torch.arange(Lx, device=x.device).repeat(B, 1)
    off = torch.arange(wmax, device=x.device)
    rows = torch.arange(B, device=x.device)[:, None].expand(B, wmax)
    m = off[None, :] < w[:, None]
    src_a, src_b = (a[:, None] + off)[m], (b[:, None] + off)[m]
    r = rows[m]
    idx[r, src_a] = src_b
    idx[r, src_b] = src_a
    return x.gather(1, idx)


def train_atlas(model, X, y, inst, rules, epochs=12, bs=128, lr=5e-4, seed=0, lam_rank=1.0, lam_inv=1.0,
                margin=1.0, n_aux=32, val_frac=0.1, tata_only=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = len(X)
    perm = rng.permutation(n)
    nv = int(val_frac * n)
    vi, ti = perm[:nv], perm[nv:]
    in_train = np.zeros(n, bool)
    in_train[ti] = True
    inst = inst[in_train[inst[:, 0]]]
    if tata_only:
        inst = inst[[rules[k]["name"] == "TBP" for k in inst[:, 1]]]
    # balance rules: sample instances with probability inversely proportional to rule frequency
    cnt = np.bincount(inst[:, 1], minlength=len(rules)).astype(float)
    wts = 1.0 / cnt[inst[:, 1]]
    wts /= wts.sum()
    lo = np.array([r["lo"] for r in rules])
    hi = np.array([r["hi"] for r in rules])
    wmax = max(r["w"] for r in rules)
    Xt = torch.as_tensor(X[ti], dtype=torch.long, device=DEV)
    yt = torch.as_tensor(y[ti], dtype=torch.float32, device=DEV)
    Xall = torch.as_tensor(X, dtype=torch.long, device=DEV)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), device=DEV, generator=g)
        for i in range(0, len(Xt), bs):
            b = order[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(model(Xt[b]), yt[b])
            if len(inst):
                pick = inst[rng.choice(len(inst), n_aux, p=wts)]
                s, k, a, w = pick[:, 0], pick[:, 1], pick[:, 2], pick[:, 3]
                ctrl = control_starts(a, w, rng)
                tgt = far_targets(lo[k], hi[k], w, n_aux, rng, avoid=ctrl)
                xc = Xall[torch.as_tensor(s, device=DEV)]
                T = lambda v: torch.as_tensor(v, device=DEV)
                z = model(xc)
                x_cf = swap_var(xc, T(a), T(tgt), T(w), wmax)
                loss = loss + lam_rank * F.relu(margin - (z - model(x_cf))).mean()
                if lam_inv > 0:
                    x_ct = swap_var(xc, T(ctrl), T(tgt), T(w), wmax)
                    loss = loss + lam_inv * (z - model(x_ct)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        auc = roc_auc_score(y[vi], predict(model, X[vi]))
        if auc > best:
            best, best_state = auc, {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    from sklearn.metrics import matthews_corrcoef
    zv = predict(model, X[vi])
    grid = np.quantile(zv, np.linspace(0.05, 0.95, 91))
    model.thr = float(grid[np.argmax([matthews_corrcoef(y[vi], zv > t) for t in grid])])
    return model


def atlas_grammar(model, X, inst, rules, seed=0, R=4):
    """Per-rule grammar discrimination on held-out instances: drop after displacing the motif
    vs |change| after a same-width flank control swap."""
    rng = np.random.default_rng(100 + seed)
    lo = np.array([r["lo"] for r in rules])
    hi = np.array([r["hi"] for r in rules])
    wmax = max(r["w"] for r in rules)
    s, k, a, w = inst[:, 0], inst[:, 1], inst[:, 2], inst[:, 3]
    xs = torch.as_tensor(X[s], dtype=torch.long, device=DEV)
    z0 = predict(model, X[s])
    T = lambda v: torch.as_tensor(v, device=DEV)
    d_m, d_c = np.zeros(len(s)), np.zeros(len(s))
    for _ in range(R):
        ctrl = control_starts(a, w, rng)
        tgt = far_targets(lo[k], hi[k], w, len(s), rng, avoid=ctrl)
        d_m += z0 - predict(model, swap_var(xs, T(a), T(tgt), T(w), wmax).cpu().numpy())
        d_c += np.abs(z0 - predict(model, swap_var(xs, T(ctrl), T(tgt), T(w), wmax).cpu().numpy()))
    d_m, d_c = d_m / R, d_c / R
    per = {}
    for kk, r in enumerate(rules):
        sel = k == kk
        if sel.sum() >= 15:
            yy = np.r_[np.ones(sel.sum()), np.zeros(sel.sum())]
            per[r["name"]] = dict(n=int(sel.sum()), delta=float(d_m[sel].mean()), ctrl=float(d_c[sel].mean()),
                                  gd=float(roc_auc_score(yy, np.r_[d_m[sel], d_c[sel]])))
    return per
