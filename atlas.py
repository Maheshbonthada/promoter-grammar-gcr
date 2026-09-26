"""JASPAR Positional Grammar Atlas of human core promoters.

For every JASPAR 2026 CORE vertebrate (non-redundant) profile, find whether its
occurrences in promoters are concentrated at a specific distance from the TSS.

Composition control: each motif is compared with K column-permuted versions of
itself (same length, same per-column content, same information content), so
positional enrichment produced by the GC/CpG gradient around the TSS is
cancelled out; only the *arrangement* of the motif can produce a signal.

Hit threshold: per-motif score quantile on genomic negative sequences
(false-positive rate FPR per position per strand), so thresholds are comparable
across motif lengths.

Only training chromosomes are used (held-out and test chromosomes excluded),
so constraints derived here can be used for training without test leakage.
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import poisson
from motifs import load_task

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
JASPAR_FILE = os.path.join(DATA, "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
EXCLUDE = {"chr8", "chr9", "chr18", "chr20", "chr21"}
TSS, L = 49, 300
K_PERM = 5
FPR = 2e-4
WIN = 10          # width of the peak window (bp)
torch.set_num_threads(16)


def parse_jaspar(path=JASPAR_FILE):
    motifs, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            mid, name = line[1:].split("\t")[:2]
            cur = dict(id=mid, name=name, rows=[])
            motifs.append(cur)
        elif line:
            cur["rows"].append([float(v) for v in line.split("[")[1].split("]")[0].split()])
    for m in motifs:
        m["pfm"] = np.array(m.pop("rows"))
    return motifs


def log_odds(pfm, pseudo=0.8):
    p = (pfm + pseudo) / (pfm + pseudo).sum(0, keepdims=True)
    return np.log2(p / 0.25)


def load_sets():
    pos, neg, seen = [], [], set()
    for t in ["promoter_all", "promoter_tata", "promoter_no_tata"]:
        for sp in ["train", "test"]:
            X, y, n = load_task(t, sp)
            for x, yy, nn in zip(X, y, n):
                if nn in seen or nn.split(":")[0] in EXCLUDE:
                    continue
                seen.add(nn)
                (pos if yy == 1 else neg).append(x)
    return np.stack(pos), np.stack(neg)


def onehot(X):
    return F.one_hot(torch.as_tensor(X, dtype=torch.long), 5)[..., :4].permute(0, 2, 1).float()


def _kernels(pwms):
    W = max(p.shape[1] for p in pwms)
    k = np.zeros((len(pwms), 4, W), dtype=np.float32)
    invalid = np.zeros((len(pwms), L), dtype=bool)          # starts where the motif would run off the end
    for i, p in enumerate(pwms):
        k[i, :, :p.shape[1]] = p
        invalid[i, L - p.shape[1] + 1:] = True
    return torch.as_tensor(k), torch.as_tensor(invalid), W


def _scores(X, kern, invalid, W):
    """Forward-strand and reverse-strand scores indexed by start in each strand's own frame: [n,2,M,L]."""
    oh = onehot(X)
    rc = torch.flip(oh, dims=[1, 2])                         # reverse complement (ACGT axis flip = complement)
    s = torch.stack([F.conv1d(F.pad(oh, (0, W - 1)), kern), F.conv1d(F.pad(rc, (0, W - 1)), kern)], 1)
    return s.masked_fill(invalid.view(1, 1, *invalid.shape), -1e9)


def positional_counts(Xpos, Xneg, pwms, bs=2048):
    """Hit counts [2, M, L] on positives by strand and start position in ORIGINAL coordinates,
    with thresholds set so that the per-position false-positive rate on background = FPR."""
    kern, invalid, W = _kernels(pwms)
    M = len(pwms)
    kth = max(1, int(FPR * len(Xneg) * 2 * (L - 10)))
    top = None
    for i in range(0, len(Xneg), bs):
        s = _scores(Xneg[i:i + bs], kern, invalid, W).permute(2, 0, 1, 3).reshape(M, -1)
        s = torch.cat([s, top], 1) if top is not None else s
        top = torch.topk(s, kth, dim=1).values
    thr = top[:, -1]
    H = torch.zeros(2, M, L)
    for i in range(0, len(Xpos), bs):
        s = _scores(Xpos[i:i + bs], kern, invalid, W)
        H += (s >= thr.view(1, 1, M, 1)).sum(0)
    H = H.numpy()
    out = np.zeros_like(H)
    out[0] = H[0]
    for m, p in enumerate(pwms):                              # rc start j  ->  original start L - w - j
        w = p.shape[1]
        out[1, m, :L - w + 1] = H[1, m, :L - w + 1][::-1]
    return out, W


def smooth(c, w=WIN):
    k = np.ones(w)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), -1, c)


def main():
    motifs = parse_jaspar()
    Xpos, Xneg = load_sets()
    print(f"{len(motifs)} JASPAR motifs; {len(Xpos)} promoters, {len(Xneg)} background sequences (training chromosomes)")
    rng = np.random.default_rng(0)
    rows, profiles, null_all = [], {}, []
    chunk = 32
    for c0 in range(0, len(motifs), chunk):
        ms = motifs[c0:c0 + chunk]
        pwms, owner = [], []
        for j, m in enumerate(ms):
            lo = log_odds(m["pfm"])
            pwms.append(lo)
            owner.append((j, -1))
            for k in range(K_PERM):
                pwms.append(lo[:, rng.permutation(lo.shape[1])])
                owner.append((j, k))
        H, W = positional_counts(Xpos, Xneg, pwms)            # [2, len(pwms), P]
        P = H.shape[-1]
        for j, m in enumerate(ms):
            w = m["pfm"].shape[1]
            valid = L - w + 1
            idx = [i for i, o in enumerate(owner) if o[0] == j]
            real = H[:, idx[0], :valid].astype(float)          # [2, valid]
            perm = H[:, idx[1:], :valid].astype(float)         # [2, K, valid]
            tot = real.sum()
            if tot < 30:
                continue
            both = real.sum(0)                                 # strand-combined profile
            null_shape = perm.sum((0, 1)) + 1e-9
            expected = null_shape / null_shape.sum() * tot     # composition-matched expectation
            o_w, e_w = smooth(both), smooth(expected)
            logp = poisson.logsf(o_w - 1, e_w)                 # P(X >= obs)
            pk = int(np.argmin(logp))
            sense = real[0, pk:pk + WIN].sum() / max(1.0, real[:, pk:pk + WIN].sum())
            # same statistic for each permuted motif against the others -> empirical null
            null_min = []
            for k in range(K_PERM):
                pk_k = perm[:, k].sum(0)
                others = np.delete(perm, k, axis=1).sum((0, 1)) + 1e-9
                ex_k = others / others.sum() * max(pk_k.sum(), 1)
                null_min.append(poisson.logsf(smooth(pk_k) - 1, smooth(ex_k)).min())
            null_all.extend(v / np.log(10) for v in null_min)
            rows.append(dict(id=m["id"], name=m["name"], width=w, hits=int(tot),
                             peak_start_rel_tss=pk - TSS, peak_obs=float(o_w[pk]), peak_exp=float(e_w[pk]),
                             enrichment=float((o_w[pk] + 1) / (e_w[pk] + 1)), log10p=float(logp[pk] / np.log(10)),
                             null_log10p_mean=float(np.mean(null_min) / np.log(10)), sense_frac=float(sense)))
            profiles[m["id"]] = dict(name=m["name"], obs=both.tolist(), exp=expected.tolist())
        print(f"  scanned {min(c0 + chunk, len(motifs))}/{len(motifs)}", flush=True)
    A = pd.DataFrame(rows)
    # empirical FDR: compare each motif's peak statistic to the pooled null statistics
    null = np.sort(np.array(null_all))
    A["emp_p"] = [(np.sum(null <= v) + 1) / (len(null) + 1) for v in A.log10p]
    A = A.sort_values("log10p")
    A["bh_q"] = np.minimum.accumulate((A.emp_p * len(A) / np.arange(1, len(A) + 1))[::-1])[::-1].clip(upper=1)
    A.to_csv(os.path.join(RES, "atlas.csv"), index=False)
    json.dump(profiles, open(os.path.join(RES, "atlas_profiles.json"), "w"))
    print(A.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
