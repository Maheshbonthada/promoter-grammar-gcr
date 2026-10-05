"""Do these benchmarks reward models for reading biology, or for reading base composition?

Scoring the TBP motif classifies promoters at AUROC 0.212 - far below chance - because the benchmark's
negatives are AT-rich while promoters are GC-rich, and the TATA box is AT-rich. A model trained on that
task is rewarded for down-weighting the best-characterised core promoter element. This script asks how far
that generalises, across every task in the benchmark and all 1,019 JASPAR CORE vertebrate profiles, with no
model training at all.

For each task and motif we measure

    raw          AUROC of the best PWM hit as a classifier
    stratified   the same AUROC computed within GC deciles and pooled, which holds composition fixed
    gc           the motif's own GC content

and then, across motifs within a task,

    rho(gc, raw)          how far a motif's apparent discriminative power is explained by its GC content
    shrinkage            how much |AUROC - 0.5| falls when composition is held fixed

A benchmark that tests biology should show motif power that survives GC stratification and no systematic
relation between a motif's GC content and its score. The opposite pattern means the task is largely a
composition test wearing a motif's name.

    python benchmark_audit.py            # -> results/benchmark_audit.csv, .json
"""
import json, os, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from atlas import parse_jaspar, log_odds
from motifs import DATA, load_task

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SUB = int(os.environ.get("SUB", 6000))          # sequences per task; scans are O(n L W)
NDEC = 10
JOBS = int(os.environ.get("JOBS", 10))


def best_hit(X, lo):
    """Best forward-strand log-odds hit per sequence."""
    n, L = X.shape
    W = lo.shape[1]
    pad = np.vstack([lo, np.full((1, W), -10.0)])
    S = np.zeros((n, L - W + 1), dtype=np.float32)
    for k in range(W):
        S += pad[X[:, k:L - W + 1 + k], k]
    return S.max(1)


def stratified_auroc(y, s, gc, ndec=NDEC):
    """AUROC within GC deciles, pooled by decile size: the motif's power with composition held fixed."""
    edges = np.quantile(gc, np.linspace(0, 1, ndec + 1))
    edges[0] -= 1e-9
    num, den = 0.0, 0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (gc > a) & (gc <= b)
        if m.sum() > 20 and 0 < y[m].sum() < m.sum():
            num += roc_auc_score(y[m], s[m]) * m.sum()
            den += m.sum()
    return float(num / den) if den else float("nan")


def one_motif(m, X, y, gc):
    lo = log_odds(m["pfm"]).astype(np.float32)
    s = best_hit(X, lo)
    p = (m["pfm"] + 0.8) / (m["pfm"] + 0.8).sum(0, keepdims=True)
    return dict(motif=m["name"], id=m["id"], width=lo.shape[1],
                motif_gc=float(p[[1, 2]].sum(0).mean()),
                raw=float(roc_auc_score(y, s)), strat=stratified_auroc(y, s, gc))


def audit_task(task, motifs, rng):
    X, y, _ = load_task(task, "train")
    if len(np.unique(y)) != 2:                  # enhancers_types / splice_sites_all are multi-class
        print(f"{task:28s} skipped: {len(np.unique(y))} classes", flush=True)
        return None
    if len(y) > SUB:
        i = rng.choice(len(y), SUB, replace=False)
        X, y = X[i], y[i]
    gc = ((X == 1) | (X == 2)).mean(1)
    gc_auroc = float(roc_auc_score(y, gc))
    rows = Parallel(n_jobs=JOBS)(delayed(one_motif)(m, X, y, gc) for m in motifs)
    d = pd.DataFrame(rows)
    d["task"] = task
    d["n"] = len(y)
    d["gc_auroc"] = gc_auroc
    return d


def main():
    motifs = parse_jaspar()
    tasks = sorted({f[:-len("_train.npz")] for f in os.listdir(DATA) if f.endswith("_train.npz")})
    tasks = [t for t in (sys.argv[1:] or tasks)]
    print(f"{len(motifs)} motifs x {len(tasks)} tasks: {tasks}", flush=True)
    rng = np.random.default_rng(0)
    out = []
    for t in tasks:
        d = audit_task(t, motifs, rng)
        if d is None:
            continue
        out.append(d)
        anti = (d.raw < 0.5).mean()
        rho = d[["motif_gc", "raw"]].corr(method="spearman").iloc[0, 1]
        shrink = 1 - (d.strat - 0.5).abs().mean() / max((d.raw - 0.5).abs().mean(), 1e-9)
        print(f"{t:28s} n={d.n.iloc[0]:6d}  GC-only AUROC {d.gc_auroc.iloc[0]:.3f}  "
              f"motifs anti-predictive {anti:5.1%}  rho(motifGC,AUROC) {rho:+.2f}  "
              f"shrinkage under GC control {shrink:5.1%}", flush=True)
    D = pd.concat(out, ignore_index=True)
    D.to_csv(os.path.join(RES, "benchmark_audit.csv"), index=False)
    summ = {t: dict(n=int(g.n.iloc[0]), gc_auroc=float(g.gc_auroc.iloc[0]),
                    frac_anti=float((g.raw < 0.5).mean()),
                    rho_gc_auroc=float(g[["motif_gc", "raw"]].corr(method="spearman").iloc[0, 1]),
                    shrinkage=float(1 - (g.strat - 0.5).abs().mean() / max((g.raw - 0.5).abs().mean(), 1e-9)))
            for t, g in D.groupby("task")}
    json.dump(summ, open(os.path.join(RES, "benchmark_audit.json"), "w"), indent=1)
    print(f"\nwrote results/benchmark_audit.csv ({len(D)} rows)", flush=True)


if __name__ == "__main__":
    main()
