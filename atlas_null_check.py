"""Stricter null for the 60 GCR-Atlas rules: local dinucleotide-preserving shuffles.

The atlas compares each motif with column-permuted copies of itself. That keeps per-column base content but
not CpG dinucleotides, so CpG-island signal near the TSS could pass as positional grammar. Here each training
promoter is shuffled within consecutive 20-bp segments with the Altschul-Erickson algorithm, which preserves
every dinucleotide count (including CpG) inside each segment, so the local GC/CpG profile along the promoter
is kept while motif instances are destroyed. A rule passes if its window holds significantly more hits in the
real promoters than in the shuffled ones (Poisson test, Bonferroni over 60 rules, enrichment >= 1.5).
Writes results/atlas_null_check.csv.
"""
import json, os, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import poisson
from atlas import parse_jaspar, log_odds, load_sets
from gcr_atlas import thresholds, find_instances
from motifs import load_task
from run_study import HOLD, chrom

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEG, N_SHUF = 20, 3


def dinuc_shuffle(s, rng):
    """Altschul-Erickson dinucleotide shuffle of one sequence (1-D int array); keeps first and last base."""
    n = len(s)
    if n < 3:
        return s.copy()
    letters = np.unique(s)
    last = s[-1]
    edges = {c: list(s[1:][s[:-1] == c]) for c in letters}
    while True:                                   # random last-exit edges must form a tree rooted at `last`
        last_edge = {c: edges[c][rng.integers(len(edges[c]))] for c in letters if c != last and edges[c]}
        ok = True
        for c in last_edge:
            seen, v = set(), c
            while v != last:
                if v in seen or v not in last_edge:
                    ok = False
                    break
                seen.add(v)
                v = last_edge[v]
            if not ok:
                break
        if ok:
            break
    order = {}
    for c in letters:
        e = list(edges[c])
        if c in last_edge:
            e.remove(last_edge[c])
            rng.shuffle(e)
            e.append(last_edge[c])
        else:
            rng.shuffle(e)
        order[c] = e
    out, v, ptr = [s[0]], s[0], {c: 0 for c in letters}
    for _ in range(n - 1):
        nxt = order[v][ptr[v]]
        ptr[v] += 1
        out.append(nxt)
        v = nxt
    return np.array(out, dtype=s.dtype)


def local_shuffle_block(X, seed):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    for i in range(len(X)):
        for a in range(0, X.shape[1], SEG):
            Y[i, a:a + SEG] = dinuc_shuffle(X[i, a:a + SEG], rng)
    return Y


def main():
    A = json.load(open(os.path.join(RES, "atlas_study.json")))
    mot = {m["id"]: m for m in parse_jaspar()}
    rules = [dict(id=r["id"], name=r["name"], pwm=log_odds(mot[r["id"]]["pfm"]), w=r["width"],
                  lo=r["window"][0] + 49, hi=r["window"][1] + 49) for r in A["rules"]]
    Xpos, _ = load_sets()
    X, y, n = load_task("promoter_all", "train")
    thr = thresholds(rules, X[~np.isin(chrom(n), list(HOLD)) & (y == 0)])
    ones = np.ones(len(Xpos), dtype=np.int8)
    count = lambda inst: np.bincount(inst[:, 1], minlength=len(rules)) if len(inst) else np.zeros(len(rules))
    obs = count(find_instances(Xpos, ones, rules, thr))
    chunks = np.array_split(np.arange(len(Xpos)), os.cpu_count())
    exp = np.zeros(len(rules))
    for s in range(N_SHUF):
        parts = Parallel(n_jobs=os.cpu_count())(delayed(local_shuffle_block)(Xpos[c], 7919 * s + j)
                                                for j, c in enumerate(chunks))
        Xs = np.concatenate(parts)
        assert (np.sort(Xs, 1) == np.sort(Xpos, 1)).all()        # composition preserved
        exp += count(find_instances(Xs, ones, rules, thr)) / N_SHUF
        print(f"  shuffle {s + 1}/{N_SHUF} done", flush=True)
    logp = poisson.logsf(obs - 1, np.maximum(exp, 1e-9)) / np.log(10)
    enr = (obs + 1) / (exp + 1)
    bonf = np.log10(0.05 / len(rules))
    R = pd.DataFrame(dict(id=[r["id"] for r in rules], name=[r["name"] for r in rules],
                          window=[f"{r['lo'] - 49}..{r['hi'] - 49}" for r in rules],
                          hits_real=obs.astype(int), hits_shuffled=np.round(exp, 1), enrichment=enr,
                          log10p=logp, passes=(logp <= bonf) & (enr >= 1.5)))
    R.to_csv(os.path.join(RES, "atlas_null_check.csv"), index=False)
    print(R.round(2).to_string())
    print(f"\n{int(R.passes.sum())} of {len(R)} rules pass the local dinucleotide-shuffle null "
          f"(Bonferroni log10 p <= {bonf:.2f}, enrichment >= 1.5)")


if __name__ == "__main__":
    main()
