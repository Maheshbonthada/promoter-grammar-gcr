"""Stricter statistics for the external validations (all two-sided; resampling units match the data).

MPRA (582 variants in 9 promoters): difference in Spearman rho vs baseline, with 95% CIs and two-sided p from
(a) a variant-level bootstrap and (b) a promoter-level (cluster) bootstrap that resamples whole promoters.
TATA-promoter subset (3 promoters): variant-level bootstrap only (too few promoters to resample).
TBP binding (14 SNPs): bootstrap over SNPs of the difference in Spearman rho vs baseline.
Holm correction within each family of GCR / GCR-Atlas vs baseline comparisons.
Writes results/robust_stats.csv.
"""
import json, os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from evaluate_mpra import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 5000


def rho(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def boot_diff(x, base, y, samples):
    d = np.array([rho(x[i], y[i]) - rho(base[i], y[i]) for i in samples])
    d = d[~np.isnan(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(max(p, 1 / len(d)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mpra_row(a, m, x, base, y, el, tata, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    var = [rng.integers(0, n, n) for _ in range(B)]
    U = np.unique(el)
    by = {u: np.where(el == u)[0] for u in U}
    clu = [np.concatenate([by[u] for u in rng.choice(U, len(U))]) for _ in range(B)]
    it = np.where(tata)[0]
    sub = [it[rng.integers(0, len(it), len(it))] for _ in range(B)]
    lo_v, hi_v, p_v = boot_diff(x, base, y, var)
    lo_c, hi_c, p_c = boot_diff(x, base, y, clu)
    lo_t, hi_t, p_t = boot_diff(x, base, y, sub)
    return dict(analysis="MPRA", arch=a, method=m, delta=rho(x, y) - rho(base, y),
                ci_lo_variant=lo_v, ci_hi_variant=hi_v, p_variant=p_v,
                ci_lo_promoter=lo_c, ci_hi_promoter=hi_c, p_promoter=p_c,
                delta_tata_prom=rho(x[it], y[it]) - rho(base[it], y[it]), ci_lo_tata_prom=lo_t,
                ci_hi_tata_prom=hi_t, p_tata_prom=p_t)


def tbp_row(a, m, g, b, ddg, seed):
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(ddg), len(ddg)) for _ in range(B)]
    d = np.array([spearmanr(g[i], ddg[i])[0] - spearmanr(b[i], ddg[i])[0] for i in samples])
    d = d[~np.isnan(d)]
    return dict(analysis="TBP", arch=a, method=m, delta=spearmanr(g, ddg)[0] - spearmanr(b, ddg)[0],
                ci_lo_variant=float(np.percentile(d, 2.5)), ci_hi_variant=float(np.percentile(d, 97.5)),
                p_variant=float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))))


def main():
    V = prepare()["V"]
    el = V.element.values
    tata = V.element.isin(["HBB", "HBG1", "MSMB"]).values
    D = np.load(os.path.join(RES, "mpra_ensemble_preds.npz"))
    y = D["effect"]
    jobs = []
    for k in D.files:
        if k == "effect" or k.endswith("|baseline"):
            continue
        a, m = k.split("|")
        jobs.append(delayed(mpra_row)(a, m, D[k], D[f"{a}|baseline"], y, el, tata, len(jobs)))
    T = json.load(open(os.path.join(RES, "tbp_binding.json")))
    ddg = pd.read_csv(os.path.join(RES, "tbp_binding_variants.csv")).ddG.values
    groups = {}
    for k, v in T["runs"].items():
        a, m, _ = k.split("|")
        groups.setdefault((a, m), []).append(np.array(v["pred"]))
    ens = {k: (np.array(P) / np.array(P).std(axis=1, keepdims=True)).mean(0) for k, P in groups.items()}
    for (a, m), g in ens.items():
        if m != "baseline":
            jobs.append(delayed(tbp_row)(a, m, g, ens[(a, "baseline")], ddg, 1000 + len(jobs)))
    rows = Parallel(n_jobs=os.cpu_count())(jobs)
    R = pd.DataFrame(rows).sort_values(["analysis", "arch", "method"]).reset_index(drop=True)
    for an, col in [("MPRA", "p_promoter"), ("MPRA", "p_tata_prom"), ("TBP", "p_variant")]:
        fam = (R.analysis == an) & R.method.isin(["gcr", "gcr_atlas"])
        R.loc[fam, col + "_holm"] = holm(R.loc[fam, col])
    R.to_csv(os.path.join(RES, "robust_stats.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
