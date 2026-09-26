"""Statistics for the external validation: seed-ensembled predictions per method,
bootstrap 95% CIs for Spearman correlation with MPRA effects, and paired bootstrap
tests against the baseline (resampling variants). Also per-promoter correlations."""
import glob, json, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata
from train import predict
from evaluate_mpra import prepare, load

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 2000


def fast_spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def main():
    C = prepare()
    V = C["V"]
    y = V.effect.values
    preds = {}
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt"))):
        arch, method, seed = os.path.basename(path)[:-3].split("__")
        *_, m = load(path)
        d = predict(m, C["Xa"]) - predict(m, C["Xr"])
        preds.setdefault((arch, method), []).append(d)
        del m
        torch.cuda.empty_cache()
    ens = {k: np.mean(v, 0) for k, v in preds.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), (B, len(y)))
    rows = []
    for (arch, method), d in sorted(ens.items()):
        boot = np.array([fast_spearman(d[i], y[i]) for i in idx])
        row = dict(arch=arch, method=method, n_seeds=len(preds[(arch, method)]),
                   spearman=fast_spearman(d, y), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5))
        base = ens.get((arch, "baseline"))
        if base is not None and method != "baseline":
            diff = boot - np.array([fast_spearman(base[i], y[i]) for i in idx])
            row.update(diff_vs_baseline=float(np.mean(diff)), diff_ci_lo=np.percentile(diff, 2.5),
                       diff_ci_hi=np.percentile(diff, 97.5), p_boot=float((np.sum(diff <= 0) + 1) / (B + 1)))
        tata_prom = V.element.isin(["HBB", "HBG1", "MSMB"]).values     # promoters with a canonical TATA box
        row["rho_TATA_promoters"] = spearmanr(d[tata_prom], y[tata_prom])[0]
        row["rho_other_promoters"] = spearmanr(d[~tata_prom], y[~tata_prom])[0]
        if base is not None and method != "baseline":
            ii = np.where(tata_prom)[0]
            bi = ii[rng.integers(0, len(ii), (B, len(ii)))]
            dt = np.array([fast_spearman(d[i], y[i]) - fast_spearman(base[i], y[i]) for i in bi])
            row.update(tata_prom_diff=float(dt.mean()), tata_prom_diff_lo=np.percentile(dt, 2.5),
                       tata_prom_diff_hi=np.percentile(dt, 97.5))
        for el in C["elements"]:
            s = (V.element == el).values
            row[f"rho_{el}"] = spearmanr(d[s], y[s])[0]
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "mpra_stats.csv"), index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    np.savez(os.path.join(RES, "mpra_ensemble_preds.npz"), **{f"{a}|{m}": v for (a, m), v in ens.items()}, effect=y)


if __name__ == "__main__":
    main()
