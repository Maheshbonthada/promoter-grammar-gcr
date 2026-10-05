"""Does the rule-specific gain replicate across architectures, and does it survive the training overlap?

The single-architecture test of real versus scrambled rules is underpowered: nine promoter clusters gave
p = 0.056, and dropping LDLR - the one MPRA promoter whose 300 bp interval is also a training sequence -
raises it to 0.184. Two things can still be done without new external data. The same contrast can be
repeated on the other three architectures, which are independent replications of the same comparison, and
the per-architecture differences can be pooled within each bootstrap resample, which removes model noise
while keeping the promoter clusters as the unit of resampling.

Pooling does not manufacture power: the clusters stay the same nine (or eight) promoters, so the confidence
interval can only narrow to the extent that architectures agree.

    python review2_controls_multi.py     # -> results/review2_controls_multi.json
"""
import json, os
import numpy as np
from scipy.stats import rankdata
from evaluate_mpra import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 5000
ARCHS = ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]
LDLR = "LDLR"                      # 300/300 bp overlap with a training window


def rho(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def fmt(delta, lo, hi, p):
    ps = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"
    sg = lambda v: f"{v:+.3f}".replace("-", "−")
    return f"{sg(delta)} [{sg(lo)}, {sg(hi)}], {ps}"


def main():
    d = np.load(os.path.join(RES, "mpra_ensemble_preds.npz"))
    C = prepare()
    y, el = C["V"].effect.values, C["V"].element.values
    have = {k for k in d.files}

    out = {}
    for label, els in [("all 9 promoters", sorted(set(el))),
                       ("excluding LDLR", sorted(set(el) - {LDLR}))]:
        els = np.array(els)
        idx_by_el = {e: np.where(el == e)[0] for e in els}
        rng = np.random.default_rng(0)
        samples = [np.concatenate([idx_by_el[els[i]] for i in rng.choice(len(els), len(els), replace=True)])
                   for _ in range(B)]
        keep = np.concatenate([idx_by_el[e] for e in els])
        out[label] = {}

        for a, b in [("gcr_atlas", "gcr_atlas_scram"), ("gcr_atlas", "baseline")]:
            per_arch, curves = {}, []
            for A in ARCHS:
                if f"{A}|{a}" not in have or f"{A}|{b}" not in have:
                    continue
                x, z = d[f"{A}|{a}"], d[f"{A}|{b}"]
                diff = np.array([rho(x[i], y[i]) - rho(z[i], y[i]) for i in samples])
                curves.append(diff)
                pt = rho(x[keep], y[keep]) - rho(z[keep], y[keep])
                p = max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / len(diff))
                per_arch[A] = fmt(pt, np.nanpercentile(diff, 2.5), np.nanpercentile(diff, 97.5), p)
                print(f"{label:16s} {a} vs {b:16s} {A:16s} {per_arch[A]}", flush=True)
            if len(curves) > 1:
                pooled = np.nanmean(np.vstack(curves), 0)
                pt = float(np.nanmean([float(c.mean()) for c in curves]))
                p = max(2 * min((pooled <= 0).mean(), (pooled >= 0).mean()), 1 / len(pooled))
                per_arch["pooled"] = fmt(pt, np.nanpercentile(pooled, 2.5), np.nanpercentile(pooled, 97.5), p)
                n_pos = sum(float(c.mean()) > 0 for c in curves)
                per_arch["direction"] = f"{n_pos} of {len(curves)} architectures positive"
                print(f"{label:16s} {a} vs {b:16s} {'POOLED':16s} {per_arch['pooled']}  "
                      f"({per_arch['direction']})\n", flush=True)
            out[label][f"{a} vs {b}"] = per_arch

    json.dump(out, open(os.path.join(RES, "review2_controls_multi.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
