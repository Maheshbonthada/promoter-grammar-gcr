"""Promoter-level tests for the falsification controls -> results/review2_controls.json.

The headline external result is that GCR-Atlas improves agreement with MPRA-measured variant effects. That
gain could come from the positional rules or from any ranking constraint over composition-preserving edits.
Here the real rules are compared with two wrong-rule controls trained identically (decoy GCR, scrambled
atlas) and with label-preserving random augmentation, resampling whole promoters as elsewhere.
"""
import json, os
import numpy as np
from scipy.stats import rankdata
from evaluate_mpra import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 5000
ARCH = "Transformer"


def rho(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def main():
    d = np.load(os.path.join(RES, "mpra_ensemble_preds.npz"))
    C = prepare()
    y = C["V"].effect.values
    el = C["V"].element.values
    els = np.unique(el)
    rng = np.random.default_rng(0)
    idx_by_el = {e: np.where(el == e)[0] for e in els}
    samples = [np.concatenate([idx_by_el[els[i]] for i in rng.choice(len(els), len(els), replace=True)])
               for _ in range(B)]

    def compare(a, b):
        x, z = d[f"{ARCH}|{a}"], d[f"{ARCH}|{b}"]
        diff = np.array([rho(x[i], y[i]) - rho(z[i], y[i]) for i in samples])
        diff = diff[~np.isnan(diff)]
        p = max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / len(diff))
        return rho(x, y) - rho(z, y), np.percentile(diff, 2.5), np.percentile(diff, 97.5), p

    def fmt(t, dd=3):
        delta, lo, hi, p = t
        ps = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"
        sg = lambda v: f"{v:+.{dd}f}".replace("-", "−")
        return f"{sg(delta)} [{sg(lo)}, {sg(hi)}], {ps}"

    out = {
        "CTRL_ATLAS_VS_BASE": fmt(compare("gcr_atlas", "baseline")),
        "CTRL_SCRAM_VS_BASE": fmt(compare("gcr_atlas_scram", "baseline")),
        "CTRL_DECOY_VS_BASE": fmt(compare("gcr_decoy", "baseline")),
        "CTRL_RANDAUG_VS_BASE": fmt(compare("randaug", "baseline")),
        "CTRL_ATLAS_VS_SCRAM": fmt(compare("gcr_atlas", "gcr_atlas_scram")),
        "CTRL_GCR_VS_DECOY": fmt(compare("gcr", "gcr_decoy")),
    }
    for k in ("gcr_atlas_scram", "gcr_decoy", "randaug", "gcr_atlas", "gcr", "baseline"):
        out[f"RHO_{k}"] = f"{rho(d[f'{ARCH}|{k}'], y):.3f}"
    gen = np.mean([rho(d[f"{ARCH}|{k}"], y) - rho(d[f"{ARCH}|baseline"], y)
                   for k in ("gcr_atlas_scram", "gcr_decoy", "randaug")])
    out["CTRL_GENERIC_MEAN"] = f"+{gen:.3f}"
    json.dump(out, open(os.path.join(RES, "review2_controls.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
