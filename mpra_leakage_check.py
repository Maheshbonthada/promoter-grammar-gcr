"""Does the external MPRA result survive removing promoters the models were trained on?

The MPRA windows were cut from hg38 around each TSS, independently of the benchmark. Comparing coordinates
shows that six of the nine overlap a training window, and one - LDLR - is the same 300 bp interval, present
in the training split with a positive label. The models have therefore seen that exact sequence.

Memorising a reference sequence does not by itself give the effect of a variant in it, but the headline
external claim should not depend on a promoter the model was trained on. This recomputes the promoter-level
bootstrap on progressively cleaner subsets.

    python mpra_leakage_check.py         # -> results/mpra_leakage_check.json
"""
import json, os
import numpy as np
from scipy.stats import rankdata
from evaluate_mpra import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 5000
ARCH = "Transformer"
EXACT = ["LDLR"]                                        # 300/300 bp in the training split
PARTIAL = ["HBB", "HBG1", "MSMB", "PKLR", "TERT"]       # 50-75 bp of 300


def rho(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def compare(d, y, el, els, a, b, rng):
    idx_by_el = {e: np.where(el == e)[0] for e in els}
    samples = [np.concatenate([idx_by_el[els[i]] for i in rng.choice(len(els), len(els), replace=True)])
               for _ in range(B)]
    x, z = d[f"{ARCH}|{a}"], d[f"{ARCH}|{b}"]
    keep = np.concatenate([idx_by_el[e] for e in els])
    diff = np.array([rho(x[i], y[i]) - rho(z[i], y[i]) for i in samples])
    diff = diff[~np.isnan(diff)]
    p = max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / len(diff))
    return dict(delta=rho(x[keep], y[keep]) - rho(z[keep], y[keep]),
                lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5), p=p, n_promoters=len(els))


def main():
    d = np.load(os.path.join(RES, "mpra_ensemble_preds.npz"))
    C = prepare()
    y, el = C["V"].effect.values, C["V"].element.values
    allels = np.unique(el)
    subsets = {
        "all 9 promoters": list(allels),
        "excluding LDLR (exact overlap)": [e for e in allels if e not in EXACT],
        "excluding all 6 overlapping": [e for e in allels if e not in EXACT + PARTIAL],
    }
    out = {}
    for name, els in subsets.items():
        rng = np.random.default_rng(0)
        out[name] = {}
        for a, b in [("gcr_atlas", "baseline"), ("gcr", "baseline"), ("gcr_atlas", "gcr_atlas_scram")]:
            r = compare(d, y, el, np.array(els), a, b, rng)
            out[name][f"{a} vs {b}"] = r
            print(f"{name:32s} {a:10s} vs {b:16s} n={r['n_promoters']}  "
                  f"delta {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.3f}", flush=True)
        print(flush=True)
    json.dump(out, open(os.path.join(RES, "mpra_leakage_check.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
