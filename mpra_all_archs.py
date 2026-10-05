"""The MPRA comparison against baseline, for every architecture rather than the Transformer alone.

mpra_leakage_check.py fixes ARCH = "Transformer", so the headline "+0.161 with LDLR excluded" is a
single-architecture result. This repeats the identical promoter-level bootstrap (same subsets, same
B, same seed) for each architecture that has GCR-Atlas predictions, so the manuscript can say which
architectures show the gain and which do not.

    python mpra_all_archs.py            # -> results/mpra_all_archs.json
"""
import json
import os

import numpy as np

import mpra_leakage_check as M
from evaluate_mpra import prepare

ARCHS = ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]


def main():
    d = np.load(os.path.join(M.RES, "mpra_ensemble_preds.npz"))
    C = prepare()
    y, el = C["V"].effect.values, C["V"].element.values
    els = np.array([e for e in np.unique(el) if e not in M.EXACT])
    out = {}
    for arch in ARCHS:
        M.ARCH = arch
        out[arch] = {}
        for a, b in [("gcr_atlas", "baseline"), ("gcr", "baseline"), ("gcr_atlas", "gcr_atlas_scram")]:
            if f"{arch}|{a}" not in d.files or f"{arch}|{b}" not in d.files:
                continue
            r = M.compare(d, y, el, els, a, b, np.random.default_rng(0))
            out[arch][f"{a} vs {b}"] = r
            print(f"{arch:15s} {a:10s} vs {b:16s} n={r['n_promoters']}  delta {r['delta']:+.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.3f}", flush=True)
    json.dump(out, open(os.path.join(M.RES, "mpra_all_archs.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
