"""Does GCR preserve sensitivity to core-promoter elements it does not protect?

GCR's invariance term penalises sensitivity at the control blocks, which overlap BREd. This measures, on the
76 probe promoters, how strongly each model responds to every single-nucleotide substitution in:

  TATA   the 7-bp TBP core found by the probe (p .. p+6)                       -- the protected element
  BREu   the 7 bp immediately upstream of it (p-7 .. p-1)
  BREd   the 7 bp immediately downstream of it (p+7 .. p+13)
  Inr    -2 .. +4 relative to the TSS (index 47 .. 53)
  distal +61 .. +100 (index 110 .. 149), clear of the DPE and MTE              -- the reference

Sensitivity is the mean absolute change in logit over the three alternative bases at each position. Two
summaries are reported for each element: an AUROC separating the per-promoter sensitivity at the element from
that at the distal reference, scored like GD (0.5 = no more sensitive than distal sequence); and the ratio of
element to distal sensitivity, so that models with different logit scales can be compared. Baseline, GCR and
the architecture-matched null are scored for seeds 0-2.

    python element_sensitivity.py ARCH [ARCH ...]      # -> results/element_sensitivity.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate_mpra import load
from gain_ci import _load_null
from run_study import build
from train import predict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "element_sensitivity.json")
TSS = 49


def regions(p):
    return {"TATA": np.arange(p, p + 7), "BREu": np.arange(p - 7, p), "BREd": np.arange(p + 7, p + 14),
            "Inr": np.arange(TSS - 2, TSS + 5), "distal": np.arange(110, 150)}


def sensitivity(model, X, tata):
    """Per promoter and region: mean |delta logit| over all substitutions in the region."""
    z0 = predict(model, X)
    out = {r: np.zeros(len(X)) for r in regions(int(tata[0]))}
    for k, (x, p) in enumerate(zip(X, tata)):
        R = regions(int(p))
        pos = np.concatenate(list(R.values()))
        muts = [(i, b) for i in pos for b in range(4) if b != x[i]]
        Xm = np.repeat(x[None], len(muts), 0)
        Xm[np.arange(len(muts)), [i for i, _ in muts]] = [b for _, b in muts]
        d = np.abs(predict(model, Xm) - z0[k])
        at = np.array([i for i, _ in muts])
        for r, idx in R.items():
            out[r][k] = d[np.isin(at, idx)].mean()
    return out


def summarise(s):
    res = {}
    for r in ("TATA", "BREu", "BREd", "Inr"):
        lab = np.r_[np.ones(len(s[r])), np.zeros(len(s["distal"]))]
        res[r] = dict(auroc_vs_distal=float(roc_auc_score(lab, np.r_[s[r], s["distal"]])),
                      ratio_to_distal=float(np.mean(s[r]) / np.mean(s["distal"])),
                      mean_abs=float(np.mean(s[r])))
    res["distal_mean_abs"] = float(np.mean(s["distal"]))
    return res


def main(archs):
    if any(a.startswith("NT-") for a in archs):
        import nt_lora
        nt_lora.register()
    if "DeepSTARR" in archs:
        import published_archs
        published_archs.register()
    *_, Xg, yg, tg = build()
    idx = np.where((tg >= 0) & (yg == 1))[0]
    X, tata = Xg[idx], tg[idx]
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for a in archs:
        paths = {}
        for m in ("baseline", "gcr"):
            for s in (0, 1, 2):
                for p in (os.path.join(HERE, "runs", "%s__%s__%d.pt" % (a, m, s)),
                          os.path.join(HERE, "runs_review2", "full__%s__%s__%d.pt" % (a, m, s))):
                    if os.path.exists(p):
                        paths["%s|%s|%d" % (a, m, s)] = p
        for s in (0, 1, 2):
            p = os.path.join(HERE, "runs_review2", "notata__%s__baseline__%d.pt" % (a, s))
            if os.path.exists(p):
                paths["%s|null|%d" % (a, s)] = p
        for key, p in sorted(paths.items()):
            if key in out:
                continue
            arch, method, seed = key.split("|")
            if "runs_review2" in p:
                model = _load_null(p, arch, method, seed)[3]
            else:
                model = load(p)[3]
            out[key] = summarise(sensitivity(model, X, tata))
            json.dump(out, open(OUT, "w"), indent=1)
            r = out[key]
            print("%-30s " % key + "  ".join("%s %.2f/%.2f" % (e, r[e]["auroc_vs_distal"], r[e]["ratio_to_distal"])
                                              for e in ("TATA", "BREu", "BREd", "Inr")), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
