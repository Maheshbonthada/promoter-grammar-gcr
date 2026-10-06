"""Small checks requested by the third review, written to one result file so the manuscript can cite them.

  * how predictive GC content and the TBP profile are once negatives are composition-matched
  * where a hand-coded TBP profile ranks the HBB beta-thalassemia variants (the reference for the models)
  * how the HBB window overlaps the training split, and in which orientation its TATA box appears there
  * seed-ensembled MPRA Spearman rho per condition, LDLR excluded (Fig. 4)

    python review3_checks.py       # -> results/review3_checks.json
"""
import json
import os
import re

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import review2_study
from evaluate_mpra import prepare
from motifs import load_pwms, load_task, scan
from mpra_leakage_check import EXACT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "review3_checks.json")
RC = str.maketrans("ACGT", "TGCA")


def gc(X):
    return ((X == 1) | (X == 2)).mean(1)


def main():
    out = {}
    pwm = load_pwms()["TBP"]

    Xtr, ytr, *_, extra = review2_study.data("negmatch")
    out["negmatch"] = {name: dict(gc_auroc=float(roc_auc_score(y, gc(X))),
                                  tbp_best_hit_auroc=float(roc_auc_score(y, scan(X, pwm).max(1))))
                       for name, X, y in [("train", Xtr, ytr), ("matched_test", extra["Xmt"], extra["ymt"])]}

    C = prepare()
    ref = C["ref"]["HBB"][None]
    for name, sl in [("pwm_anywhere", slice(None)), ("pwm_at_minus30", slice(14, 25))]:
        f = lambda Z: scan(Z, pwm)[:, sl].max(1)
        dh = f(C["hbb_X"]) - f(ref)[0]
        out.setdefault("hbb_thal_percentile", {})[name] = float(100 * np.mean([(dh > v).mean() for v in dh[C["thal"]]]))

    w = json.load(open(os.path.join(HERE, "data", "mpra_windows.json")))["HBB"]
    a, b = w["start0"], w["start0"] + 300
    X, y, names = load_task("promoter_all", "train")
    core = w["seq"][w["seq"].find("GGGCATAAAAG"):][:17]
    for x, lab, n in zip(X, y, names):
        m = re.match(r"(chr[^:]+):(\d+)-(\d+)", str(n))
        if m and m.group(1) == w["chrom"] and int(m.group(2)) < b and int(m.group(3)) > a:
            s = "".join("ACGT"[int(c)] for c in x)
            out["hbb_overlap"] = dict(train_window=str(n), label=int(lab),
                                      overlap_bp=int(min(b, int(m.group(3))) - max(a, int(m.group(2)))),
                                      tata_region_forward_index=s.find(core),
                                      tata_region_revcomp_index=s.find(core[::-1].translate(RC)))

    d = np.load(os.path.join(HERE, "results", "mpra_ensemble_preds.npz"))
    V = prepare()["V"]
    keep = ~np.isin(V.element.values, EXACT)
    rho = lambda p, q: float(np.corrcoef(rankdata(p), rankdata(q))[0, 1])
    out["mpra_rho_ldlr_excluded"] = {k: rho(d[k][keep], V.effect.values[keep]) for k in d.files}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "mpra_rho_ldlr_excluded"}, indent=1))


if __name__ == "__main__":
    main()
