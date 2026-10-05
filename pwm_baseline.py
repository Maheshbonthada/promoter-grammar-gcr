"""Hand-coded baselines: what does the biology give you without training a neural network at all?

Every method compared elsewhere in this study is a variant of our own training recipe. That leaves the
obvious question unanswered: if the claim is that models should use the TBP motif at -30, how well does
simply scoring the TBP motif at -30 do? A method that does not beat its own hand-coded rule has not earned
its complexity.

Four position-free or position-fixed scorers are put through the identical evaluation as the neural models
(classification AUROC, then the saturation-mutagenesis MPRA):

    GC                 GC content alone
    PWM anywhere       best TBP log-odds hit anywhere in the window   (position-blind)
    PWM at -30         best TBP log-odds hit inside the canonical window  (position-fixed)
    GC + PWM at -30    logistic combination of the two, fitted on the training split

The contrast between "anywhere" and "at -30" is the hand-coded version of the paper's whole claim, and the
contrast between "at -30" and the trained models says whether GCR buys anything a PWM does not.

    python pwm_baseline.py              # -> results/pwm_baseline.json
"""
import json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from counterfactuals import TATA_WIN
from evaluate_mpra import prepare
from motifs import load_pwms, scan
from run_study import build

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TBP = load_pwms()["TBP"]


def gc(X):
    return ((X == 1) | (X == 2)).mean(1)


def pwm_window(X, lo=None, hi=None):
    """Best TBP log-odds hit, either anywhere in the window or inside the canonical -30 window."""
    S = scan(X, TBP)
    return S.max(1) if lo is None else S[:, lo:hi].max(1)


class Hand(torch.nn.Module):
    """A hand-coded scorer wrapped so train.predict() and evaluate_mpra.score() can call it unchanged."""

    def __init__(self, fn):
        super().__init__()
        self.fn, self.thr = fn, 0.0

    def forward(self, x):
        return torch.as_tensor(self.fn(x.detach().cpu().numpy().astype(np.int64)), dtype=torch.float32)


def main():
    Xtr, ytr, _, Xte, yte, _, _, _, _ = build()
    lo, hi = TATA_WIN

    feats = lambda X: np.c_[gc(X), pwm_window(X, lo, hi)]
    comp = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(feats(Xtr), ytr)

    SCORERS = {
        "GC": gc,
        "PWM anywhere": lambda X: pwm_window(X),
        "PWM at -30": lambda X: pwm_window(X, lo, hi),
        "GC + PWM at -30": lambda X: comp.decision_function(feats(X)),
    }

    C = prepare()
    V = C["V"]
    out = {}
    for name, fn in SCORERS.items():
        m = Hand(fn)
        auroc = float(roc_auc_score(yte, fn(Xte)))
        d = fn(C["Xa"]) - fn(C["Xr"])
        per_el = [float(spearmanr(d[V.element == e], V.effect[V.element == e])[0]) for e in C["elements"]]
        out[name] = dict(auroc=auroc, spearman_all=float(spearmanr(d, V.effect)[0]),
                         spearman_promoter_mean=float(np.nanmean(per_el)),
                         spearman_tata=float(spearmanr(d[V.tata_box.values],
                                                       V.effect[V.tata_box.values])[0]))
        print(f"{name:18s} AUROC {auroc:.3f}   MPRA rho {out[name]['spearman_all']:+.3f}   "
              f"TATA-box variants rho {out[name]['spearman_tata']:+.3f}", flush=True)

    json.dump(out, open(os.path.join(RES, "pwm_baseline.json"), "w"), indent=1)

    S = json.load(open(os.path.join(RES, "mpra_runs.csv"))) if False else None
    print("\nfor comparison, the trained Transformer (promoter-level rho):", flush=True)
    print("  baseline 0.088 | decoy 0.134 | scrambled 0.137 | GCR 0.185 | GCR-Atlas 0.265", flush=True)


if __name__ == "__main__":
    main()
