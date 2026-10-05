"""What grammar discrimination certifies: the composition-only scorers put through the same measurement.

Proposition. Both edits behind grammar discrimination are block swaps of equal width, so they permute the
bases of a window without changing the multiset. Any scorer that reads a window only through its
mononucleotide composition therefore returns the same score before and after either edit: the TATA
displacement and the position-matched control both give a score change of exactly zero, the two sets of
changes are identical, and GD is exactly 0.5. The same holds for TATA recognition, whose control is matched
for the number and direction of substituted bases. A score above 0.5 therefore certifies that a model reads
something beyond base composition.

The converse does not hold, which is the point of the architecture-matched nulls: a model can score well
above 0.5 while knowing nothing about the TATA box (see review2_study.py). GD bounds the shortcut from
below; the null bounds it from above.

This script checks the proposition on the fitted composition baselines by running them through exactly the
same evaluate() used for the neural models, so nothing about the measurement differs.

    python composition_gd.py            # -> results/composition_gd.json
"""
import json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from composition_baseline import kmer_counts
from run_study import build
from train import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KMAX = int(os.environ.get("KMAX", 5))


class CompositionScorer(torch.nn.Module):
    """A fitted composition model wrapped so that train.predict() can call it unchanged.

    Features are counted over the whole window, so every position enters the same way; the wrapper exists
    only to put the baseline through the identical measurement code path as the neural models.
    """

    def __init__(self, clf, k):
        super().__init__()
        self.clf, self.k, self.thr = clf, k, 0.0

    def forward(self, x):
        X = x.detach().cpu().numpy().astype(np.int64)
        F = gc_feat(X) if self.k == 0 else np.concatenate([kmer_counts(X, j) for j in range(1, self.k + 1)], 1)
        return torch.as_tensor(self.clf.decision_function(F), dtype=torch.float32)


def gc_feat(X):
    return (((X == 1) | (X == 2)).mean(1, keepdims=True)).astype(np.float32)


def main():
    Xtr, ytr, _, Xte, yte, tte, Xg, yg, tg = build()
    print(f"train {len(ytr)} | test {len(yte)} | grammar pool {len(yg)} "
          f"({int(((tg >= 0) & (yg == 1)).sum())} canonical-TATA promoters)", flush=True)
    out = {}
    for k in range(0, KMAX + 1):
        name = "GC content" if k == 0 else f"k-mers, k = 1..{k}"
        feat = (lambda X: gc_feat(X)) if k == 0 else \
            (lambda X: np.concatenate([kmer_counts(X, j) for j in range(1, k + 1)], 1))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(feat(Xtr), ytr)
        m = CompositionScorer(clf, k)
        r_test = evaluate(m, Xte, yte, tte, seed=0, R=1)
        r_gram = evaluate(m, Xg, yg, tg, seed=0)
        out[name] = dict(auroc=r_test["auroc"], n_features=feat(Xtr[:2]).shape[1],
                         grammar_disc=r_gram["grammar_disc"], heldout_gd=r_gram["heldout_gd"],
                         delta_tata=r_gram["delta_tata"], delta_ctrl=r_gram["delta_ctrl"],
                         max_abs_delta=float(max(abs(r_gram["delta_tata"]), abs(r_gram["delta_ctrl"]))))
        print(f"{name:20s} AUROC {out[name]['auroc']:.3f}  GD {out[name]['grammar_disc']:.3f}  "
              f"TATA recognition {out[name]['heldout_gd']:.3f}  "
              f"max |delta| {out[name]['max_abs_delta']:.2e}", flush=True)
    json.dump(out, open(os.path.join(RES, "composition_gd.json"), "w"), indent=1)
    gd1 = out["k-mers, k = 1..1"]["grammar_disc"]
    print(f"\nmononucleotide model: GD = {gd1:.6f} (proposition predicts exactly 0.5)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
