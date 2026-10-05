"""Direct test of the composition shortcut: how well do position-free features classify promoters?

Logistic regression on (i) GC content, (ii) k-mer frequencies (k = 1..K) counted over the whole window,
which ignore where anything is. Same training set (three chromosomes held out) and the official chr20/21
test split as the neural models. Writes results/composition_baseline.json.
"""
import json, os, sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run_study import build

HERE = os.path.dirname(os.path.abspath(__file__))


def kmer_counts(X, k):
    """Frequencies of all 4^k k-mers per sequence (windows containing N are skipped)."""
    n, L = X.shape
    code = np.zeros((n, L - k + 1), dtype=np.int64)
    bad = np.zeros((n, L - k + 1), dtype=bool)
    for j in range(k):
        s = X[:, j:L - k + 1 + j].astype(np.int64)
        bad |= s > 3
        code = code * 4 + np.minimum(s, 3)
    out = np.zeros((n, 4 ** k), dtype=np.float32)
    for i in range(n):
        out[i] = np.bincount(code[i][~bad[i]], minlength=4 ** k)
    return out / np.maximum(out.sum(1, keepdims=True), 1)


def fit(name, Ftr, ytr, Fte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    clf.fit(Ftr, ytr)
    return name, float(roc_auc_score(yte, clf.decision_function(Fte))), Ftr.shape[1]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    Xtr, ytr, _, Xte, yte, _, _, _, _ = build()
    gc = lambda X: (((X == 1) | (X == 2)).mean(1, keepdims=True)).astype(np.float32)
    feats = {"GC content": (gc(Xtr), gc(Xte))}
    for k in range(1, K + 1):
        Ktr = np.hstack([kmer_counts(Xtr, j) for j in range(1, k + 1)])
        Kte = np.hstack([kmer_counts(Xte, j) for j in range(1, k + 1)])
        feats[f"k-mers, k = 1..{k}"] = (Ktr, Kte)
    res = Parallel(n_jobs=min(len(feats), os.cpu_count()))(
        delayed(fit)(n, a, ytr, b, yte) for n, (a, b) in feats.items())
    out = {n: dict(auroc=a, n_features=f) for n, a, f in res}
    json.dump(out, open(os.path.join(HERE, "results", "composition_baseline.json"), "w"), indent=1)
    for n, v in out.items():
        print(f"{n:22s} features {v['n_features']:5d}  test AUROC {v['auroc']:.3f}")


if __name__ == "__main__":
    main()
