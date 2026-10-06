"""Extra training runs requested by the second review.

  negmatch  Composition-matched negatives: the genomic negatives are replaced by the training promoters
            themselves, each shuffled within consecutive 20-bp segments with a dinucleotide-preserving shuffle.
            GC content, CpG and their profile along the window are identical in both classes, so a model can
            only separate them by motifs. Does the baseline then learn where the TATA box belongs?
  notata    The main training set with every canonical-TATA promoter removed. Its TATA recognition score is
            the architecture-matched reference level for a model without TATA knowledge from training.
  strictnull  As notata, but sequences with a canonical TATA box at -30 are removed from both classes.

    python review2_study.py EXPERIMENT ARCH [ARCH ...]      # METHODS=baseline,gcr  SEEDS=0,1,2
Writes results/review2_study.json (merge on write; parallel workers are safe) and weights to runs_review2/.
"""
import json, os, sys, time
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from models import ARCHS
from train import train, evaluate, predict
from run_study import build, LR, EPOCHS, BS
from atlas_null_check import local_shuffle_block
from counterfactuals import canonical_tata

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "review2_study.json")
RUNS = os.path.join(HERE, "runs_review2")
os.makedirs(RUNS, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
METHODS = os.environ.get("METHODS", "baseline").split(",")
KEYS = ("grammar_disc", "heldout_gd", "delta_tata", "delta_ctrl", "delta_mut", "delta_mut_ctrl", "tuning_curve")


def shuffled(X, seed):
    """Local (20-bp) dinucleotide shuffle of every row, in parallel.

    Each worker imports torch through this module, so on Windows too many workers exhaust the paging
    file (WinError 1455). SHUFFLE_JOBS caps them; the shuffle itself is pure NumPy and cheap."""
    n_jobs = int(os.environ.get("SHUFFLE_JOBS", 6))
    chunks = np.array_split(np.arange(len(X)), n_jobs)
    parts = Parallel(n_jobs=n_jobs)(delayed(local_shuffle_block)(X[c], seed * 1000 + i) for i, c in enumerate(chunks))
    return np.concatenate(parts)


def gc(X):
    return ((X == 1) | (X == 2)).mean(1, keepdims=True)


def data(exp):
    Xtr, ytr, ttr, Xte, yte, tte, Xg, yg, tg = build()
    extra = {}
    if exp == "negmatch":
        P = Xtr[ytr == 1]
        Xtr = np.concatenate([P, shuffled(P, 1)])
        ytr = np.r_[np.ones(len(P)), np.zeros(len(P))].astype(np.int8)
        ttr = np.r_[canonical_tata(P), -np.ones(len(P), dtype=int)]
        Pte = Xte[yte == 1]
        extra["Xmt"] = np.concatenate([Pte, shuffled(Pte, 2)])
        extra["ymt"] = np.r_[np.ones(len(Pte)), np.zeros(len(Pte))].astype(np.int8)
        clf = LogisticRegression(max_iter=1000).fit(gc(Xtr), ytr)
        extra["gc_only_matched_auroc"] = float(roc_auc_score(extra["ymt"], clf.decision_function(gc(extra["Xmt"]))))
    elif exp == "notata":
        keep = ~((ttr >= 0) & (ytr == 1))
        Xtr, ytr, ttr = Xtr[keep], ytr[keep], ttr[keep]
    elif exp == "strictnull":
        # notata removes canonical-TATA positives only, so the model still sees a TATA box at -30 in
        # negatives. This removes it from both classes: no training sequence of either label carries a
        # canonical TATA box at the canonical position.
        keep = ~(ttr >= 0)
        Xtr, ytr, ttr = Xtr[keep], ytr[keep], ttr[keep]
    return Xtr, ytr, ttr, Xte, yte, tte, Xg, yg, tg, extra


def main(exp, archs):
    Xtr, ytr, ttr, Xte, yte, tte, Xg, yg, tg, extra = data(exp)
    print(f"[{exp}] train {len(ytr)} ({int(ytr.sum())} pos, {int(((ttr >= 0) & (ytr == 1)).sum())} canonical TATA)"
          + (f"; GC-only AUROC on matched test {extra['gc_only_matched_auroc']:.3f}" if extra else ""), flush=True)
    if os.environ.get("DATA_ONLY"):
        return
    if any(a.startswith("NT-") for a in archs):
        from nt_model import NT_ARCHS
        ARCHS.update(NT_ARCHS)
    for arch in archs:
        for method in METHODS:
            for seed in SEEDS:
                key = f"{exp}|{arch}|{method}|{seed}"
                disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
                if key in disk:
                    continue
                t0 = time.time()
                m = train(ARCHS[arch](), Xtr, ytr, ttr, method=method, epochs=EPOCHS.get(arch, 12),
                          lr=LR[arch], seed=seed, bs=BS.get(arch, 128))
                r = evaluate(m, Xg, yg, tg, seed=seed)
                row = dict(exp=exp, arch=arch, method=method, seed=seed,
                           auroc=float(roc_auc_score(yte, predict(m, Xte))), **{k: r[k] for k in KEYS})
                if "Xmt" in extra:
                    row["auroc_matched"] = float(roc_auc_score(extra["ymt"], predict(m, extra["Xmt"])))
                    row["gc_only_matched_auroc"] = extra["gc_only_matched_auroc"]
                torch.save(dict(state=m.state_dict(), thr=m.thr), os.path.join(RUNS, key.replace("|", "__") + ".pt"))
                disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
                disk[key] = row
                json.dump(disk, open(PATH, "w"), indent=1)
                sh = -np.mean(np.asarray(r["tuning_curve"])[np.abs(np.arange(-14, 27, 2)) >= 8])
                print(f"{key:40s} auroc {row['auroc']:.3f}" + (f" matched {row['auroc_matched']:.3f}" if "Xmt" in extra else "")
                      + f" GD {r['grammar_disc']:.2f} shift {sh:.2f} TATA-rec {r['heldout_gd']:.2f} ({time.time() - t0:.0f}s)",
                      flush=True)
                del m
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
