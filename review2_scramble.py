"""Falsification control for GCR-Atlas: the same machinery with deliberately wrong rule positions.

The headline external result is that GCR-Atlas raises the Transformer's agreement with MPRA-measured variant
effects. That gain could come from the positional rules, or from any regularizer that ranks a sequence above
a composition-preserving edit of itself. To separate the two, each atlas rule keeps its motif, its threshold
and its width, but its window is moved SHIFT bp away from the position where the atlas found it enriched.
Instances are then taken from the wrong windows, so the model is taught that each motif belongs where it is
not positionally constrained. Everything else - loss, margin, instance balancing, epochs, seeds - is
identical to GCR-Atlas.

If the scrambled-rule models match GCR-Atlas on the MPRA, the gain is not rule-specific.

    python review2_scramble.py ARCH [ARCH ...]        # SEEDS=0,1,2,3,4
Writes checkpoints to runs/ as ARCH__gcr_atlas_scram__SEED.pt, so evaluate_mpra.py, stats.py and
robust_stats.py pick them up with no changes.
"""
import os, sys, time
import numpy as np
import torch
from models import ARCHS
from motifs import load_task
from run_study import build, chrom, HOLD, LR, EPOCHS, BS
from gcr_atlas import select_rules, dedupe_rules, thresholds, find_instances, train_atlas, L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
SHIFT = int(os.environ.get("SHIFT", 80))


def scramble(rules):
    """Move every rule window away from its enriched position, keeping it inside the sequence."""
    out = []
    for r in rules:
        w = r["hi"] - r["lo"]
        lo = r["lo"] + SHIFT
        if lo + w + r["w"] >= L:                       # wrap back upstream instead of running off the end
            lo = max(0, r["lo"] - SHIFT)
        out.append(dict(r, lo=int(lo), hi=int(lo + w)))
    return out


def main(archs):
    rules = select_rules()
    Xtr, ytr, _, _, _, _, _, _, _ = build()
    X, y, n = load_task("promoter_all", "train")
    thr = thresholds(rules, X[~np.isin(chrom(n), list(HOLD)) & (y == 0)])
    inst = find_instances(Xtr, ytr, rules, thr)
    keep = dedupe_rules(rules, inst)
    remap = {k: i for i, k in enumerate(keep)}
    rules, thr = [rules[k] for k in keep], thr[keep]
    bad = scramble(rules)
    inst_bad = find_instances(Xtr, ytr, bad, thr)
    real = np.array([(s, remap[k], a, w) for s, k, a, w in inst if k in remap], dtype=np.int64)
    print(f"{len(rules)} rules | instances: real windows {len(real)}, shifted windows {len(inst_bad)} "
          f"(shift {SHIFT} bp)", flush=True)
    for arch in archs:
        if arch.startswith("NT-") and arch not in ARCHS:
            from nt_model import NT_ARCHS
            ARCHS[arch] = NT_ARCHS[arch]
        for seed in SEEDS:
            path = os.path.join(RUNS, f"{arch}__gcr_atlas_scram__{seed}.pt")
            if os.path.exists(path):
                continue
            t0 = time.time()
            m = train_atlas(ARCHS[arch](), Xtr, ytr, inst_bad, bad, epochs=EPOCHS.get(arch, 12),
                            lr=LR[arch], seed=seed, bs=BS.get(arch, 128))
            torch.save(dict(state=m.state_dict(), thr=m.thr), path)
            print(f"{arch}|gcr_atlas_scram|{seed} done ({time.time() - t0:.0f}s)", flush=True)
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["Transformer"])
