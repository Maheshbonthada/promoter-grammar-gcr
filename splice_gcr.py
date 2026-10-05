"""Does the positional grammar result generalize beyond promoters? Splice sites as a second element class.

The promoter work rests on one element, the TATA box, whose positional constraint is real but soft: it sits
near -30 in a minority of promoters. Splice sites give a much harder positional rule in a different task.
In the benchmark's 600-bp windows the donor GT sits at index 301 in 51% of positives against a 5% background
rate, and the acceptor AG at index 298 in 54% against 7%. If accurate models still fail to use position when
the rule is this sharp, the promoter finding is not a quirk of a weak motif; and if GCR transfers here with
no change other than the coordinates, the method is not specific to TBP.

Everything mirrors the promoter code: the same architectures, the same composition-preserving block swaps,
the same ranking-plus-invariance loss, the same grammar-discrimination measure.

The null condition needs a different construction. Every splice-site positive contains the core dinucleotide,
so there is no equivalent of "promoters without a TATA box" to train on. Instead each window is rolled by a
random circular offset, which leaves every local feature intact but destroys absolute position. A model
trained that way can learn the motif and cannot learn where it belongs, which is the reference we need.

    python splice_gcr.py [donors|acceptors]      # -> results/splice_gcr.json
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import train as T
from models import CNNGlobalPool, SmallTransformer
from motifs import load_task
from train import predict, train as train_promoter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RUNS = os.path.join(HERE, "runs_splice")
os.makedirs(RUNS, exist_ok=True)

L = 600
BLOCK = 8
CORE = {"donors": ((2, 3), 301), "acceptors": ((0, 2), 298)}      # (dinucleotide, modal index)
SEARCH = 4                                                        # +/- bp around the modal index
FAR_TARGETS = np.r_[60:250, 360:550 - BLOCK]                      # far from the junction, both sides
CTRL_STARTS = np.r_[270:278, 320:328]                             # junction-flanking, element-free
SHIFTS = np.arange(-20, 21, 2)
LR = {"CNN-GlobalPool": 1e-3, "Transformer": 5e-4}
EPOCHS = int(os.environ.get("EPOCHS", 12))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ARCH_LIST = os.environ.get("ARCHS", "CNN-GlobalPool,Transformer").split(",")


def core_positions(X, dinuc, modal):
    """Start index of the BLOCK containing the core dinucleotide, or -1 if it is not where it belongs."""
    a, b = dinuc
    lo, hi = modal - SEARCH, modal + SEARCH + 1
    hit = (X[:, lo:hi] == a) & (X[:, lo + 1:hi + 1] == b)
    pos = np.where(hit.any(1), hit.argmax(1) + lo, -1)
    return np.where(pos >= 0, pos - 1, -1)                        # one flanking base, as in the promoter block


def roll(X, rng):
    """Random circular shift per sequence: keeps all local content, removes absolute position."""
    off = rng.integers(0, L, len(X))
    idx = (np.arange(L)[None] + off[:, None]) % L
    return np.take_along_axis(X, idx, 1)


def swap_np(x, a, b, w=BLOCK):
    y = x.copy()
    y[a:a + w], y[b:b + w] = x[b:b + w], x[a:a + w]
    return y


def displace(X, pos, rng):
    t = rng.choice(FAR_TARGETS, len(X))
    return np.stack([swap_np(x, p, q) for x, p, q in zip(X, pos, t)])


def control(X, rng):
    s, t = rng.choice(CTRL_STARTS, len(X)), rng.choice(FAR_TARGETS, len(X))
    return np.stack([swap_np(x, p, q) for x, p, q in zip(X, s, t)])


def move_np(x, a, b, w=BLOCK):
    blk = x[a:a + w]
    rest = np.concatenate([x[:a], x[a + w:]])
    return np.concatenate([rest[:b], blk, rest[b:]])


def evaluate(model, X, y, pos, seed, R=8):
    rng = np.random.default_rng(1000 + seed)
    z = predict(model, X)
    out = {"auroc": float(roc_auc_score(y, z))}
    m = (pos >= 0) & (y == 1)
    Xc, pc, zc = X[m], pos[m], z[m]
    d_el = np.mean([zc - predict(model, displace(Xc, pc, rng)) for _ in range(R)], 0)
    d_ct = np.mean([zc - predict(model, control(Xc, rng)) for _ in range(R)], 0)
    lab = np.r_[np.ones(len(Xc)), np.zeros(len(Xc))]
    curve = [float(np.mean(predict(model, np.stack([move_np(x, p, p + s) for x, p in zip(Xc, pc)])) - zc))
             if s else 0.0 for s in SHIFTS]
    out.update(n_probe=int(m.sum()),
               grammar_disc=float(roc_auc_score(lab, np.r_[d_el, d_ct])),
               shift_sensitivity=float(-np.mean([c for s, c in zip(SHIFTS, curve) if abs(s) >= 8])),
               tuning_curve=curve)
    return out


def main(task="donors"):
    # train.train() reads these as module globals; they carry promoter coordinates, so retarget them.
    T.FAR_TARGETS, T.CTRL_STARTS = FAR_TARGETS, CTRL_STARTS
    dinuc, modal = CORE[task]
    name = f"splice_sites_{task}"
    Xtr, ytr, _ = load_task(name, "train")
    Xte, yte, _ = load_task(name, "test")
    ptr, pte = core_positions(Xtr, dinuc, modal), core_positions(Xte, dinuc, modal)
    print(f"{name}: train {len(ytr)} ({int(((ptr >= 0) & (ytr == 1)).sum())} positives with the core at "
          f"{modal}+/-{SEARCH}), test {len(yte)} (probe {int(((pte >= 0) & (yte == 1)).sum())})", flush=True)

    path = os.path.join(RES, "splice_gcr.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for cond in ("full", "null"):
        rng = np.random.default_rng(7)
        Xt = Xtr if cond == "full" else roll(Xtr, rng)
        pt = ptr if cond == "full" else core_positions(Xt, dinuc, modal)
        for arch, cls in [(a, {"CNN-GlobalPool": CNNGlobalPool, "Transformer": SmallTransformer}[a])
                          for a in ARCH_LIST]:
            for method in ("baseline", "gcr"):
                if cond == "null" and method == "gcr":
                    continue                                   # the null needs no rule: it has none to learn
                for seed in SEEDS:
                    key = f"{task}|{cond}|{arch}|{method}|{seed}"
                    if key in out:
                        continue
                    t0 = time.time()
                    model = cls(L=L) if arch == "Transformer" else cls()
                    model = train_promoter(model, Xt, ytr, pt, method=method, epochs=EPOCHS,
                                           lr=LR[arch], seed=seed)
                    out[key] = evaluate(model, Xte, yte, pte, seed)
                    json.dump(out, open(path, "w"), indent=1)
                    r = out[key]
                    print(f"{key:44s} auroc {r['auroc']:.3f}  GD {r['grammar_disc']:.3f}  "
                          f"shift {r['shift_sensitivity']:+.2f}  ({time.time() - t0:.0f}s)", flush=True)
                    del model
                    torch.cuda.empty_cache()

    print("\nmean GD:", flush=True)
    for cond in ("full", "null"):
        for arch in ("CNN-GlobalPool", "Transformer"):
            for method in ("baseline", "gcr"):
                v = [r["grammar_disc"] for k, r in out.items()
                     if k.startswith(f"{task}|{cond}|{arch}|{method}|")]
                if v:
                    print(f"  {cond:5s} {arch:16s} {method:9s} {np.mean(v):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "donors")
