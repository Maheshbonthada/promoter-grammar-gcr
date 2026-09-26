"""Invariants the method relies on. Run: python -m pytest tests -q"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from counterfactuals import (swap, move, displace_tata, control_swap, random_translocation,
                             canonical_tata, BLOCK, TSS, TATA_WIN, FAR_TARGETS, CTRL_STARTS)
from train import _swap_t
from gcr_atlas import swap_var, far_targets, control_starts

RNG = np.random.default_rng(0)
X = RNG.integers(0, 4, (256, 300)).astype(np.int8)


def counts(A):
    return np.stack([(A == b).sum(1) for b in range(5)], 1)


def test_edits_preserve_composition_exactly():
    tata = RNG.integers(TATA_WIN[0], TATA_WIN[1], len(X))
    for Y in (displace_tata(X, tata, RNG), control_swap(X, RNG), random_translocation(X, RNG, p=1.0)):
        assert Y.shape == X.shape
        assert (counts(Y) == counts(X)).all()


def test_swap_and_move_semantics():
    x = np.arange(300)
    y = swap(x, 10, 200)
    assert (y[10:18] == x[200:208]).all() and (y[200:208] == x[10:18]).all()
    z = move(x, 10, 100)
    assert (z[100:108] == x[10:18]).all() and sorted(z) == sorted(x)


def test_gpu_swaps_match_cpu():
    xt = torch.as_tensor(X[:64]).long()
    a = torch.as_tensor(RNG.integers(0, 40, 64))
    b = torch.as_tensor(RNG.integers(100, 300 - 20, 64))
    y = _swap_t(xt, a, b).numpy()
    ref = np.stack([swap(X[i].astype(np.int64), int(a[i]), int(b[i])) for i in range(64)])
    assert (y == ref).all()
    w = torch.as_tensor(RNG.integers(6, 20, 64))
    yv = swap_var(xt, a, b, w, int(w.max())).numpy()
    refv = np.stack([swap(X[i].astype(np.int64), int(a[i]), int(b[i]), int(w[i])) for i in range(64)])
    assert (yv == refv).all()


def test_targets_and_controls_are_clear_of_core_promoter():
    assert FAR_TARGETS.min() >= TSS + 50 and FAR_TARGETS.max() + BLOCK <= 300
    # Controls avoid TATA boxes starting at 14..21 (97% of training TATA promoters; later
    # starts can overlap the start-29 control, a documented <1% effect) and the Inr.
    for s in CTRL_STARTS:
        assert s + BLOCK <= TATA_WIN[0] or s >= 21 + BLOCK
        assert s + BLOCK <= TSS - 2
    lo = np.full(200, 15); hi = np.full(200, 20); w = np.full(200, 8)
    c = control_starts(lo, w, RNG)
    t = far_targets(lo, hi, w, 200, RNG, avoid=c)
    assert ((t + w <= lo) | (t >= hi + w)).all()                      # target never overlaps the motif
    assert ((t + w <= c) | (t >= c + w)).all()                        # nor the control block
    assert ((t + w <= TSS - 8) | (t >= TSS + 12)).all()               # nor the TSS core


def test_canonical_tata_detects_planted_box():
    Y = X[:32].copy()
    Y[:, 19:26] = [3, 0, 3, 0, 0, 0, 0]                               # TATAAAA at -30
    assert (canonical_tata(Y) == 19).all()
