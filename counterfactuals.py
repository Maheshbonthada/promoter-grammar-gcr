"""Composition-preserving counterfactual edits of promoter windows.

Window coordinates (derived from the data, see README): index 49 = TSS (+1),
canonical TATA box starts at index ~19 (-30).

Every edit is a block swap or a block move, so nucleotide composition (and
hence GC / AT content) is exactly preserved: a model that only reads
composition cannot tell an edited sequence from the original.
"""
import numpy as np
from motifs import load_pwms, scan

TSS = 49
BLOCK = 8                      # 7-bp TBP core + 1 flank
TATA_WIN = (14, 25)            # search window for canonical TATA starts (-35..-25)
FAR_TARGETS = np.arange(100, 300 - BLOCK)          # >= +51, clear of Inr / DPE / MTE
CTRL_STARTS = np.r_[0:5, 29:34]  # core-upstream blocks that avoid BRE, TATA, Inr


def canonical_tata(X, frac=0.7):
    """Start index of the best canonical TATA hit, or -1."""
    pwm = load_pwms()["TBP"]
    S = scan(X, pwm)[:, TATA_WIN[0]:TATA_WIN[1]]
    pos = S.argmax(1) + TATA_WIN[0]
    ok = S.max(1) >= frac * pwm.max(0).sum()
    return np.where(ok, pos, -1)


def swap(x, a, b, w=BLOCK):
    y = x.copy()
    y[a:a + w], y[b:b + w] = x[b:b + w], x[a:a + w]
    return y


def move(x, a, b, w=BLOCK):
    """Remove x[a:a+w] and re-insert it so it starts at b (intervening bases shift)."""
    blk = x[a:a + w]
    rest = np.concatenate([x[:a], x[a + w:]])
    return np.concatenate([rest[:b], blk, rest[b:]])


def displace_tata(X, tata_pos, rng):
    """Swap the TATA block with a random far-downstream block."""
    t = rng.choice(FAR_TARGETS, len(X))
    return np.stack([swap(x, p, q) for x, p, q in zip(X, tata_pos, t)])


def control_swap(X, rng):
    """Position-matched control: swap a non-TATA core-upstream block with the same far targets."""
    s = rng.choice(CTRL_STARTS, len(X))
    t = rng.choice(FAR_TARGETS, len(X))
    return np.stack([swap(x, p, q) for x, p, q in zip(X, s, t)])


def random_translocation(X, rng, p=0.5):
    """EvoAug-style label-preserving augmentation: swap two random blocks anywhere."""
    Y = X.copy()
    for k in np.where(rng.random(len(X)) < p)[0]:
        a, b = rng.choice(300 - BLOCK, 2, replace=False)
        while abs(int(a) - int(b)) < BLOCK:
            a, b = rng.choice(300 - BLOCK, 2, replace=False)
        Y[k] = swap(X[k], a, b)
    return Y
