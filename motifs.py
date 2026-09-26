"""PWM scanning with JASPAR profiles."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TSS = 249  # index of the TSS in the 300-bp promoter windows


def load_task(task, split):
    d = np.load(os.path.join(DATA, f"{task}_{split}.npz"))
    return d["X"], d["y"], d["name"]


def load_pwms(pseudo=0.8):
    mats = json.load(open(os.path.join(DATA, "jaspar_motifs.json")))
    out = {}
    for k, m in mats.items():
        pfm = np.array(m["pfm"]) + pseudo
        ppm = pfm / pfm.sum(0, keepdims=True)
        out[k] = np.log2(ppm / 0.25)          # 4 x W log-odds
    return out


def scan(X, pwm):
    """Forward-strand log-odds score at every start position. X: [n, L] int (4 = N).
    Returns [n, L-W+1]."""
    n, L = X.shape
    W = pwm.shape[1]
    pad = np.vstack([pwm, np.full((1, W), -10.0)])      # N scores very low
    S = np.zeros((n, L - W + 1))
    for k in range(W):
        S += pad[X[:, k:L - W + 1 + k], k]
    return S


def best_hit(X, pwm, lo=0, hi=None):
    S = scan(X, pwm)
    hi = S.shape[1] if hi is None else hi
    sub = S[:, lo:hi]
    pos = sub.argmax(1) + lo
    return pos, sub.max(1)
