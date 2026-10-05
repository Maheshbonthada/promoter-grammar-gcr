"""Extra tests on the saved models, requested by the second review (no retraining).

  TATA implantation (gain of function, never used in training): the TATA core TATAAAA is written into
      held-out promoters that have no canonical TATA box, at every start from -44 to +20. The same is done
      with a scrambled copy of the same seven bases (the permutation with the lowest JASPAR TBP score), so
      the difference between the two removes the effect of adding A/T bases. A model that knows where the
      TATA box belongs should gain most when it is placed at -30.
  Logit scale: the s.d. of the model's logits on the official test set, so that shift sensitivity can also
      be reported in scale-free units (logit drop / logit s.d.).

    python review2_probe.py ARCH [ARCH ...]      -> results/review2_probe.json (merge on write)
"""
import glob, itertools, json, os, sys, time
import numpy as np
import torch
from motifs import load_pwms, scan
from train import predict
from run_study import build
from evaluate_mpra import load

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "review2_probe.json")
TSS, N_PROBE = 49, 300
STARTS = np.arange(5, 70, 2)                 # window indices; offsets -44 .. +20, index 19 = -30
TATA = np.array([3, 0, 3, 0, 0, 0, 0])      # TATAAAA
NATIVE = 19


def scrambled_core():
    """Permutation of TATAAAA with the lowest TBP (MA0108.3) score."""
    pwm = load_pwms()["TBP"]
    perms = np.array(sorted(set(itertools.permutations(TATA.tolist()))))
    best = []
    for p in perms:
        x = np.full((1, 40), 1)
        x[0, 16:23] = p
        best.append(scan(x, pwm).max())
    return perms[int(np.argmin(best))]


def implant(X, core, s):
    Y = X.copy()
    Y[:, s:s + len(core)] = core
    return Y


def probe_set():
    _, _, _, Xte, yte, _, Xg, yg, tg = build()
    idx = np.where((yg == 1) & (tg < 0))[0]
    idx = np.random.default_rng(0).choice(idx, N_PROBE, replace=False)
    return Xg[idx], Xte


def revcomp_block(X, pos, w=7):
    """Reverse-complement the motif in place: same composition at the same position, opposite orientation."""
    Y = X.copy()
    for k, p in enumerate(pos):
        blk = X[k, p:p + w]
        Y[k, p:p + w] = np.where(blk < 4, 3 - blk, 4)[::-1]
    return Y


def strand_test(m, Xt, pos):
    """TBP reads the TATA box in one orientation. Reverse-complementing it preserves position, length and
    composition of the window, so a model that only detects A/T-rich content should not respond."""
    z0 = predict(m, Xt)
    return float(np.mean(z0 - predict(m, revcomp_block(Xt, pos))))


def run(m, X, Xte, ctrl):
    z0 = predict(m, X)
    gT = np.stack([predict(m, implant(X, TATA, s)) - z0 for s in STARTS], 1)     # [n, positions]
    gC = np.stack([predict(m, implant(X, ctrl, s)) - z0 for s in STARTS], 1)
    spec = gT - gC
    near = np.abs(STARTS - NATIVE) <= 2
    far = np.abs(STARTS - NATIVE) >= 10
    pref = spec[:, near].mean(1) - spec[:, far].mean(1)
    zt = predict(m, Xte)
    return dict(implant_curve=spec.mean(0).tolist(), implant_tata_curve=gT.mean(0).tolist(),
                implant_ctrl_curve=gC.mean(0).tolist(),
                implant_peak=int(STARTS[np.argmax(spec.mean(0))] - TSS),
                implant_pref=float(pref.mean()), implant_pref_frac=float((pref > 0).mean()),
                logit_sd=float(zt.std()))


def main(archs):
    X, Xte = probe_set()
    ctrl = scrambled_core()
    print("scrambled control core:", "".join("ACGT"[b] for b in ctrl), flush=True)
    paths = [p for p in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt")) +
                               glob.glob(os.path.join(HERE, "runs_review2", "*.pt")))
             if any(os.path.basename(p).startswith(a + "__") or f"__{a}__" in os.path.basename(p) for a in archs)]
    for p in paths:
        key = "|".join(os.path.basename(p)[:-3].split("__"))
        disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
        if key in disk:
            continue
        t0 = time.time()
        if "runs_review2" in p:                      # exp__arch__method__seed
            arch = os.path.basename(p).split("__")[1]
            _, _, _, m = load_as(p, arch)
        else:
            _, _, _, m = load(p)
        row = run(m, X, Xte, ctrl)
        disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
        disk[key] = row
        json.dump(disk, open(PATH, "w"), indent=1)
        print(f"{key:40s} peak {row['implant_peak']:+d} pref {row['implant_pref']:+.2f} "
              f"({100 * row['implant_pref_frac']:.0f}% of promoters) logit s.d. {row['logit_sd']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()


def load_as(path, arch):
    from models import ARCHS
    from train import DEV
    if arch.startswith("NT-"):
        from nt_model import NT_ARCHS
        ARCHS.update(NT_ARCHS)
    ck = torch.load(path, map_location=DEV)
    m = ARCHS[arch]().to(DEV)
    m.load_state_dict(ck["state"])
    m.eval()
    return None, None, None, m


if __name__ == "__main__":
    main(sys.argv[1:])
