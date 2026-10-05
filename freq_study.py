"""How rare can a regulatory rule be before a model stops learning it?

Training sets of fixed size (N_POS promoters + N_NEG background, training
chromosomes only) in which the fraction of promoters carrying a canonical TATA
box at -30 is set to f. Grammar is measured on the same held-out probe as the
main study. Baseline vs GCR, 3 seeds each.
"""
import json, os, time
import numpy as np
from motifs import load_task
from counterfactuals import canonical_tata
from models import ARCHS
from train import train, evaluate
from run_study import build, chrom, HOLD

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "freq_study.json")
FRACS = [0.0, 0.01, 0.03, 0.1, 0.16]   # 0.16 x 3000 = 480 <= 502 available
N_POS = N_NEG = 3000
SEEDS = [0, 1, 2]
ARCH = "Transformer"


def pools():
    """Unique training-chromosome promoters split by canonical-TATA status, plus background."""
    Xs, ys, seen = [], [], set()
    for t in ["promoter_all", "promoter_tata", "promoter_no_tata"]:
        X, y, n = load_task(t, "train")
        for x, yy, nn in zip(X, y, n):
            if nn in seen or nn.split(":")[0] in HOLD:
                continue
            seen.add(nn)
            Xs.append(x)
            ys.append(yy)
    X, y = np.stack(Xs), np.array(ys)
    t = canonical_tata(X)
    return X[(y == 1) & (t >= 0)], X[(y == 1) & (t < 0)], X[y == 0]


def main():
    Xtata, Xother, Xneg = pools()
    print(f"pools: canonical-TATA promoters {len(Xtata)}, other promoters {len(Xother)}, background {len(Xneg)}")
    _, _, _, Xte, yte, tte, Xg, yg, tg = build()
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for f in (FRACS[::-1] if os.environ.get("REVERSE") else FRACS):     # REVERSE=1: second parallel worker
        for method in ["baseline", "gcr"]:
            for seed in SEEDS:
                key = f"{f}|{method}|{seed}"
                out.update(json.load(open(PATH)) if os.path.exists(PATH) else {})   # parallel workers share PATH
                if key in out or (method == "gcr" and f == 0.0):
                    continue
                rng = np.random.default_rng(seed)
                k = int(round(f * N_POS))
                P = np.concatenate([Xtata[rng.choice(len(Xtata), k, replace=False)],
                                    Xother[rng.choice(len(Xother), N_POS - k, replace=False)]])
                Nn = Xneg[rng.choice(len(Xneg), N_NEG, replace=False)]
                X = np.concatenate([P, Nn])
                y = np.r_[np.ones(N_POS), np.zeros(N_NEG)].astype(np.int8)
                t0 = time.time()
                m = train(ARCHS[ARCH](), X, y, canonical_tata(X), method=method, epochs=20, lr=5e-4, seed=seed)
                r_test = evaluate(m, Xte, yte, tte, seed=seed, R=1)
                r = evaluate(m, Xg, yg, tg, seed=seed)
                out[key] = dict(frac=f, n_tata=k, method=method, seed=seed, auroc=r_test["auroc"],
                                **{kk: r[kk] for kk in ("delta_tata", "delta_ctrl", "delta_ctrl_abs", "delta_mut",
                                                      "delta_mut_ctrl", "grammar_disc", "grammar_disc_asym",
                                                      "heldout_gd")})
                disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
                disk[key] = out[key]
                json.dump(disk, open(PATH, "w"), indent=1)
                print(f"f={f:<5} n_tata={k:<4} {method:8s} seed {seed}: auroc {r_test['auroc']:.3f} "
                      f"GD {r['grammar_disc']:.2f} dMUT {r['delta_mut']:+.2f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
