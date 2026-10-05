"""Hyperparameter sensitivity of GCR (Transformer, seeds 0-2): ranking weight, invariance weight, margin."""
import json, os, time
from models import ARCHS
from train import train, evaluate
from run_study import build
from evaluate_mpra import prepare, score

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "sensitivity.json")
GRID = [dict(lam_rank=r, lam_inv=i, margin=m) for r, i, m in
        [(1, 1, 1), (0.3, 1, 1), (3, 1, 1), (1, 0.3, 1), (1, 3, 1), (1, 1, 0.5), (1, 1, 2)]]
SEEDS = [0, 1, 2]


def main():
    Xtr, ytr, ttr, Xte, yte, tte, Xg, yg, tg = build()
    C = prepare()
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for seed in (SEEDS[::-1] if os.environ.get("REVERSE") else SEEDS):   # REVERSE=1: second parallel worker
      for cfg in (GRID[::-1] if os.environ.get("GRID_REVERSE") else GRID):   # extra parallel workers
        key = "rank{lam_rank}_inv{lam_inv}_m{margin}".format(**cfg) + f"_s{seed}"
        out.update(json.load(open(PATH)) if os.path.exists(PATH) else {})   # parallel workers share PATH
        if key in out:
            continue
        t = time.time()
        m = train(ARCHS["Transformer"](), Xtr, ytr, ttr, method="gcr", epochs=12, lr=5e-4, seed=seed, **cfg)
        rt = evaluate(m, Xte, yte, tte, seed=seed, R=1)
        rg = evaluate(m, Xg, yg, tg, seed=seed)
        s = score(m, C)
        out[key] = dict(**cfg, seed=seed, auroc=rt["auroc"], grammar_disc=rg["grammar_disc"],
                        heldout_gd=rg["heldout_gd"], delta_mut=rg["delta_mut"],
                        delta_ctrl_abs=rg["delta_ctrl_abs"], mpra_rho=s["spearman_all"],
                        thal_pct=s["thal_damage_percentile"])
        disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
        disk[key] = out[key]
        json.dump(disk, open(PATH, "w"), indent=1)
        r = out[key]
        print(f"{key:22s} auroc {r['auroc']:.3f} GD {r['grammar_disc']:.2f} dMUT {r['delta_mut']:+.2f} "
              f"MPRA {r['mpra_rho']:+.3f} thal {r['thal_pct']:.0f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
