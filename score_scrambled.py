"""Grammar discrimination for the scrambled-atlas control models.

review2_scramble.py trained these models and evaluate_mpra.py scored them against the MPRA, but neither
measured TATA grammar discrimination, so the claim that this control did not learn the TATA rule had no
number behind it. This scores the saved checkpoints with the same evaluate() call that produced every
GD in study.json, on the same held-out probe set.

    python score_scrambled.py            # -> results/scrambled_gd.json
"""
import glob
import json
import os
import time

import numpy as np

from evaluate_mpra import load
from run_study import build
from train import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "scrambled_gd.json")


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    _, _, _, _, _, _, Xg, yg, tg = build()
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*__gcr_atlas_scram__*.pt"))):
        key = "|".join(os.path.basename(p)[:-3].split("__"))
        if key in out:
            continue
        t = time.time()
        _, _, seed, m = load(p)
        r = evaluate(m, Xg, yg, tg, seed=int(seed))
        out[key] = dict(grammar_disc=r["grammar_disc"], auroc=r["auroc"], heldout_gd=r["heldout_gd"])
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"{key:34s} GD {r['grammar_disc']:.3f}  AUROC {r['auroc']:.3f}  ({time.time() - t:.0f}s)", flush=True)
        del m
    for arch in sorted({k.split("|")[0] for k in out}):
        v = [x["grammar_disc"] for k, x in out.items() if k.startswith(arch + "|")]
        print(f"{arch:16s} mean GD {np.mean(v):.3f} +/- {np.std(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
