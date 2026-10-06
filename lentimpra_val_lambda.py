"""Would the lentiMPRA constraint weight have been chosen the same way on the validation split?

The weight lambda = 0.1 was chosen after the exploratory runs at 1 and 0.3 had been scored on the held-out
test split. This scores every saved CNN checkpoint (baseline, and GCR at lambda = 1, 0.3 and 0.1) on the
validation split instead, which no selection has touched, so the choice can be checked against data that
was legitimately available for it.

    python lentimpra_val_lambda.py       # -> results/lentimpra_val_lambda.json
"""
import glob
import json
import os

import numpy as np
import torch

from lentimpra_gcr import CNNGlobalPool, canonical_tata, evaluate, load, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "lentimpra_val_lambda.json")


def main():
    Xv, yv, prom, rev = load("validation")
    tv = canonical_tata(Xv)
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "runs_lentimpra", "full__CNN-GlobalPool__*.pt"))):
        key = "|".join(os.path.basename(p)[:-3].split("__"))
        model = CNNGlobalPool()
        model.load_state_dict(torch.load(p, map_location=DEV))
        model.to(DEV)
        seed = int(key.split("|")[-1])
        r = evaluate(model, Xv, yv, prom, rev, tv, seed)
        out[key] = {k: r[k] for k in ("pearson_all", "grammar_disc", "n_test") if k in r}
        print("%-40s val r %.3f  GD %.3f" % (key, r["pearson_all"], r.get("grammar_disc", np.nan)), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
