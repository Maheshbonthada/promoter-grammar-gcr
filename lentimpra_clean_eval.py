"""Re-score the lentiMPRA checkpoints on test sequences never seen in training, and per cell line.

zhangtaolab/human_lentiMPRA ships HepG2, K562 and WTC11 as separate folders, and load_dataset() without
data_dir concatenates all three. The models were therefore trained on the three cell lines pooled, and
11.9% of unique test sequences also occur in training, measured in another cell line. This repeats the
original evaluate() on (a) the test sequences absent from training and (b) each cell line, so the
lentiMPRA conclusions can be checked against both.

    python lentimpra_clean_eval.py       # -> results/lentimpra_clean_eval.json
"""
import glob
import hashlib
import json
import os

import numpy as np
import torch

from lentimpra_gcr import (L, CNNGlobalPool, SmallTransformer, canonical_tata, evaluate, load, DEV)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "lentimpra_clean_eval.json")
CELLS = [("HepG2", 15379), ("K562", 23056), ("WTC11", 6836)]   # row order of the pooled test split


def seq_hash(X):
    return np.array([hashlib.md5(r.astype(np.int8).tobytes()).hexdigest() for r in X])


def main():
    Xtr = np.load(os.path.join(HERE, "data", "lentimpra_train.npz"))["X"]
    Xte, yte, prom, rev = load("test")
    tte = canonical_tata(Xte)
    clean = ~np.isin(seq_hash(Xte), np.unique(seq_hash(Xtr)))
    cell = np.concatenate([[c] * n for c, n in CELLS])
    subsets = {"all": np.ones(len(yte), bool), "unseen_in_train": clean}
    subsets.update({c: cell == c for c, _ in CELLS})
    print("test %d | unseen in train %d | canonical-TATA promoters: all %d, unseen %d"
          % (len(yte), clean.sum(), ((tte >= 0) & prom).sum(), ((tte >= 0) & prom & clean).sum()), flush=True)

    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in sorted(glob.glob(os.path.join(HERE, "runs_lentimpra", "*.pt"))):
        key = "|".join(os.path.basename(p)[:-3].split("__"))
        if key in out:
            continue
        arch, seed = key.split("|")[1], int(key.split("|")[-1])
        model = SmallTransformer(L=L) if arch == "Transformer" else CNNGlobalPool()
        model.load_state_dict(torch.load(p, map_location=DEV))
        model.to(DEV)
        out[key] = {}
        for name, m in subsets.items():
            r = evaluate(model, Xte[m], yte[m], prom[m], rev[m], tte[m], seed)
            out[key][name] = {k: r[k] for k in ("pearson_all", "grammar_disc", "n_test") if k in r}
        json.dump(out, open(OUT, "w"), indent=1)
        a, c = out[key]["all"], out[key]["unseen_in_train"]
        print("%-34s r all %.3f unseen %.3f | GD all %.3f unseen %.3f"
              % (key, a["pearson_all"], c["pearson_all"], a.get("grammar_disc", np.nan),
                 c.get("grammar_disc", np.nan)), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
