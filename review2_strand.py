"""Strand test: does the model read the TATA box in a specific orientation?

TBP binds the TATA box in one orientation, so the rule is not only positional. Reverse-complementing the
7-bp motif in place keeps its position, its length and its GC content (the motif is entirely A/T), and
changes the orientation. TATAAAA is not palindromic - its reverse complement is TTTTATA - so the edit is
informative, but it also reverses the A/T ordering, which a model could respond to for reasons unrelated to
orientation. We therefore apply the same edit to a non-TATA core-upstream block and report the difference,
as for every other test in this study. Neither edit is used in training.

    python review2_strand.py ARCH [ARCH ...]      -> results/review2_strand.json (merge on write)
"""
import glob, json, os, sys, time
import numpy as np
import torch
from run_study import build
from train import predict
from evaluate_mpra import load
from review2_probe import revcomp_block, load_as
from counterfactuals import CTRL_STARTS

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "review2_strand.json")
MOTIF_W = 7


def main(archs):
    _, _, _, _, _, _, Xg, yg, tg = build()
    idx = np.where((yg == 1) & (tg >= 0))[0]
    Xt, pos = Xg[idx], tg[idx]
    print(f"{len(idx)} held-out canonical-TATA promoters", flush=True)
    paths = [p for p in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt")) +
                               glob.glob(os.path.join(HERE, "runs_review2", "*.pt")))
             if any(os.path.basename(p).startswith(a + "__") or f"__{a}__" in os.path.basename(p) for a in archs)]
    for p in paths:
        key = "|".join(os.path.basename(p)[:-3].split("__"))
        disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
        if key in disk:
            continue
        t0 = time.time()
        if "runs_review2" in p:
            _, _, _, m = load_as(p, os.path.basename(p).split("__")[1])
        else:
            _, _, _, m = load(p)
        z0 = predict(m, Xt)
        d = z0 - predict(m, revcomp_block(Xt, pos, MOTIF_W))
        rng = np.random.default_rng(0)
        cpos = rng.choice(CTRL_STARTS, len(Xt))
        c = z0 - predict(m, revcomp_block(Xt, cpos, MOTIF_W))
        disk = json.load(open(PATH)) if os.path.exists(PATH) else {}
        disk[key] = dict(strand_drop=float(d.mean()), strand_drop_sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                         strand_ctrl=float(c.mean()), strand_net=float(d.mean() - c.mean()),
                         strand_frac_down=float((d > 0).mean()), n=int(len(d)))
        json.dump(disk, open(PATH, "w"), indent=1)
        print(f"{key:40s} TATA {d.mean():+.3f} ctrl {c.mean():+.3f} net {d.mean() - c.mean():+.3f} ({time.time() - t0:.0f}s)",
              flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
