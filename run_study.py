"""Main study: does JASPAR-guided Grammar-Consistency Regularization (GCR) teach
promoter models the TATA-box positional rule when TATA promoters are rare?

Protocol
  train   : promoter_all train, minus held-out chromosomes HOLD (+10% random validation)
  test    : official promoter_all test (chr20, chr21)            -> AUROC, MCC
  grammar : canonical-TATA promoters on HOLD + chr20/21 from
            promoter_all and promoter_tata (de-duplicated)       -> Core-Promoter Grammar Test
Resumable: finished runs are stored in results/study.json, weights in runs/.
"""
import json, os, sys, time
import numpy as np
import torch
from motifs import load_task
from counterfactuals import canonical_tata
from models import ARCHS
from train import train, evaluate, METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RUNS = os.path.join(HERE, "runs")
os.makedirs(RES, exist_ok=True)
os.makedirs(RUNS, exist_ok=True)
HOLD = {"chr8", "chr9", "chr18"}
LR = {"CNN-GlobalPool": 1e-3, "CNN-Dense": 1e-4, "Transformer": 5e-4, "NT-v2-50M": 5e-5, "NT-v2-100M": 5e-5}
EPOCHS = {"NT-v2-50M": 3, "NT-v2-100M": 3}   # pretrained models: short fine-tuning; others 12 epochs
BS = {"NT-v2-100M": 64}             # 8 GB GPU: halve the batch for the 100M model
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]


def chrom(names):
    return np.array([n.split(":")[0] for n in names])


def build():
    Xa, ya, na = load_task("promoter_all", "train")
    keep = ~np.isin(chrom(na), list(HOLD))
    Xtr, ytr = Xa[keep], ya[keep]
    Xte, yte, nte = load_task("promoter_all", "test")
    # grammar probe pool: held-out chromosomes from every promoter table, de-duplicated by name
    pool_X, pool_y, seen = [], [], set()
    for t in ["promoter_all", "promoter_tata"]:
        for sp in ["train", "test"]:
            X, y, n = load_task(t, sp)
            m = np.isin(chrom(n), list(HOLD | {"chr20", "chr21"}))
            for x, yy, nn in zip(X[m], y[m], n[m]):
                if nn not in seen:
                    seen.add(nn)
                    pool_X.append(x)
                    pool_y.append(yy)
    Xg, yg = np.stack(pool_X), np.array(pool_y)
    return Xtr, ytr, canonical_tata(Xtr), Xte, yte, canonical_tata(Xte), Xg, yg, canonical_tata(Xg)


def main():
    Xtr, ytr, ttr, Xte, yte, tte, Xg, yg, tg = build()
    info = dict(n_train=len(ytr), train_canonical_pos=int(((ttr >= 0) & (ytr == 1)).sum()),
                train_pos=int(ytr.sum()), n_test=len(yte),
                grammar_probe_canonical_pos=int(((tg >= 0) & (yg == 1)).sum()), hold=sorted(HOLD))
    print(info, flush=True)
    path = os.path.join(RES, "study.json")
    out = json.load(open(path)) if os.path.exists(path) else {"info": info, "runs": {}}
    args = [a for a in sys.argv[1:] if not a.startswith("--methods=")]
    meth = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("--methods=")), METHODS)
    from nt_model import NT_ARCHS
    ARCHS.update({a: NT_ARCHS[a] for a in args if a in NT_ARCHS})
    for arch in args or list(ARCHS):
        for method in meth:
            for seed in SEEDS:
                key = f"{arch}|{method}|{seed}"
                if key in out["runs"]:
                    continue
                t = time.time()
                model = train(ARCHS[arch](), Xtr, ytr, ttr, method=method, epochs=EPOCHS.get(arch, 12),
                              lr=LR[arch], seed=seed, bs=BS.get(arch, 128))
                r_test = evaluate(model, Xte, yte, tte, seed=seed, R=1)
                r_gram = evaluate(model, Xg, yg, tg, seed=seed)
                out["runs"][key] = dict(auroc=r_test["auroc"], mcc=r_test["mcc"],
                                        **{k: v for k, v in r_gram.items() if k not in ("auroc", "mcc")})
                torch.save(dict(state=model.state_dict(), thr=model.thr),
                           os.path.join(RUNS, key.replace("|", "__") + ".pt"))
                disk = json.load(open(path)) if os.path.exists(path) else {"info": info, "runs": {}}
                disk["runs"][key] = out["runs"][key]           # merge: parallel jobs share this file
                json.dump(disk, open(path, "w"), indent=1)
                r = out["runs"][key]
                print(f"{key:32s} auroc {r['auroc']:.3f} mcc {r['mcc']:.3f} dTATA {r['delta_tata']:+.2f} "
                      f"|dCTRL| {r['delta_ctrl_abs']:.2f} dMUT {r['delta_mut']:+.2f} GD {r['grammar_disc']:.2f} "
                      f"({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
