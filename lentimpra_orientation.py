"""The strand claim, tested against measured data instead of an in-silico probe.

The promoter study argues that GCR models acquire orientation sensitivity they were never trained on,
because reverse-complementing the TATA motif in place changes their score far more than it changes a
baseline's. That is an in-silico argument: nothing in it says the models' orientation preference matches
what orientation actually does to expression.

lentiMPRA tested many sequences in both orientations. In the held-out split, 1,964 sequences appear as
exact reverse-complement pairs with two measured activities, 598 of them promoters. The measured effect is
real and substantial: the two orientations correlate at 0.74, and 29% of pairs differ by more than 0.5.

So the claim becomes falsifiable. For each pair, the measured orientation effect is the difference in
activity between the two strands, and the predicted effect is the difference in model score. If GCR's
orientation sensitivity is biologically meaningful rather than an artefact of the training edit, GCR models
should track the measured differences better than baselines - and better than a model that never saw a
canonical TATA box at all.

    python lentimpra_orientation.py      # -> results/lentimpra_orientation.json
"""
import collections, json, os
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from models import CNNGlobalPool, SmallTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
RUNS = os.path.join(HERE, "runs_lentimpra")
DEV = torch.device("cpu")           # inference only, a few thousand sequences: keep the GPU free


def pairs():
    d = np.load(os.path.join(DATA, "lentimpra_test.npz"), allow_pickle=True)
    X, y, ids, rev, prom = d["X"].astype(np.int64), d["y"], d["name"], d["reversed"], d["promoter"]
    base = np.array([str(i).split("_Reversed")[0] for i in ids])
    c = collections.Counter(base)
    both = [b for b, v in c.items() if v == 2]
    f, r = [], []
    for b in both:
        i = np.where(base == b)[0]
        a, z = (i[0], i[1]) if not rev[i[0]] else (i[1], i[0])
        f.append(a)
        r.append(z)
    f, r = np.array(f), np.array(r)
    rc = np.where(X[f] < 4, 3 - X[f], 4)[:, ::-1]
    ok = (rc == X[r]).all(1)                       # keep only exact reverse-complement pairs
    return X[f][ok], X[r][ok], y[f][ok] - y[r][ok], prom[f][ok]


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    return np.concatenate([model(torch.as_tensor(X[i:i + bs], dtype=torch.long, device=DEV)).float().numpy()
                           for i in range(0, len(X), bs)])


def main():
    Xf, Xr, d_meas, prom = pairs()
    print(f"{len(Xf)} exact reverse-complement pairs ({int(prom.sum())} promoters); "
          f"measured orientation effect sd {d_meas.std():.3f}", flush=True)

    out = {}
    for fn in sorted(os.listdir(RUNS)):
        if not fn.endswith(".pt"):
            continue
        key = fn[:-3].replace("__", "|")
        arch = key.split("|")[1]
        model = (SmallTransformer(L=200) if arch == "Transformer" else CNNGlobalPool()).to(DEV)
        model.load_state_dict(torch.load(os.path.join(RUNS, fn), map_location=DEV))
        d_pred = predict(model, Xf) - predict(model, Xr)
        out[key] = dict(
            pearson=float(pearsonr(d_pred, d_meas)[0]), spearman=float(spearmanr(d_pred, d_meas)[0]),
            pearson_promoters=float(pearsonr(d_pred[prom], d_meas[prom])[0]),
            sign_agreement=float(np.mean(np.sign(d_pred) == np.sign(d_meas))),
            pred_sd=float(d_pred.std()), n=int(len(d_meas)))
        r = out[key]
        print(f"{key:36s} r {r['pearson']:+.3f}  (promoters {r['pearson_promoters']:+.3f})  "
              f"sign agreement {r['sign_agreement']:.3f}", flush=True)
        del model

    json.dump(out, open(os.path.join(RES, "lentimpra_orientation.json"), "w"), indent=1)
    print("\nmean by condition:", flush=True)
    for pref in sorted({k.rsplit("|", 1)[0] for k in out}):
        v = [r["pearson"] for k, r in out.items() if k.rsplit("|", 1)[0] == pref]
        p = [r["pearson_promoters"] for k, r in out.items() if k.rsplit("|", 1)[0] == pref]
        print(f"  {pref:34s} r {np.mean(v):+.3f} (promoters {np.mean(p):+.3f})  n_seeds={len(v)}", flush=True)


if __name__ == "__main__":
    main()
