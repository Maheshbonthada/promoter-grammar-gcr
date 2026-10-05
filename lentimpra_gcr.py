"""Does enforcing positional grammar improve prediction of measured expression?

Every external claim in the promoter study rests on 9 saturation-mutagenesis promoters, and one of those
(LDLR) is also a training sequence. That is the binding constraint on the whole paper: no number of internal
controls can make 8 clusters into strong evidence, and the endpoint is variant ranking rather than activity.

lentiMPRA (Agarwal et al., Nature 2025) replaces both problems at once. It measures the regulatory activity
of ~453,000 200-bp sequences, split here into 362,163 training and 45,271 held-out, with a continuous
readout. About 14% are gene promoters, and those carry a positional anchor: the modal best TBP hit sits at
index ~68, consistent with a TATA box about 30 bp upstream of a TSS near index 98.

So the question the promoter study could only ask on 9 promoters can be asked on 45,271 held-out sequences,
against measured expression rather than a classification proxy:

    does a model that is forced to use the position of the TATA box predict activity better?

Conditions mirror the promoter study exactly. "null" removes every canonical-TATA promoter from training,
giving the score expected with no TATA knowledge.

    SEEDS=0 python lentimpra_gcr.py                 # fast single-seed read
    SEEDS=0,1,2 python lentimpra_gcr.py             # full
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from models import CNNGlobalPool, SmallTransformer
from motifs import load_pwms, scan

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

L, BLOCK = 200, 8
TATA_WIN = (62, 76)                       # search window for the canonical TATA start (measured)
FAR_TARGETS = np.arange(130, L - BLOCK)   # downstream body, clear of the core promoter
CTRL_STARTS = np.r_[45:53, 80:88]         # core-flanking, element-free
EPOCHS = int(os.environ.get("EPOCHS", 6))
BS = 128
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
ARCHS = os.environ.get("ARCHS", "CNN-GlobalPool").split(",")
CONDS = os.environ.get("CONDS", "full,null").split(",")
METHODS = os.environ.get("METHODS", "baseline,gcr").split(",")


def load(split):
    d = np.load(os.path.join(DATA, f"lentimpra_{split}.npz"))
    return d["X"].astype(np.int64), d["y"], d["promoter"], d["reversed"]


def canonical_tata(X, frac=0.7):
    pwm = load_pwms()["TBP"]
    S = scan(X, pwm)[:, TATA_WIN[0]:TATA_WIN[1]]
    pos = S.argmax(1) + TATA_WIN[0]
    return np.where(S.max(1) >= frac * pwm.max(0).sum(), pos, -1)


def _swap_t(x, a, b, w=BLOCK):
    B, Ln = x.shape
    idx = torch.arange(Ln, device=x.device).repeat(B, 1)
    off = torch.arange(w, device=x.device)
    rows = torch.arange(B, device=x.device)[:, None]
    idx[rows, a[:, None] + off] = b[:, None] + off
    idx[rows, b[:, None] + off] = a[:, None] + off
    return x.gather(1, idx)


def _rand_from(choices, n, g):
    c = torch.as_tensor(np.asarray(choices), device=DEV)
    return c[torch.randint(len(c), (n,), device=DEV, generator=g)]


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    return np.concatenate([model(torch.as_tensor(X[i:i + bs], dtype=torch.long, device=DEV)).float().cpu().numpy()
                           for i in range(0, len(X), bs)])


def swap_np(x, a, b, w=BLOCK):
    y = x.copy()
    y[a:a + w], y[b:b + w] = x[b:b + w], x[a:a + w]
    return y


LAM = float(os.environ.get("LAM", 1.0))         # constraint weight; the promoter study swept this too
MARGIN = float(os.environ.get("MARGIN", 1.0))


def train(model, X, y, tata, method, seed, lr=1e-3, lam_rank=LAM, lam_inv=LAM, margin=MARGIN, n_aux=32):
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    model.to(DEV)
    Xt = torch.as_tensor(X, dtype=torch.long, device=DEV)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    canon = np.where(tata >= 0)[0]
    Xc = Xt[canon]
    Pc = torch.as_tensor(tata[canon], device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(len(Xt), device=DEV, generator=g)
        for i in range(0, len(Xt), BS):
            b = order[i:i + BS]
            loss = F.mse_loss(model(Xt[b]), yt[b])
            if method == "gcr" and len(Xc):
                k = torch.randint(len(Xc), (n_aux,), device=DEV, generator=g)
                xc, pc = Xc[k], Pc[k]
                x_cf = _swap_t(xc, pc, _rand_from(FAR_TARGETS, n_aux, g))
                z, z_cf = model(xc), model(x_cf)
                loss = loss + lam_rank * F.relu(margin - (z - z_cf)).mean()
                x_ct = _swap_t(xc, _rand_from(CTRL_STARTS, n_aux, g), _rand_from(FAR_TARGETS, n_aux, g))
                loss = loss + lam_inv * (z - model(x_ct)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def evaluate(model, Xte, yte, prom, rev, tte, seed, R=4):
    z = predict(model, Xte)
    out = dict(pearson_all=float(pearsonr(z, yte)[0]), spearman_all=float(spearmanr(z, yte)[0]),
               pearson_promoters=float(pearsonr(z[prom], yte[prom])[0]),
               spearman_promoters=float(spearmanr(z[prom], yte[prom])[0]),
               n_test=int(len(yte)), n_promoters=int(prom.sum()))
    m = (tte >= 0) & prom
    if m.sum() > 10:
        rng = np.random.default_rng(1000 + seed)
        Xc, pc, zc = Xte[m], tte[m], z[m]
        d_el = np.mean([zc - predict(model, np.stack(
            [swap_np(x, p, q) for x, p, q in zip(Xc, pc, rng.choice(FAR_TARGETS, len(Xc)))])) for _ in range(R)], 0)
        d_ct = np.mean([zc - predict(model, np.stack(
            [swap_np(x, s, q) for x, s, q in zip(Xc, rng.choice(CTRL_STARTS, len(Xc)),
                                                 rng.choice(FAR_TARGETS, len(Xc)))])) for _ in range(R)], 0)
        from sklearn.metrics import roc_auc_score
        lab = np.r_[np.ones(len(Xc)), np.zeros(len(Xc))]
        out.update(n_probe=int(m.sum()),
                   grammar_disc=float(roc_auc_score(lab, np.r_[d_el, d_ct])),
                   delta_tata=float(d_el.mean()), delta_ctrl=float(d_ct.mean()))
    return out


def main():
    Xtr, ytr, ptr, _ = load("train")
    Xte, yte, pte_prom, rev = load("test")
    mu, sd = ytr.mean(), ytr.std()
    ytr_z = (ytr - mu) / sd
    ttr, tte = canonical_tata(Xtr), canonical_tata(Xte)
    print(f"train {len(ytr)} ({int((ttr >= 0).sum())} with a canonical TATA at {TATA_WIN}), "
          f"test {len(yte)} ({int(pte_prom.sum())} promoters, "
          f"{int(((tte >= 0) & pte_prom).sum())} of them canonical-TATA)", flush=True)

    path = os.path.join(RES, "lentimpra_gcr.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for cond in CONDS:
        keep = np.ones(len(ytr), bool) if cond == "full" else ~(ttr >= 0)
        for arch in ARCHS:
            for method in METHODS:
                if cond == "null" and method != "baseline":
                    continue
                for seed in SEEDS:
                    tag = "" if (method == "baseline" or LAM == 1.0) else f"|lam{LAM:g}"
                    key = f"{cond}|{arch}|{method}{tag}|{seed}"
                    if key in out:
                        continue
                    t0 = time.time()
                    model = SmallTransformer(L=L) if arch == "Transformer" else CNNGlobalPool()
                    model = train(model, Xtr[keep], ytr_z[keep], ttr[keep], method, seed)
                    out[key] = evaluate(model, Xte, yte, pte_prom, rev, tte, seed)
                    out[key]["n_train"] = int(keep.sum())
                    json.dump(out, open(path, "w"), indent=1)
                    torch.save(model.state_dict(),
                               os.path.join(HERE, "runs_lentimpra", key.replace("|", "__") + ".pt"))
                    r = out[key]
                    print(f"{key:34s} r={r['pearson_all']:.3f} (promoters {r['pearson_promoters']:.3f})  "
                          f"GD {r.get('grammar_disc', float('nan')):.3f}  ({time.time() - t0:.0f}s)", flush=True)
                    del model
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "runs_lentimpra"), exist_ok=True)
    main()
