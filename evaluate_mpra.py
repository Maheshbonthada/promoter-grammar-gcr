"""External validation on measured variant effects.

* Saturation-mutagenesis MPRA of 9 disease promoters (Kircher et al. 2019; HF
  gonzalobenegas/sat_mut_mpra; label = log2 expression effect of the alt allele).
* beta-thalassemia: HBB TATA-box positions -31..-28 (all 12 SNVs), ranked among
  all 897 possible SNVs of the HBB window by predicted damage.
* Cancer: TERT promoter C228T (chr5:1,295,113 G>A, hg38), ranked among all SNVs of
  the TERT window by predicted activation (C250T lies outside the -49..+250 window).
"""
import glob, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from models import ARCHS
from train import predict
from mpra_variants import variant_table
from counterfactuals import canonical_tata

HERE = os.path.dirname(os.path.abspath(__file__))
LUT = {c: i for i, c in enumerate("ACGT")}
BETA_THAL_POS = [-28, -29, -30, -31]
TERT_C228T = (4, "T")          # window index (sense strand) and alt base; see README


def enc(s):
    return np.array([LUT.get(c, 4) for c in s], dtype=np.int64)


def saturate(x):
    sat = [(i, b) for i in range(len(x)) for b in range(4) if b != x[i]]
    Xs = np.repeat(x[None], len(sat), 0)
    Xs[np.arange(len(sat)), [i for i, _ in sat]] = [b for _, b in sat]
    return Xs, np.array([i for i, _ in sat]), np.array([b for _, b in sat])


def prepare():
    V, win = variant_table()
    ref = {el: enc(w["seq"]) for el, w in win.items()}
    Xr = np.stack([ref[e] for e in V.element])
    Xa = Xr.copy()
    Xa[np.arange(len(V)), V.idx.values] = [LUT[a] for a in V.alt]
    tpos = {el: int(canonical_tata(r[None])[0]) for el, r in ref.items()}
    V["tata_box"] = [tpos[e] >= 0 and tpos[e] <= i < tpos[e] + 8 for e, i in zip(V.element, V.idx)]
    hbb_X, hbb_i, _ = saturate(ref["HBB"])
    tert_X, tert_i, tert_b = saturate(ref["TERT"])
    c228t = np.where((tert_i == TERT_C228T[0]) & (tert_b == LUT[TERT_C228T[1]]))[0][0]
    return dict(V=V, elements=list(win), ref=ref, Xr=Xr, Xa=Xa, hbb_X=hbb_X,
                thal=np.isin(hbb_i - 49, BETA_THAL_POS), tert_X=tert_X, c228t=c228t)


def score(model, C):
    V = C["V"]
    d = predict(model, C["Xa"]) - predict(model, C["Xr"])
    per_el = [spearmanr(d[V.element == e], V.effect[V.element == e])[0] for e in C["elements"]]
    dh = predict(model, C["hbb_X"]) - predict(model, C["ref"]["HBB"][None])[0]
    thal_pct = 100 * np.mean([(dh > v).mean() for v in dh[C["thal"]]])          # % of SNVs predicted less damaging
    dt = predict(model, C["tert_X"]) - predict(model, C["ref"]["TERT"][None])[0]
    tb = V.tata_box.values
    return dict(spearman_all=float(spearmanr(d, V.effect)[0]),
                spearman_mean_element=float(np.nanmean(per_el)),
                tata_n=int(tb.sum()),
                tata_spearman=float(spearmanr(d[tb], V.effect[tb])[0]),
                tata_mean_delta=float(d[tb].mean()),
                tata_sign_agree=float(np.mean(np.sign(d[tb]) == np.sign(V.effect[tb]))),
                thal_mean_delta=float(dh[C["thal"]].mean()),
                thal_damage_percentile=float(thal_pct),
                tert_c228t_delta=float(dt[C["c228t"]]),
                tert_c228t_activation_percentile=float(100 * (dt < dt[C["c228t"]]).mean()))


def load(path):
    arch, method, seed = os.path.basename(path)[:-3].split("__")
    if arch.startswith("NT-") and arch not in ARCHS:
        from nt_model import NT_ARCHS
        ARCHS[arch] = NT_ARCHS[arch]
    from train import DEV
    ck = torch.load(path, map_location=DEV)
    m = ARCHS[arch]().to(DEV)
    m.load_state_dict(ck["state"])
    m.thr = ck.get("thr", 0.0)
    m.eval()
    return arch, method, int(seed), m


def main():
    C = prepare()
    out_csv = os.path.join(HERE, "results", "mpra_runs.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    have = set(zip(done.get("arch", []), done.get("method", []), done.get("seed", [])))
    rows = done.to_dict("records")
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt"))):
        a, me, s = os.path.basename(path)[:-3].split("__")
        if (a, me, int(s)) in have:
            continue
        arch, method, seed, m = load(path)
        rows.append(dict(arch=arch, method=method, seed=seed, **score(m, C)))
        del m
        torch.cuda.empty_cache()
    R = pd.DataFrame(rows)
    R.to_csv(out_csv, index=False)
    C["V"].to_csv(os.path.join(HERE, "results", "mpra_variants_used.csv"), index=False)
    cols = ["spearman_all", "spearman_mean_element", "tata_mean_delta", "tata_sign_agree",
            "thal_mean_delta", "thal_damage_percentile", "tert_c228t_delta", "tert_c228t_activation_percentile"]
    print(R.groupby(["arch", "method"])[cols].mean().round(3).to_string())
    return R


if __name__ == "__main__":
    main()
