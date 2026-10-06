"""Confidence intervals for the gain over null that include probe sampling, not only seed variation.

Grammar discrimination is computed on 76 probe promoters, so a model's GD carries sampling error from which
promoters happen to be in the probe. This recomputes the per-promoter TATA and control displacement effects
for every baseline, GCR and null checkpoint (the same edits and random draws as train.evaluate, checked
against the stored GD), then resamples promoters: in each replicate the same promoters are drawn for every
model, GD is recomputed per model, averaged over seeds, and the gain over null is taken. The normalised gain
divides by the room above the null, (GD - null) / (1 - null).

    python gain_ci.py        # -> results/gain_ci.json
"""
import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from counterfactuals import control_swap, displace_tata
from evaluate_mpra import load
from run_study import build
from train import predict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "gain_ci.json")
ARCHS = ["Transformer", "CNN-GlobalPool", "CNN-Dense", "NT-v2-50M", "NT-v2-100M", "NT-v2-500M-LoRA"]
B = 5000
R = 8


def effects(model, X, tata, seed):
    """Per-promoter displacement effects, drawn exactly as in train.evaluate."""
    rng = np.random.default_rng(1000 + seed)
    z = predict(model, X)
    d_tata = np.mean([z - predict(model, displace_tata(X, tata, rng)) for _ in range(R)], 0)
    d_ctrl = np.mean([z - predict(model, control_swap(X, rng)) for _ in range(R)], 0)
    return d_tata, d_ctrl


def gd(d_tata, d_ctrl):
    lab = np.r_[np.ones(len(d_tata)), np.zeros(len(d_ctrl))]
    return roc_auc_score(lab, np.r_[d_tata, d_ctrl])


def main():
    import nt_lora
    nt_lora.register()          # the 500M LoRA model is not in NT_ARCHS
    *_, Xg, yg, tg = build()
    idx = np.where((tg >= 0) & (yg == 1))[0]
    X, tata = Xg[idx], tg[idx]
    study = json.load(open(os.path.join(HERE, "results", "study.json")))["runs"]
    nulls = json.load(open(os.path.join(HERE, "results", "review2_study.json")))
    cache = json.load(open(OUT))["effects"] if os.path.exists(OUT) else {}
    paths = {}
    for a in ARCHS:
        for m in ("baseline", "gcr"):
            paths.update({"%s|%s|%s" % (a, m, os.path.basename(p)[:-3].split("__")[-1]): p
                          for p in glob.glob(os.path.join(HERE, "runs", "%s__%s__*.pt" % (a, m)))})
        paths.update({"notata|%s|baseline|%s" % (a, os.path.basename(p)[:-3].split("__")[-1]): p
                      for p in glob.glob(os.path.join(HERE, "runs_review2", "notata__%s__baseline__*.pt" % a))})
    for key, p in sorted(paths.items()):
        stored = (nulls if key.startswith("notata") else study).get(key, {}).get("grammar_disc")
        if key in cache or stored is None:
            continue
        arch, method, seed = key.split("|")[-3:]
        _, _, _, model = load(p) if not key.startswith("notata") else _load_null(p, arch, method, seed)
        dt, dc = effects(model, X, tata, int(seed))
        g = gd(dt, dc)
        # the pretrained models are not bit-reproducible on GPU; anything beyond rounding is an error
        assert abs(g - stored) < 3e-3, (key, g, stored)
        cache[key] = dict(d_tata=dt.tolist(), d_ctrl=dc.tolist(), gd_stored=stored)
        json.dump(dict(effects=cache), open(OUT, "w"))
        print("%-36s GD %.4f (stored %.4f)" % (key, g, stored), flush=True)
        del model
        torch.cuda.empty_cache()

    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(idx), len(idx)) for _ in range(B)]
    summary = {}
    for a in ARCHS:
        groups = {m: [k for k in cache if k.startswith(m + "|" + a + "|" if m == "notata" else a + "|" + m + "|")]
                  for m in ("baseline", "gcr", "notata")}
        arr = {m: [(np.array(cache[k]["d_tata"]), np.array(cache[k]["d_ctrl"])) for k in ks] for m, ks in groups.items()}

        def mean_gd(m, s=slice(None)):
            return np.mean([gd(t[s], c[s]) for t, c in arr[m]])

        boot = np.array([[mean_gd("baseline", s), mean_gd("gcr", s), mean_gd("notata", s)] for s in draws])
        # hierarchical: seeds resampled within each condition as well as promoters
        srng = np.random.default_rng(1)
        per = {m: np.array([[gd(t[s], c[s]) for t, c in arr[m]] for s in draws]) for m in arr}
        pick = {m: srng.integers(0, per[m].shape[1], (B, per[m].shape[1])) for m in per}
        hb = {m: np.take_along_axis(per[m], pick[m], 1).mean(1) for m in per}
        b, g, n = mean_gd("baseline"), mean_gd("gcr"), mean_gd("notata")
        out = dict(n_seeds={m: len(v) for m, v in groups.items()}, n_promoters=int(len(idx)))
        for name, est, rep in [("gain", b - n, boot[:, 0] - boot[:, 2]),
                               ("gcr_gain", g - n, boot[:, 1] - boot[:, 2]),
                               ("norm_gain", (b - n) / (1 - n), (boot[:, 0] - boot[:, 2]) / (1 - boot[:, 2])),
                               ("gcr_norm_gain", (g - n) / (1 - n), (boot[:, 1] - boot[:, 2]) / (1 - boot[:, 2]))]:
            lo, hi = np.percentile(rep, [2.5, 97.5])
            out[name] = dict(est=float(est), lo=float(lo), hi=float(hi), p_le0=float(np.mean(rep <= 0)))
        for name, rep in [("gain_hier", hb["baseline"] - hb["notata"]),
                          ("norm_gain_hier", (hb["baseline"] - hb["notata"]) / (1 - hb["notata"]))]:
            lo, hi = np.percentile(rep, [2.5, 97.5])
            out[name] = dict(lo=float(lo), hi=float(hi), p_le0=float(np.mean(rep <= 0)))
        summary[a] = out
        print("%-16s gain %+.3f [%+.3f, %+.3f]  norm %.2f [%.2f, %.2f]  | GCR norm %.2f [%.2f, %.2f]"
              % (a, out["gain"]["est"], out["gain"]["lo"], out["gain"]["hi"], out["norm_gain"]["est"],
                 out["norm_gain"]["lo"], out["norm_gain"]["hi"], out["gcr_norm_gain"]["est"],
                 out["gcr_norm_gain"]["lo"], out["gcr_norm_gain"]["hi"]), flush=True)
        print("%-16s   with seeds resampled too: gain [%+.3f, %+.3f] P(<=0) %.3f  norm [%.2f, %.2f]"
              % ("", out["gain_hier"]["lo"], out["gain_hier"]["hi"], out["gain_hier"]["p_le0"],
                 out["norm_gain_hier"]["lo"], out["norm_gain_hier"]["hi"]), flush=True)
    json.dump(dict(summary=summary, effects=cache), open(OUT, "w"))


def _load_null(path, arch, method, seed):
    """Null checkpoints are named notata__ARCH__baseline__SEED; load() expects ARCH__METHOD__SEED."""
    from models import ARCHS as A
    if arch.startswith("NT-") and arch not in A:
        from nt_model import NT_ARCHS
        A[arch] = NT_ARCHS[arch]
    from train import DEV
    ck = torch.load(path, map_location=DEV)
    m = A[arch]().to(DEV)
    m.load_state_dict(ck["state"])
    m.thr = ck.get("thr", 0.0)
    m.eval()
    return arch, method, int(seed), m


if __name__ == "__main__":
    main()
