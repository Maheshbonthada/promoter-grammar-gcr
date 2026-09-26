"""Multi-rule study: GCR-Atlas (all positionally constrained JASPAR motifs) vs GCR-TATA vs baseline.

Same data protocol as run_study.py. Baseline and GCR-TATA models are re-used from
runs/ (identical data and seeds); GCR-Atlas models are trained here.
Outputs results/atlas_study.json.
"""
import json, os, sys, time
import numpy as np
import torch
from motifs import load_task
from models import ARCHS
from train import evaluate
from run_study import build, chrom, HOLD
from gcr_atlas import select_rules, dedupe_rules, thresholds, find_instances, train_atlas, atlas_grammar
from evaluate_mpra import prepare, score, load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "atlas_study.json")
RUNS = os.path.join(HERE, "runs")
LR = {"CNN-GlobalPool": 1e-3, "Transformer": 5e-4, "NT-v2-50M": 5e-5, "NT-v2-100M": 5e-5}
EPOCHS = {"NT-v2-50M": 3, "NT-v2-100M": 3}
BS = {"NT-v2-100M": 64}             # 8 GB GPU: halve the batch for the 100M model
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]


def train_negatives():
    X, y, n = load_task("promoter_all", "train")
    keep = ~np.isin(chrom(n), list(HOLD))
    return X[keep & (y == 0)]


def main():
    archs = sys.argv[1:] or ["Transformer", "CNN-GlobalPool"]
    rules = select_rules()
    Xtr, ytr, ttr, Xte, yte, tte, Xg, yg, tg = build()
    thr = thresholds(rules, train_negatives())
    inst_tr = find_instances(Xtr, ytr, rules, thr)
    keep = dedupe_rules(rules, inst_tr)
    remap = {k: i for i, k in enumerate(keep)}
    print(f"{len(rules)} candidate windows -> {len(keep)} non-redundant rules", flush=True)
    rules, thr = [rules[k] for k in keep], thr[keep]
    inst_tr = np.array([(s, remap[k], a, w) for s, k, a, w in inst_tr if k in remap], dtype=np.int64)
    print("rules:", ", ".join(f"{r['name']}@{r['lo'] - 49 + 3}..{r['hi'] - 49 - 3}" for r in rules), flush=True)
    inst_g = find_instances(Xg, yg, rules, thr)
    print(f"rule instances: train {len(inst_tr)}, held-out probe {len(inst_g)}", flush=True)
    C = prepare()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out["rules"] = [dict(id=r["id"], name=r["name"], window=[r["lo"] - 49, r["hi"] - 49], width=r["w"],
                         enrichment=r["enrichment"]) for r in rules]
    out.setdefault("runs", {})
    for arch in archs:
        if arch.startswith("NT-") and arch not in ARCHS:
            from nt_model import NT_ARCHS
            ARCHS[arch] = NT_ARCHS[arch]
        for method in os.environ.get("ATLAS_METHODS", "baseline,gcr,gcr_atlas").split(","):
            for seed in SEEDS:
                key = f"{arch}|{method}|{seed}"
                if key in out["runs"]:
                    continue
                path = os.path.join(RUNS, f"{arch}__{method}__{seed}.pt")
                t = time.time()
                if method == "gcr_atlas" and not os.path.exists(path):
                    m = train_atlas(ARCHS[arch](), Xtr, ytr, inst_tr, rules, epochs=EPOCHS.get(arch, 12),
                                    lr=LR[arch], seed=seed, bs=BS.get(arch, 128))
                    torch.save(dict(state=m.state_dict(), thr=m.thr), path)
                elif os.path.exists(path):
                    _, _, _, m = load(path)
                else:
                    print(f"  skip {key}: no checkpoint", flush=True)
                    continue
                r_test = evaluate(m, Xte, yte, tte, seed=seed, R=1)
                r_tata = evaluate(m, Xg, yg, tg, seed=seed)
                per = atlas_grammar(m, Xg, inst_g, rules, seed=seed)
                res = dict(auroc=r_test["auroc"], mcc=r_test["mcc"],
                           tata_gd=r_tata["grammar_disc"], tata_dmut=r_tata["delta_mut"],
                           atlas_gd_mean=float(np.mean([v["gd"] for v in per.values()])),
                           atlas_per_rule=per, **score(m, C))
                out["runs"][key] = res
                disk = json.load(open(OUT)) if os.path.exists(OUT) else {}
                disk.setdefault("runs", {})[key] = res          # merge: parallel jobs share this file
                disk["rules"] = out["rules"]
                json.dump(disk, open(OUT, "w"), indent=1)
                print(f"{key:30s} auroc {res['auroc']:.3f} TATA-GD {res['tata_gd']:.2f} atlas-GD {res['atlas_gd_mean']:.2f} "
                      f"MPRA rho {res['spearman_all']:+.3f} thal-pct {res['thal_damage_percentile']:.0f} "
                      f"C228T-pct {res['tert_c228t_activation_percentile']:.0f} ({time.time() - t:.0f}s)", flush=True)
                del m
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
