"""Re-score saved checkpoints with the corrected grammar metrics (no retraining).

    python rescore.py OUT_NAME ARCH [ARCH ...]     # one worker; writes results/rescore_OUT_NAME.json
    python rescore.py --merge                      # merge all workers into study.json / atlas_study.json

grammar_disc is now AUROC(signed TATA-displacement effect vs signed control effect), whose null is 0.5.
The original version compared the signed effect with the absolute control effect (null ~ 0.25); it is kept
as grammar_disc_asym. heldout_gd uses only edits never seen in training (TATA double substitution vs a
matched double substitution in a non-TATA block).
"""
import glob, json, os, sys, time
import numpy as np
import torch
from train import evaluate
from run_study import build, HOLD, chrom
from evaluate_mpra import load
from motifs import load_task
from gcr_atlas import select_rules, thresholds, find_instances, dedupe_rules, atlas_grammar

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GRAMMAR_KEYS = ("grammar_disc", "grammar_disc_asym", "heldout_gd", "delta_tata", "delta_ctrl", "delta_ctrl_abs",
                "delta_mut", "delta_mut_ctrl", "tuning_curve")


def atlas_setup(Xtr, ytr, Xg, yg):
    """The same 60 rules and held-out instances as run_atlas_study.py."""
    rules = select_rules()
    X, y, n = load_task("promoter_all", "train")
    thr = thresholds(rules, X[~np.isin(chrom(n), list(HOLD)) & (y == 0)])
    inst_tr = find_instances(Xtr, ytr, rules, thr)
    keep = dedupe_rules(rules, inst_tr)
    rules, thr = [rules[k] for k in keep], thr[keep]
    return rules, find_instances(Xg, yg, rules, thr)


def worker(name, archs):
    out_path = os.path.join(RES, f"rescore_{name}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    Xtr, ytr, _, _, _, _, Xg, yg, tg = build()
    rules, inst_g = atlas_setup(Xtr, ytr, Xg, yg)
    print(f"[{name}] {len(rules)} rules, {len(inst_g)} held-out instances", flush=True)
    paths = [p for p in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt")))
             if os.path.basename(p).split("__")[0] in archs]
    for p in paths:
        key = "|".join(os.path.basename(p)[:-3].split("__"))
        if key in out:
            continue
        t = time.time()
        arch, method, seed, m = load(p)
        r = evaluate(m, Xg, yg, tg, seed=int(seed))
        per = atlas_grammar(m, Xg, inst_g, rules, seed=int(seed))
        out[key] = dict(**{k: r[k] for k in GRAMMAR_KEYS}, atlas_per_rule=per,
                        atlas_gd_mean=float(np.mean([v["gd"] for v in per.values()])),
                        atlas_gd_asym_mean=float(np.mean([v["gd_asym"] for v in per.values()])))
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"[{name}] {key:28s} GD {r['grammar_disc']:.2f} (asym {r['grammar_disc_asym']:.2f}) "
              f"held-out GD {r['heldout_gd']:.2f} atlas GD {out[key]['atlas_gd_mean']:.2f} ({time.time() - t:.0f}s)",
              flush=True)
        del m
        torch.cuda.empty_cache()


def merge():
    new = {}
    for f in glob.glob(os.path.join(RES, "rescore_*.json")):
        new.update(json.load(open(f)))
    S = json.load(open(os.path.join(RES, "study.json")))
    A = json.load(open(os.path.join(RES, "atlas_study.json")))
    ns = na = 0
    for key, v in new.items():
        if key in S["runs"]:
            S["runs"][key].update({k: v[k] for k in GRAMMAR_KEYS})
            ns += 1
        if key in A["runs"]:
            A["runs"][key].update(tata_gd=v["grammar_disc"], tata_gd_asym=v["grammar_disc_asym"],
                                  tata_heldout_gd=v["heldout_gd"], tata_dmut=v["delta_mut"],
                                  atlas_gd_mean=v["atlas_gd_mean"], atlas_gd_asym_mean=v["atlas_gd_asym_mean"],
                                  atlas_per_rule=v["atlas_per_rule"])
            na += 1
    S["info"]["grammar_metric"] = ("grammar_disc = AUROC(signed displacement vs signed control effect), null 0.5; "
                                   "grammar_disc_asym = original signed-vs-absolute version, null ~0.25")
    json.dump(S, open(os.path.join(RES, "study.json"), "w"), indent=1)
    json.dump(A, open(os.path.join(RES, "atlas_study.json"), "w"), indent=1)
    print(f"merged {len(new)} checkpoints: {ns} into study.json, {na} into atlas_study.json")


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge()
    else:
        worker(sys.argv[1], sys.argv[2:])
