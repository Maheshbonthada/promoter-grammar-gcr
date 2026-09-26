"""Generate every results table (markdown) directly from the result files -> paper/tables.md"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "paper", "tables.md")
NAMES = {"baseline": "Baseline", "randaug": "Random-swap aug.", "hardneg": "Hard negatives",
         "gcr_rank": "GCR-rank (ours)", "gcr": "GCR (ours)", "gcr_atlas": "GCR-Atlas (ours)"}
ORDER = list(NAMES)
ARCHS = ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def table_main():
    S = json.load(open(os.path.join(RES, "study.json")))["runs"]
    rows = ["| Model | Method | n seeds | Test AUROC | Test MCC | Grammar discr. | Δ TATA displaced | abs Δ control | Δ point mutation |",
            "|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for m in ORDER:
            v = [S[k] for k in S if k.startswith(f"{a}|{m}|")]
            if not v:
                continue
            g = lambda key, d=3: ms([x[key] for x in v], d)
            rows.append(f"| {a} | {NAMES[m]} | {len(v)} | {g('auroc')} | {g('mcc')} | {g('grammar_disc', 2)} | "
                        f"{g('delta_tata', 2)} | {g('delta_ctrl_abs', 2)} | {g('delta_mut', 2)} |")
    return "### Table 1. Main comparison (mean ± s.d. over seeds; grammar probe = 76 held-out TATA promoters)\n\n" + "\n".join(rows)


def table_freq():
    p = os.path.join(RES, "freq_study.json")
    if not os.path.exists(p):
        return ""
    F = pd.DataFrame(json.load(open(p)).values())
    rows = ["| TATA promoters in training | Method | Grammar discr. | Δ point mutation | Test AUROC |", "|---|---|---|---|---|"]
    for n in sorted(F.n_tata.unique()):
        for m in ["baseline", "gcr"]:
            s = F[(F.n_tata == n) & (F.method == m)]
            if len(s):
                rows.append(f"| {n} ({100 * n / 3000:.0f}%) | {NAMES[m]} | {ms(s.grammar_disc, 2)} | "
                            f"{ms(s.delta_mut, 2)} | {ms(s.auroc)} |")
    return "### Table 2. Rare-rule experiment (Transformer; 3,000 promoters + 3,000 background)\n\n" + "\n".join(rows)


def table_external():
    p = os.path.join(RES, "atlas_study.json")
    if not os.path.exists(p):
        return ""
    S = json.load(open(p))["runs"]
    st = pd.read_csv(os.path.join(RES, "mpra_stats.csv")) if os.path.exists(os.path.join(RES, "mpra_stats.csv")) else None
    rows = ["| Model | Method | n seeds | Test AUROC | TATA grammar | 60-rule grammar | MPRA ρ (per seed) | MPRA ρ, seed ensemble [95% CI] | Δρ vs baseline [95% CI] | β-thal damage pct | TERT C228T activation pct |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for m in ["baseline", "gcr", "gcr_atlas"]:
            v = [S[k] for k in S if k.startswith(f"{a}|{m}|")]
            if not v:
                continue
            g = lambda key, d=3: ms([x[key] for x in v], d)
            e, dd = "", ""
            if st is not None:
                r = st[(st.arch == a) & (st.method == m)]
                if len(r):
                    r = r.iloc[0]
                    e = f"{r.spearman:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                    if m != "baseline" and not pd.isna(r.get("diff_vs_baseline", np.nan)):
                        dd = f"{r.diff_vs_baseline:+.3f} [{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]"
            rows.append(f"| {a} | {NAMES[m]} | {len(v)} | {g('auroc')} | {g('tata_gd', 2)} | {g('atlas_gd_mean', 2)} | "
                        f"{g('spearman_all')} | {e} | {dd} | {g('thal_damage_percentile', 0)} | "
                        f"{g('tert_c228t_activation_percentile', 0)} |")
    return ("### Table 3. External validation on measured variant effects (582 MPRA variants in 9 disease promoters) "
            "and multi-rule grammar\n\n" + "\n".join(rows))


def table_tata_promoters():
    p = os.path.join(RES, "mpra_stats.csv")
    if not os.path.exists(p):
        return ""
    st = pd.read_csv(p)
    if "rho_TATA_promoters" not in st:
        return ""
    rows = ["| Model | Method | ρ TATA promoters (HBB, HBG1, MSMB) | Δ vs baseline [95% CI] | ρ other promoters | ρ HBB | ρ HBG1 | ρ MSMB |",
            "|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for m in ORDER:
            r = st[(st.arch == a) & (st.method == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            dd = "" if m == "baseline" or pd.isna(r.get("tata_prom_diff", np.nan)) else \
                f"{r.tata_prom_diff:+.3f} [{r.tata_prom_diff_lo:+.3f}, {r.tata_prom_diff_hi:+.3f}]"
            rows.append(f"| {a} | {NAMES[m]} | {r.rho_TATA_promoters:.3f} | {dd} | {r.rho_other_promoters:.3f} | "
                        f"{r.rho_HBB:.3f} | {r.rho_HBG1:.3f} | {r.rho_MSMB:.3f} |")
    return "### Table 4. Targeted effect: TATA-box promoters vs other promoters (seed-ensembled predictions)\n\n" + "\n".join(rows)


def table_atlas(top=30):
    A = pd.read_csv(os.path.join(RES, "atlas.csv"))
    rules = json.load(open(os.path.join(RES, "atlas_study.json")))["rules"] if os.path.exists(
        os.path.join(RES, "atlas_study.json")) else []
    rows = ["| Rank | JASPAR ID | TF | Peak window (bp from TSS) | Enrichment (obs/exp) | log10 p | Used as GCR-Atlas rule |",
            "|---|---|---|---|---|---|---|"]
    used = {r["id"] for r in rules}
    for i, r in enumerate(A.head(top).itertuples(), 1):
        rows.append(f"| {i} | {r.id} | {r.name} | {r.peak_start_rel_tss:+d}..{r.peak_start_rel_tss + 9:+d} | "
                    f"{r.enrichment:.2f} | {r.log10p:.1f} | {'yes' if r.id in used else ''} |")
    return f"### Table 5. JASPAR 2026 positional atlas: top {top} of 1,019 motifs\n\n" + "\n".join(rows)


def table_sensitivity():
    p = os.path.join(RES, "sensitivity.json")
    if not os.path.exists(p):
        return ""
    S = json.load(open(p))
    rows = ["| λ rank | λ invariance | margin | Test AUROC | Grammar discr. | Δ point mutation | MPRA ρ | β-thal damage pct |",
            "|---|---|---|---|---|---|---|---|"]
    for v in S.values():
        rows.append(f"| {v['lam_rank']} | {v['lam_inv']} | {v['margin']} | {v['auroc']:.3f} | {v['grammar_disc']:.2f} | "
                    f"{v['delta_mut']:+.2f} | {v['mpra_rho']:.3f} | {v['thal_pct']:.0f} |")
    return ("### Table 6. GCR hyperparameter sensitivity (Transformer, seed 0; baseline: AUROC 0.940, "
            "grammar 0.40, MPRA ρ 0.043, β-thal pct 44)\n\n" + "\n".join(rows))


def table_tbp():
    p = os.path.join(RES, "tbp_binding.json")
    if not os.path.exists(p):
        return ""
    T = json.load(open(p))
    ref = T["reference"]["TBP_PWM"]
    rows = ["| Model | Method | n seeds | Spearman ρ vs ΔΔG (p) | Pearson r | Weakened-binding SNPs called damaging |",
            "|---|---|---|---|---|---|",
            f"| JASPAR TBP PWM (reference) | – | – | {ref['spearman']:+.2f} ({ref['spearman_p']:.3f}) | "
            f"{ref['pearson']:+.2f} | {ref['frac_damaging_called']:.0%} |"]
    for a in ARCHS:
        for m in ORDER:
            r = T["ensemble"].get(f"{a}|{m}")
            if r is None:
                continue
            rows.append(f"| {a} | {NAMES[m]} | {r['n_seeds']} | {r['spearman']:+.2f} ({r['spearman_p']:.3f}) | "
                        f"{r['pearson']:+.2f} | {r['frac_damaging_called']:.0%} |")
    n_weak = sum(1 for v in pd.read_csv(os.path.join(RES, "tbp_binding_variants.csv")).ddG if v > 0.5925 * np.log(2))
    return (f"### Table 7. Molecular validation: model effects vs measured TBP–DNA binding free energy "
            f"({T['n']} disease TATA-box SNPs in 9 promoters; ΔΔG = RT ln(K_D,mut/K_D,wt) from EMSA; "
            f"seed-ensembled predictions; last column: {n_weak} SNPs with ≥ 2-fold weaker binding)\n\n" + "\n".join(rows))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    parts = [t for t in [table_main(), table_freq(), table_external(), table_tata_promoters(), table_atlas(), table_sensitivity(), table_tbp()] if t]
    open(OUT, "w", encoding="utf-8").write("# Results tables (auto-generated by make_tables.py)\n\n" + "\n\n".join(parts) + "\n")
    print(open(OUT, encoding="utf-8").read())
