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


SHIFTS = np.arange(-14, 27, 2)            # as in train.py


def shift_sens(curve):
    """Mean logit drop when the TATA block is moved by >= 8 bp (never used in training)."""
    return float(-np.asarray(curve)[np.abs(SHIFTS) >= 8].mean())


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def table_main():
    S = json.load(open(os.path.join(RES, "study.json")))["runs"]
    rows = ["| Model | Method | n seeds | Test AUROC | Test MCC | Grammar discr. | Shift sensitivity | TATA recognition | Δ TATA displaced | Δ control swap | Δ TATA double substitution | Δ control double substitution |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for m in ORDER:
            v = [S[k] for k in S if k.startswith(f"{a}|{m}|")]
            if not v:
                continue
            g = lambda key, d=3: ms([x[key] for x in v], d)
            rows.append(f"| {a} | {NAMES[m]} | {len(v)} | {g('auroc')} | {g('mcc')} | {g('grammar_disc', 2)} | "
                        f"{ms([shift_sens(x['tuning_curve']) for x in v], 2)} | "
                        f"{g('heldout_gd', 2)} | {g('delta_tata', 2)} | {g('delta_ctrl', 2)} | {g('delta_mut', 2)} | "
                        f"{g('delta_mut_ctrl', 2)} |")
    return ("### Table 1. Main comparison (mean ± s.d. over seeds; grammar probe = 76 held-out canonical-TATA promoters; "
            "grammar discrimination = AUROC of TATA-edit vs control-edit effects, 0.5 = no positional knowledge; "
            "shift sensitivity = mean logit drop when the TATA box is moved by at least 8 bp (up to 14 bp upstream or 26 bp downstream), never used in training; "
            "TATA recognition = AUROC of TATAAAA→TGTGAAA vs a matched double substitution in a non-TATA block, never "
            "used in training, with a reference level without TATA knowledge given by the 0-TATA rows of the rare-rule "
            "table; Δ = drop in logit)\n\n" + "\n".join(rows))


def table_freq():
    p = os.path.join(RES, "freq_study.json")
    if not os.path.exists(p):
        return ""
    F = pd.DataFrame(json.load(open(p)).values())
    rows = ["| TATA promoters in training | Method | Grammar discr. | TATA recognition (held-out) | Δ TATA double substitution | Test AUROC |",
            "|---|---|---|---|---|---|"]
    for n in sorted(F.n_tata.unique()):
        for m in ["baseline", "gcr"]:
            s = F[(F.n_tata == n) & (F.method == m)]
            if len(s):
                rows.append(f"| {n} ({100 * n / 3000:.0f}%) | {NAMES[m]} | {ms(s.grammar_disc, 2)} | "
                            f"{ms(s.heldout_gd, 2)} | {ms(s.delta_mut, 2)} | {ms(s.auroc)} |")
    return "### Table 2. Rare-rule experiment (Transformer; 3,000 promoters + 3,000 background)\n\n" + "\n".join(rows)


def table_external():
    p = os.path.join(RES, "atlas_study.json")
    if not os.path.exists(p):
        return ""
    S = json.load(open(p))["runs"]
    st = pd.read_csv(os.path.join(RES, "mpra_stats.csv")) if os.path.exists(os.path.join(RES, "mpra_stats.csv")) else None
    robust = robust_rules()
    rows = ["| Model | Method | n seeds | Test AUROC | TATA grammar | Rule grammar (49 scorable rules) | Rule grammar (rules passing the stricter null) | MPRA ρ (per seed) | MPRA ρ, seed ensemble [95% CI] | Δρ vs baseline [95% CI] | β-thal damage pct | TERT C228T activation pct |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
            rob = ms([np.mean([r["gd"] for k, r in x["atlas_per_rule"].items() if k in robust]) for x in v], 2) \
                if robust else ""
            rows.append(f"| {a} | {NAMES[m]} | {len(v)} | {g('auroc')} | {g('tata_gd', 2)} | {g('atlas_gd_mean', 2)} | {rob} | "
                        f"{g('spearman_all')} | {e} | {dd} | {g('thal_damage_percentile', 0)} | "
                        f"{g('tert_c228t_activation_percentile', 0)} |")
    return ("### Table 3. External validation on measured variant effects (582 MPRA variants in 9 disease promoters) "
            "and multi-rule grammar\n\n" + "\n".join(rows))


def robust_rules():
    """Rule keys ('NAME@lo..hi') that pass the local dinucleotide-shuffle null (atlas_null_check.py)."""
    p = os.path.join(RES, "atlas_null_check.csv")
    if not os.path.exists(p):
        return set()
    N = pd.read_csv(p)
    return {f"{r.name}@{r.window}" for r in N[N.passes].itertuples()}


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
    S = pd.DataFrame(json.load(open(p)).values())
    rows = ["| λ rank | λ invariance | margin | n seeds | Test AUROC | Grammar discr. | TATA recognition (held-out) | Δ TATA double substitution | MPRA ρ | β-thal damage pct |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for (r, i, mg), g in S.groupby(["lam_rank", "lam_inv", "margin"], sort=False):
        rows.append(f"| {r:g} | {i:g} | {mg:g} | {len(g)} | {ms(g.auroc)} | {ms(g.grammar_disc, 2)} | "
                    f"{ms(g.heldout_gd, 2)} | {ms(g.delta_mut, 2)} | {ms(g.mpra_rho)} | {ms(g.thal_pct, 0)} |")
    return "### Table 6. GCR hyperparameter sensitivity (Transformer; mean ± s.d. over seeds)\n\n" + "\n".join(rows)


def table_tbp():
    p = os.path.join(RES, "tbp_binding.json")
    if not os.path.exists(p):
        return ""
    T = json.load(open(p))
    ref = T["reference"]["TBP_PWM"]
    rs = pd.read_csv(os.path.join(RES, "robust_stats.csv")) if os.path.exists(os.path.join(RES, "robust_stats.csv")) else None
    rows = ["| Model | Method | n seeds | Spearman ρ vs ΔΔG (p) | Δρ vs baseline [95% CI] (two-sided p) | Pearson r | Weakened-binding SNPs called damaging |",
            "|---|---|---|---|---|---|---|",
            f"| JASPAR TBP PWM (reference) | – | – | {ref['spearman']:+.2f} ({ref['spearman_p']:.3f}) | – | "
            f"{ref['pearson']:+.2f} | {ref['frac_damaging_called']:.0%} |"]
    for a in ARCHS:
        for m in ORDER:
            r = T["ensemble"].get(f"{a}|{m}")
            if r is None:
                continue
            dd = ""
            if rs is not None and m != "baseline":
                q = rs[(rs.analysis == "TBP") & (rs.arch == a) & (rs.method == m)]
                if len(q):
                    q = q.iloc[0]
                    dd = f"{q.delta:+.2f} [{q.ci_lo_variant:+.2f}, {q.ci_hi_variant:+.2f}] ({q.p_variant:.2f})"
            rows.append(f"| {a} | {NAMES[m]} | {r['n_seeds']} | {r['spearman']:+.2f} ({r['spearman_p']:.3f}) | {dd} | "
                        f"{r['pearson']:+.2f} | {r['frac_damaging_called']:.0%} |")
    n_weak = sum(1 for v in pd.read_csv(os.path.join(RES, "tbp_binding_variants.csv")).ddG if v > 0.5925 * np.log(2))
    return (f"### Table 7. Molecular validation: model effects vs measured TBP–DNA binding free energy "
            f"({T['n']} disease TATA-box SNPs in 9 promoters; ΔΔG = RT ln(K_D,mut/K_D,wt) from EMSA; "
            f"seed-ensembled predictions; last column: {n_weak} SNPs with ≥ 2-fold weaker binding)\n\n" + "\n".join(rows))


def table_composition():
    p = os.path.join(RES, "composition_baseline.json")
    if not os.path.exists(p):
        return ""
    C = json.load(open(p))
    rows = ["| Position-free features | n features | Test AUROC (chr20/21) |", "|---|---|---|"]
    rows += [f"| {k} | {v['n_features']} | {v['auroc']:.3f} |" for k, v in C.items()]
    return ("### Table 8. The composition shortcut: logistic regression on position-free features, same training set "
            "and test split as the neural models (neural models: AUROC 0.940–0.951)\n\n" + "\n".join(rows))


def table_robust_mpra():
    p = os.path.join(RES, "robust_stats.csv")
    if not os.path.exists(p):
        return ""
    R = pd.read_csv(p)
    R = R[R.analysis == "MPRA"]
    rows = ["| Model | Method | Δρ vs baseline | 95% CI, variant bootstrap (p) | 95% CI, promoter bootstrap (p) | Holm p (promoter) | Δρ TATA promoters [95% CI] (p) | Holm p (TATA promoters) |",
            "|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for m in ORDER:
            q = R[(R.arch == a) & (R.method == m)]
            if not len(q):
                continue
            q = q.iloc[0]
            h1 = "" if pd.isna(q.get("p_promoter_holm")) else f"{q.p_promoter_holm:.3f}"
            h2 = "" if pd.isna(q.get("p_tata_prom_holm")) else f"{q.p_tata_prom_holm:.3f}"
            rows.append(f"| {a} | {NAMES[m]} | {q.delta:+.3f} | [{q.ci_lo_variant:+.3f}, {q.ci_hi_variant:+.3f}] ({q.p_variant:.3f}) | "
                        f"[{q.ci_lo_promoter:+.3f}, {q.ci_hi_promoter:+.3f}] ({q.p_promoter:.3f}) | {h1} | "
                        f"{q.delta_tata_prom:+.3f} [{q.ci_lo_tata_prom:+.3f}, {q.ci_hi_tata_prom:+.3f}] ({q.p_tata_prom:.3f}) | {h2} |")
    return ("### Table 9. Stricter statistics for the MPRA comparison (two-sided; the promoter bootstrap resamples the 9 "
            "promoters; Holm correction over the GCR and GCR-Atlas comparisons)\n\n" + "\n".join(rows))


def table_atlas_null():
    p = os.path.join(RES, "atlas_null_check.csv")
    if not os.path.exists(p):
        return ""
    N = pd.read_csv(p)
    rows = ["| Rule (TF, window) | Hits in promoters | Hits after local dinucleotide shuffle | Enrichment | log10 p | Passes |",
            "|---|---|---|---|---|---|"]
    rows += [f"| {r.name} {r.window} | {r.hits_real} | {r.hits_shuffled:.1f} | {r.enrichment:.2f} | {r.log10p:.1f} | "
             f"{'yes' if r.passes else 'no'} |" for r in N.itertuples()]
    return (f"### Table 10. Stricter null for the 60 GCR-Atlas rules: promoters shuffled within 20-bp segments preserving "
            f"all dinucleotides, including CpG ({int(N.passes.sum())} of {len(N)} rules pass; Bonferroni, "
            f"enrichment ≥ 1.5)\n\n" + "\n".join(rows))


def _probe():
    p = os.path.join(RES, "review2_probe.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def table_matched_negatives():
    p = os.path.join(RES, "review2_study.json")
    if not os.path.exists(p):
        return ""
    R, P = json.load(open(p)), _probe()
    S = json.load(open(os.path.join(RES, "study.json")))["runs"]
    EXP = {"main": "Genomic negatives (main study)", "negmatch": "Composition-matched negatives",
           "notata": "No canonical-TATA promoters in training"}
    ARCHS = ["Transformer", "CNN-GlobalPool", "CNN-Dense", "NT-v2-50M", "NT-v2-100M"]
    rows = ["| Training set | Model | Method | n seeds | Test AUROC | AUROC, matched test | Grammar discr. | "
            "Shift sensitivity | TATA recognition | TATA implant preference |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for e in EXP:
        for a in ARCHS:
            for m in ["baseline", "gcr"]:
                if e == "main":
                    ks = [k for k in S if k.startswith(f"{a}|{m}|")]
                    v, pv = [S[k] for k in ks], [P[k] for k in ks if k in P]
                    if not any(k.startswith(f"negmatch|{a}|{m}|") for k in R):
                        continue
                else:
                    ks = [k for k in R if k.startswith(f"{e}|{a}|{m}|")]
                    v, pv = [R[k] for k in ks], [P[k] for k in ks if k in P]
                if not v:
                    continue
                mt = ms([x["auroc_matched"] for x in v]) if "auroc_matched" in v[0] else "–"
                rows.append(f"| {EXP[e]} | {a} | {NAMES[m].replace(' (ours)', '')} | {len(v)} | {ms([x['auroc'] for x in v])} | "
                            f"{mt} | {ms([x['grammar_disc'] for x in v], 2)} | "
                            f"{ms([shift_sens(x['tuning_curve']) for x in v], 2)} | {ms([x['heldout_gd'] for x in v], 2)} | "
                            f"{ms([x['implant_pref'] for x in pv], 2) if pv else '–'} |")
    gc = next((x["gc_only_matched_auroc"] for x in R.values() if "gc_only_matched_auroc" in x), None)
    return ("### Table 11. Composition-matched negatives and no-TATA reference models. Composition-matched negatives are "
            "the training promoters shuffled within 20-bp segments preserving all dinucleotides, so GC and CpG content and "
            f"their profile along the window are identical in both classes (GC content alone: AUROC {gc:.3f} on the matched "
            "test set). Test AUROC uses the official chr20/21 split with genomic negatives; the matched test set pairs the "
            "same test promoters with their own local shuffles. No-TATA models were trained on the main training set with "
            "all 392 canonical-TATA promoters removed; their TATA recognition is the architecture-matched reference level "
            "without TATA knowledge from training. TATA implant preference is defined with the positional robustness tests\n\n" + "\n".join(rows))


def table_positional_robustness():
    P = _probe()
    if not P:
        return ""
    RS = {}
    for f in ("rescore_nt.json", "rescore_small.json"):
        RS.update(json.load(open(os.path.join(RES, f))))
    cut = lambda c, s: float(-np.asarray(c)[np.abs(SHIFTS) >= s].mean())
    rows = ["| Model | Method | n seeds | Shift sens. ≥ 4 bp | ≥ 8 bp | ≥ 12 bp | Logit s.d. (test set) | "
            "Shift sens. ≥ 8 bp / logit s.d. | Implant peak at −30 ± 2 bp | Implant preference (logits) | "
            "Implant preference / logit s.d. |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for m in ORDER:
            ks = [k for k in RS if k.startswith(f"{a}|{m}|") and k in P]
            if not ks:
                continue
            c, p = [RS[k]["tuning_curve"] for k in ks], [P[k] for k in ks]
            sd = [x["logit_sd"] for x in p]
            peak = sum(abs(x["implant_peak"] + 30) <= 2 for x in p)
            rows.append(f"| {a} | {NAMES[m]} | {len(ks)} | {ms([cut(x, 4) for x in c], 2)} | {ms([cut(x, 8) for x in c], 2)} | "
                        f"{ms([cut(x, 12) for x in c], 2)} | {ms(sd, 2)} | "
                        f"{ms([cut(x, 8) / s for x, s in zip(c, sd)], 2)} | {peak} of {len(ks)} | "
                        f"{ms([x['implant_pref'] for x in p], 2)} | {ms([x['implant_pref'] / x['logit_sd'] for x in p], 2)} |")
    return ("### Table 12. Robustness of the positional tests (held-out probe; no edit used in training). Shift "
            "sensitivity at three cut-offs, and in units of each model's logit s.d. on the official test set. TATA "
            "implantation: TATAAAA was written into 300 held-out promoters without a canonical TATA box at every start from "
            "−44 to +20 bp, and the gain was compared with that of a scrambled copy of the same bases (AAAATTA) at the same "
            "position. Implant peak: number of seeds whose TATA-specific gain peaks within 2 bp of −30. Implant preference: "
            "TATA-specific gain within 2 bp of −30 minus the mean gain at positions at least 10 bp away\n\n" + "\n".join(rows))


def table_controls():
    """Falsification controls: the same loss with a deliberately wrong rule."""
    import pandas as _pd
    S = json.load(open(os.path.join(RES, "study.json")))["runs"]
    mp = os.path.join(RES, "mpra_stats.csv")
    M = _pd.read_csv(mp) if os.path.exists(mp) else None
    ST = json.load(open(os.path.join(RES, "review2_strand.json"))) if os.path.exists(
        os.path.join(RES, "review2_strand.json")) else {}
    P = _probe()
    LBL = {"baseline": "Baseline", "gcr": "GCR (real rule)", "gcr_decoy": "Decoy GCR (wrong position)",
           "gcr_atlas": "GCR-Atlas (real rules)", "gcr_atlas_scram": "Scrambled-atlas GCR (wrong positions)"}
    rows = ["| Model | Training | n seeds | Test AUROC | Grammar discr. | TATA implant preference | MPRA ρ (seed ensemble) |",
            "|---|---|---|---|---|---|---|"]
    any_row = False
    for a_ in ARCHS:
        for m in ["baseline", "gcr", "gcr_decoy", "gcr_atlas", "gcr_atlas_scram"]:
            v = [x for k, x in S.items() if k.startswith(f"{a_}|{m}|")]
            if not v:
                continue
            any_row = True
            pv = [P[k] for k in S if k.startswith(f"{a_}|{m}|") and k in P]
            rho = ""
            if M is not None:
                q = M[(M.arch == a_) & (M.method == m)]
                rho = f"{q.spearman.iloc[0]:.3f}" if len(q) else ""
            rows.append(f"| {a_} | {LBL.get(m, m)} | {len(v)} | {ms([x['auroc'] for x in v])} | "
                        f"{ms([x['grammar_disc'] for x in v], 2)} | "
                        f"{ms([x['implant_pref'] for x in pv], 2) if pv else '–'} | {rho} |")
    if not any_row:
        return ""
    return ("### Table 13. Falsification controls. The decoy and scrambled-atlas models use the identical loss, "
            "margin, auxiliary batch and seeds, and differ only in the position the rule protects: a fixed "
            "downstream block at +71 (decoy), or every atlas window moved 80 bp from where the atlas found it "
            "enriched (scrambled). A gain that survives the wrong rule is not evidence for the rule\n\n" + "\n".join(rows))


def table_strand():
    p = os.path.join(RES, "review2_strand.json")
    if not os.path.exists(p):
        return ""
    ST = json.load(open(p))
    rows = ["| Model | Method | n seeds | Δ TATA reverse-complement | Δ control block | Net |", "|---|---|---|---|---|---|"]
    for a_ in ARCHS:
        for m in ORDER + ["gcr_decoy"]:
            v = [x for k, x in ST.items() if k.startswith(f"{a_}|{m}|")]
            if not v:
                continue
            rows.append(f"| {a_} | {NAMES.get(m, m)} | {len(v)} | {ms([x['strand_drop'] for x in v], 2)} | "
                        f"{ms([x['strand_ctrl'] for x in v], 2)} | {ms([x['strand_net'] for x in v], 2)} |")
    return ("### Table 14. Strand test (held-out edit). The 7-bp TATA core is reverse-complemented in place, "
            "preserving position, width and GC content; the same edit is applied to a non-TATA core-upstream "
            "block and the difference is reported as the net response\n\n" + "\n".join(rows))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    parts = [t for t in [table_main(), table_freq(), table_external(), table_tata_promoters(), table_atlas(), table_sensitivity(),
                         table_tbp(), table_composition(), table_robust_mpra(), table_atlas_null(),
                         table_matched_negatives(), table_positional_robustness(),
                         table_controls(), table_strand()] if t]
    open(OUT, "w", encoding="utf-8").write("# Results tables (auto-generated by make_tables.py)\n\n" + "\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")
