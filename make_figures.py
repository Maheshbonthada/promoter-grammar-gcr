"""Figures for the report (static PNG, light surface, fixed method->colour mapping)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.environ.get("FIG_DIR", os.path.join(HERE, "figures"))
DPI = int(os.environ.get("FIG_DPI", 170))   # 600 for journal submission
os.makedirs(FIG, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"baseline": "#2a78d6", "gcr": "#eb6834", "randaug": "#1baf7a", "hardneg": "#eda100",
         "gcr_rank": "#e87ba4", "gcr_atlas": "#4a3aa7"}
LABEL = {"baseline": "Baseline", "randaug": "Random-swap aug. (EvoAug-style)", "hardneg": "Hard negatives",
         "gcr_rank": "GCR (rank only)", "gcr": "GCR (ours)", "gcr_atlas": "GCR-Atlas (ours)"}
ORDER = ["baseline", "randaug", "hardneg", "gcr_rank", "gcr"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.titlesize": 10.5, "axes.titlelocation": "left"})


def study_df():
    S = json.load(open(os.path.join(RES, "study.json")))["runs"]
    rows = []
    for k, v in S.items():
        a, m, s = k.split("|")
        rows.append(dict(arch=a, method=m, seed=int(s), **{kk: vv for kk, vv in v.items() if kk != "tuning_curve"},
                         curve=v["tuning_curve"]))
    return pd.DataFrame(rows)


def _heldout_reference():
    """Reference level of TATA recognition without TATA knowledge from training, per architecture: models trained on
    the main training set with every canonical-TATA promoter removed (review2_study.py). Falls back to the
    Transformers of the rare-rule experiment trained with 0 canonical-TATA promoters."""
    ref = {}
    p = os.path.join(RES, "review2_study.json")
    if os.path.exists(p):
        R = json.load(open(p))
        for a in ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]:
            v = [r["heldout_gd"] for k, r in R.items() if k.startswith(f"notata|{a}|baseline|")]
            if v:
                ref[a] = float(np.mean(v))
    if ref:
        return ref
    p = os.path.join(RES, "freq_study.json")
    if not os.path.exists(p):
        return {}
    v = [r["heldout_gd"] for r in json.load(open(p)).values() if r["n_tata"] == 0 and "heldout_gd" in r]
    return {a: float(np.mean(v)) for a in ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]} if v else {}


HO_REF = _heldout_reference()


def fig_main(D):
    archs = [a for a in ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"] if a in set(D.arch)]
    fig, axes = plt.subplots(len(archs), 4, figsize=(13, 2.5 * len(archs)), squeeze=False)
    for i, a in enumerate(archs):
        for j, (col, title, ref) in enumerate([("grammar_disc", "Grammar discrimination (0.5 = none)", 0.5),
                                               ("heldout_gd", "TATA recognition, held-out edits", 0.5),
                                               ("delta_mut", "TATA double substitution (logit drop)", 0.0),
                                               ("auroc", "Test AUROC (chr20/21)", None)]):
            ax = axes[i, j]
            ms = [m for m in ORDER if m in set(D[D.arch == a].method)]
            for x, m in enumerate(ms):
                v = D[(D.arch == a) & (D.method == m)][col].values
                ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=COLOR[m],
                           edgecolor=SURFACE, linewidth=1, zorder=3)
                ax.hlines(v.mean(), x - 0.3, x + 0.3, color=COLOR[m], lw=2, zorder=2)
            if ref is not None:
                ax.axhline(ref, color=INK2, lw=0.8, ls=":")
            if col == "heldout_gd" and a in HO_REF:               # same architecture trained with no TATA promoters
                ax.axhline(HO_REF[a], color=INK2, lw=0.8, ls="--")
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([LABEL[m].split(" (")[0] if m != "gcr_rank" else "GCR-rank" for m in ms],
                               rotation=25, ha="right", fontsize=8)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(a, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_main_comparison.png"), dpi=DPI)
    plt.close(fig)


def fig_tuning(D):
    from train import SHIFTS
    archs = [a for a in ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"] if a in set(D.arch)]
    fig, axes = plt.subplots(1, len(archs), figsize=(3.4 * len(archs), 2.8), squeeze=False, sharey=True)
    for ax, a in zip(axes[0], archs):
        for m in ["baseline", "gcr"]:
            c = np.array(D[(D.arch == a) & (D.method == m)].curve.tolist())
            if len(c):
                ax.plot(SHIFTS, c.mean(0), color=COLOR[m], lw=2, marker="o", ms=3.5, label=LABEL[m])
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_title(a)
        ax.set_xlabel("TATA box moved by (bp); 0 = native -30")
    axes[0, 0].set_ylabel("Change in promoter score (logit)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Positional tuning: shifts never used in training", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_tuning_curves.png"), dpi=DPI)
    plt.close(fig)


def fig_frequency():
    p = os.path.join(RES, "freq_study.json")
    if not os.path.exists(p):
        return
    F = pd.DataFrame(json.load(open(p)).values())
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9))
    for ax, col, title in [(axes[0], "grammar_disc", "Grammar discrimination"),
                           (axes[1], "heldout_gd", "Held-out grammar discrimination"),
                           (axes[2], "auroc", "Test AUROC")]:
        for m in ["baseline", "gcr"]:
            g = F[F.method == m].groupby("n_tata")[col]
            mu, lo, hi = g.mean(), g.min(), g.max()
            ax.fill_between(mu.index, lo, hi, color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(mu.index, mu.values, color=COLOR[m], lw=2, marker="o", ms=4, label=LABEL[m])
        ax.set_xscale("symlog", linthresh=30)
        ticks = sorted(F.n_tata.unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("TATA promoters in training (of 3,000; 30 = 1%, 480 = 16%)")
        ax.set_title(title)
    for ax in axes[:2]:
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.text(F.n_tata.max(), 0.52, "no positional knowledge", ha="right", va="bottom", fontsize=7.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.55))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_rare_rule_curve.png"), dpi=DPI)
    plt.close(fig)


def fig_atlas(top=24):
    A = pd.read_csv(os.path.join(RES, "atlas.csv"))
    P = json.load(open(os.path.join(RES, "atlas_profiles.json")))
    rules = json.load(open(os.path.join(RES, "atlas_study.json")))["rules"] if os.path.exists(
        os.path.join(RES, "atlas_study.json")) else None
    ids = list(dict.fromkeys([r["id"] for r in rules]))[:top] if rules else A.id.head(top).tolist()
    names = [A.set_index("id").loc[i, "name"] for i in ids]
    M = []
    for i in ids:
        o, e = np.convolve(P[i]["obs"], np.ones(5), "same"), np.convolve(P[i]["exp"], np.ones(5), "same")
        M.append(np.log2((o + 1) / (e + 1))[:150])
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(ids) + 1.2))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                   extent=[-49 - 0.5, 100 - 0.5, len(ids) - 0.5, -0.5])
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Motif start relative to TSS (bp)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log2 observed / composition-matched expected", fontsize=8)
    ax.set_title("JASPAR 2026 positional grammar atlas of human core promoters (rules used by GCR-Atlas)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_atlas_heatmap.png"), dpi=DPI)
    plt.close(fig)


def fig_hbb(arch="Transformer", seeds=(0, 1, 2)):
    """Predicted per-position effect (mean over the 3 alternative bases) along the HBB promoter,
    baseline vs GCR, with the MPRA-measured effects of the same promoter."""
    import torch
    from evaluate_mpra import prepare, load, saturate
    from train import predict
    C = prepare()
    ref = C["ref"]["HBB"]
    Xs, pos, _ = saturate(ref)
    V = C["V"][C["V"].element == "HBB"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    for m in ["baseline", "gcr"]:
        curves = []
        for s in seeds:
            p = os.path.join(HERE, "runs", f"{arch}__{m}__{s}.pt")
            if not os.path.exists(p):
                continue
            *_, net = load(p)
            d = predict(net, Xs) - predict(net, ref[None])[0]
            curves.append(np.array([d[pos == i].mean() for i in range(300)]))
            del net
            torch.cuda.empty_cache()
        if curves:
            c = np.mean(curves, 0)
            axes[0].plot(np.arange(300) - 49, c, color=COLOR[m], lw=1.6, label=LABEL[m])
    for ax in axes:
        ax.axvspan(-31, -24, color="#eda100", alpha=0.18, lw=0)
    axes[0].axhline(0, color=INK2, lw=0.7)
    axes[0].set_ylabel("Predicted effect\n(logit, mean of 3 alts)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_title("HBB promoter: model-predicted vs MPRA-measured variant effects (TATA box shaded)")
    axes[0].text(-33, axes[0].get_ylim()[0] * 0.8, "TATA box\n(-31..-25):\nbeta-thal\nalleles", ha="right",
                 va="center", fontsize=7.5, color=INK2)
    axes[1].axhline(0, color=INK2, lw=0.7)
    axes[1].scatter(V.rel_tss, V.effect, s=14, color=INK2, zorder=3)
    axes[1].set_ylabel("MPRA effect\n(log2 alt/ref)")
    axes[1].set_xlabel("Position relative to HBB TSS (bp)")
    axes[1].set_xlim(-50, 110)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_hbb_beta_thalassemia.png"), dpi=DPI)
    plt.close(fig)


def fig_mpra():
    """Seed-ensembled Spearman with measured MPRA effects, 95% bootstrap CI, per architecture and method."""
    st = pd.read_csv(os.path.join(RES, "mpra_stats.csv"))
    archs = [a for a in ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"] if a in set(st.arch)]
    methods = ["baseline", "randaug", "hardneg", "gcr_rank", "gcr", "gcr_atlas"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, (col, lo, hi, title) in zip(axes, [("spearman", "ci_lo", "ci_hi", "All 582 variants, 9 disease promoters"),
                                                ("rho_TATA_promoters", None, None, "TATA promoters (HBB, HBG1, MSMB)")]):
        w = 0.8 / len(methods)
        for i, a in enumerate(archs):
            for j, m in enumerate(methods):
                r = st[(st.arch == a) & (st.method == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = i + (j - (len(methods) - 1) / 2) * w
                ax.bar(x, r[col], width=w * 0.92, color=COLOR[m], label=LABEL[m] if i == 0 else None)
                if lo:
                    ax.plot([x, x], [r[lo], r[hi]], color=INK2, lw=0.9)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels(archs)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_title(title)
    axes[0].set_ylabel("Spearman ρ with measured effect")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Agreement with saturation-mutagenesis MPRA (seed ensembles; bars = 95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig6_mpra_validation.png"), dpi=DPI)
    plt.close(fig)


def _tbp_ensemble(T, arch, method):
    P = np.array([v["pred"] for k, v in T["runs"].items() if k.startswith(f"{arch}|{method}|")])
    return (P / P.std(axis=1, keepdims=True)).mean(0) if len(P) else None


def fig_tbp():
    """Model variant effects vs measured TBP-DNA binding free energy (EMSA K_D)."""
    p = os.path.join(RES, "tbp_binding.json")
    if not os.path.exists(p):
        return
    T = json.load(open(p))
    V = pd.read_csv(os.path.join(RES, "tbp_binding_variants.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    for m, mk in [("baseline", "o"), ("gcr", "s")]:
        e = _tbp_ensemble(T, "NT-v2-50M", m)
        if e is not None:
            r = T["ensemble"]["NT-v2-50M|" + m]["spearman"]
            ax.scatter(V.ddG, e, s=30, marker=mk, color=COLOR[m], label=f"{LABEL[m]} (ρ = {r:+.2f})", zorder=3)
    e_gcr = _tbp_ensemble(T, "NT-v2-50M", "gcr")
    for i, v in V.iterrows():
        if v.gene != "HBB" and e_gcr is not None:
            ax.annotate(v.gene, (v.ddG, e_gcr[i]), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel("Measured ΔΔG of TBP binding (kcal/mol; > 0 = weaker)")
    ax.set_ylabel("Predicted loss of promoter score (z)")
    ax.set_title("NT-v2-50M: 14 disease TATA-box SNPs")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    archs = [a for a in ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]
             if any(k.startswith(a + "|") for k in T["ensemble"])]
    methods = ["baseline", "gcr", "gcr_atlas"]
    w = 0.8 / len(methods)
    for i, a in enumerate(archs):
        for j, m in enumerate(methods):
            r = T["ensemble"].get(f"{a}|{m}")
            if r is None:
                continue
            ax.bar(i + (j - 1) * w, r["spearman"], width=w * 0.92, color=COLOR[m], label=LABEL[m] if i == 0 else None)
    ref = T["reference"]["TBP_PWM"]["spearman"]
    ax.axhline(ref, color=INK2, lw=1, ls="--")
    ax.set_ylim(min(0, ax.get_ylim()[0]), ref + 0.25)
    ax.text(len(archs) - 0.5, ref + 0.02, "JASPAR TBP PWM", ha="right", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs)
    ax.set_ylabel("Spearman ρ with measured ΔΔG")
    ax.set_title("Agreement with TBP–DNA binding energetics (seed ensembles)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_tbp_binding.png"), dpi=DPI)
    plt.close(fig)


def fig_toc():
    """JCIM table-of-contents graphic (3.25 x 1.75 in)."""
    S = json.load(open(os.path.join(RES, "study.json")))["runs"]
    gd = {m: np.mean([v["grammar_disc"] for k, v in S.items() if k.startswith(f"NT-v2-50M|{m}|")])
          for m in ["baseline", "gcr"]}
    fig = plt.figure(figsize=(3.25, 1.75))
    ax = fig.add_axes([0, 0, 0.62, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    for y, lab, tx, col in [(4.3, "real promoter", 2.2, COLOR["gcr"]), (1.5, "same letters, TATA moved", 7.0, INK2)]:
        ax.plot([0.3, 9.7], [y, y], color=INK2, lw=1.2)
        ax.add_patch(plt.Rectangle((tx, y - 0.35), 1.3, 0.7, color=col, zorder=3))
        ax.text(tx + 0.65, y, "TATA", ha="center", va="center", fontsize=6, color="white", weight="bold", zorder=4)
        ax.annotate("", xy=(4.9, y + 0.9), xytext=(4.9, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1))
        ax.text(5.1, y + 0.55, "TSS", fontsize=5.5, color=INK)
        ax.text(0.3, y - 0.95, lab, fontsize=6, color=INK2)
    ax.text(0.3, 5.6, "GCR: rank  real > displaced", fontsize=7, weight="bold", color=INK)
    ax.text(2.85, 4.85, "−30", fontsize=5.5, color=INK2, ha="center")
    ax2 = fig.add_axes([0.72, 0.2, 0.26, 0.62])
    ax2.bar([0, 1], [gd["baseline"], gd["gcr"]], color=[COLOR["baseline"], COLOR["gcr"]], width=0.7)
    ax2.axhline(0.5, color=INK2, lw=0.7, ls="--")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["NT-v2", "+GCR"], fontsize=6)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=5.5)
    ax2.set_title("TATA grammar", fontsize=6.5)
    ax2.grid(False)
    fig.savefig(os.path.join(FIG, "toc_graphic.png"), dpi=max(300, DPI))
    plt.close(fig)


def fig_implant():
    """TATA implantation into held-out promoters without a canonical TATA box (review2_probe.py)."""
    p = os.path.join(RES, "review2_probe.json")
    if not os.path.exists(p):
        return
    P = json.load(open(p))
    from review2_probe import STARTS, TSS
    x = STARTS - TSS
    archs = ["Transformer", "CNN-GlobalPool", "NT-v2-50M", "NT-v2-100M"]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 2.9), squeeze=False)
    for ax, a in zip(axes[0], archs):
        for m in ["baseline", "gcr", "gcr_atlas"]:
            c = np.array([v["implant_curve"] for k, v in P.items() if k.startswith(f"{a}|{m}|")])
            if len(c):
                ax.plot(x, c.mean(0), color=COLOR[m], lw=2, marker="o", ms=3, label=LABEL[m])
                if len(c) > 1:
                    ax.fill_between(x, c.min(0), c.max(0), color=COLOR[m], alpha=0.15, lw=0)
        ax.axvline(-30, color=INK2, lw=0.8, ls=":")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(a)
        ax.set_xlabel("Start of implanted TATAAAA (bp from TSS)")
    axes[0, 0].set_ylabel("TATA-specific gain (logit)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_tata_implant.png"), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig_tbp()
    fig_toc()
    fig_mpra()
    D = study_df()
    fig_hbb()
    fig_main(D)
    fig_tuning(D)
    fig_frequency()
    fig_atlas()
    fig_implant()
    print("figures written to", FIG)
