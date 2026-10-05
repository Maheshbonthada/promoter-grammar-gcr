# Grammar-Consistency Regularization (GCR) for Promoter Models

**Links:** code https://github.com/Maheshbonthada/promoter-grammar-gcr · model weights https://huggingface.co/Sravankumarbonthada/promoter-grammar-gcr

**Question:** do deep-learning promoter models learn *where* regulatory elements must sit, or
only *what* the sequence is made of?

**Answer:** they recognize *what* a regulatory element is, but often not *where* it must be. On a benchmark
that GC content alone classifies with AUROC 0.90, promoter models including the pretrained Nucleotide
Transformer reach AUROC 0.94–0.95 while using the position of the TATA box only weakly.

**Fix:** GCR builds JASPAR-guided, composition-preserving counterfactuals and trains with a ranking
constraint. Every model becomes position-aware, including for shifts never used in training, without
losing accuracy.

**Start here:**
* **Manuscript methods and tables:** `paper/methods.md`, `paper/tables.md` (all numbers regenerated from
  `results/` by `make_tables.py`).
* `paper/PAPER.md` and `paper/PAPER_JCIM.md` are earlier drafts and are superseded (see the correction note below).

## Headline results (5 seeds for main configurations, 3 for NT-v2-100M; details in `paper/tables.md`)

| Model | Method | Test AUROC | Positional grammar (0.5 = none) | Shift sensitivity (held-out, logits) | Rule grammar, 49 atlas rules (0.5 = none) | ρ with MPRA effects | ρ with TBP binding ΔΔG | β-thal alleles, damage percentile |
|---|---|---|---|---|---|---|---|---|
| **Nucleotide Transformer v2 (pretrained, 50M)** | Baseline | 0.940 | 0.53 | 0.13 | 0.50 | 0.247 | 0.07 | 69 |
|  | GCR | 0.946 | 0.99 | 1.31 | 0.53 | 0.222 | 0.53 | 89 |
|  | GCR-Atlas | 0.939 | 0.95 | 0.55 | 0.83 | 0.313 | 0.41 | 90 |
| **Nucleotide Transformer v2 (pretrained, 100M)** | Baseline | 0.951 | 0.75 | 0.49 | 0.53 | 0.255 | 0.40 | 86 |
|  | GCR | 0.952 | 0.99 | 1.64 | 0.53 | 0.251 | 0.52 | 91 |
|  | GCR-Atlas | 0.947 | 0.92 | 0.50 | 0.85 | 0.324 | 0.35 | 88 |
| Transformer (from scratch) | Baseline | 0.942 | 0.60 | 0.65 | 0.48 | 0.088 | 0.11 | 42 |
|  | GCR | 0.938 | 0.99 | 2.42 | 0.51 | 0.185 | 0.44 | 93 |
|  | GCR-Atlas | 0.934 | 0.97 | 0.94 | 0.77 | 0.265 | 0.34 | 74 |
| CNN (global pooling) | Baseline | 0.944 | 0.91 | 1.13 | 0.51 | 0.308 | 0.49 | 57 |
|  | GCR | 0.940 | 0.98 | 1.45 | 0.53 | 0.310 | 0.49 | 94 |
|  | GCR-Atlas | 0.931 | 0.96 | 0.54 | 0.59 | 0.336 | 0.33 | 82 |

Positional grammar = AUROC of TATA-displacement vs matched control effects (0.5 = no positional knowledge).
Shift sensitivity = mean logit drop when the TATA box is moved by at least 8 bp (never used in training).
Reference: the JASPAR TBP matrix alone reaches ρ = 0.66 with the measured binding ΔΔG.

**Takeaways:**
* **Accurate models recognize the TATA box but largely ignore its position.** Against architecture-matched
  nulls every baseline gains little (+0.01 to +0.16). GC content alone gives AUROC 0.90, so position is not
  needed for this task.
* **GCR makes all four models position-aware** without losing accuracy, and the ranking term alone
  (GCR-rank) is enough.
* **GCR-Atlas teaches 49 scorable JASPAR rules** to attention models (0.5 → 0.77–0.85), with trade-offs on
  the TATA rule.
* **External measurements:** significant MPRA gains for the from-scratch Transformer and on TATA-box
  promoters for the pretrained models (promoter-level bootstrap, Holm-corrected). TBP binding
  agreement moves in the expected direction but is **not significant** with 14 SNPs.

**Correction (29 September 2026):** earlier versions computed grammar discrimination as AUROC(signed
displacement effect vs *absolute* control effect), whose null value is about 0.25, not 0.5, and stored
atlas scores by motif name (so motifs with two rules overwrote each other). Both are fixed; all models were
re-scored (`rescore.py`), and the old values are kept as `grammar_disc_asym`. Statistics are now two-sided
with promoter-level resampling (`robust_stats.py`); new analyses: `composition_baseline.py`,
`atlas_null_check.py`.

## Architecture-matched nulls (added 29 September 2026)

Positional scores have **no single chance level**. Training each architecture on the same data with every
canonical-TATA promoter removed (`review2_study.py`) gives the score expected without TATA knowledge:

| Model | Positional grammar (its own null) | Baseline gain | GCR gain | TATA implant preference (baseline → GCR) |
|---|---|---|---|---|
| NT-v2-50M | 0.53 → 0.99 (0.52) | +0.01 | **+0.46** | +0.03 → +1.25 |
| NT-v2-100M | 0.75 → 0.99 (0.58) | +0.16 | **+0.40** | +0.26 → +1.45 |
| Transformer | 0.60 → 0.99 (0.48) | +0.12 | **+0.51** | +0.21 → +1.82 |
| CNN-GlobalPool | 0.91 → 0.98 (0.78) | +0.12 | **+0.19** | +0.55 → +1.50 |

The global-pooling CNN reaches 0.78 **without ever seeing a TATA promoter**, because its translation-invariant
readout responds to any edit near the core promoter. An earlier version of this README described the CNN as the
one model that had learned the positional rule; measured against its own null it gains the same +0.12 as the
Transformer, and that claim is withdrawn.

Two further controls (`review2_study.py`, `review2_probe.py`):

* **Composition-matched negatives** — promoters versus their own dinucleotide-preserving local shuffles, where
  GC content alone gives AUROC 0.50. Baselines stay weak; GCR still learns the rule.
* **Motif implantation** — writing TATAAAA into promoters that lack it, against a scrambled control of the same
  bases. Null models prefer the native −30 position in **0 of 12** runs, baselines in **5 of 18**, GCR in
  **18 of 18**.

## Reproduce

Requires Python 3.10, PyTorch with CUDA, `datasets`, `transformers`, `scikit-learn`, `scipy`,
`pandas` and `matplotlib`.

```bash
python prepare_data.py                 # NT promoter tasks + JASPAR motifs  -> data/
python run_study.py Transformer CNN-GlobalPool          # main comparison     -> results/study.json
python run_study.py NT-v2-50M NT-v2-100M --methods=baseline,gcr   # pretrained DNA LMs (SEEDS=0,1,2,3,4 env var)
python tbp_binding.py                  # TBP-DNA binding validation      -> results/tbp_binding.json
python -m pytest tests -q              # unit tests
python freq_study.py                   # rare-rule experiment               -> results/freq_study.json
python atlas.py                        # JASPAR 2026 positional atlas       -> results/atlas.csv
python run_atlas_study.py Transformer CNN-GlobalPool    # GCR-Atlas + MPRA -> results/atlas_study.json
python evaluate_mpra.py                # MPRA / beta-thal / TERT per checkpoint
python stats.py                        # bootstrap CIs                      -> results/mpra_stats.csv
python rescore.py nt NT-v2-50M NT-v2-100M; python rescore.py small Transformer CNN-GlobalPool; python rescore.py --merge
python robust_stats.py                 # two-sided, promoter-level, Holm    -> results/robust_stats.csv
python composition_baseline.py         # GC / k-mer logistic regression     -> results/composition_baseline.json
python atlas_null_check.py             # dinucleotide-shuffle null          -> results/atlas_null_check.csv
python review2_study.py notata Transformer CNN-GlobalPool NT-v2-50M NT-v2-100M   # architecture-matched nulls
python review2_study.py negmatch Transformer CNN-GlobalPool NT-v2-50M           # composition-matched negatives
python review2_probe.py Transformer CNN-GlobalPool NT-v2-50M NT-v2-100M         # motif implantation
python make_tables.py && python make_figures.py        # paper/tables.md, figures/
```

Trained checkpoints (all 81 models) are on Hugging Face (https://huggingface.co/Sravankumarbonthada/promoter-grammar-gcr). Download them into `runs/` to
re-evaluate without retraining:

```bash
huggingface-cli download Sravankumarbonthada/promoter-grammar-gcr --include "runs/*" --local-dir .
```

Place the JASPAR bulk file `JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt` in `data/`
(download it from jaspar.elixir.no).

## Files

| File | Purpose |
|---|---|
| `prepare_data.py`, `motifs.py` | Data download, PWM scanning |
| `counterfactuals.py` | Composition-preserving swaps and moves, canonical-TATA calls |
| `models.py`, `nt_model.py` | CNN, Transformer, Nucleotide Transformer v2 wrapper |
| `train.py` | Training methods and the Core-Promoter Grammar Test |
| `run_study.py`, `freq_study.py` | Main and rare-rule experiments |
| `atlas.py`, `gcr_atlas.py`, `run_atlas_study.py` | JASPAR positional atlas and multi-rule GCR |
| `mpra_variants.py`, `evaluate_mpra.py`, `stats.py`, `robust_stats.py` | External validation and statistics |
| `rescore.py` | Re-scores saved checkpoints with the corrected grammar metrics |
| `composition_baseline.py`, `atlas_null_check.py` | Composition shortcut; stricter atlas null |
| `review2_study.py`, `review2_probe.py` | Architecture-matched nulls, composition-matched negatives, motif implantation |
| `make_tables.py`, `make_figures.py` | All tables and figures |

## Notes
* **Held-out data:** chromosomes 8, 9 and 18 (grammar probe) and 20 and 21 (official test) are
  never used in training. The atlas uses training chromosomes only.
* **Dropped architecture:** the CNN-Dense model was dropped to prioritise the pretrained-model
  experiments on a single RTX 3050.
* **AI assistance:** the code was developed with Claude Code.
