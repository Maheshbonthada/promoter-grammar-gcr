# Grammar-Consistency Regularization (GCR) for Promoter Models

**Links:** code https://github.com/Maheshbonthada/promoter-grammar-gcr · model weights https://huggingface.co/Sravankumarbonthada/promoter-grammar-gcr

**Question:** do deep-learning promoter models learn *where* regulatory elements must sit, or
only *what* the sequence is made of?

**Answer:** often not. Promoter models, including the pretrained Nucleotide Transformer, reach AUROC 0.94–0.95 while
ignoring the TATA-box position rule. It therefore misjudges β-thalassemia TATA mutations.

**Fix:** GCR builds JASPAR-guided, composition-preserving counterfactuals and trains with a
ranking constraint. The model learns the rule from as few as 30 examples, without losing accuracy,
and agrees better with measured mutation effects.

**Start here:**
* **Paper draft:** `paper/PAPER.md`, with methods in `paper/methods.md` and tables in
  `paper/tables.md`.
* **Manuscript (JCIM format):** `paper/PAPER_JCIM.md`.

## Headline results (5 seeds for main configurations, 3 for NT-v2-100M; details in `paper/tables.md`)

| Model | Method | Test AUROC | TATA grammar (0.5 = chance) | 60-rule grammar | ρ with MPRA effects (seed ensemble) | ρ with TBP binding ΔΔG | β-thal TATA alleles, damage percentile |
|---|---|---|---|---|---|---|---|
| **Nucleotide Transformer v2 (pretrained, 50M)** | Baseline | 0.940 | 0.34 | 0.23 | 0.247 | 0.07 | 69 |
| | GCR (ours) | **0.946** | **0.98** | 0.27 | 0.222 | **0.53** | **89** |
| | GCR-Atlas (ours) | 0.939 | 0.93 | **0.71** | 0.313 (p = 0.06) | 0.41 | **90** |
| **Nucleotide Transformer v2 (pretrained, 100M)** | Baseline | 0.951 | 0.58 | 0.28 | 0.255 | 0.40 | 86 |
| | GCR (ours) | **0.952** | **0.98** | 0.30 | 0.251 | **0.52** | **91** |
| Transformer (from scratch) | Baseline | 0.942 | 0.39 | 0.25 | 0.088 | 0.11 | 42 |
| | GCR (ours) | 0.938 | **0.99** | 0.30 | 0.185 (p = 0.007) | **0.44** | **93** |
| | GCR-Atlas (ours) | 0.934 | 0.96 | **0.64** | **0.265** (p < 0.001) | 0.34 | 74 |
| CNN (global pooling) | Baseline | 0.944 | 0.85 | 0.24 | 0.308 | 0.49 | 57 |
| | GCR (ours) | 0.940 | 0.97 | 0.27 | 0.310 | 0.49 | **94** |
| | GCR-Atlas (ours) | 0.931 | 0.94 | 0.34 | 0.336 | 0.33 | 82 |

Reference: the JASPAR TBP matrix alone reaches ρ = 0.66 with the measured binding ΔΔG.

**Takeaways:**
* **Every model is "accurate but grammar-blind".**
  - The 50M pretrained model is never better than chance.
  - The 100M model is less blind, but still unreliable.
* **GCR fixes the TATA rule** in all four models without losing accuracy. It makes the models flag
  the β-thalassemia alleles, and it brings their predictions closer to measured TBP–DNA binding
  energies.
* **GCR-Atlas generalises this to 60 JASPAR rules.**
  - It gives large MPRA gains for the from-scratch Transformer.
  - For the pretrained models and the CNN, the MPRA gains are targeted to TATA-box promoters; the
    overall gain is small and not significant.


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
python make_tables.py && python make_figures.py        # paper/tables.md, figures/
```

Trained checkpoints (all 78 models) are on Hugging Face (https://huggingface.co/Sravankumarbonthada/promoter-grammar-gcr). Download them into `runs/` to
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
| `mpra_variants.py`, `evaluate_mpra.py`, `stats.py` | External validation and statistics |
| `make_tables.py`, `make_figures.py` | All tables and figures |

## Notes
* **Held-out data:** chromosomes 8, 9 and 18 (grammar probe) and 20 and 21 (official test) are
  never used in training. The atlas uses training chromosomes only.
* **Dropped architecture:** the CNN-Dense model was dropped to prioritise the pretrained-model
  experiments on a single RTX 3050.
* **AI assistance:** the code was developed with Claude Code.
