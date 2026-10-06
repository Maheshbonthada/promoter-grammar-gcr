# Promoter grammar audit: architecture-matched null models and Grammar-Consistency Regularization (GCR)

Code and result files for the manuscript *Accurate Promoter Models Only Partly Learn TATA-Box Position:
An Audit With Architecture-Matched Null Models* (S. K. Bonthada). Every number in the manuscript and its
supplement is generated from the files in `results/`.

**Links:** code https://github.com/Maheshbonthada/promoter-grammar-gcr · model weights
https://huggingface.co/Sravankumarbonthada/promoter-grammar-gcr

## Question and answer

**Question.** Do accurate DNA sequence models use the *position* of a regulatory element, or only what the
sequence is made of? We test this for the TATA box, whose position near −30 is fixed by the structure of
the TBP–DNA complex.

**Answer.** Only in part. On a promoter benchmark that GC content alone classifies at AUROC 0.90, six models
with test AUROC 0.940–0.962 capture **2% to 69%** of the room above their own null models. Removing the
composition confound from training does not close that gap, and making the element more common closes it
only in part. A ranking constraint over composition-preserving counterfactuals (GCR) closes it in six of
seven architectures. Its benefit on external measurements is limited, and it failed for one published
architecture.

## How positional use is measured

* **Grammar discrimination (GD):** AUROC separating the score changes caused by moving the TATA block from
  those caused by moving a neighbouring control block of the same width. Both edits are block swaps, which
  preserve the window's base composition exactly, so any mononucleotide-composition scorer gets GD = 0.5.
  It is not blind to k-mer composition (k-mer scorers give 0.471–0.530).
* **Architecture-matched null:** the same architecture retrained with every canonical-TATA promoter removed.
  GD has no universal chance level: nulls range from 0.483 (Transformer) to 0.801 (CNN-Dense), and 0.844
  for DeepSTARR, so each model must be read against its own null.
* **Share of the room above the null:** (GD − null) / (1 − null).

## Main results

GD on 76 held-out canonical-TATA promoters (chromosomes 8, 9, 18, 20, 21). Gain intervals resample probe
promoters and seeds (`gain_ci.py`).

| Model | Test AUROC | Null GD | Baseline GD | Gain [95% CI] | Share | GCR GD | GCR share |
|---|---|---|---|---|---|---|---|
| Transformer | 0.942 | 0.483 | 0.602 | +0.119 [−0.018, +0.254] | 23% | 0.995 | 99% |
| CNN-GlobalPool | 0.944 | 0.784 | 0.906 | +0.122 [+0.060, +0.187] | 56% | 0.979 | 90% |
| CNN-Dense | 0.948 | 0.801 | 0.939 | +0.138 [+0.066, +0.213] | 69% | 0.999 | 99% |
| NT-v2-50M | 0.940 | 0.525 | 0.535 | +0.010 [−0.126, +0.154] | 2% | 0.987 | 97% |
| NT-v2-100M | 0.951 | 0.584 | 0.747 | +0.163 [+0.037, +0.292] | 39% | 0.985 | 96% |
| NT-v2-500M (LoRA) | 0.962 | 0.657 | 0.797 | +0.140 [+0.061, +0.224] | 41% | 0.991 | 98% |
| DeepSTARR (published design, retrained here) | 0.944 | 0.844 | 0.925 | +0.082 [+0.030, +0.143] | 52% | 0.806 | −24% |

**Why the baselines fall short**

* The TATA motif is **anti-predictive** on this benchmark (AUROC 0.212): negatives are AT-rich and promoters
  GC-rich.
* That penalty is **not the main explanation**. With composition-matched negatives (each promoter's own
  20-bp-segment dinucleotide shuffle; GC content alone gives AUROC 0.500) the baselines capture 13%–30% of
  the room above matched nulls.
* **Rarity contributes but does not explain it either.** Only 2.9% of training promoters carry a canonical
  TATA box. With matched negatives, Transformer GD rises from 0.527 (no TATA promoters) to 0.718 (10%) and
  stays there at 16%; GCR reaches 0.971 with 1%.

**What GCR does and does not do**

* Raises the share to 90%–99% in the six main architectures, for a change in test AUROC of at most 0.006,
  with positional responses to shifts and motif insertions never used in training.
* **Fails for DeepSTARR** (GD 0.925 → 0.806). GCR scores a sequence and its counterfactuals in separate
  forward passes; DeepSTARR applies batch normalisation to its dense features. A joint forward pass
  (`GCR_JOINT=1`) gave 0.865 on average but was unstable across seeds.
* **External benefit is limited.** Against MPRA-measured variant effects (*LDLR* excluded), correct rules
  beat deliberately displaced rules in four architectures, but beat the unconstrained model significantly
  in only one after Holm correction (Transformer, p = 0.026). On the three promoters that do not overlap
  training, the effect is absent.
* **Costs.** On lentiMPRA, 3% of predictive correlation at λ = 0.1 and 11% at the default λ = 1 (the
  validation split selects the same λ). GCR lowers sensitivity at BREd and the Inr by 15%–37% in some
  architectures. A hand-coded TBP profile ranks the *HBB* β-thalassemia variants (98.8th percentile) and
  predicts TBP binding better than any trained model.

## Reproduce

Requires Python 3.10, PyTorch with CUDA, `datasets`, `transformers`, `peft`, `scikit-learn`, `scipy`,
`pandas` and `matplotlib`. Everything ran on one 8 GB RTX 3050. Place
`JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt` (from jaspar.elixir.no) in `data/`.

```bash
python prepare_data.py                    # benchmark + JASPAR motifs -> data/
python run_study.py Transformer CNN-GlobalPool CNN-Dense            # baselines and GCR -> results/study.json
python run_study.py NT-v2-50M NT-v2-100M --methods=baseline,gcr    # pretrained models
python nt_lora.py                         # NT-v2-500M with LoRA
python review2_study.py notata ARCH ...   # architecture-matched nulls     -> results/review2_study.json
python review2_study.py strictnull ARCH   # TATA at -30 removed from both classes
python review2_study.py negmatch ARCH     # composition-matched negatives (METHODS=baseline,gcr)
python review2_study.py negmatchnull ARCH # their matched nulls
METHODS=baseline,gcr python review2_study.py full DeepSTARR   # published architecture
python review2_probe.py ARCH ...          # motif implantation
python gain_ci.py                         # promoter-and-seed intervals    -> results/gain_ci.json
python freq_study.py; MATCHED=1 python freq_study.py   # TATA-frequency sweeps
python element_sensitivity.py ARCH ...    # BREu / BREd / Inr sensitivity  -> results/element_sensitivity.json
python composition_baseline.py; python composition_gd.py   # composition-only scorers
python run_atlas_study.py ARCH ...        # GCR-Atlas (60 JASPAR positional rules)
python evaluate_mpra.py; python mpra_all_archs.py; python mpra_leakage_check.py   # MPRA validation
python tbp_binding.py; python review3_checks.py            # TBP binding; HBB and other checks
python lentimpra_gcr.py; python lentimpra_clean_eval.py; python lentimpra_val_lambda.py   # lentiMPRA
python splice_gcr.py                      # splice-donor transfer
python -m pytest tests -q                 # unit tests
```

Checkpoints for the main models are on Hugging Face; download them into `runs/` to re-evaluate without
retraining:

```bash
huggingface-cli download Sravankumarbonthada/promoter-grammar-gcr --include "runs/*" --local-dir .
```

Every other model is reproducible from the code and the seeds recorded in the result files.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py`, `motifs.py` | Data download, PWM scanning |
| `counterfactuals.py` | Composition-preserving block swaps, canonical-TATA calls, control positions |
| `models.py`, `nt_model.py`, `nt_lora.py`, `published_archs.py` | CNNs, Transformer, Nucleotide Transformer v2, DeepSTARR |
| `train.py` | Training methods (baseline, GCR, controls) and grammar discrimination |
| `run_study.py`, `review2_study.py`, `freq_study.py` | Main runs, nulls, matched negatives, frequency sweeps |
| `gain_ci.py`, `element_sensitivity.py`, `review2_probe.py` | Intervals, core-element sensitivity, implantation |
| `atlas.py`, `gcr_atlas.py`, `run_atlas_study.py`, `atlas_null_check.py` | JASPAR positional atlas and GCR-Atlas |
| `evaluate_mpra.py`, `mpra_all_archs.py`, `mpra_leakage_check.py`, `robust_stats.py` | MPRA validation and statistics |
| `lentimpra_gcr.py`, `lentimpra_clean_eval.py`, `lentimpra_val_lambda.py` | lentiMPRA replication |
| `results/` | Every result file the manuscript cites |

`paper/` and `figures/` hold **earlier drafts** of this work. They predate the corrected metric, the
architecture-matched nulls and several corrections, and contain claims the manuscript no longer makes.
The manuscript supersedes them.

## Notes

* **Held-out data:** chromosomes 8, 9 and 18 (with 20 and 21) supply the 76-promoter GD probe; 20 and 21
  are the official test split. None of these is used in training.
* **Known limitations** (all stated in the manuscript): the external MPRA covers eight promoter clusters
  after excluding *LDLR*; the lentiMPRA data pool three cell lines, and 11.9% of test sequences also occur
  in training (results hold without them); pretrained models such as Enformer cannot be audited this way,
  because no matched null can be built for them.
* **AI assistance:** analysis code was written with the help of Claude (Anthropic), as disclosed in the
  manuscript.
