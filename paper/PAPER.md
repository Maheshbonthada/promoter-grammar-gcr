# Accurate but grammar-blind: promoter models and a genomic language model ignore positional regulatory logic, and JASPAR-guided grammar-consistency training restores it

*Earlier preprint draft (3 seeds). **Superseded by `PAPER_JCIM.md`**, which uses 5 seeds, adds
NT-v2-100M and the TBP–DNA binding validation, and corrects the NT-v2 MPRA result: with 5 seeds
the overall GCR-Atlas gain is +0.065 (p = 0.06), not +0.132 (p = 0.001).*

## Abstract

**Question.** Deep-learning models of regulatory DNA reach high accuracy on promoter benchmarks.
Do they learn the positional rules of promoter architecture?

**Test.** We use a coordinate frame derived from the data itself and **composition-preserving
counterfactuals** built from JASPAR motifs. With these we show that three promoter models reach
AUROC ≈ 0.94 while ignoring the textbook TATA-box rule (TATA at −30):

* a Transformer trained from scratch;
* a global-pooling CNN;
* the pretrained genomic language model **Nucleotide Transformer v2**.

The pretrained model is completely blind to the rule. Its grammar discrimination is 0.22 (0.5 =
chance), and TATA-destroying mutations leave its predictions unchanged.

**Rarity is not the cause.** A controlled experiment rules out rarity alone: a Transformer did not
learn the rule whether TATA promoters made up 0%, 1%, 3%, 10% or 16% of training data. This points
to shortcut learning from sequence composition.

**Method: Grammar-Consistency Regularization (GCR).** GCR adds two terms, both built from JASPAR:

* a *ranking* constraint: the original promoter must outscore the same sequence with the motif moved
  out of its preferred window;
* an *invariance* constraint: the original must score the same as a position-matched control edit.

Both edits preserve composition exactly.

**GCR results:**

* The rule is learned in all three architectures (grammar discrimination 0.97–1.00), including
  from only **30 examples**.
* It generalises to edits never used in training.
* Accuracy is maintained: pretrained model AUROC 0.941 → 0.947. Hard negatives, by contrast, cost
  accuracy and were less stable.

**JASPAR 2026 atlas and GCR-Atlas.** We scanned all 1,019 JASPAR 2026 vertebrate motifs against a
composition-matched null. The atlas recovers known core-promoter architecture: ETS at −25,
AP-1/CREB and SP/KLF at the TSS, YY downstream, and TBP at −30. From it we derived 60
non-redundant rules; training on all of them is GCR-Atlas.

**External validation.** On a saturation-mutagenesis MPRA of 9 disease promoters (582 measured
variants), GCR-Atlas gave the best agreement with measured effects:

* pretrained model: Spearman ρ 0.204 → **0.336** (+0.132 [0.044, 0.223], p = 0.001), the best
  model overall;
* Transformer: ρ 0.044 → 0.262.

**Disease alleles.**

* **β-thalassemia:** the HBB TATA-box alleles (−31…−28) move from the 44th to the 94th percentile of
  predicted damage (Transformer + GCR). They are ranked consistently high in every architecture.
* **Cancer:** the TERT C228T mutation is ranked more strongly activating by the from-scratch models;
  results for the pretrained model are mixed.

**Conclusion.** High benchmark accuracy does not imply that regulatory grammar has been learned.
Curated motif knowledge from JASPAR, used as a composition-preserving training constraint, is a
cheap and effective remedy.

---

## 1. Introduction

**Promoter grammar.** Promoters are built from short elements whose function depends on *where*
they sit relative to the TSS. The TATA box, bound by TBP, is the classic example. It works about
30 bp upstream of the TSS; moved elsewhere, it no longer positions the pre-initiation complex
correctly.

**Evidence that models miss it.** Sequence-to-function models and DNA language models reach high
accuracy on promoter benchmarks. But a recent *Mechanistic Invariance Test* (arXiv 2604.06549) reported
that genomic language models learn compositional regularities ("AT-rich ⇒ promoter-like") rather
than positional constraints. Its benchmark centres on promoter elements such as the −10 box. The
authors proposed remedies, including *compositional supervision* that asks models to tell
structured sequences from scrambled ones, but did not test them. Three gaps follow:

1. **Whether** such a remedy works has not been shown.
2. **Whether** the failure matters for human disease-variant interpretation is unknown.
3. **Why** the failure arises (rarity or shortcut) has not been established.

**Our approach.** We focus on a setting where both the rule and its clinical relevance are
unambiguous. TATA-box point mutations in the *HBB* promoter (−28 to −31) cause β-thalassemia,
one of the most common monogenic disorders in the Middle East. Our contributions:

1. **A composition-controlled test of positional grammar.** It uses block swaps, which preserve
   nucleotide composition exactly, with position-matched controls. It rests on a coordinate frame
   derived from the data rather than assumed.
2. **A controlled rare-rule experiment.** It shows that a Transformer does not learn the TATA rule
   at any natural frequency (0–16%).
3. **GCR.** A training constraint that teaches the rule from a handful of examples without losing
   accuracy. It is compared against augmentation (EvoAug-style) and hard-negative alternatives.
4. **External validation.** On measured variant effects (MPRA) and on disease alleles
   (β-thalassemia *HBB*, cancer *TERT* C228T).
5. **A JASPAR 2026 positional grammar atlas.** Built with a composition-matched null in the
   spirit of BiasAway, and used to generalise GCR to 60 rules (GCR-Atlas).

## 2. Results

### 2.1 A data-derived coordinate frame
We did not rely on the documented window convention. In the benchmark's 300-bp windows, the CA
initiator dinucleotide peaks at indices 48–49 (3.4× its background frequency), and JASPAR TBP
(MA0108.3) hits peak at index 19. The TSS is therefore at index 49, and the canonical TATA box
starts at −30, in agreement with experimental knowledge.

Only 20% of `promoter_tata` positives, and 2.9% of `promoter_all` training positives, carry a
canonical TATA box by our 88%-precision criterion. In general promoter data, the TATA rule is
**rare**.

### 2.2 High accuracy without positional grammar
See Table 1 and Fig. 1 for the main comparison. We trained three architectures on `promoter_all`
(26.7k sequences, three held-out chromosomes) and tested them on the official chr20/21 split.

**Transformer baseline.** Its test AUROC is 0.940, yet it is essentially blind to TATA position:

* grammar discrimination 0.40 ± 0.17 (chance 0.5);
* displacing the TATA box changes the logit by 0.26, about the same as a harmless position-matched
  control swap (0.21);
* the TATA-destroying point mutation TATAAAA→TGTGAAA changes it by only 0.13.

**CNN with global pooling.** It learns partial grammar (0.86 ± 0.04).

**Pretrained NT-v2.** It is the most blind of the three (0.22 ± 0.03; Section 2.8).

**Generic augmentation (random block swaps).** It does not help, and it slightly *hurts*
positional grammar in the Transformer (0.33). This is expected: it teaches invariance to
arbitrary rearrangements.

### 2.3 Rarity does not explain the failure
To test whether the failure is simply a matter of frequency, we held the training set at 3,000
promoters and 3,000 background sequences and varied the number of canonical-TATA promoters
(Table 2, Fig. 3). The baseline Transformer did not learn the rule at **any** frequency tested
(grammar discrimination 0.13–0.36 at 0–16%), and its predictions ignored TATA point mutations.

Composition cues separate promoters from background so well that there is little gradient signal
to learn a rare, position-specific element. The failure is a **shortcut-learning** problem, and
it does not go away with modestly more data.

### 2.4 GCR teaches the rule without sacrificing accuracy
GCR adds two terms, built from JASPAR MA0108.3, to the standard loss:

* a **ranking** term, requiring that swapping the TATA box to a far-downstream position lowers
  the promoter score;
* an **invariance** term, requiring that swapping a neighbouring non-TATA block to the same kind
  of position does not change the score.

Both edits preserve composition exactly.

**Results (Table 1):**

* Grammar discrimination reached 0.97–1.00 in all three architectures: Transformer, CNN and
  pretrained NT-v2.
* Test AUROC stayed within 0.006 of baseline; for NT-v2 it rose by 0.006. GCR-rank, the variant without the invariance
  term, even slightly improved AUROC (0.946 vs 0.940).
* **In the rare-rule experiment, 30 TATA promoters (1%) were enough** for GCR to learn the rule
  (0.98 ± 0.01), at unchanged accuracy (AUROC 0.918 vs 0.917).
* **Accuracy cost in small data.** Across frequencies, GCR's AUROC ranged from 0.013 below to
  0.005 above baseline. The largest cost was at 16% (0.914 vs 0.927), where GCR constraints apply
  to many more training sequences. In the full-data main study the cost was ≤ 0.006.

**Hard negatives teach the rule but cost accuracy.** Labelling TATA-displaced sequences as
negatives also taught the rule (0.97–0.98), but reduced AUROC (Transformer 0.923, CNN 0.932, NT-v2 0.936). The
label is biologically unjustified: a promoter without a positioned TATA box is often still a
promoter. The ranking formulation claims only what is known, that displacement *weakens* the
promoter.

**Generalisation to edits never used in training.**

* **Point mutations:** GCR models respond strongly (Δ 1.0–2.9 logits, vs 0.13 for the
  Transformer baseline).
* **Positional tuning curves** (Fig. 2): these peak at the native −30 position and fall off
  smoothly over ±10 bp, even though training used only far (≥ +51 bp) displacements.

### 2.5 External validation on measured variant effects and disease alleles
We placed 9 promoters from a saturation-mutagenesis MPRA (Kircher et al. 2019) into the training
frame using RefSeq TSSs. That gives 582 measured variants, 26 of them in TATA boxes. We scored each
model by the Spearman correlation between predicted and measured effects (Table 3, Table 4,
Fig. 5, Fig. 6).

**Transformer.**

* The baseline shows essentially no correlation: ρ = 0.044 [−0.035, 0.125].
* GCR raised it to 0.199 (+0.154 [+0.080, +0.232]).
* GCR-Atlas raised it to 0.262 (+0.217 [+0.132, +0.299]).
* The β-thalassemia TATA alleles (HBB −31…−28) moved from the 44th to the **94th percentile** of
  predicted damage.
* TERT C228T, the recurrent cancer mutation that creates an ETS/GABPA site, moved from the 30th to
  the 67th percentile of predicted activation.

**CNN.** It already correlates moderately with MPRA (ρ = 0.315), and GCR did not change the
overall correlation (−0.008 [−0.066, +0.052]). The effect is **targeted**:

* The β-thalassemia alleles moved from the 57th to the 95th percentile of predicted damage.
* Correlation on the TATA-box promoters rose (see Table 4).

**Mechanism.** Fig. 5 shows how this happens. For *HBB*, the GCR model's predicted-effect profile
has a sharp trough over the TATA box, where the MPRA measured uniformly negative effects. The
baseline shows no trough there.

### 2.6 A JASPAR 2026 positional grammar atlas
We scanned all 1,019 JASPAR 2026 CORE vertebrate non-redundant profiles across 16.9k promoters
from training chromosomes (Table 5, Fig. 4). Each motif was compared with column-permuted copies
of itself, which removes enrichment caused by the TSS-proximal GC/CpG gradient.

The strongest positional preferences reproduce known promoter architecture:

* **ETS family** (ELK1, ETS1, GABPA-like, FEV): −23…−28;
* **SP/KLF GC-boxes, AP-1/ATF/CREB and bHLH E-boxes (MAX, USF):** at the TSS;
* **YY:** downstream (+4 to +10);
* **TBP:** −30, with a second, reverse-strand cluster near −15.

Under Bonferroni correction, with a margin over each motif's own permutations, 180 motifs are
positionally constrained. Greedy de-duplication by shared instances left **60 non-redundant rules**.

### 2.7 GCR-Atlas: many rules at once
Training the Transformer with all 60 rules (GCR-Atlas) gave the following (Table 3):

* **Multi-rule grammar discrimination** rose from 0.25 to 0.62. It rose strongly for ETS, AP-1/CREB,
  MAX, TBP and bHLH rules (e.g. CREB3L4 0.14→0.97, FOS 0.26→0.96).
* **Rules that did not improve:** the SP/KLF GC-box rules and several downstream NF-Y windows.
  This is consistent with the known position- and orientation-flexibility of GC-boxes, and with the
  weaker support for downstream NF-Y windows.
* **TATA grammar was preserved** (0.96).
* **Correlation with MPRA** was the highest of all methods (ρ = 0.262).
* **Accuracy** was unchanged (AUROC 0.937).

**CNN.** The benefit depends on architecture. In the global-pooling CNN, GCR-Atlas gave:

* only a modest gain in multi-rule grammar (0.24 → 0.34);
* a small gain in per-seed MPRA correlation (0.28 → 0.30);
* a small accuracy cost (AUROC 0.944 → 0.931).

A translation-invariant readout can encode positions only through local context within its 71-bp
receptive field. This limits how many positional rules it can absorb at once.

### 2.8 A pretrained genomic language model (NT-v2-50M)
To test whether the failure affects published genomic language models, we fully fine-tuned the
pretrained Nucleotide Transformer v2 (50M parameters, multi-species) under the same protocol
(Table 1, Table 3).

**Baseline: accurate but blind.** Fine-tuned NT-v2 reached AUROC 0.941 ± 0.002, yet it was
**completely blind to TATA position**:

* grammar discrimination 0.22 ± 0.03;
* displacing the TATA box changed the logit by −0.01 ± 0.03;
* the TATA-destroying point mutation changed it by 0.00 ± 0.01.

Pretraining on multi-species genomes does not give the model this positional rule, and neither
does fine-tuning on promoters.

**GCR fixes it, with no accuracy cost.**

* Grammar discrimination 0.99 ± 0.01.
* Point-mutation effect +1.35 ± 0.12.
* Control changes stay small (0.06).
* AUROC was, if anything, higher: 0.947 ± 0.002.

**Hard negatives** taught the rule less reliably and cost accuracy:

* grammar discrimination 0.95 ± 0.06 (one seed only 0.86);
* point-mutation effect 0.86 ± 0.55 (range 0.20–1.54);
* AUROC 0.936 ± 0.006, about 0.01 below GCR.

On the pretrained model, GCR was the most accurate and most stable way to inject the rule.

**On measured variant effects, the gain is targeted.**

* The HBB β-thalassemia TATA alleles were ranked consistently high: 88 ± 4th percentile across
  seeds, against 63 ± 17 for the baseline, which varied widely between seeds.
* Overall MPRA correlation was unchanged (per-seed ρ 0.18 vs 0.18).
* TERT C228T, which creates an ETS site rather than a TATA site, stayed erratic for both
  (0–69th percentile). This is expected: TATA-only GCR does not teach ETS grammar.

**GCR-Atlas on NT-v2 gives the best model overall.** Training the pretrained model with all 60
atlas rules gave:

* **Multi-rule grammar:** raised from 0.21 to 0.70 ± 0.02, the highest of any architecture.
* **TATA grammar:** 0.95 ± 0.03.
* **Accuracy:** preserved (AUROC 0.939 ± 0.002).
* **MPRA agreement:** the seed-ensembled correlation with measured MPRA effects rose from 0.204 to
  **0.336** (+0.132 [0.044, 0.223], paired bootstrap p = 0.001). This is the highest of any model
  in the study, above the CNN baseline (0.315).
* **Breadth of the gain:** it holds on TATA promoters (0.34 → 0.67) and on the other six promoters
  (0.18 → 0.26). It is largest for TERT (0.04 → 0.44), HBG1 (−0.01 → 0.69) and MSMB
  (0.47 → 0.70).

So a single-rule constraint helps mainly on TATA-dependent variants, whereas the atlas-wide
constraint improves the pretrained model's variant-effect predictions broadly.

### 2.9 Robustness to hyperparameters and run-to-run noise
We varied the ranking weight (0.3–3), the invariance weight (0.3–3) and the margin (0.5–2) for the
Transformer (Table 6). Every setting kept:

* grammar discrimination 0.97–1.00;
* β-thalassemia damage percentile 90–99;
* MPRA ρ 0.10–0.17 (all well above baseline 0.04);
* AUROC 0.935–0.949.

The only visible trade-offs came at the extremes. A strong invariance term (λ = 3) lowered MPRA ρ
to 0.10, and a large margin (m = 2) cost ~0.005 AUROC.

Re-running the default configuration with the same seed gave AUROC 0.949, against 0.940 in the
main study. GPU nondeterminism alone therefore moves AUROC by up to ~0.01. The AUROC differences
between GCR and baseline in the main study (≤ 0.006) are within this noise.

## 3. Discussion

**Main finding.** High benchmark accuracy does not mean a model has learned regulatory grammar.
In our setting, three promoter models, including a fine-tuned pretrained genomic language model,
reached AUROC ≈ 0.94 while ignoring a positional rule that is textbook biology and clinically
consequential. The pretrained model was the most grammar-blind of the three (0.22). Rarity alone did not explain the failure:
the rule stayed unlearned even at 16% frequency, which points to shortcut learning. Composition
cues are sufficient for the benchmark, so the optimiser has no reason to encode position.

**Relation to prior proposals.** GCR can be seen as a concrete, biologically constrained
realisation of the "compositional supervision" direction suggested by the Mechanistic Invariance
Test. It differs from generic structured-vs-scrambled discrimination in four ways:

1. The counterfactuals preserve composition *exactly* (block swaps).
2. It uses a ranking rather than a label.
3. Position-matched controls rule out edit detection.
4. The rules come from JASPAR or from a composition-controlled atlas.

The related idea of training with explanation or counterfactual constraints ("right for the right
reasons") is established in other ML domains. To our knowledge, its use for regulatory grammar with
disease-variant validation is new; this claim rests on a limited literature search.

**Why GCR works.** GCR supplies exactly the missing signal. Its counterfactuals keep every
composition cue fixed, so only positional information can explain the required score difference.
It needs only a motif (from JASPAR) and a positional window (from the literature or from our
atlas). It adds no parameters and can be applied to any differentiable model. Its advantages over
the two closest alternatives:

* **vs augmentation:** augmentation teaches invariance to exactly the rearrangements that matter;
* **vs hard negatives:** hard negatives assert a biologically false label and cost accuracy.

**Relation to the Khan Lab's resources.**

* **JASPAR 2026** added deep-learning models *to* the database. GCR goes the other way, putting
  JASPAR knowledge *into* deep-learning models.
* **The atlas's composition-matched null** follows the same principle as BiasAway.
* **UniBind's** high-confidence direct TF–DNA interactions would give an experimentally anchored
  alternative to PWM-derived instances.

### Limitations
* **Window size.** The windows span only −49…+250, so upstream elements such as CCAAT (−60 to
  −100) are largely outside them.
* **Positional enrichment is not proof of function.** Some atlas motifs (TIGD5, ZNF367) behave like
  AT-rich TATA mimics.
* **Task.** A promoter classifier is not an expression model. Absolute MPRA correlations remain
  modest (≤ 0.35), and for a CNN that already had partial grammar, GCR's benefit was targeted
  rather than global.
* **Scale.** 3 seeds per configuration; one pretrained model size.
* **Hyperparameters.** Not tuned: λ = 1, margin 1.
* **Rule selection** uses PWM thresholds.

### Future work
1. Longer windows with full EPDnew/FANTOM promoter sets.
2. UniBind-anchored rule instances.
3. GCR for MPRA-trained expression models and Enformer-class models.
4. Non-coding cancer mutations in promoters (e.g. PCAWG).
5. A positional-grammar annotation layer for JASPAR.

## Figure legends
*(files in `figures/`)*

- **Fig. 1 (`fig1_main_comparison.png`). Grammar, mutation sensitivity and accuracy by method.**
  Columns: grammar discrimination on 76 held-out TATA promoters (0.5 = chance), effect of a
  TATA-destroying point mutation never used in training, and test AUROC on chr20/21. Rows:
  Transformer, CNN-GlobalPool and NT-v2-50M. Dots are seeds; bars are means.
- **Fig. 2 (`fig2_tuning_curves.png`). Positional tuning curves.** Change in promoter score when the
  TATA box is moved by −14…+26 bp. None of these shifts was used in training. Baseline NT-v2 is
  flat, i.e. blind to position. GCR models peak at the native −30 position.
- **Fig. 3 (`fig3_rare_rule_curve.png`). Rare-rule experiment.** Grammar discrimination and AUROC as
  the number of canonical-TATA promoters among 3,000 training promoters varies. Bands show min–max
  over 3 seeds.
- **Fig. 4 (`fig4_atlas_heatmap.png`). JASPAR 2026 positional grammar atlas.** log2
  observed/expected hit density along the promoter for motifs used as GCR-Atlas rules. The
  expectation comes from column-permuted versions of each motif (composition-matched).
- **Fig. 5 (`fig5_hbb_beta_thalassemia.png`). HBB promoter.** Top: mean predicted effect of all
  single-nucleotide changes along the promoter (Transformer, baseline vs GCR, 3-seed average).
  Bottom: MPRA-measured effects. The shaded region is the TATA box (−31…−25), where β-thalassemia
  alleles occur.
- **Fig. 6 (`fig6_mpra_validation.png`). Agreement with measured variant effects.** Spearman
  correlation between predicted and MPRA-measured effects, using seed-ensembled predictions. Left:
  all 582 variants, with 95% bootstrap CIs. Right: the three TATA-box promoters.

## 4. Methods
See `methods.md` (full details, including every threshold, split and definition).

## 5. Data and code availability
All data are public:

* NT benchmark (Hugging Face);
* JASPAR 2026 (REST API / bulk download);
* UCSC hg38 API;
* MPRA (`gonzalobenegas/sat_mut_mpra`).

All code, results (JSON/CSV) and figures are in `DNA/grammar/`, and every table is regenerated by
`make_tables.py`.

## Author contributions and disclosure
Code was developed with AI assistance (Claude Code). All analyses are reproducible from the
provided scripts.
