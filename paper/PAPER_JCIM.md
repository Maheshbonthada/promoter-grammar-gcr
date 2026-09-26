# Composition-Preserving Counterfactual Regularization Teaches DNA Language Models the Positional Grammar of TBP–Promoter Recognition

[Author names]¹, …
¹ [Affiliation]
Corresponding author: [e-mail]

*Manuscript prepared for the Journal of Chemical Information and Modeling (Article). All numbers
are generated from `results/` by `make_tables.py` (see `paper/tables.md`); 5 seeds for the main
configurations, 3 elsewhere.*

## Abstract

Deep-learning models of DNA, including pretrained genomic language models, reach high accuracy on
promoter benchmarks and are increasingly used to interpret disease variants. We ask whether they
learn a basic physical rule of promoter function: the TATA-binding protein (TBP) must bind about
30 bp upstream of the transcription start site (TSS).

Using composition-preserving counterfactuals (same nucleotides, motif moved) with position-matched
controls, we show that four promoter models reach AUROC 0.94–0.95 while largely ignoring this
rule: a from-scratch Transformer, a CNN, and the pretrained Nucleotide Transformer v2 at two sizes.
The 50M-parameter pretrained model is never better than chance (grammar discrimination
0.34 ± 0.15; 0.5 = chance), and the 100M model is unreliable (0.58 ± 0.10). A controlled experiment
shows that the failure is shortcut learning, not rarity.

We introduce **Grammar-Consistency Regularization (GCR)**, a model-agnostic loss built from JASPAR
binding profiles. GCR ranks each native promoter above its counterfactuals and keeps it invariant
to control edits. It restores the rule in every model (grammar discrimination 0.97–0.99), including
both pretrained model sizes, without loss of accuracy. A JASPAR 2026-wide positional atlas
extends GCR to 60 rules (GCR-Atlas).

GCR improves agreement with independent physical and functional measurements:

* **TBP–DNA binding free energies (EMSA) of 14 disease TATA-box SNPs:** Spearman ρ 0.07 → 0.53
  for NT-v2-50M. All 9 binding-weakening SNPs are called damaging, against 6 of 9 for the baseline.
* **Saturation-mutagenesis MPRA effects (582 variants in 9 disease promoters):**
  - Transformer, ρ 0.09 → 0.27 with GCR-Atlas (p < 0.001);
  - TATA-box promoters, gains for both pretrained models;
  - overall MPRA agreement of the pretrained models, gains that are small and not significant.

Curated binding-site knowledge, imposed as a composition-preserving constraint, makes sequence
models respect the molecular grammar they are used to interpret.

**Keywords:** genomic language models; transcription factor binding; TATA-binding protein;
counterfactual regularization; JASPAR; variant effect prediction; shortcut learning

## INTRODUCTION

**Position is part of the binding code.** Transcription starts when general transcription factors
assemble on the core promoter. TBP binds the minor groove of the TATA box and bends the DNA by
about 80°.¹,² This positions TFIIB and RNA polymerase II so that transcription starts about 30 bp
downstream in humans.³ The TATA–TSS spacing is therefore a physical constraint set by the geometry
of the pre-initiation complex. A TATA box at the wrong distance cannot place the polymerase
correctly.

**Clinical relevance.** Single-nucleotide changes in the *HBB* TATA box (−31 to −28) weaken TBP
binding up to 9-fold⁴ and cause β-thalassemia,⁵ one of the most common monogenic disorders
worldwide.

**Are the models learning this?** Deep-learning models of regulatory DNA, from CNNs to pretrained
genomic language models such as the Nucleotide Transformer,⁶ now reach high accuracy on promoter
benchmarks. JCIM has published deep-learning predictors of TF binding sites.⁷,⁸ Whether such
models learn the positional rules of binding is less clear. A recent Mechanistic Invariance Test⁹
reported that genomic language models rely on compositional regularities (AT-rich ⇒
promoter-like) rather than positional logic. It proposed "compositional supervision" as a remedy,
but did not implement or test it. Shortcut learning of this kind is common in deep learning.¹⁰

**What we do.** We work in the setting where the rule is physically best characterised (TBP) and
where binding measurements exist for disease variants. We make six contributions:

1. **A composition-controlled test of positional grammar.** It uses exact-composition block swaps
   with position-matched controls, in a coordinate frame derived from the data.
2. **Evidence that the failure is shortcut learning.** Three architectures, including two sizes
   of a pretrained DNA language model, are accurate but blind to TATA position, and the failure is
   not explained by rarity.
3. **GCR.** A ranking-plus-invariance constraint built from JASPAR,¹¹ compared against
   augmentation¹² and hard negatives.
4. **A JASPAR 2026 positional atlas.** Built with a composition-matched null (in the spirit of
   BiasAway¹³), it generalises GCR to 60 rules.
5. **Validation against TBP–DNA binding free energies.** EMSA measurements of 14 disease SNPs.⁴
6. **Validation against measured regulatory effects.** A saturation-mutagenesis MPRA¹⁴ and
   disease alleles.

## METHODS

The full protocol, with every threshold and split, is in the Supporting Information (`methods.md`).
The key elements are summarised below.

**Data and coordinate frame.** We used the human promoter tasks of the Nucleotide Transformer
benchmark (300-bp windows from EPDnew promoters¹⁵ and non-promoter genomic windows).

* **TSS position:** index 49. We derived this from the data: CA-initiator enrichment peaks at
  indices 48–49, and TBP-motif hits peak at index 19 (−30).
* **Training set:** `promoter_all` minus chr8, chr9 and chr18 (26,699 sequences).
* **Test set:** the official chr20/21 split.
* **Grammar probe:** all canonical-TATA promoters on held-out chromosomes (n = 76).

**Counterfactuals.** All edits are block swaps, so nucleotide composition is preserved exactly.

* **TATA displacement:** the 8-bp TATA block is swapped to ≥ +51.
* **Position-matched control:** a same-length non-TATA block from the core upstream region is
  swapped to the same kind of target.
* **Tests never used in training:** point mutation (TATAAAA→TGTGAAA) and positional shifts
  (−14 to +26 bp).

**Grammar discrimination (GD)** is the AUROC separating per-sequence TATA-displacement effects from
control effects. It is 0.5 for a model without positional knowledge.

**Models.**

* CNN with global pooling (receptive field 71 bp).
* Transformer (conv stem, learned positions, 3 layers).
* Pretrained Nucleotide Transformer v2, 50M and 100M parameters, fully fine-tuned.⁶

**Objectives.** All methods use binary cross-entropy (BCE). Each step adds an auxiliary batch of 32
canonical-TATA positives *x*, with displaced (*x̃*) and control (*x̂*) versions.

* **GCR:** L = BCE + λ_r · mean[max(0, m − (z(x) − z(x̃)))] + λ_i · mean[(z(x) − z(x̂))²],
  with λ_r = λ_i = m = 1.
* **Comparators:**
  - hard negatives, BCE(z(x̃), 0);
  - EvoAug-style random block swaps;¹²
  - GCR-rank, which drops the invariance term.
* **GCR-Atlas** applies the same loss to instances of 60 JASPAR 2026 rules (below), with
  inverse-frequency sampling.

**JASPAR 2026 positional atlas.**

* **Scan:** all 1,019 CORE vertebrate non-redundant profiles, on both strands, over 16.9k promoters
  and 27.9k background windows from training chromosomes. Thresholds give a false-positive rate of
  2 × 10⁻⁴.
* **Composition-matched null:** positional enrichment is tested against five column-permuted
  copies of each motif (Poisson test, Bonferroni correction).
* **Rules:** windows are kept if enrichment ≥ 1.5, hits ≥ 100, and the signal is ≥ 5 orders of
  magnitude beyond the motif's own permutations. Greedy de-duplication by shared instances then
  leaves 60 rules.

**TBP–DNA binding validation.**

* **Data:** K_D values measured by EMSA (full-length recombinant human TBP, 26-bp duplexes, 25 °C)⁴
  for 14 SNPs in 9 promoters, including a binding-neutral SNP (*F9* −26G>C) and a binding-gain SNP
  (*IL1B* −31C>T).
* **Energy:** ΔΔG = RT ln(K_D,mut/K_D,wt).
* **Placement:** each SNP is placed in its hg38 promoter (MANE TSS), with the reference allele
  verified against the genome.
* **Prediction:** the model's loss of promoter score, logit(ref) − logit(alt), standardised per
  seed and ensembled.
* **Reference:** the change in the best JASPAR TBP log-odds score.

**MPRA and disease alleles.**

* **MPRA:** 582 variants from a saturation-mutagenesis MPRA¹⁴ fall in the windows of 9 promoters
  (RefSeq TSS; reference alleles verified).
* **Metric:** Spearman ρ between predicted and measured effects, using seed-ensembled predictions.
  CIs come from 2,000 bootstrap resamples; comparisons use a paired bootstrap.
* **β-thalassemia:** the *HBB* TATA alleles are ranked among all 897 SNVs of the window.
* **TERT:** C228T¹⁶ is scored in the same way.

**Seeds and software.**

* **Seeds:** 5 per main configuration (Transformer, CNN and NT-v2-50M; baseline, GCR and
  GCR-Atlas) and 3 elsewhere, including all NT-v2-100M configurations.
* **Software:** PyTorch 2.3 and Transformers 4.57.
* **Unit tests:** they verify exact composition preservation, GPU/CPU kernel agreement and
  target/control separation.

## RESULTS AND DISCUSSION

### Accurate promoter models are blind to TBP position
We trained each architecture on `promoter_all` with three chromosomes held out, and tested on the
official chr20/21 split (Table 1, Figure 1). All reached AUROC 0.940–0.951, yet none used TATA
position reliably:

| Model | Grammar discrimination (baseline) | Δ logit, TATA point mutation |
|---|---|---|
| Transformer | 0.39 ± 0.14 | 0.12 ± 0.15 |
| NT-v2-50M | 0.34 ± 0.15 (range 0.17–0.55) | 0.18 ± 0.22 |
| NT-v2-100M | 0.58 ± 0.10 | 0.51 ± 0.18 |
| CNN, global pooling | 0.85 ± 0.03 (partial) | 0.81 ± 0.35 |

For the Transformer and NT-v2-50M, displacing the TATA box changed the score by about as much as a
harmless control edit. Doubling the pretrained model to 100M parameters improved grammar only
partly: the bigger model is less blind, but it is not reliable.

Random block-swap augmentation (EvoAug-style¹²) did not help (Transformer 0.36 ± 0.06). This is
expected: it teaches invariance to exactly the rearrangements that matter.

### Rarity does not explain the failure
The baseline Transformer did not learn the rule when canonical-TATA promoters made up 0%, 1%, 3%,
10% or 16% of 3,000 training promoters (GD 0.13–0.36). GCR learned it from 30 examples (GD
0.98 ± 0.01) at unchanged AUROC (Table 2). Composition cues separate promoters from background so
well that a rare positional element provides almost no gradient signal. This is shortcut learning.

### GCR restores the rule without cost to accuracy
GCR raised grammar discrimination to 0.97–0.99 in all four models (Table 1). Accuracy was
unchanged within GPU run-to-run noise (± 0.01):

| Model | AUROC, baseline → GCR | Grammar, baseline → GCR | Δ point mutation, baseline → GCR |
|---|---|---|---|
| Transformer | 0.942 → 0.938 | 0.39 → 0.99 | 0.12 → 1.62 |
| CNN | 0.944 → 0.940 | 0.85 → 0.97 | 0.81 → 1.06 |
| NT-v2-50M | 0.940 → 0.946 | 0.34 → 0.98 | 0.18 → 1.55 |
| NT-v2-100M | 0.951 → 0.952 | 0.58 → 0.98 | 0.51 → 1.71 |

**Generalisation beyond the training edits.** The point mutation and small positional shifts
(Figure 2) were never used in training. GCR models respond strongly to the point mutation. Their
tuning curves peak at the native −30 position, although training used only far (≥ +51 bp)
displacements.

**Comparison with hard negatives.** Labelling displaced sequences as negatives also taught the
rule (0.95–0.98), but it cost accuracy:

* Transformer AUROC 0.920 ± 0.005;
* NT-v2-50M 0.936 ± 0.005, with one seed at GD 0.86.

That label is biologically unjustified, since a promoter with a misplaced TATA box is weakened,
not abolished. The ranking claims only what is known.

### Model effects track TBP–DNA binding energetics
If a model has learned *where* TBP must bind, its predicted effects of TATA-box SNPs should follow
how strongly those SNPs change TBP–DNA binding. We compared seed-ensembled predictions with ΔΔG
computed from EMSA dissociation constants for 14 disease SNPs⁴ (Table 7, Figure 7). For reference,
the JASPAR TBP matrix, a direct binding model, reaches ρ = 0.66.

| Model | ρ with ΔΔG, baseline → GCR | Binding-weakening SNPs (≥ 2-fold) called damaging |
|---|---|---|
| NT-v2-50M | 0.07 → 0.53 (p = 0.049) | 6/9 → 9/9 |
| Transformer | 0.11 → 0.44 | 3/9 → 8/9 |
| NT-v2-100M | 0.40 → 0.52 | 9/9 → 9/9 |
| CNN | 0.49 → 0.49 | 7/9 → 9/9 |

All models, baseline and GCR, predict the binding-gain SNP *IL1B* −31C>T (TATAAAAA created,
ΔΔG = −0.84 kcal/mol) to *increase* promoter score, which is correct. The GCR Transformer and
NT-v2 models predict the binding-neutral *F9* −26G>C near zero.

The largest binding loss, *TPI1* −24T>G, is under-called by every neural model. It changes the
fifth base of TATATAA and leaves the TATA tetranucleotide intact. The training point mutations,
by contrast, disrupt the tetranucleotide itself. With only 14 SNPs, the confidence intervals are
wide. The direction is nevertheless consistent
across architectures: GCR moves predictions toward measured binding energetics, and closest to
the physical reference for the pretrained model.

### A JASPAR 2026 positional atlas and GCR-Atlas
The atlas recovers known core-promoter architecture from sequence alone (Table 5, Figure 4):

* ETS motifs at −28…−23;
* SP/KLF, AP-1/CREB and E-box motifs at the TSS;
* YY1 downstream;
* TBP at −30.

Of the 1,019 motifs, 180 pass all criteria, giving 60 non-redundant rules. GCR-Atlas taught these
rules to attention-based models:

* **Mean 60-rule GD:** Transformer 0.25 → 0.64; NT-v2-50M 0.23 → 0.71; NT-v2-100M 0.28 → 0.76.
* **TATA rule and accuracy:** both preserved.

The global-pooling CNN gained less (0.24 → 0.34). Its translation-invariant readout can encode
position only through local context.

### Agreement with measured regulatory effects
We placed 582 variants from a saturation-mutagenesis MPRA¹⁴ of 9 disease promoters in the
training frame. We then compared seed-ensembled predictions with the measured effects (Table 3,
Table 4, Figures 5–6).

**Transformer: large, significant gains.**

* Baseline: ρ = 0.088 [0.008, 0.168].
* GCR: 0.185 (Δ = +0.097 [+0.021, +0.171]).
* GCR-Atlas: 0.265 (Δ = +0.177 [+0.097, +0.258], p < 0.001).

**CNN and pretrained models: targeted gains.** For the CNN (0.308) and the pretrained models
(0.247 for 50M, 0.255 for 100M), overall MPRA agreement changed little:

* NT-v2-50M with GCR-Atlas: Δ = +0.065 [−0.019, +0.149], p = 0.06;
* NT-v2-100M with GCR-Atlas: ρ 0.255 → 0.324, Δ = +0.068 [−0.011, +0.148], p = 0.04;
* TATA-only GCR on the pretrained models: no change.

Both pretrained sizes thus show the same small overall gain with GCR-Atlas (+0.065 and +0.068),
with confidence intervals that include zero; we do not claim it as significant.

On the three TATA-box promoters (*HBB*, *HBG1*, *MSMB*), however, the gains were significant:

* NT-v2-50M with GCR-Atlas: ρ 0.47 → 0.65 (Δ = +0.175 [+0.074, +0.286]);
* NT-v2-100M with GCR: 0.40 → 0.51 (Δ = +0.115 [+0.007, +0.221]);
* NT-v2-100M with GCR-Atlas: 0.40 → 0.58 (Δ = +0.181 [+0.058, +0.307]);
* CNN with GCR-Atlas: Δ = +0.096 [+0.002, +0.192].

**β-thalassemia alleles.** The *HBB* TATA-box alleles (−31…−28) rose in rank among all 897 SNVs of
the *HBB* window, as percentiles of predicted damage:

| Model | Baseline → GCR |
|---|---|
| Transformer | 42 → 93 |
| CNN | 57 → 94 |
| NT-v2-50M | 69 → 89 |
| NT-v2-100M | 86 → 91 |

The baseline percentiles varied widely between seeds (e.g. NT-v2-50M 69 ± 17), whereas GCR
percentiles were stable (89 ± 3).

**TERT C228T,** which creates an ETS site rather than a TATA site, remained erratic for all
methods.

In summary, the MPRA benefit is concentrated where the enforced rules operate. Improving
genome-wide variant-effect prediction will require rules and training targets beyond promoter
classification.

### Robustness and limitations
**Robustness.** Across ranking and invariance weights of 0.3–3 and margins of 0.5–2, GCR kept:

* GD at 0.97–1.00;
* the β-thalassemia damage percentile at 90–99.

GPU nondeterminism moves AUROC by up to ~0.01, which is larger than any GCR-vs-baseline AUROC
difference.

**Limitations.**

* **Window length.** The windows span −49…+250, so distal elements such as CCAAT boxes are
  outside them.
* **Task.** The models are promoter classifiers, not expression models, so absolute MPRA
  correlations are modest.
* **TBP data.** The binding set is small (n = 14), and its EMSA used 26-bp duplexes rather than
  chromatin.
* **Atlas.** Positional enrichment does not prove function.
* **Controls.** For < 1% of control draws, the upstream control block overlaps a late-starting TATA
  box. This makes the grammar test conservative.

**Relation to prior work.** GCR is, to our knowledge, the first implemented and tested form of the
compositional supervision proposed by the Mechanistic Invariance Test.⁹ It differs from generic
structured-vs-scrambled training in four ways:

1. The counterfactuals preserve composition exactly.
2. It uses a ranking, not a label.
3. Controls rule out edit detection.
4. The rules come from JASPAR.

It follows the "right for the right reasons" principle¹⁷ with biophysically grounded constraints.

## CONCLUSIONS
High benchmark accuracy does not mean that a DNA model has learned the physical grammar of
protein–DNA recognition. The widely used pretrained Nucleotide Transformer ignores where TBP must
bind, and scores binding-weakening disease mutations as neutral. GCR fixes this at no cost in
accuracy or parameters, by encoding curated binding-site knowledge as composition-preserving
counterfactual constraints. The result is closer agreement with measured TBP binding energetics
and, for TATA-dependent promoters, with measured regulatory effects.

The approach applies to any differentiable sequence model and any positional binding rule.
Natural extensions are:

* longer promoter windows;
* expression models;
* rule instances anchored in experimentally supported TF–DNA interactions (UniBind¹⁸).

## ASSOCIATED CONTENT
**Supporting Information:**

* full methods;
* Tables S1–S7 (all results, including per-rule atlas statistics, hyperparameter sensitivity and
  per-variant TBP data);
* Figures S1–S7.

## DATA AND SOFTWARE AVAILABILITY
All input data are public:

* Nucleotide Transformer benchmark (Hugging Face `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`);
* JASPAR 2026;
* UCSC hg38 (REST API);
* MPRA (`gonzalobenegas/sat_mut_mpra`);
* TBP K_D values (ref 4, Table 1; transcribed in `tbp_binding.py`).

Code (MIT license), unit tests, result files (JSON/CSV) and scripts that regenerate every table
and figure are available at https://github.com/Maheshbonthada/promoter-grammar-gcr. All trained checkpoints (81 models across 4 architectures and 6 training
methods, 3–5 seeds each) are available at https://huggingface.co/Sravankumarbonthada/promoter-grammar-gcr (CC BY-NC-SA 4.0, following the Nucleotide Transformer
license).

## AUTHOR INFORMATION
[Names, ORCID, contributions.] **Notes:** The authors declare no competing financial interest.
Code development used an AI coding assistant (Claude Code); all analyses were verified by the
authors.

## REFERENCES
*(ACS style; check volume and page details before submission)*
1. Kim, J. L.; Nikolov, D. B.; Burley, S. K. Co-crystal structure of TBP recognizing the minor groove of a TATA element. *Nature* **1993**, *365*, 520–527.
2. Kim, Y.; Geiger, J. H.; Hahn, S.; Sigler, P. B. Crystal structure of a yeast TBP/TATA-box complex. *Nature* **1993**, *365*, 512–520.
3. Bucher, P. Weight matrix descriptions of four eukaryotic RNA polymerase II promoter elements derived from 502 unrelated promoter sequences. *J. Mol. Biol.* **1990**, *212*, 563–578.
4. Savinkova, L.; Drachkova, I.; Arshinova, T.; Ponomarenko, P.; Ponomarenko, M.; Kolchanov, N. An experimental verification of the predicted effects of promoter TATA-box polymorphisms associated with human diseases on interactions between the TATA boxes and TATA-binding protein. *PLoS One* **2013**, *8*, e54626.
5. Antonarakis, S. E.; Orkin, S. H.; et al. β-Thalassemia in American Blacks: novel mutations in the "TATA" box and an acceptor splice site. *Proc. Natl. Acad. Sci. U.S.A.* **1984**, *81*, 1154–1158.
6. Dalla-Torre, H.; et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nat. Methods* **2025**, *22*, 287–297.
7. [Authors]. Prediction of transcription factor binding sites on cell-free DNA based on deep learning. *J. Chem. Inf. Model.* **2024**, *64*, 4002.
8. [Authors]. Multiview deep learning framework for precise prediction of transcription factor binding sites. *J. Chem. Inf. Model.* **2025**, *65*, 9762.
9. [Authors]. The Mechanistic Invariance Test. *arXiv* **2026**, 2604.06549.
10. Geirhos, R.; et al. Shortcut learning in deep neural networks. *Nat. Mach. Intell.* **2020**, *2*, 665–673.
11. [JASPAR 2026 consortium]. JASPAR 2026: expansion of transcription factor binding profiles and integration of deep learning models. *Nucleic Acids Res.* **2026**.
12. Lee, N. K.; Tang, Z.; Toneyan, S.; Koo, P. K. EvoAug: improving generalization and interpretability of genomic deep neural networks with evolution-inspired data augmentations. *Genome Biol.* **2023**, *24*, 105.
13. Khan, A.; Riudavets Puig, R.; Boddie, P.; Mathelier, A. BiasAway: command-line and web server to generate nucleotide composition-matched DNA background sequences. *Bioinformatics* **2021**, *37*, 1607–1609.
14. Kircher, M.; et al. Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution. *Nat. Commun.* **2019**, *10*, 3583.
15. Meylan, P.; Dreos, R.; Ambrosini, G.; Groux, R.; Bucher, P. EPD in 2020: enhanced data visualization and extension to ncRNA promoters. *Nucleic Acids Res.* **2020**, *48*, D65–D69.
16. Huang, F. W.; et al. Highly recurrent TERT promoter mutations in human melanoma. *Science* **2013**, *339*, 957–959.
17. Ross, A. S.; Hughes, M. C.; Doshi-Velez, F. Right for the right reasons: training differentiable models by constraining their explanations. *Proc. IJCAI* **2017**, 2662–2670.
18. Riudavets Puig, R.; Boddie, P.; Khan, A.; Castro-Mondragon, J. A.; Mathelier, A. UniBind: maps of high-confidence direct TF–DNA interactions across nine species. *BMC Genomics* **2021**, *22*, 482.
