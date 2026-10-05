## Methods

### Data and coordinate frame
**Promoter sequences.** We used the promoter tasks of the revised Nucleotide Transformer benchmark (`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`, tasks `promoter_all`, `promoter_tata` and `promoter_no_tata`). These are 300-bp human windows from EPDnew promoters (Meylan et al. 2020) (positives) and from genomic non-promoter regions (negatives). The official test split holds out chr20 and chr21.

**TSS position, derived from the data.** We located the TSS empirically rather than relying on the documented window convention: the initiator dinucleotide CA (C at −1, A at +1) peaks at window indices 48–49, at 3.4× its median frequency, and the TATA box (JASPAR MA0108.3, TATAAAA) peaks at index 19. The windows therefore span −49 to +250 around the TSS (index 49 = +1), and the canonical TATA box starts at −30, as expected from experimental work.

**Canonical TATA.** A sequence has a canonical TATA box if MA0108.3 (log-odds, pseudocount 0.8, uniform background) scores at least 0.7 × the maximum score at a start between −35 and −25 (indices 14–24). This holds for 20% of `promoter_tata` positives and 2.9% of `promoter_all` training positives.

**Splits.** The main study trains on `promoter_all` (train split) minus chr8, chr9 and chr18 (26,699 sequences; 13,611 positives; 392 with a canonical TATA box). 10% is held out at random for early stopping and for choosing the decision threshold. The test set is the official chr20/21 split (1,584 sequences). The grammar probe contains every promoter window from chr8, chr9, chr18, chr20 and chr21 across all promoter tables, de-duplicated by genomic coordinate (76 canonical-TATA promoters). No probe or test sequence is seen during training.

### Composition-preserving counterfactual edits
Every training edit swaps or moves blocks, so nucleotide composition is preserved exactly.

| Edit | Definition |
|---|---|
| TATA displacement | Swap the 8-bp TATA block (motif + 1 flank) with a random block starting at ≥ +51 (index ≥ 100) |
| Position-matched control | Swap a non-TATA 8-bp block from the core upstream region (starts −49 to −45 or −20 to −16, avoiding BRE, TATA and Inr) with the same kind of far target. For the 2.8% of training TATA boxes that start at −27 or later, the −20 control can overlap the last bases of the TATA block (< 1% of control draws), which makes the test conservative |
| Double substitution (never used in training) | TATAAAA → TGTGAAA (A→G at positions 2 and 4) |
| Control double substitution (never used in training) | Two A/T→G/C transitions at random positions of a non-TATA core-upstream block (same starts as the control swap); if the block has fewer than two A/T bases, two random bases are changed by transitions |
| Positional shift (never used in training) | Move the TATA block by s ∈ {−14, …, +26} bp; intervening bases shift by 8 bp |

### Models
* **CNN-GlobalPool:** conv(4→128, k = 15) + 3 residual dilated convolutions (dilation 2/4/8; receptive field 71 bp) → global max + mean pooling → MLP; a translation-invariant readout.
* **CNN-Dense:** the same convolutional trunk as CNN-GlobalPool with a position-aware readout (max-pool by 5, flatten, dense), used only to test whether the elevated null of the CNN depends on the translation-invariant readout. It is trained at a learning rate of 1e-4, at which all seeds converge.
* **Transformer:** convolutional stem (k = 9, stride 3) + learned absolute positions + 3 pre-norm encoder layers (d = 96, 4 heads) + CLS readout.
* **NT-v2-50M:** the pretrained Nucleotide Transformer v2 with 50M parameters (`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`), fully fine-tuned with a CLS + mean-pool head; our GPU-side 6-mer tokenizer reproduces the official tokenizer exactly.
* **NT-v2-100M:** the same wrapper around the 100M-parameter model (`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`), trained with batch size 64 because of the 8 GB GPU (all other settings identical) and with activation checkpointing.

### Training objectives
All methods share the binary cross-entropy (BCE) loss, the AdamW optimizer (weight decay 0.01), batch size 128 and early stopping on validation AUROC. The Transformer and CNN train for 12 epochs and NT-v2 for 3. Learning rates are 1 × 10⁻³ (CNN), 5 × 10⁻⁴ (Transformer) and 5 × 10⁻⁵ (NT-v2), fixed for every training method, so no method received extra tuning. The baselines reach test MCC 0.73–0.76, in line with published results for this task, so they are not undertrained relative to the literature. GCR costs 1.4–2.3× the baseline training time, from the extra passes over 32 auxiliary sequences per step. Every step also draws an auxiliary batch of 32 canonical-TATA training positives *x* with counterfactuals *x̃* (TATA displaced) and *x̂* (position-matched control). Writing *z* for the model's logit:

| Method | Loss |
|---|---|
| Baseline | BCE |
| Random-swap augmentation (EvoAug-style (Lee et al. 2023)) | BCE on sequences in which, with probability 0.5, two random non-overlapping 8-bp blocks are swapped, keeping the label |
| Hard negatives | BCE + BCE(*z*(*x̃*), 0) |
| GCR-rank | BCE + λ_r mean max(0, *m* − (*z*(*x*) − *z*(*x̃*))) |
| GCR | GCR-rank + λ_i mean (*z*(*x*) − *z*(*x̂*))² |

with λ_r = λ_i = *m* = 1. GCR asks only that a displaced TATA box lowers the promoter score (a ranking) and that position-matched controls leave it unchanged (invariance); it never claims that the edited sequence is a non-promoter, which is what the hard-negative baseline assumes. For every method, the decision threshold is chosen on the validation split to maximize the Matthews correlation coefficient.

### Core-Promoter Grammar Test
Measured on the 76 held-out canonical-TATA promoters; displacement and control effects are averaged over 8 random targets per sequence.

* **Δ_TATA and Δ_ctrl:** mean logit drop after TATA displacement and after the control swap.
* **Grammar discrimination (GD):** AUROC separating the per-sequence Δ_TATA values from the per-sequence Δ_ctrl values, both signed. It is 0.5 when displacing the TATA box lowers the score no more than a control edit does. (An earlier version of this analysis compared Δ_TATA with |Δ_ctrl|; that asymmetric statistic has a null value near 0.25, not 0.5, and is not used here.)
* **TATA recognition:** the same AUROC for the double substitution versus the control double substitution. Neither edit is used in training, and both change two bases in the same A/T→G/C direction.
* **Tuning curve:** mean score change as a function of TATA shift.

### Architecture-matched null models
For each architecture we trained a null model on the same data and settings with all 392 canonical-TATA
promoters removed from the training set (3 seeds), and additionally for CNN-Dense. Its scores define the level expected without TATA
knowledge from training, and replace the single reference level used previously. For the pretrained models
this removes TATA promoters only from fine-tuning, not from unsupervised pretraining, so their nulls are
conservative.

### Falsification controls: wrong rules
A ranking constraint over composition-preserving edits could improve a model for reasons unrelated to the
rule it encodes. Two controls use the identical loss, margin, auxiliary batch size, schedule and seeds, and
change only the rule.

* **Decoy GCR.** The protected block is not the TATA box but a fixed block at +71, in the downstream body,
  where no core-promoter element is positionally constrained; it is swapped with blocks further downstream.
* **Scrambled-atlas GCR.** Every atlas rule keeps its motif, threshold and width, but its window is moved
  80 bp from the position at which the atlas found it enriched (wrapping upstream where the shift would run
  off the sequence). Instances are taken from these wrong windows, so each motif is taught to belong where
  it is not positionally constrained.

Both are evaluated on the same held-out probe and the same external measurements as the real rules. A gain
that also appears with the wrong rule is not evidence for the rule.

### Strand test
TBP reads the TATA box in one orientation, so the rule is not purely positional. Reverse-complementing the
7-bp motif in place preserves its position, its width and its GC content (the motif is entirely A/T) and
changes the orientation; because TATAAAA is not palindromic, the edit is informative. It also reverses the
A/T ordering, so the same edit is applied to a non-TATA core-upstream block and the difference is reported.
Neither edit is used in training.

### Composition-matched negatives
To test whether the composition shortcut depends on the negative set, we replaced the genomic negatives with
the training promoters themselves, each shuffled within consecutive 20-bp segments by the Altschul-Erickson
algorithm, which preserves every dinucleotide count within each segment. Both classes then have identical
base and dinucleotide composition and identical composition profiles along the window, and a logistic
regression on GC content reaches AUROC 0.50 on the matched test set. Models were trained with the same
settings (3 seeds) and evaluated on the same held-out probe.

### Motif implantation
As a gain-of-function test using an edit never seen in training, the core TATAAAA was written into 300
held-out promoters without a canonical TATA box, at every start from -44 to +20 bp in 2-bp steps. The same
was done with the permutation of those seven bases that scores lowest with the JASPAR TBP matrix (AAAATTA),
and the difference between the two isolates the response to the motif from the response to its base
composition. We report the position of the maximum of this difference, and the implantation preference: its
mean within 2 bp of -30 minus its mean at positions at least 10 bp away.

### Robustness of the positional measures
Shift sensitivity is reported at cut-offs of 4, 8 and 12 bp, and also divided by each model's logit standard
deviation on the test set, because the GCR margin fixes a logit scale.

### Composition baseline
To measure how much of the task composition alone can solve, we fit L2-regularized logistic regressions (C = 0.1, standardized features) on the same training set and evaluated them on the same test split. Features were GC content, or frequencies of all k-mers for k = 1 up to k = 6 (best: k ≤ 5), counted over the whole window and therefore blind to position.

### Rare-rule experiment
Training sets are fixed at 3,000 promoters and 3,000 background sequences from training chromosomes. The number of canonical-TATA promoters is set to 0, 30, 90, 300 or 480 (0–16%; 480 is close to all 502 available). We train the Transformer with baseline or GCR, 3 seeds each, for 20 epochs, and evaluate on the same probe.

### JASPAR 2026 positional grammar atlas
**Scanning.** All 1,019 JASPAR 2026 CORE vertebrate non-redundant profiles (JASPAR 2026) were scanned on both strands over 16,916 promoters and 27,865 background windows from training chromosomes only. Per-motif thresholds give a false-positive rate on background of 2 × 10⁻⁴ per position per strand.

**Composition control.** Hit positions (sense-strand coordinates relative to the TSS) are compared with 5 column-permuted versions of the same motif, thresholded the same way. These keep length, per-column base content and information content, so they reproduce positional enrichment caused by the GC gradient around the TSS, following the principle of composition-matched backgrounds (BiasAway; Khan et al. 2021). They do not preserve CpG dinucleotides; see the stricter null below.

**Test statistic and rule selection.** For each 10-bp window we compute a Poisson upper-tail test of the observed hits against the permutation-derived expectation. A motif window counts as a positional rule if it passes Bonferroni correction (1,019 motifs × ~290 windows; log10 *p* ≤ −6.8), has enrichment ≥ 1.5 and ≥ 100 hits, and is at least 5 orders of magnitude more significant than the mean of the motif's own permutations. Up to two non-overlapping windows are kept per motif. Rules are then de-duplicated greedily, most significant first: a rule is dropped if more than 50% of its training instances coincide (same sequence, start within ±4 bp) with instances of rules already kept. This leaves 60 rules.

**Stricter null.** Each training promoter was shuffled within consecutive 20-bp segments with the Altschul–Erickson algorithm, which preserves every dinucleotide count (including CpG) within each segment, so the local GC and CpG profile along the promoter is kept while motif instances are destroyed (3 shuffles). A rule passes if its window holds more hits in real than in shuffled promoters (Poisson test, Bonferroni over 60 rules, enrichment ≥ 1.5).

### GCR-Atlas
GCR is generalized to all atlas rules. Instances are every motif hit inside a rule's window in a training positive, sampled with weights inversely proportional to the rule's frequency. Each instance is swapped (at its own width) to a random start at least 20 bp outside the window and clear of the TSS core; the control moves a same-width block from the motif's flank, never overlapping the motif or its target, to the same kind of target. Rule-level GD on held-out probe instances uses signed effects, as above, and is averaged over rules with at least 15 held-out instances.

### External validation on measured variant effects
**MPRA variants.** We used a saturation-mutagenesis MPRA of disease-associated regulatory elements (Kircher et al. 2019) (processed table `gonzalobenegas/sat_mut_mpra`; label = log2 alt/ref expression). We kept elements with a RefSeq TSS (UCSC `ncbiRefSeqCurated`, hg38) inside the assayed region: F9, GP1BB, HBB, HBG1, HNF4A, LDLR, MSMB, PKLR and TERT. Each promoter is placed in the training frame (sense strand, TSS at index 49) with sequence from the UCSC API, and every reference allele is checked against the genome. 582 variants fall in the −49 to +250 windows, 26 of them inside canonical TATA boxes (HBB, HBG1, MSMB). The predicted effect is logit(alt) − logit(ref), and the metric is the Spearman correlation with the measured effect.

**β-Thalassemia.** All 12 SNVs at HBB −31 to −28, the TATA-box positions where β-thalassemia alleles are reported (Antonarakis et al. 1984), are ranked by predicted damage among all 897 SNVs of the HBB window; the damage percentile is the share of SNVs predicted to be less damaging.

**TERT cancer mutation.** C228T (Huang et al. 2013) (chr5:1,295,113 G>A, hg38; sense-strand C>T at −45) creates an ETS/GABPA site and is ranked by predicted activation among all SNVs of the TERT window.

### Molecular validation: TBP–DNA binding free energy
Equilibrium dissociation constants (*K*_D) of recombinant human TBP binding to 26-bp duplexes carrying disease-associated TATA-box SNPs were taken from EMSA measurements (Savinkova et al. 2013) (Table 1 of that study); ΔΔ*G* = *RT* ln(*K*_D,mut/*K*_D,wt) at 298 K, positive values meaning weaker binding. The 14 SNPs lie in 9 promoters: *HBB* (7 β-thalassemia alleles, −31 to −27) and *HBD*, *CYP2A6*, *SOD1*, *TPI1*, *F9*, *IL1B* and *NOS2* (one each), including a binding-neutral SNP (*F9* −26G>C) and a binding-gain SNP (*IL1B* −31C>T). Each promoter is placed in the training frame using the MANE Select TSS; the published TATA string is located within ±30 bp of its reported position (up to 21 bp away, for *CYP2A6*, where annotations disagree), and the reference base is checked against hg38. hg38 carries the minor allele of *NOS2* −21T>C, so that SNP is scored in reverse with the *K*_D values swapped. Three published SNPs were excluded before analysis because their sequences could not be placed unambiguously (*TF* −21C>T, *MBL2* and εψ-globin). The prediction is logit(ref) − logit(alt); each seed's predictions are standardized and then averaged. The reference predictor is the change in the best JASPAR TBP (MA0108.3) log-odds score between −45 and −10. It is expected to be strong, because it is a direct model of TBP binding, whereas the neural models are promoter classifiers never trained on binding data; it is a physical reference rather than a competitor.

### Statistics
Transformer, CNN and NT-v2-50M use 5 seeds for baseline, GCR and GCR-Atlas; other configurations, including all NT-v2-100M models and the hyperparameter grid, use 3 seeds. We report mean ± s.d. over seeds. External comparisons use seed-ensembled predictions. All tests are two-sided with 5,000 bootstrap resamples. For the MPRA, differences in Spearman ρ are tested by resampling whole promoters (9 clusters), with a variant-level bootstrap reported alongside; the TATA-promoter subset (3 promoters) can only be resampled at the variant level. For the TBP analysis, differences in ρ are tested by resampling SNPs. Holm correction is applied within each family of GCR and GCR-Atlas versus baseline comparisons. These tests were added after the initial analysis, in response to the concern that variant-level resampling overstates significance.

### Software tests
`tests/test_counterfactuals.py` checks exact composition preservation for every swap edit, agreement of the GPU swap kernels with the reference CPU implementation, that displacement targets never overlap the motif, its control block or the TSS core, and recovery of a planted TATA box.

### Implementation and AI assistance
All models were implemented in PyTorch 2.3 and Transformers 4.57 and trained on a single 8 GB GPU. Code was written with the help of an AI coding assistant (Claude Code); the author reviewed all code and verified all analyses. No AI tool was used to generate data or results.
