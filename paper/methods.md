## Methods

### Data and coordinate frame
**Promoter sequences.** We used the promoter tasks of the revised Nucleotide Transformer benchmark
(`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`, tasks `promoter_all`, `promoter_tata`
and `promoter_no_tata`). These are 300-bp human windows from EPDnew promoters (positives) and from
genomic non-promoter regions (negatives). The official test split holds out chr20 and chr21.

**TSS position, derived from the data.** We did not rely on the documented window convention. We
located the TSS empirically using two signals:

* the initiator dinucleotide CA (C at −1, A at +1) peaks at window indices 48–49, at 3.4× its
  median frequency;
* the TATA box (JASPAR MA0108.3, TATAAAA) peaks at index 19.

The windows therefore span −49…+250 around the TSS (index 49 = +1), and the canonical TATA box
starts at −30, as expected from experimental work.

**Canonical TATA.** A sequence has a canonical TATA box if MA0108.3 (log-odds, pseudocount 0.8,
uniform background) scores ≥ 0.7 × the maximum score at a start between −35 and −25 (indices
14–24). This call has 88% precision against genomic negatives. It holds for 20% of `promoter_tata`
positives and 2.9% of `promoter_all` training positives.

**Splits.** The main study trains on `promoter_all` (train split) minus chr8, chr9 and chr18
(26,699 sequences; 13,611 positives; 392 with a canonical TATA). 10% is held out at random for
early stopping and for choosing the decision threshold. The **test set** is the official chr20/21
split (1,584 sequences). The **grammar probe** contains every promoter window from chr8, chr9,
chr18, chr20 and chr21 across all promoter tables, de-duplicated by genomic coordinate
(76 canonical-TATA promoters). No probe or test sequence is seen during training.

### Composition-preserving counterfactual edits
Every edit **swaps or moves blocks**, so nucleotide composition is preserved exactly. A model that
reads only composition (GC/AT content, CpG density) therefore cannot tell an edited sequence from
the original. The edits are:

| Edit | Definition |
|---|---|
| TATA displacement | Swap the 8-bp TATA block (motif + 1 flank) with a random block starting ≥ +51 (index ≥ 100) |
| Position-matched control | Swap a non-TATA 8-bp block from the core upstream region (starts −49…−45 or −20…−16, avoiding BRE, TATA and Inr) with the same kind of far target. For the 2.8% of training TATA boxes that start at −27 or later, the start −20 control can overlap the TATA block's last bases (affects < 1% of control draws) |
| Point mutation (never used in training) | TATAAAA → TGTGAAA (positions 2 and 4 A→G) |
| Positional shift (never used in training) | Move the TATA block by s ∈ {−14, …, +26} bp; intervening bases shift by 8 bp |

### Models
* **CNN-GlobalPool:** conv(4→128, k = 15) + 3 residual dilated convs (dilation 2/4/8; receptive
  field 71 bp) → global max + mean pooling → MLP. Translation-invariant readout.
* **Transformer:** conv stem (k = 9, stride 3) + learned absolute positions + 3 pre-norm encoder
  layers (d = 96, 4 heads) + CLS readout. Its inductive bias is closest to a DNA language model.
* **NT-v2-50M:** the pretrained Nucleotide Transformer v2, 50M parameters, multi-species
  (`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`), fully fine-tuned with a CLS +
  mean-pool head.
  - Our GPU-side 6-mer tokenizer reproduces the official tokenizer exactly.
* **NT-v2-100M:** the same wrapper around the 100M-parameter model
  (`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`), used as a model-size check. Because
  of the 8 GB GPU, it is trained with batch size 64 (all other settings identical).

### Training objectives
All methods share the binary cross-entropy (BCE) loss, the AdamW optimiser (weight decay 0.01),
batch size 128, and early stopping on validation AUROC. Transformer and CNN train for 12 epochs;
NT-v2 for 3. Every step also draws an auxiliary batch of 32 canonical-TATA training positives *x*
with counterfactuals *x̃* (TATA displaced) and *x̂* (position-matched control). Writing *z* for the
model's logit:

| Method | Loss |
|---|---|
| Baseline | BCE |
| Random-swap aug. (EvoAug-style) | BCE on sequences where, with p = 0.5, two random non-overlapping 8-bp blocks are swapped, keeping the label |
| Hard negatives | BCE + BCE(*z*(*x̃*), 0) |
| GCR-rank | BCE + mean max(0, m − (*z*(*x*) − *z*(*x̃*))), m = 1 |
| **GCR** | GCR-rank + mean (*z*(*x*) − *z*(*x̂*))² |

GCR asks only that a displaced TATA box lowers the promoter score (a *ranking*), and that
position-matched controls leave it unchanged (*invariance*). It never claims that the edited
sequence is a non-promoter; that stronger claim is biologically unjustified, and it is exactly
what the hard-negative baseline assumes.

For every method, the decision threshold is chosen on the validation split to maximise MCC.

### Core-Promoter Grammar Test (CPGT)
Measured on held-out canonical-TATA promoters; displacement and control deltas are averaged over
8 random targets.

* **Δ_TATA:** mean logit drop after TATA displacement.
* **|Δ_ctrl|:** mean absolute change after a position-matched control swap.
* **Grammar discrimination (GD):** AUROC separating the per-sequence Δ_TATA values from the
  |Δ_ctrl| values (0.5 = no positional knowledge).
* **Δ_mut:** logit drop after the TATA-destroying point mutation.
* **Tuning curve:** mean score change as a function of TATA shift.

### Rare-rule experiment
Training sets are fixed at 3,000 promoters and 3,000 background sequences (training chromosomes
only). The number of canonical-TATA promoters is set to 0, 30, 90, 300 or 480 (0–16%); 480 is close to all
502 available. We train
the Transformer with baseline or GCR, 3 seeds each, for 20 epochs, and evaluate on the same probe.

### JASPAR 2026 positional grammar atlas
**Scanning.** All 1,019 JASPAR 2026 CORE vertebrate non-redundant profiles were scanned on both
strands over 16,916 promoters and 27,865 background windows from training chromosomes only.
Per-motif thresholds are set so that the false-positive rate on background is 2 × 10⁻⁴ per
position per strand.

**Composition control.** Hit positions (in sense-strand coordinates relative to the TSS) are
compared with 5 **column-permuted versions of the same motif**, thresholded the same way. These
keep length, per-column content and information content, so they reproduce any positional
enrichment caused by the GC/CpG gradient around the TSS. This follows the same principle as
composition-matched backgrounds (BiasAway).

**Test statistic.** For each 10-bp window we compute a Poisson upper-tail test of the observed
hits against the permutation-derived expectation, scaled to the motif's total hits.

**Rule selection.** A motif window counts as a positional rule if it passes all of:

* Bonferroni correction (1,019 motifs × ~290 windows; log10 p ≤ −6.8);
* enrichment ≥ 1.5;
* ≥ 100 hits;
* ≥ 5 orders of magnitude more significant than the mean of its own permutations.

Up to two non-overlapping windows are kept per motif. Rules are then de-duplicated greedily, most
significant first: a rule is dropped if > 50% of its training instances coincide (same sequence,
start within ±4 bp) with instances of rules already kept. This removes TF-family redundancy, e.g.
the many ETS profiles.

### GCR-Atlas
GCR generalises to all atlas rules as follows:

* **Instances:** every motif hit inside a rule's window in a training positive.
* **Sampling:** instances are drawn with weights inversely proportional to how common their rule
  is.
* **Displacement:** each instance is swapped (at its own width) to a random start ≥ 20 bp outside
  the window and clear of the TSS core.
* **Controls:** a same-width block from the motif's flank, never overlapping the motif or its
  target, is moved to the same kind of target.
* **Evaluation:** per-rule grammar discrimination on held-out probe instances.

### External validation on measured variant effects
**MPRA variants.** We used a saturation-mutagenesis MPRA of disease-associated regulatory elements
(Kircher et al. 2019; processed table `gonzalobenegas/sat_mut_mpra`; label = log2 alt/ref
expression). We kept elements with a RefSeq TSS (UCSC `ncbiRefSeqCurated`, hg38) inside the
assayed region: F9, GP1BB, HBB, HBG1, HNF4A, LDLR, MSMB, PKLR and TERT.

* Each promoter is placed in the training frame (sense strand, TSS at index 49) with sequence
  from the UCSC API. We assert that every reference allele matches the genome.
* **Result:** 582 variants fall in the −49…+250 windows, 26 of them inside canonical TATA boxes
  (HBB, HBG1, MSMB).
* **Predicted effect:** logit(alt) − logit(ref).
* **Metric:** Spearman correlation with the measured effect.

**β-thalassemia.** All 12 SNVs at HBB −31…−28, the TATA-box positions where β-thalassemia alleles
(e.g. −28 A>G, −29 A>G, −30 T>A, −31 A>G) are reported. Each is ranked by predicted damage among
all 897 SNVs of the HBB window; the damage percentile is the share of SNVs predicted to be less
damaging.

**TERT cancer mutation.** C228T (chr5:1,295,113 G>A, hg38; sense-strand C>T at −45), which creates
an ETS/GABPA site (CCGGAA). It is ranked by predicted activation among all SNVs of the TERT
window. C250T (−67) lies outside the window and is not scored.

### Molecular validation: TBP–DNA binding free energy
* **Measurements.** Equilibrium dissociation constants (K_D) of recombinant human TBP binding to
  26-bp duplexes carrying disease-associated TATA-box SNPs, measured by EMSA (Savinkova et al.,
  PLoS One 2013, Table 1). ΔΔG = RT ln(K_D,mut / K_D,wt) at 298 K; positive values mean weaker
  binding.
* **Variants.** 14 SNPs in 9 promoters:
  - *HBB* (7 β-thalassemia alleles, −31…−27);
  - *HBD*, *CYP2A6*, *SOD1*, *TPI1*, *F9*, *IL1B* and *NOS2* (one each).
  They include a binding-neutral SNP (*F9* −26G>C) and a binding-*gain* SNP (*IL1B* −31C>T).
* **Placement.** Each promoter is placed in the training frame using the MANE Select TSS (UCSC
  `ncbiRefSeqCurated`, hg38).
  - The published TATA string is located within ±30 bp of its reported position, and the reference
    base is checked against hg38.
  - hg38 carries the minor allele of *NOS2* −21T>C, so that SNP is scored in the reverse direction
    with the K_D values swapped.
  - *TF* −21C>T is excluded because its published sequence is absent from the hg38 promoter.
  - *MBL2* and εψ-globin are excluded because their published strings are ambiguous.
* **Scoring.**
  - **Prediction:** logit(ref) − logit(alt), i.e. predicted loss of promoter score.
  - **Seed ensembles:** each seed's predictions are standardised, then averaged.
  - **Metrics:** Spearman and Pearson correlations with ΔΔG, and the fraction of the 9 SNPs with
    ≥ 2-fold weaker binding that are predicted to be damaging.
* **Reference predictor.** The change in the best JASPAR TBP (MA0108.3) log-odds score between
  −45 and −10. It is a physics-like reference that models binding directly.

### Software tests
`tests/test_counterfactuals.py` checks:
* exact composition preservation for every edit type;
* that the GPU swap kernels agree with the reference CPU implementation;
* that displacement targets never overlap the motif, its control block or the TSS core;
* recovery of a planted TATA box.

### Statistics
* Transformer, CNN and NT-v2-50M use 5 seeds for the main methods; the remaining comparisons use
  3 seeds, and NT-v2-100M uses 3. We report mean ± s.d. and individual seeds.
* Comparisons between methods use per-seed values, and sequence-level paired statistics where
  noted.
