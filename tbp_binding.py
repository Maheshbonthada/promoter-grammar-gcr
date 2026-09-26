"""Molecular validation: do model variant effects track measured TBP-DNA binding free energy?

Measured equilibrium dissociation constants (EMSA, human TBP, 26-bp duplexes) for disease
TATA-box SNPs from Savinkova, Drachkova, Ponomarenko et al., PLoS One 2013
(PMC3570547, Table 1).  ddG = RT ln(KD_mut / KD_wt) (kcal/mol, 25 C); positive = weaker binding.

Each variant is placed in its real promoter (hg38, MANE/RefSeq TSS, training frame:
300 bp, TSS at index 49, sense strand).  The wild-type TATA string from the table is located
near its published position and the SNP is applied inside it, checking the reference base.
Model prediction = logit(ref) - logit(alt) (predicted loss of promoter activity).
Output: results/tbp_binding.json, results/tbp_binding_variants.csv
"""
import glob, html, json, os, re, sys, urllib.request
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "tbp_windows.json")
COMP = str.maketrans("ACGT", "TGCA")
API = "https://api.genome.ucsc.edu"
RT = 0.5925  # kcal/mol at 298 K

# gene, WT string (sense), SNP position rel. TSS, ref, alt, offset of the SNP inside the string,
# KD_wt (nM), KD_mut (nM), disease   -- PMC3570547 Table 1
TABLE = [
    ("HBB", "CATAAAA", -31, "A", "G", 1, 44, 99, "beta-thalassemia"),
    ("HBB", "CATAAAA", -30, "T", "A", 2, 44, 116, "beta-thalassemia"),
    ("HBB", "CATAAAA", -30, "T", "C", 2, 44, 111, "beta-thalassemia"),
    ("HBB", "CATAAAA", -29, "A", "G", 3, 44, 390, "beta-thalassemia"),
    ("HBB", "CATAAAA", -28, "A", "G", 4, 44, 87, "beta-thalassemia"),
    ("HBB", "CATAAAA", -28, "A", "C", 4, 44, 300, "beta-thalassemia"),
    ("HBB", "CATAAAA", -27, "A", "T", 5, 44, 64, "beta-thalassemia"),
    ("HBD", "CATAAAA", -31, "A", "G", 1, 46, 120, "delta-thalassemia"),
    ("CYP2A6", "TATAAA", -48, "T", "G", 2, 17, 80, "lung cancer"),
    ("SOD1", "TATAAA", -27, "A", "G", 1, 40, 170, "ALS"),
    ("TPI1", "TATATAA", -24, "T", "G", 4, 4.8, 150, "neurological disorder"),
    ("F9", "TTGTACTT", -26, "G", "C", 2, 510, 500, "hemophilia B Leyden"),
    ("IL1B", "CATAAAAA", -31, "C", "T", 0, 29, 7, "lung cancer"),
    # hg38 already carries the C allele of NOS2 -21T>C: scored as C>T with KDs swapped
    ("NOS2", "TATAAATA", -21, "C", "T", 8, 1.6, 1.8, "infection resistance"),
]
# The table's short strings are abbreviated; each WT string above is the part verified in hg38.
# TF -21C>T is omitted: its published string (TTTATA) is absent from the hg38 TF promoter.


def _get(path):
    return json.load(urllib.request.urlopen(API + path, timeout=60))


def tss_of(gene):
    """MANE Select transcript -> (chrom, 1-based TSS, strand)."""
    d = _get(f"/search?search={gene}&genome=hg38")
    for m in d.get("positionMatches", []):
        if m.get("trackName") != "mane":
            continue
        for x in m["matches"]:
            name = html.unescape(x["posName"])
            if name.split(" ")[0] != gene:
                continue
            nm = re.search(r"(NM_\d+)", name).group(1)
            chrom, rng = x["position"].split(":")
            lo, hi = map(int, rng.split("-"))
            rows = _get(f"/getData/track?genome=hg38;track=ncbiRefSeqCurated;chrom={chrom};start={lo - 10};end={hi + 10}")
            for r in rows.get("ncbiRefSeqCurated", []):
                if r["name"].split(".")[0] == nm:
                    return chrom, (r["txStart"] + 1 if r["strand"] == "+" else r["txEnd"]), r["strand"], nm
    raise ValueError(gene)


def windows():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for gene in sorted({t[0] for t in TABLE}):
        chrom, tss1, strand, nm = tss_of(gene)
        if strand == "+":
            s0, e0 = tss1 - 1 - 49, tss1 - 1 + 251
        else:
            s0, e0 = tss1 - 251, tss1 + 49
        dna = _get(f"/getData/sequence?genome=hg38;chrom={chrom};start={s0};end={e0}")["dna"].upper()
        seq = dna if strand == "+" else dna.translate(COMP)[::-1]
        out[gene] = dict(chrom=chrom, tss=tss1, strand=strand, tx=nm, seq=seq)
        print(f"  {gene}: {nm} {chrom}:{tss1} ({strand})", flush=True)
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def variants():
    win = windows()
    rows = []
    for gene, wt, pos, ref, alt, off, kd_wt, kd_mut, dis in TABLE:
        seq = win[gene]["seq"]
        exp_idx = 49 + pos                      # published SNP index (no position 0)
        hits = [m.start() + off for m in re.finditer(f"(?={wt})", seq)
                if 0 <= m.start() + off < 300 and seq[m.start() + off] == ref]
        hits = [h for h in hits if abs(h - exp_idx) <= 30]   # TSS annotations differ
        if not hits:
            print(f"  {gene} {pos}{ref}>{alt}: '{wt}' not found near index {exp_idx} -> skipped")
            continue
        idx = min(hits, key=lambda h: abs(h - exp_idx))
        rows.append(dict(gene=gene, snp=f"{pos}{ref}>{alt}", idx=idx, shift=idx - exp_idx, ref=ref, alt=alt,
                         kd_wt=kd_wt, kd_mut=kd_mut, ddG=RT * np.log(kd_mut / kd_wt), disease=dis,
                         seq=seq))
    return pd.DataFrame(rows)


def encode(s):
    return np.array(["ACGTN".index(c) if c in "ACGT" else 4 for c in s], dtype=np.int64)


@torch.no_grad()
def predict_delta(m, V):
    R = np.stack([encode(s) for s in V.seq])
    A = R.copy()
    A[np.arange(len(V)), V.idx.values] = [ "ACGT".index(a) for a in V.alt]
    f = lambda Z: m(torch.as_tensor(Z).to(next(m.parameters()).device)).float().cpu().numpy()
    return f(R) - f(A)


def pwm_delta(V):
    """Physics-style reference: change in best JASPAR TBP (MA0108) log-odds score in -40..-15."""
    from motifs import load_pwms
    pwm = load_pwms()["TBP"]
    w = pwm.shape[1]

    def best(s):
        x = encode(s)
        return max(sum(pwm[x[i + j], j] for j in range(w) if x[i + j] < 4)
                   for i in range(max(0, 49 - 45), 49 - 10))
    out = []
    for _, v in V.iterrows():
        a = v.seq[:v.idx] + v.alt + v.seq[v.idx + 1:]
        out.append(best(v.seq) - best(a))
    return np.array(out)


def summarize(pred, ddg):
    rs, ps = spearmanr(pred, ddg)
    rp, pp = pearsonr(pred, ddg)
    strong = ddg > RT * np.log(2)             # >= 2-fold weaker binding
    return dict(spearman=float(rs), spearman_p=float(ps), pearson=float(rp), pearson_p=float(pp),
                frac_damaging_called=float((pred[strong] > 0).mean()),
                mean_pred_strong=float(pred[strong].mean()), mean_pred_neutral_or_gain=float(pred[~strong].mean()))


def main():
    from evaluate_mpra import load
    V = variants()
    print(V[["gene", "snp", "idx", "shift", "ddG", "disease"]].to_string(), flush=True)
    ddg = V.ddG.values
    out = {"n": len(V), "reference": {"TBP_PWM": summarize(pwm_delta(V), ddg)}, "runs": {}}
    paths = sorted(glob.glob(os.path.join(HERE, "runs", "*.pt")))
    archs = sys.argv[1:]
    preds = {}
    for p in paths:
        arch, method, seed = os.path.basename(p)[:-3].split("__")
        if archs and arch not in archs:
            continue
        _, _, _, m = load(p)
        d = predict_delta(m, V)
        preds[(arch, method, int(seed))] = d
        out["runs"][f"{arch}|{method}|{seed}"] = dict(pred=d.tolist(), **summarize(d, ddg))
        del m
        torch.cuda.empty_cache()
    ens = {}
    for arch, method in sorted({(a, me) for a, me, _ in preds}):
        D = np.stack([v for (a, me, _), v in preds.items() if a == arch and me == method])
        D = D / D.std(axis=1, keepdims=True).clip(1e-6)       # scale-free seed ensemble
        ens[f"{arch}|{method}"] = dict(n_seeds=len(D), **summarize(D.mean(0), ddg))
    out["ensemble"] = ens
    json.dump(out, open(os.path.join(HERE, "results", "tbp_binding.json"), "w"), indent=1)
    V.drop(columns="seq").to_csv(os.path.join(HERE, "results", "tbp_binding_variants.csv"), index=False)
    print(f"reference TBP PWM: rho {out['reference']['TBP_PWM']['spearman']:+.2f}")
    for k, v in ens.items():
        print(f"{k:28s} seeds {v['n_seeds']} rho {v['spearman']:+.2f} (p {v['spearman_p']:.3f}) "
              f"r {v['pearson']:+.2f} damaging-called {v['frac_damaging_called']:.2f}")


if __name__ == "__main__":
    main()
