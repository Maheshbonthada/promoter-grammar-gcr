"""External validation on measured variant effects: saturation-mutagenesis MPRA of
disease promoters (Kircher et al. 2019, processed by G. Benegas: HF
gonzalobenegas/sat_mut_mpra; label = log2 expression effect of the alt allele).

Each promoter is placed in the training frame (300 bp, TSS at index 49, sense
strand) using the RefSeq TSS from the UCSC API.
"""
import json, os, urllib.request
import numpy as np
import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "mpra_windows.json")
COMP = str.maketrans("ACGTacgt", "TGCAtgca")
UCSC = "https://api.genome.ucsc.edu/getData"


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


def fetch_windows(df):
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for el, g in df.groupby("element"):
        chrom = "chr" + str(g.chrom.iloc[0])
        lo, hi = int(g.pos.min()), int(g.pos.max())
        tr = _get(f"{UCSC}/track?genome=hg38;track=ncbiRefSeqCurated;chrom={chrom};start={lo - 300};end={hi + 300}")
        rows = tr.get("ncbiRefSeqCurated", [])
        cands = []
        for r in rows:
            tss1 = r["txStart"] + 1 if r["strand"] == "+" else r["txEnd"]          # 1-based TSS
            if lo - 150 <= tss1 <= hi + 150:
                cands.append((tss1, r["strand"], r["name2"], r["name"]))
        if not cands:
            print(f"  {el}: no RefSeq TSS inside element -> treated as enhancer, skipped")
            continue
        tss1, strand, gene, tx = sorted(set(cands))[0]
        if strand == "+":
            s0, e0 = tss1 - 1 - 49, tss1 - 1 + 251
        else:
            s0, e0 = tss1 - 251, tss1 + 49
        dna = _get(f"{UCSC}/sequence?genome=hg38;chrom={chrom};start={s0};end={e0}")["dna"].upper()
        seq = dna if strand == "+" else dna.translate(COMP)[::-1]
        out[el] = dict(chrom=chrom, tss=tss1, strand=strand, gene=gene, tx=tx, start0=s0, seq=seq)
        print(f"  {el}: {gene} {tx} {chrom}:{tss1} ({strand})")
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def variant_table():
    """One row per MPRA variant that falls inside a promoter window, with ref/alt windows."""
    df = load_dataset("gonzalobenegas/sat_mut_mpra")["test"].to_pandas()
    win = fetch_windows(df)
    rows = []
    for _, v in df.iterrows():
        w = win.get(v.element)
        if w is None:
            continue
        g0 = int(v.pos) - 1                                           # 0-based genomic
        if w["strand"] == "+":
            i, ref, alt = g0 - w["start0"], v.ref, v.alt
        else:
            i, ref, alt = (w["start0"] + 300 - 1) - g0, v.ref.translate(COMP), v.alt.translate(COMP)
        if not 0 <= i < 300:
            continue
        assert w["seq"][i] == ref, (v.element, v.pos, w["seq"][i], ref)
        rows.append(dict(element=v.element, gene=w["gene"], pos=int(v.pos), idx=i, rel_tss=i - 49,
                         ref=ref, alt=alt, effect=float(v.label)))
    return pd.DataFrame(rows), win


if __name__ == "__main__":
    t, win = variant_table()
    print(t.groupby("element").agg(n=("effect", "size"), gene=("gene", "first")).to_string())
    print("total variants in windows:", len(t))
