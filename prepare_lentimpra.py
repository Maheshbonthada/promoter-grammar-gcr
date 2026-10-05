"""Prepare the lentiMPRA expression data: the external endpoint the promoter study could not afford.

The saturation-mutagenesis MPRA gives 9 promoters, which is the binding constraint on every external claim
in the promoter study. lentiMPRA (Agarwal et al., Nature 2025) measured the regulatory activity of ~453,000
200-bp sequences, of which ~65,000 are gene promoters, with a continuous activity readout rather than a
classification proxy.

The promoter subset carries a positional anchor: scanning the TBP profile over these windows puts the modal
best hit at index 67-70, about 3.4x above uniform, consistent with a TATA box roughly 30 bp upstream of a
TSS near index 98. That anchor is what makes the positional rule transferable here.

    python prepare_lentimpra.py          # -> data/lentimpra_{train,validation,test}.npz
"""
import os
import numpy as np
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LUT = np.full(256, 4, dtype=np.int8)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i
    LUT[ord(c.lower())] = i


def encode(seqs):
    return np.stack([LUT[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])


def main():
    ds = load_dataset("zhangtaolab/human_lentiMPRA")
    for split in ds:
        out = os.path.join(DATA, f"lentimpra_{split}.npz")
        if os.path.exists(out):
            print(f"skip {split} (exists)", flush=True)
            continue
        d = ds[split]
        ids = np.array(d["id"])
        seqs = d["sequence"]
        lens = {len(s) for s in seqs}
        assert len(lens) == 1, f"ragged sequences: {sorted(lens)[:5]}"
        X = encode(seqs)
        y = np.array([float(v) for v in d["label"]], dtype=np.float32)
        promoter = np.array([i.startswith("ENSG") for i in ids])
        reversed_ = np.array(["_Reversed" in i for i in ids])
        np.savez_compressed(out, X=X, y=y, promoter=promoter, reversed=reversed_, name=ids)
        print(f"{split:11s} n={len(y):7d}  L={lens.pop()}  promoters={int(promoter.sum()):6d}  "
              f"reversed={int(reversed_.sum()):6d}  activity {y.mean():+.3f} +/- {y.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
