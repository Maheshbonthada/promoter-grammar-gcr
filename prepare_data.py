"""Download promoter tasks (Nucleotide Transformer benchmark, revised) and JASPAR motifs.

Sequences are 300 bp windows; for positives the TSS sits at index 249
(window = -249..+50 relative to the TSS, EPDnew convention).
"""
import json, os, urllib.request
import numpy as np
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

TASKS = ["promoter_tata", "promoter_no_tata", "promoter_all"]
# JASPAR 2024/2026 CORE profiles used in this study
MOTIFS = {"TBP": "MA0108.3", "SP1": "MA0079.5", "NFYA": "MA0060.3"}
LUT = np.full(256, 4, dtype=np.int8)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i
    LUT[ord(c.lower())] = i


def encode(seqs):
    return np.stack([LUT[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])


def main():
    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    for split in ["train", "test"]:
        d = ds[split].filter(lambda x: x["task"] in TASKS)
        for t in TASKS:
            s = d.filter(lambda x: x["task"] == t)
            np.savez_compressed(os.path.join(DATA, f"{t}_{split}.npz"), X=encode(s["sequence"]),
                                y=np.array(s["label"], dtype=np.int8), name=np.array(s["name"]))
            print(split, t, len(s))
    mats = {}
    for name, mid in MOTIFS.items():
        m = json.load(urllib.request.urlopen(f"https://jaspar.elixir.no/api/v1/matrix/{mid}/?format=json", timeout=30))
        pfm = np.array([m["pfm"][b] for b in "ACGT"], dtype=float)       # 4 x W counts
        mats[name] = dict(id=mid, pfm=pfm.tolist())
        print(name, mid, "width", pfm.shape[1])
    json.dump(mats, open(os.path.join(DATA, "jaspar_motifs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
