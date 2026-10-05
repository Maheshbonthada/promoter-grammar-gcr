"""Download every task in the Nucleotide Transformer benchmark, not only the promoter ones.

The promoter analysis showed that the TBP motif anti-predicts the promoter label, because the negatives are
AT-rich relative to GC-rich promoters. Whether that is a quirk of one task or a property of how the whole
benchmark family is built can only be answered by looking at all of them.

    python prepare_all_tasks.py         # -> data/<task>_<split>.npz for every task
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
    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    tasks = sorted(set(ds["train"]["task"]))
    print(f"{len(tasks)} tasks: {tasks}", flush=True)
    for split in ["train", "test"]:
        d = ds[split]
        arr_task = np.array(d["task"])
        seqs, labels = np.array(d["sequence"], dtype=object), np.array(d["label"], dtype=np.int8)
        names = np.array(d["name"]) if "name" in d.column_names else np.array([""] * len(d))
        for t in tasks:
            out = os.path.join(DATA, f"{t}_{split}.npz")
            if os.path.exists(out):
                print(f"  skip {t} {split} (exists)", flush=True)
                continue
            m = arr_task == t
            lens = {len(s) for s in seqs[m]}
            if len(lens) != 1:                      # ragged tasks are padded out; record and skip encoding
                print(f"  {t} {split}: ragged lengths {sorted(lens)[:5]}... n={m.sum()}", flush=True)
                continue
            np.savez_compressed(out, X=encode(list(seqs[m])), y=labels[m], name=names[m])
            print(f"  {t:28s} {split:5s} n={m.sum():7d} len={lens.pop()}", flush=True)


if __name__ == "__main__":
    main()
