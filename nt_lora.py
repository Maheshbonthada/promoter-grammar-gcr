"""Nucleotide Transformer v2 250M with LoRA adapters: the model-scale check the study was missing.

Reviewers reasonably ask whether weak positional knowledge is an artefact of small models. The study fully
fine-tunes up to 100M parameters, which is the limit of an 8 GB card. LoRA removes that limit for the
forward-heavy regime here: 300-bp windows become 50 six-mer tokens, so memory is dominated by weights
rather than activations, and freezing the backbone means no optimizer state for 250M parameters.

Only the attention query and value projections get adapters, which is the usual default, plus the
classification head. Gradient checkpointing keeps the GCR auxiliary batches inside memory.

    python nt_lora.py                    # trains NT-v2-250M-LoRA through the normal study pipeline
"""
import os
import sys

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from nt_model import NTClassifier

R = int(os.environ.get("LORA_R", 16))
ALPHA = int(os.environ.get("LORA_ALPHA", 32))


class NT250LoRA(NTClassifier):
    # The 250M repository ships only a pytorch_model.bin, which transformers refuses to load under
    # torch 2.3 (CVE-2025-32434); the 500M repository ships safetensors, so the scale check uses that -
    # five times the study's largest fully fine-tuned model rather than two and a half.
    name = "NT-v2-500M-LoRA"
    hf_id = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

    def __init__(self):
        super().__init__()
        from torch.utils.checkpoint import checkpoint
        for layer in self.encoder.encoder.layer:
            fwd = layer.forward
            layer.forward = (lambda f: lambda h, attention_mask=None, *a, **k:
                             checkpoint(f, h, attention_mask, *a, use_reentrant=False, **k)
                             if torch.is_grad_enabled() else f(h, attention_mask, *a, **k))(fwd)
        targets = sorted({n.split(".")[-1] for n, m in self.encoder.named_modules()
                          if isinstance(m, nn.Linear) and n.split(".")[-1] in ("query", "value")})
        if not targets:                                    # fall back to whatever linear names exist
            targets = sorted({n.split(".")[-1] for n, m in self.encoder.named_modules()
                              if isinstance(m, nn.Linear)})[:2]
        self.encoder = get_peft_model(
            self.encoder, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.05, bias="none",
                                     target_modules=targets))
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"{self.name}: LoRA on {targets}, r={R} | trainable {trainable/1e6:.2f}M "
              f"of {total/1e6:.0f}M ({100*trainable/total:.2f}%)", flush=True)


def register():
    """Make the LoRA architecture visible to the study runners, with its own hyperparameters."""
    import models, run_study
    models.ARCHS[NT250LoRA.name] = NT250LoRA
    run_study.LR[NT250LoRA.name] = float(os.environ.get("LR", 3e-4))     # adapters tolerate a higher rate
    run_study.EPOCHS[NT250LoRA.name] = int(os.environ.get("EPOCHS", 3))
    run_study.BS[NT250LoRA.name] = int(os.environ.get("BS", 32))
    return NT250LoRA.name


def main():
    name = register()
    if os.environ.get("MODE", "study") == "null":
        # Architecture-matched null: retrain with every canonical-TATA promoter removed. Without it the
        # model's grammar-discrimination score has no reference, which is the study's whole point.
        import review2_study
        review2_study.main("notata", [name])
    else:
        import run_study
        sys.argv = ["run_study.py", name, f"--methods={os.environ.get('METHODS', 'baseline,gcr')}"]
        run_study.main()


if __name__ == "__main__":
    main()
