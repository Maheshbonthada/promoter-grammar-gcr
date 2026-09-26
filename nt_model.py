"""Pretrained genomic language model (Nucleotide Transformer v2, 50M) as a promoter
classifier with the same interface as models.py: forward(X[B,300] int) -> logits[B]."""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

NT_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"


class NTClassifier(nn.Module):
    name = "NT-v2-50M"
    hf_id = NT_NAME

    def __init__(self):
        super().__init__()
        tok = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        mlm = AutoModelForMaskedLM.from_pretrained(self.hf_id, trust_remote_code=True)
        self.encoder = mlm.esm
        d = mlm.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * d, 1))
        # 6-mer code (base-4) -> vocabulary id; chunks containing N map to <unk>
        vocab = tok.get_vocab()
        table = np.full(4 ** 6, tok.unk_token_id, dtype=np.int64)
        for code in range(4 ** 6):
            kmer = "".join("ACGT"[(code >> (2 * (5 - j))) & 3] for j in range(6))
            table[code] = vocab.get(kmer, tok.unk_token_id)
        self.register_buffer("kmer_table", torch.as_tensor(table), persistent=False)
        self.register_buffer("weights4", torch.tensor([4 ** (5 - j) for j in range(6)]), persistent=False)
        self.cls_id, self.unk_id = tok.cls_token_id, tok.unk_token_id

    def tokenize(self, X):
        B, L = X.shape
        ch = X[:, : L - L % 6].reshape(B, -1, 6)
        ids = self.kmer_table[(ch.clamp(max=3) * self.weights4).sum(-1)]
        ids = torch.where((ch == 4).any(-1), torch.full_like(ids, self.unk_id), ids)
        return torch.cat([torch.full((B, 1), self.cls_id, device=X.device), ids], 1)

    def forward(self, X):
        ids = self.tokenize(X)
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        h = h.float()
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)).squeeze(-1)


class NT100Classifier(NTClassifier):
    """Same wrapper, larger pretrained model (model-size check).
    Activation checkpointing per encoder layer keeps GCR steps inside 8 GB of GPU memory
    (numerically identical gradients; ~30% more compute)."""
    name = "NT-v2-100M"
    hf_id = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

    def __init__(self):
        super().__init__()
        from torch.utils.checkpoint import checkpoint
        for layer in self.encoder.encoder.layer:
            fwd = layer.forward
            layer.forward = (lambda f: lambda h, attention_mask=None, *a, **k:
                             checkpoint(f, h, attention_mask, *a, use_reentrant=False, **k)
                             if torch.is_grad_enabled() else f(h, attention_mask, *a, **k))(fwd)


NT_ARCHS = {c.name: c for c in (NTClassifier, NT100Classifier)}
