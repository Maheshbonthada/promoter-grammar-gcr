"""Three promoter classifiers spanning common inductive biases."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(X):
    """X: [B, L] long (0-3, 4 = N) -> [B, 4, L] float (N = all zeros)."""
    return F.one_hot(X, 5)[..., :4].permute(0, 2, 1).float()


class ConvTrunk(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, c, 15, padding=7), nn.BatchNorm1d(c), nn.ReLU())
        self.res = nn.ModuleList([
            nn.Sequential(nn.Conv1d(c, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c), nn.ReLU())
            for d in (2, 4, 8)])                                  # receptive field 71 bp

    def forward(self, x):
        h = self.stem(one_hot(x))
        for blk in self.res:
            h = h + blk(h)
        return h


class CNNGlobalPool(nn.Module):
    """Translation-invariant readout (DeepBind-style global max pool)."""
    name = "CNN-GlobalPool"

    def __init__(self, c=128):
        super().__init__()
        self.trunk = ConvTrunk(c)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.trunk(x)
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], 1)).squeeze(-1)


class CNNDense(nn.Module):
    """Position-aware readout (flatten -> dense), as in classic promoter CNNs."""
    name = "CNN-Dense"

    def __init__(self, c=128, L=300):
        super().__init__()
        self.trunk = ConvTrunk(c)
        self.pool = nn.MaxPool1d(5)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c * (L // 5), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.pool(self.trunk(x)).flatten(1)).squeeze(-1)


class SmallTransformer(nn.Module):
    """Conv stem + transformer with learned absolute positions (gLM-like inductive bias)."""
    name = "Transformer"

    def __init__(self, d=96, layers=3, heads=4, L=300, stride=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, d, 9, padding=4), nn.GELU(), nn.Conv1d(d, d, stride, stride=stride))
        self.pos = nn.Parameter(torch.randn(L // stride, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        h = self.stem(one_hot(x)).transpose(1, 2) + self.pos
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        return self.head(self.enc(h)[:, 0]).squeeze(-1)


ARCHS = {m.name: m for m in (CNNGlobalPool, CNNDense, SmallTransformer)}
