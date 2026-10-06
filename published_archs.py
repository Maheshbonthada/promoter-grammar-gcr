"""A published architecture, re-implemented and trained on this benchmark, so the audit is not limited to
designs chosen for this study.

DeepSTARR (de Almeida et al., Nature Genetics 2022): four convolution blocks (256 filters of width 7, then
60 of width 3, 60 of width 5 and 120 of width 3; each with batch normalisation, ReLU and max-pooling by 2),
flattened into two dense layers of 256 units with batch normalisation, ReLU and dropout 0.4. The layer
sizes follow the published model; the single-output head and binary cross-entropy are the only changes,
because the task here is classification. The learning rate is not tuned: 1e-3, as for CNN-GlobalPool.

Register before training or loading:  import published_archs; published_archs.register()
"""
import torch.nn as nn

from models import one_hot


class DeepSTARR(nn.Module):
    name = "DeepSTARR"

    def __init__(self, L=300):
        super().__init__()
        layers, c_in = [], 4
        for c, k in ((256, 7), (60, 3), (60, 5), (120, 3)):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        n = 120 * (L // 16)
        self.head = nn.Sequential(nn.Linear(n, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.conv(one_hot(x)).flatten(1)).squeeze(-1)


def register():
    import models
    import run_study
    models.ARCHS[DeepSTARR.name] = DeepSTARR
    run_study.LR[DeepSTARR.name] = 1e-3
    return DeepSTARR.name
