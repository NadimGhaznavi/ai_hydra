# ai_hydra/net/LinearPolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import torch
import torch.nn as nn

from ai_hydra.constants.DNNet import DNetDef, DLinear
from ai_hydra.constants.DHydra import DHydra


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(DHydra.RANDOM_SEED)

        self.net = nn.Sequential(
            nn.Linear(DNetDef.INPUT_SIZE, DLinear.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(DLinear.HIDDEN_SIZE, DLinear.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=DLinear.DROPOUT_P),
            nn.Linear(DLinear.HIDDEN_SIZE, DLinear.OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.net(x)
