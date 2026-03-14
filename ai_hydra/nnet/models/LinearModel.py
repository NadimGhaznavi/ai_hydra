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

        self._hidden_size = None
        self._dropout_p = None

    def forward(self, x):
        return self.net(x)

    def _init_model(self):
        self.net = nn.Sequential(
            nn.Linear(DNetDef.INPUT_SIZE, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self._dropout_p),
            nn.Linear(self._hidden_size, DLinear.OUTPUT_SIZE),
        )

    def set_params(self, hidden_size: int, dropout_p: float) -> None:
        self._hidden_size = hidden_size
        self._dropout_p = dropout_p
        self._init_model()
