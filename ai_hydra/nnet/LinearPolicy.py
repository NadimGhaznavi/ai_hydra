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
from typing import Sequence

from ai_hydra.constants.DNet import DNetDef, DLinear
from ai_hydra.nnet.HydraPolicy import HydraPolicy


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(DNetDef.INPUT_SIZE, DLinear.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(DLinear.HIDDEN_SIZE, DLinear.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(DLinear.HIDDEN_SIZE, DLinear.OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.net(x)


class LinearPolicy(HydraPolicy):
    def __init__(self, model: LinearModel, device: torch.device | None = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.eval()

    def select_action(self, state: Sequence[float]) -> int:
        with torch.no_grad():
            x = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.model(x)  # [1, 3]
            return int(q.argmax(dim=1).item())
