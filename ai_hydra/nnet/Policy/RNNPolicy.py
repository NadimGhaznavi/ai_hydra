# ai_hydra/nnet/Policy/RNNPolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import torch
from typing import Sequence

from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.models.RNNModel import RNNModel


class RNNPolicy(HydraPolicy):
    def __init__(self, model: RNNModel, device: torch.device | None = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.eval()

    def select_action(self, state: Sequence[float], board=None) -> int:
        with torch.no_grad():
            x = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )

            q = self.model(x)  # [3]
            return int(q.argmax().item())
