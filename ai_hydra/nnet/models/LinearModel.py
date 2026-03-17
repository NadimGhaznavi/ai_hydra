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
from ai_hydra.constants.DHydra import DHydraLog

from ai_hydra.utils.HydraLog import HydraLog


class LinearModel(nn.Module):
    def __init__(self, log_level: DHydraLog):
        super().__init__()
        torch.manual_seed(DHydra.RANDOM_SEED)

        self.log = HydraLog(
            client_id="LinearModel",
            log_level=log_level,
            to_console=True,
        )

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
        self.log.info(f"Setting hidden size to {hidden_size}")
        self.log.info(f"Setting dropout layer p-value to {dropout_p}")
        self._init_model()
