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

from ai_hydra.constants.DNNet import DNetDef
from ai_hydra.constants.DHydra import DModule
from ai_hydra.constants.DHydra import DHydraLog

from ai_hydra.utils.HydraLog import HydraLog


class LinearModel(nn.Module):
    def __init__(self, log_level: DHydraLog, seed: int):
        super().__init__()

        self._seed = seed
        torch.manual_seed(seed)

        self.log = HydraLog(
            client_id=DModule.LINEAR_MODEL,
            log_level=log_level,
            to_console=True,
        )
        self.hidden_block = nn.Sequential()

        self._hidden_size = None
        self._dropout_p = None

    def forward(self, x):
        return self.net(x)

    def forward_sequence(self, x):
        return self.net(x)

    def _init_model(self):
        self.net = nn.Sequential(
            nn.Linear(DNetDef.INPUT_SIZE, self._hidden_size),
            nn.ReLU(),
            self.hidden_block,
            nn.Dropout(p=self._dropout_p),
            nn.Linear(self._hidden_size, DNetDef.OUTPUT_SIZE),
        )

    def set_params(
        self, hidden_size: int, dropout_p: float, layers: int
    ) -> None:
        self._hidden_size = hidden_size
        self._dropout_p = dropout_p
        self.log.info(f"Set hidden size: {hidden_size}")
        self.log.info(f"Set dropout layer p-value: {dropout_p}")
        self.log.info(f"Set layers: {layers}")

        # Add hidden layers
        while layers:
            self.hidden_block.append(
                nn.Linear(self._hidden_size, self._hidden_size),
            )
            self.hidden_block.append(nn.ReLU())
            layers -= 1

        self._init_model()
