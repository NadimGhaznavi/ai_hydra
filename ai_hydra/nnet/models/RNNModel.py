# ai_hydra/net/RNNModel.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import torch
import torch.nn as nn

from ai_hydra.constants.DNNet import DNetDef, DRNN
from ai_hydra.constants.DHydra import DModule, DHydraLog
from ai_hydra.constants.DHydraTui import DField

from ai_hydra.utils.HydraLog import HydraLog


class RNNModel(nn.Module):
    def __init__(self, log_level: DHydraLog, seed: int):
        super().__init__()
        torch.manual_seed(seed)

        self.log = HydraLog(
            client_id=DModule.RNN_MODEL,
            log_level=log_level,
            to_console=True,
        )

        self._hidden_size = None
        self._rnn_dropout = None
        self._rnn_layers = None

    def _init_model(self):
        input_size = DNetDef.INPUT_SIZE
        output_size = DRNN.OUTPUT_SIZE

        self.m_in = nn.Sequential(
            nn.Linear(input_size, self._hidden_size),
            nn.ReLU(),
        )
        self.m_rnn = nn.RNN(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            nonlinearity=DField.TANH,
            num_layers=self._rnn_layers,
            dropout=self._rnn_dropout,
            batch_first=True,
        )
        self.m_out = nn.Linear(self._hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, F]
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # [1, T, F]

        x = self.m_in(x)  # [B, T, H]
        x, _ = self.m_rnn(x)  # [B, T, H]
        x = self.m_out(x)  # [B, T, A]

        return x[:, -1, :]  # [B, A]

    def forward_sequence(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, F]
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # [1, T, F]

        x = self.m_in(x)  # [B, T, H]
        x, _ = self.m_rnn(x)  # [B, T, H]
        x = self.m_out(x)  # [B, T, A]
        return x

    def reset_parameters(self):
        def _reset(m):
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(_reset)

    def set_params(
        self, hidden_size: int, dropout_p: float, layers: int
    ) -> None:
        self._hidden_size = hidden_size
        self._rnn_dropout = dropout_p
        self._rnn_layers = layers
        self.log.info(f"Setting hidden size to {hidden_size}")
        self.log.info(f"Setting dropout layer p-value to {dropout_p}")
        self.log.info(f"Setting the number of RNN layers to {layers}")
        self._init_model()
