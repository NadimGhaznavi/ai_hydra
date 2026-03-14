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
from ai_hydra.constants.DHydra import DHydra


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(DHydra.RANDOM_SEED)

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
            nonlinearity="tanh",
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
        self, hidden_size: int, dropout_p: float, rnn_layers: int
    ) -> None:
        self._hidden_size = hidden_size
        self._rnn_dropout = dropout_p
        self._rnn_layers = rnn_layers
        self._init_model()
