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


class RNNModel2(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(DHydra.RANDOM_SEED)

        input_size = DNetDef.INPUT_SIZE
        hidden_size = DRNN.HIDDEN_SIZE
        output_size = DRNN.OUTPUT_SIZE
        rnn_layers = DRNN.RNN_LAYERS
        rnn_dropout = DRNN.P_VALUE

        self.m_in = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.m_rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.m_out = nn.Linear(hidden_size, output_size)

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
