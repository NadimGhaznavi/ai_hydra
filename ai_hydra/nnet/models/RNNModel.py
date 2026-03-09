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
import torch.nn.functional as F

from ai_hydra.constants.DNNet import DNetDef, DRNN
from ai_hydra.constants.DHydra import DHydra


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
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
            nonlinearity=DRNN.NON_LINEARITY,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
        )
        self.m_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Accept a single state or a full sequence of states.
        # Expected shapes:
        #   [input_size]              -> one state
        #   [seq_len, input_size]     -> one full game / one sequence

        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, input_size]

        x = self.m_in(x)  # [seq_len, hidden_size]
        x = x.unsqueeze(1)  # [seq_len, 1, hidden_size]
        x, h_n = self.m_rnn(x)  # [seq_len, 1, hidden_size]
        x = self.m_out(x)  # [seq_len, 1, output_size]

        return x[-1, 0, :]

    def forward_sequence(self, x):
        """
        Return Q-values for every timestep in the sequence.

        Expected input shapes:
            [input_size]           -> one state
            [seq_len, input_size]  -> one sequence / one game

        Returns:
            [seq_len, output_size]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, input_size]

        x = self.m_in(x)  # [seq_len, hidden_size]
        x = x.unsqueeze(1)  # [seq_len, 1, hidden_size]
        x, _ = self.m_rnn(x)  # [seq_len, 1, hidden_size]
        x = self.m_out(x)  # [seq_len, 1, output_size]
        return x[:, 0, :]  # [seq_len, output_size]

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
