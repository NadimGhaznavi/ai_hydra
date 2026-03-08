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
        x = self.m_in(x)
        inputs = x.view(1, -1, DRNN.HIDDEN_SIZE)
        x, h_n = self.m_rnn(inputs)
        x = self.m_out(x)
        return x[len(x) - 1]

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
