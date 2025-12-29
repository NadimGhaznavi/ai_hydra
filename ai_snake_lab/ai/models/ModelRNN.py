"""
ai_snake_lab/ai/models/ModelRNN.py

    AI Snake Game Simulator
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_snake_lab.constants.DSim import DSim
from ai_snake_lab.constants.DModelLRNN import DModelRNN


class ModelRNN(nn.Module):
    def __init__(self, seed: int):
        super(ModelRNN, self).__init__()
        torch.manual_seed(seed)
        input_size = DSim.STATE_SIZE
        hidden_size = DModelRNN.HIDDEN_SIZE
        output_size = DSim.OUTPUT_SIZE
        rnn_layers = DModelRNN.RNN_LAYERS
        rnn_dropout = DModelRNN.P_VALUE
        self.m_in = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.m_rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            nonlinearity=DModelRNN.NON_LINEARITY,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
        )
        self.m_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.m_in(x)
        inputs = x.view(1, -1, DModelRNN.HIDDEN_SIZE)
        x, h_n = self.m_rnn(inputs)
        x = self.m_out(x)
        return x[len(x) - 1]

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
