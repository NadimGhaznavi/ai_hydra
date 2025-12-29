"""
ai_snake_lab/ai/models/ModelL.py

    AI Snake Lab
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
from ai_snake_lab.constants.DModelL import DModelL


class ModelL(nn.Module):
    def __init__(self, seed: int):
        super(ModelL, self).__init__()
        torch.manual_seed(seed)
        input_size = DSim.STATE_SIZE  # Size of the "state" as tracked by the GameBoard
        hidden_size = DModelL.HIDDEN_SIZE
        output_size = DSim.OUTPUT_SIZE
        p_value = DModelL.P_VALUE
        self.input_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.hidden_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.dropout_block = nn.Dropout(p=p_value)
        self.output_block = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_block(x)
        x = self.hidden_block(x)
        x = self.hidden_block(x)
        x = self.dropout_block(x)
        x = self.output_block(x)
        return x

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
