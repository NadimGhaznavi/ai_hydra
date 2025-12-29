"""
ai_snake_lab/ai/models/ModelT.py

    AI Snake Game Simulator
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0

    Transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_snake_lab.constants.DSim import DSim


class ModelT(nn.Module):

    def __init__self(seed: int):
        super(ModelT, self).__init__()
        torch.manual_seed(seed)

        input_size = DSim.STATE_SIZE
