"""
constants/DModelL.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup


class DModelL(ConstGroup):
    """Linear Model constants"""

    HIDDEN_SIZE: int = 170  # Default: The number of nodes in the hidden layer
    LEARNING_RATE: float = 0.00003  # Default: Learning rate
    MODEL: str = "linear"
    P_VALUE: float = 0.1  # Default: Dropout layer value - 20%
