"""
constants/DEpsilon.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup


class DEpsilon(ConstGroup):
    """Epsilon Defaults"""

    EPSILON: str = "epsilon"
    EPSILON_N: str = "epsilon_n"
    EPSILON_INITIAL: float = 0.99  # Initial value for Epsilon
    EPSILON_MIN: float = 0.1  # Minimum value for Epsilon
    EPSILON_DECAY: float = 0.95  # How quickly Epsilon decays
