"""
constants/DDef.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup


class DDef(ConstGroup):
    """Defaults"""

    DOT_DB: str = ".db"  # .db files
    MOVE_DELAY: float = 0.01  # Delay between game moves (in the training loop)
