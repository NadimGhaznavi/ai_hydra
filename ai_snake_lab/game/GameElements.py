"""
ai_snake_lab/game/GameElements.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from enum import Enum


class Direction(Enum):
    """
    A simple Enum class that represents a direction in the
    Snake game. It has four values:
    """

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
