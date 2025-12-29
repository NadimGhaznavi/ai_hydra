"""
constants/DFile.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup


class DFile(ConstGroup):
    """Files"""

    CSS_FILE: str = "AISim.tcss"  # AI Snake Lab Textual CSS file
    RUNTIME_DB: str = (
        "runtime.db"  # The SQLite DB file that stores games and game frames
    )
    CLIENT_LOG: str = "SimClient.log"
