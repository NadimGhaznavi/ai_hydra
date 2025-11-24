"""
constants/DLabLogger.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import logging

from ai_snake_lab.utils.ConstGroup import ConstGroup


class DLog(ConstGroup):
    """Logging Constants"""

    INFO: str = "info"
    DEBUG: str = "debug"
    WARNING: str = "warning"
    ERROR: str = "error"
    CRITICAL: str = "critical"


LOG_LEVELS: dict = {
    DLog.INFO: logging.INFO,
    DLog.DEBUG: logging.DEBUG,
    DLog.WARNING: logging.WARNING,
    DLog.ERROR: logging.ERROR,
    DLog.CRITICAL: logging.CRITICAL,
}
