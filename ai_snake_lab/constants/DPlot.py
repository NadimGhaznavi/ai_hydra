"""
constants/DPlot.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup


class Plot(ConstGroup):
    """Ploting constants"""

    # Method used to thin data that's being plotted, otherwise the plot gets "blurry"
    AVERAGE: str = "average"
    SLIDING: str = "sliding"

    # The average is calculated by dividing the MAX_DATA_POINTS by this number
    AVG_DIVISOR: int = 40
    MAX_GAMESCORE_DATA_POINTS: int = (
        200  # Maximum number of data points in the gamescore plot
    )
    MAX_LOSS_DATA_POINTS: int = 75  # Max number of data points in the loss plot
