# ai_hydra/constants/DEpsilonNice.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final

from ai_hydra.constants.DHydraTui import DField


class DEpsilonNice:
    WINDOW: Final[str] = "window"
    EPOCH: Final[str] = "epoch"
    CALLS: Final[str] = "calls"
    TRIGGERED: Final[str] = "triggered"
    FATAL_SUGGESTED: Final[str] = "fatal_suggested"
    OVERRIDES: Final[str] = "overrides"
    NO_SAFE_ALTERNATIVE: Final[str] = "no_safe_alternative"
    TRIGGER_RATE: Final[str] = "trigger_rate"
    OVERRIDE_RATE: Final[str] = "override_rate"
