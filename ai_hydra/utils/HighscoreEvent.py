# ai_hydra/utils/HighscoreEvent.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class HighscoreEvent:
    epoch: int
    highscore: int
    epsilon: float
    elapsed_time: str
