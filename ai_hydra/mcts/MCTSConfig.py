# ai_hydra/mcst/Node.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from dataclasses import dataclass

import random

from ai_hydra.game.GameHelper import RewardCfg


@dataclass
class MCTSConfig:
    reward_cfg: RewardCfg
    mmm: int
    search_depth: int
    explore_p_value: float
    gate_p_value: float
    iterations: int
    rng: random.Random
    steps: int
    score_threshold: int
    food_ends_episode: bool = False
