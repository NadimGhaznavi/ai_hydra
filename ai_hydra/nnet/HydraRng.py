# ai_hydra/nnet/HydraRng.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Optional

import random

from ai_hydra.constants.DHydra import DHydra, DHydraLog, DModule
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ


class HydraRng:
    """
    Centralized Random for deterministic runs.
    """

    def __init__(self, log_level: DHydraLog, pub_func):

        self._master_seed = DHydra.RANDOM_SEED

        self.log = HydraLog(
            client_id=DModule.HYDRA_RNG,
            log_level=log_level,
            to_console=True,
        )
        self.event = HydraEventMQ(
            client_id=DModule.EPSILON_ALGO, pub_func=pub_func
        )

        self._init_rng()

    def clone_rng(self, parent_rng: random.Random) -> random.Random:
        """
        Clone RNG state exactly (for future clone rollouts).
        """
        clone = random.Random()
        clone.setstate(parent_rng.getstate())
        return clone

    def _init_rng(self):
        self.seed_rng = random.Random(self._master_seed)

    def get_rng_state(self, rng: random.Random) -> object:
        """
        Export RNG state (useful later for replay/persistence).
        """
        return rng.getstate()

    def get_seed(self):
        return self._master_seed

    def new_rng(self, seed: Optional[int] = None) -> tuple[int, random.Random]:
        """
        Create a brand-new RNG stream for a session.

        If seed is None, mint a deterministic seed from seed_rng.
        Returns (seed, rng).
        """
        if seed is None:
            seed = self.seed_rng.getrandbits(64)
        else:
            seed = int(seed)

        return seed, random.Random(seed)

    def set_rng_state(self, rng: random.Random, state: object) -> None:
        """
        Restore RNG state (useful later for replay/persistence).
        """
        rng.setstate(state)

    def set_seed(self, seed: int):
        self._master_seed = seed
        self._init_rng()
