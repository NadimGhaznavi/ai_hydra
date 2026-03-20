# ai_hydra/nnet/ATHStore.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from __future__ import annotations

from random import Random


from ai_hydra.constants.DReplayMemory import DMemDef, DMemField, ATH_GEARBOX
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DEvent import EV_TYPE

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ, EventMsg

from ai_hydra.nnet.ATH.ATHCommon import Episode, GearMeta

MAX_BUCKETS = DMemDef.MAX_BUCKETS


class ATHDataStore:
    def __init__(self, log_level: DHydraLog, pub_func):

        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_DATA_STORE,
            log_level=log_level,
            to_console=True,
        )
        # Remote logging: ZeroMQ "Events" PUB/SUB topic
        self.event = HydraEventMQ(
            client_id=DModule.ATH_DATA_STORE, pub_func=pub_func
        )

        self._episodes_by_bucket: dict[int, list[Episode]] = {
            idx: [] for idx in range(MAX_BUCKETS)
        }
        self._warmed_buckets: list[int] = []
        self._games: list[Episode] = []
        self._cur_gear: int | None = None
        self._cur_game = []
        self._batch_size: int | None = None

        self._stored_frames = 0

    def append(self, t: Transition):
        self._cur_game.append(t)

    def get_cur_ep_size(self) -> int:
        return len(self._cur_game)

    def finalize_game(self, gear_meta: dict[int, GearMeta]) -> None:
        frames = tuple(self._cur_game)
        self._cur_game = []
        ep_size = len(frames)
        self._stored_frames += ep_size

        self._games.append(
            Episode(
                frames=frames,
                size=ep_size,
                gear_meta=gear_meta,
            )
        )

    def get_bucket_counts(self) -> dict[int, int]:
        return {
            idx: len(episodes)
            for idx, episodes in self._episodes_by_bucket.items()
            if episodes
        }

    def get_eps_by_bucket_idx(self, bucket_idx: int):
        return self._episodes_by_bucket[bucket_idx]

    def get_stored_frame_count(self) -> int | None:
        return self._stored_frames

    def get_warmed_buckets(self):
        return self._warmed_buckets

    def pop_first_game(self):
        episode = self._games.pop(0)
        self._stored_frames -= episode.size

    def update_warmed_buckets_list(self):
        self._warmed_buckets = [
            idx
            for idx, episodes in self._episodes_by_bucket.items()
            if episodes
        ]

    def update_episodes_by_bucket(self):
        for ep in self._games:
            meta = ep.gear_meta[self._cur_gear]
            usable_buckets = min(meta.num_chunks, MAX_BUCKETS)

            for bucket_idx in range(usable_buckets):
                self._episodes_by_bucket[bucket_idx].append(ep)

    def reset_bucket_indexes(self):
        self._episodes_by_bucket = {idx: [] for idx in range(MAX_BUCKETS)}

    def set_gear(self, gear: int):
        self._cur_gear = gear
        _, self._batch_size = ATH_GEARBOX[gear]
