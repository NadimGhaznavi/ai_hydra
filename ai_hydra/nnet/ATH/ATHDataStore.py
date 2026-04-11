# ai_hydra/nnet/ATHStore.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from __future__ import annotations

from ai_hydra.constants.DHydra import DHydraLog, DModule

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

from ai_hydra.nnet.ATH.ATHCommon import (
    Episode,
    GearMeta,
    assert_valid_gear,
)


class ATHDataStore:
    """
    Canonical storage for ATH replay memory.

    Responsibilities:
    - accumulate transitions for the in-progress episode
    - store finalized episodes
    - track total stored frame count
    - maintain derived per-gear bucket indexes

    Notes:
    - _games is the canonical source of truth
    - bucket indexes are derived and must be rebuilt from _games
    """

    def __init__(
        self,
        log_level: DHydraLog,
        max_buckets: int,
        max_gear: int,
        max_training_frames: int,
    ):
        if not isinstance(max_buckets, int):
            raise TypeError(
                f"max_buckets must be int, got {type(max_buckets).__name__}"
            )
        if not isinstance(max_gear, int):
            raise TypeError(
                f"max_gear must be int, got {type(max_gear).__name__}"
            )
        if not isinstance(max_training_frames, int):
            raise TypeError(
                "max_training_frames must be int, got "
                f"{type(max_training_frames).__name__}"
            )

        if max_buckets < 1:
            raise ValueError(f"max_buckets must be >= 1, got {max_buckets}")
        if max_gear < 1:
            raise ValueError(f"max_gear must be >= 1, got {max_gear}")
        if max_training_frames < 4:
            raise ValueError(
                f"max_training_frames must be >= 4, got {max_training_frames}"
            )

        self._max_buckets = max_buckets
        self._max_gear = max_gear
        self._max_training_frames = max_training_frames

        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_DATA_STORE,
            log_level=log_level,
            to_console=True,
        )

        # Canonical storage
        self._games: list[Episode] = []
        self._cur_game: list[Transition] = []
        self._stored_frames = 0

        # Current gear context for derived indexes / sampling views
        self._cur_gear: int | None = None

        # Derived index state
        self._episodes_by_bucket: dict[int, list[Episode]] = {
            idx: [] for idx in range(self._max_buckets)
        }
        self._warmed_buckets: list[int] = []

    def append(self, t: Transition) -> None:
        """
        Append a transition to the current in-progress episode.

        This method does NOT finalize the episode. Finalization must be
        triggered separately (typically when a terminal transition is seen).

        Parameters:
            t (Transition):
                The transition to append to the current episode.

        Invariants:
            - _cur_game represents a single episode under construction.
            - No assumptions are made about episode completion here.
        """
        if not isinstance(t, Transition):
            raise TypeError(f"t must be Transition, got {type(t).__name__}")

        self._cur_game.append(t)

    def discard_cur_game(self):
        self._cur_game = []

    def finalize_game(self, gear_meta: dict[int, GearMeta]) -> None:
        """
        Finalize the current in-progress episode and add it to canonical storage.

        This method:
        - converts the current mutable episode buffer into an immutable Episode
        - validates metadata consistency
        - updates total stored frame count
        - clears the in-progress episode buffer

        Parameters:
            gear_meta (dict[int, GearMeta]):
                Per-gear metadata for this episode. Must contain valid entries
                for all expected gears.

        Invariants:
            - _cur_game must contain at least one transition
            - gear_meta must be consistent with episode size
            - Episode will enforce its own internal invariants
        """
        if not self._cur_game:
            raise ValueError(
                "Cannot finalize empty episode (_cur_game is empty)"
            )

        if not isinstance(gear_meta, dict):
            raise TypeError(
                f"gear_meta must be dict, got {type(gear_meta).__name__}"
            )

        frames = tuple(self._cur_game)
        ep_size = len(frames)

        # Validate gear_meta before committing
        if not gear_meta:
            raise ValueError("gear_meta cannot be empty")

        for gear, meta in gear_meta.items():
            if not isinstance(gear, int):
                raise TypeError(
                    f"gear_meta key must be int, got {type(gear).__name__}"
                )
            if not isinstance(meta, GearMeta):
                raise TypeError(
                    f"gear_meta[{gear}] must be GearMeta, got {type(meta).__name__}"
                )

            # Ensure chunking is consistent with episode size
            expected_chunks = ep_size // meta.seq_length
            if meta.num_chunks != expected_chunks:
                raise ValueError(
                    f"Gear {gear}: num_chunks mismatch "
                    f"(expected {expected_chunks}, got {meta.num_chunks})"
                )

        # Commit to canonical storage
        episode = Episode(
            frames=frames,
            size=ep_size,
            gear_meta=gear_meta,
        )

        self._games.append(episode)
        self._stored_frames += ep_size

        # Clear current episode buffer
        self._cur_game = []

    def get_cur_ep_size(self) -> int:
        """
        Return the size (number of transitions) of the current in-progress episode.

        Returns:
            int: Number of transitions currently accumulated in _cur_game.

        Invariants:
            - _cur_game represents a single episode under construction
            - result is always >= 0
        """
        return len(self._cur_game)

    def get_bucket_counts(self) -> dict[int, int]:
        """
        Return the number of episodes assigned to each bucket.

        Returns:
            dict[int, int]:
                Mapping of bucket index -> number of episodes in that bucket.

        Invariants:
            - Keys are exactly 0 .. max_buckets - 1
            - Values are >= 0
            - Reflects current derived bucket index state
        """

        expected_keys = set(range(self._max_buckets))
        actual_keys = set(self._episodes_by_bucket.keys())

        if actual_keys != expected_keys:
            raise RuntimeError(
                "Bucket index is corrupted: "
                f"expected keys {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        return {
            idx: len(self._episodes_by_bucket[idx])
            for idx in range(self._max_buckets)
        }

    def get_eps_by_bucket_idx(self, bucket_idx: int) -> list[Episode]:
        """
        Return the list of episodes assigned to a given bucket index.

        Parameters:
            bucket_idx (int):
                Bucket index in range [0, max_buckets - 1].

        Returns:
            list[Episode]:
                Episodes assigned to the specified bucket.

        Invariants:
            - bucket_idx must be within valid range
            - returned list is part of derived bucket index (not canonical storage)
        """

        if not isinstance(bucket_idx, int):
            raise TypeError(
                f"bucket_idx must be int, got {type(bucket_idx).__name__}"
            )

        if bucket_idx < 0 or bucket_idx >= self._max_buckets:
            raise IndexError(
                f"bucket_idx must be in [0, {self._max_buckets - 1}], "
                f"got {bucket_idx}"
            )

        return self._episodes_by_bucket[bucket_idx]

    def get_stored_frame_count(self) -> int:
        """
        Return the total number of frames stored across all finalized episodes.

        Returns:
            int: Total frame count (always >= 0).

        Invariants:
            - equals sum(ep.size for ep in _games)
            - never negative
        """

        if not isinstance(self._stored_frames, int):
            raise TypeError(
                f"_stored_frames must be int, got {type(self._stored_frames).__name__}"
            )

        if self._stored_frames < 0:
            raise RuntimeError(
                "Stored frame count is negative — memory corruption detected"
            )

        return self._stored_frames

    def get_warmed_buckets(self) -> list[int]:
        """
        Return the list of bucket indices that currently contain at least one episode.

        Returns:
            list[int]:
                Sorted list of bucket indices in range [0, max_buckets - 1].

        Invariants:
            - values are unique
            - values are within valid bucket range
            - list reflects current derived bucket index state
        """

        if not isinstance(self._warmed_buckets, list):
            raise TypeError(
                f"_warmed_buckets must be list, got {type(self._warmed_buckets).__name__}"
            )

        for idx in self._warmed_buckets:
            if not isinstance(idx, int):
                raise TypeError(
                    f"warmed bucket index must be int, got {type(idx).__name__}"
                )
            if idx < 0 or idx >= self._max_buckets:
                raise RuntimeError(
                    f"Invalid warmed bucket index: {idx} (max={self._max_buckets})"
                )

        # Return a copy to prevent external mutation
        return list(self._warmed_buckets)

    def pop_first_game(self) -> None:
        """
        Remove the oldest episode from canonical storage.

        This method:
        - removes the first (oldest) episode from _games
        - updates the stored frame count accordingly

        Invariants:
            - _games must not be empty
            - _stored_frames must remain consistent with total stored frames
        """

        if not self._games:
            raise ValueError(
                "Cannot pop from empty replay memory (_games is empty)"
            )

        episode = self._games.pop(0)

        if not isinstance(episode, Episode):
            raise TypeError(
                f"Corrupted state: expected Episode, got {type(episode).__name__}"
            )

        if episode.size < 0:
            raise ValueError(
                f"Corrupted episode: size must be >= 0, got {episode.size}"
            )

        self._stored_frames -= episode.size

        if self._stored_frames < 0:
            raise RuntimeError(
                "Stored frame count went negative — memory corruption detected"
            )

    def update_warmed_buckets_list(self) -> None:
        """
        Recompute the list of bucket indices that contain at least one episode.

        Invariants:
            - _episodes_by_bucket must have keys 0 .. max_buckets - 1
            - result is sorted, unique, and within valid range
        """

        expected_keys = set(range(self._max_buckets))
        actual_keys = set(self._episodes_by_bucket.keys())

        if actual_keys != expected_keys:
            raise RuntimeError(
                "Bucket index is corrupted: "
                f"expected keys {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        warmed: list[int] = []

        # Deterministic ordering
        for idx in range(self._max_buckets):
            episodes = self._episodes_by_bucket[idx]
            if episodes:
                warmed.append(idx)

        # Optional sanity checks (cheap, keep them)
        for idx in warmed:
            if idx < 0 or idx >= self._max_buckets:
                raise RuntimeError(
                    f"Invalid warmed bucket index: {idx} (max={self._max_buckets})"
                )

        # Assign once (avoid partial updates on error)
        self._warmed_buckets = warmed

    def update_episodes_by_bucket(self) -> None:
        """
        Rebuild the per-bucket episode index for the current gear.

        For the active gear, an episode is added to bucket `i` if that episode
        has at least `i + 1` full chunks under that gear.

        Invariants:
            - _cur_gear must be set before calling this method
            - _episodes_by_bucket must already be in a reset/empty state
            - every episode in _games must contain GearMeta for _cur_gear
        """
        if self._cur_gear is None:
            raise ValueError("ATHDataStore._cur_gear is not set")

        expected_keys = set(range(self._max_buckets))
        actual_keys = set(self._episodes_by_bucket.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                "Bucket index is corrupted: "
                f"expected keys {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        # This method is intended to populate a freshly reset bucket index.
        for bucket_idx, episodes in self._episodes_by_bucket.items():
            if episodes:
                raise RuntimeError(
                    "update_episodes_by_bucket() called without resetting bucket "
                    f"index first: bucket {bucket_idx} already contains data"
                )

        for ep in self._games:
            if not isinstance(ep, Episode):
                raise TypeError(
                    f"Corrupted state: expected Episode, got {type(ep).__name__}"
                )

            if self._cur_gear not in ep.gear_meta:
                raise KeyError(f"Missing gear meta for gear {self._cur_gear}")

            meta = ep.gear_meta[self._cur_gear]

            if not isinstance(meta, GearMeta):
                raise TypeError(
                    f"gear_meta[{self._cur_gear}] must be GearMeta, got "
                    f"{type(meta).__name__}"
                )

            num_populated_buckets = min(meta.num_chunks, self._max_buckets)

            if num_populated_buckets < 0:
                raise RuntimeError(
                    f"Invalid num_populated_buckets={num_populated_buckets} for gear "
                    f"{self._cur_gear}"
                )

            for bucket_idx in range(num_populated_buckets):
                self._episodes_by_bucket[bucket_idx].append(ep)

    def reset_bucket_indexes(self) -> None:
        """
        Reset the derived bucket index to an empty state.

        This method:
        - clears all bucket-to-episode mappings
        - does NOT modify canonical storage (_games)
        - does NOT update warmed buckets (caller must do that)

        Invariants:
            - resulting structure has keys 0 .. max_buckets - 1
            - all bucket lists are empty
        """

        self._episodes_by_bucket = {
            idx: [] for idx in range(self._max_buckets)
        }

        # Optional sanity check (cheap and useful)
        if len(self._episodes_by_bucket) != self._max_buckets:
            raise RuntimeError(
                "Bucket index reset failed: incorrect number of buckets"
            )

        for idx, episodes in self._episodes_by_bucket.items():
            if episodes:
                raise RuntimeError(
                    f"Bucket {idx} not empty after reset — corruption detected"
                )

    def set_gear(self, gear: int) -> None:
        """
        Set the current active gear for derived bucket indexing.

        Parameters:
            gear (int):
                Active ATH gear. Must be in the valid 1-based range.

        Invariants:
            - gear must be valid for this store's configured max_gear
            - this method does NOT rebuild bucket indexes
            - this method does NOT mutate canonical storage
        """
        assert_valid_gear(gear=gear, max_gear=self._max_gear)
        self._cur_gear = gear
