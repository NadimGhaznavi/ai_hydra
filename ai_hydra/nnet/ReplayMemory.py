# ai_hydra/nnet/ReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Deque, TypeVar

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_CHUNKS = DMemory.MAX_CHUNKS
MIN_CHUNKS = DMemory.MIN_CHUNKS

MAX_MEM_SIZE = DMemory.MAX_MEM_SIZE  # Max Linear transitions
MIN_FRAMES = DMemory.MIN_FRAMES  # Min transitions before returning samples

# Bucket all chunks at or above this chunk index into one final bucket.
# Example with 4:
#   0 -> first chunk in a game
#   1 -> second chunk
#   2 -> third chunk
#   3 -> fourth chunk
#   4 -> fifth and beyond
MAX_CHUNK_BUCKET = 19

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RNNChunk:
    """
    A fixed-length training chunk for RNN training.

    Attributes:
        transitions:
            Always exactly seq_length transitions.
        valid_len:
            Number of real timesteps in the chunk.
            For normal chunks this equals seq_length.
            For cold-start padded chunks this is < seq_length, and the
            real transitions are RIGHT-aligned in the chunk.
    """

    transitions: list[Transition]
    valid_len: int


class ReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
        rnn: bool = False,
        seq_length: int | None = None,
    ):
        self.log = HydraLog(
            client_id="ReplayMemory",
            log_level=log_level,
            to_console=True,
        )
        self._rng = rng

        # Linear model settings
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)

        # RNN model settings
        self._rnn = rnn
        self._cur_game: list[Transition] = []
        self._seq_len = seq_length
        self._memory_cold = True
        self._memory_not_full = True
        self._memory_not_balanced = True

        # Global cap across all RNN buckets
        self._max_chunks = MAX_CHUNKS
        self._num_chunks = 0

        # Counter
        self._samples_delivered = 0

        # bucket_idx -> rolling list of chunks
        self._chunks_by_idx: dict[int, list[RNNChunk]] = {
            idx: [] for idx in range(MAX_CHUNK_BUCKET + 1)
        }

        if self._rnn:
            if seq_length is None:
                raise ValueError("Sequence length must be set")
            self.log.info("Initialized for RNN model training")
            self.log.info(f"Setting sequence length to {seq_length}")
            self.log.info(
                f"Setting maximum number of stored sequence to {MAX_CHUNKS}"
            )
            self.log.info(
                f"Setting maximum chunk bucket index to {MAX_CHUNK_BUCKET}"
            )
        else:
            self.log.info("Initialized for Linear model training")
            self.log.info(
                f"Setting maximum number of stored transitions to {MAX_MEM_SIZE}"
            )

    def append(self, t: Transition) -> None:
        """Add a transition into memory."""

        ### IMPORTANT NOTE
        ###
        ### In RNN mode, lookahead must be fixed for the whole episode.
        ### If lookahead were allowed to change per step, an episode could be
        ### split across LH and NLH buffers, breaking temporal continuity of
        ### the training sequences.
        if not self._rnn:
            self._memory.append(t)
            if len(self._memory) >= MAX_MEM_SIZE and self._memory_not_full:
                self.log.info(
                    f"Memory has been filled to capacity: "
                    f"{MAX_MEM_SIZE} transitions"
                )
                self._memory_not_full = False
            return

        self._cur_game.append(t)
        if t.done:
            self._finalize_game()

    def _finalize_game(self) -> None:
        """
        Finalize one completed episode.

        Cold-start policy:
          - while memory is still cold, accept short games by left-padding
          - once memory has warmed up, revert to strict full-chunk behavior
        """
        game = self._cur_game
        self._cur_game = []

        if not game:
            return

        new_chunks = self._chunk_from_end(
            game,
            allow_padding=self._memory_cold,
        )

        for bucket_idx, chunk in new_chunks:
            self._chunks_by_idx[bucket_idx].append(chunk)
            self._num_chunks += 1

        self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        if self._num_chunks <= self._max_chunks:
            return

        if self._memory_not_full:
            self.log.info(
                f"Memory has been filled to capacity: "
                f"{self._max_chunks} sequences"
            )
            self._memory_not_full = False

        overflow = self._num_chunks - self._max_chunks

        # Simple pruning policy:
        # prune oldest chunks from the most overrepresented non-empty bucket first
        for _ in range(overflow):
            bucket_to_prune = self._largest_bucket_idx()
            if bucket_to_prune is None:
                break
            del self._chunks_by_idx[bucket_to_prune][0]
            self._num_chunks -= 1

    def _log_chunk_distro(self) -> None:
        is_balanced = True
        log_str = "Sequence distribution: ["
        target = MAX_CHUNKS // (MAX_CHUNK_BUCKET + 1)

        for idx in self._chunks_by_idx.keys():
            bucket_size = len(self._chunks_by_idx[idx])
            log_str += f"{bucket_size}|"
            if bucket_size != target:
                is_balanced = False

        self.log.debug(log_str[:-1] + "]")

        if is_balanced:
            self._memory_not_balanced = False
            self.log.info("Memory buckets are balanced")

    def _largest_bucket_idx(self) -> int | None:
        largest_idx: int | None = None
        largest_size = 0

        for idx, chunks in self._chunks_by_idx.items():
            size = len(chunks)
            if size > largest_size:
                largest_size = size
                largest_idx = idx

        return largest_idx

    def _chunk_from_end(
        self,
        game: list[Transition],
        *,
        allow_padding: bool,
    ) -> list[tuple[int, RNNChunk]]:
        """
        Split a completed game into fixed-length chunks aligned from the end.

        Behavior:
          - If len(game) >= seq_len:
              split into full seq_len chunks aligned from the end
          - If len(game) < seq_len:
              * while memory is cold: left-pad into one chunk in bucket 0
              * once warm: discard

        Returns:
            List of (bucket_idx, RNNChunk)
        """
        n = len(game)
        chunks: list[tuple[int, RNNChunk]] = []

        if n < self._seq_len:
            if not allow_padding:
                return []

            padded = self._left_pad_game(game)
            chunks.append(
                (
                    0,
                    RNNChunk(
                        transitions=padded,
                        valid_len=n,
                    ),
                )
            )
            return chunks

        rem = n % self._seq_len
        start = rem if rem != 0 else 0
        chunk_idx = 0

        for i in range(start, n, self._seq_len):
            chunk = game[i : i + self._seq_len]
            if len(chunk) == self._seq_len:
                bucket_idx = min(chunk_idx, MAX_CHUNK_BUCKET)
                chunks.append(
                    (
                        bucket_idx,
                        RNNChunk(
                            transitions=chunk,
                            valid_len=self._seq_len,
                        ),
                    )
                )
                chunk_idx += 1

        return chunks

    def _left_pad_game(self, game: list[Transition]) -> list[Transition]:
        """
        Left-pad a short game to seq_len by repeating the first transition.

        The real game remains RIGHT-aligned, which fits the existing
        end-aligned chunk semantics.
        """
        if not game:
            raise ValueError("Cannot pad an empty game")

        pad_count = self._seq_len - len(game)
        first = game[0]

        # Cheap bootstrap padding.
        # This reuses the same Transition reference, which is fine as long as
        # Transition objects are treated as immutable after creation.
        return [first] * pad_count + game

    def num_chunks(self) -> int:
        return self._num_chunks

    def chunk_bucket_counts(self) -> dict[int, int]:
        return {
            idx: len(chunks) for idx, chunks in self._chunks_by_idx.items()
        }

    def _warmed_bucket_indices(self) -> list[int]:
        warmed: list[int] = []
        for idx, chunks in self._chunks_by_idx.items():
            if len(chunks) > 0:
                warmed.append(idx)
        return warmed

    def sample_chunks(
        self,
        batch_size: int,
    ) -> list[RNNChunk] | None:
        """
        Sample chunks across warmed chunk-index buckets.

        Sampling policy:
          - require enough chunks in aggregate
          - divide batch approximately evenly across non-empty buckets
          - redistribute remainder to buckets with spare capacity
        """
        if self._num_chunks < max(batch_size, MIN_CHUNKS):
            return None

        warmed = self._warmed_bucket_indices()
        if not warmed:
            return None

        if self._memory_cold:
            self.log.debug("Memory has warmed up")
            self._memory_cold = False

        samples: list[RNNChunk] = []

        # First pass: even split across warmed buckets
        base = batch_size // len(warmed)
        remainder = batch_size % len(warmed)

        requested: dict[int, int] = {}
        for idx in warmed:
            requested[idx] = base

        # distribute remainder from later buckets backward so late-game buckets
        # get slight preference when possible
        warmed_desc = sorted(warmed, reverse=True)
        for i in range(remainder):
            requested[warmed_desc[i % len(warmed_desc)]] += 1

        # Sample what we can from each bucket
        shortage = 0
        for idx in warmed:
            bucket = self._chunks_by_idx[idx]
            want = requested[idx]
            take = min(want, len(bucket))
            if take > 0:
                samples.extend(self._rng.sample(bucket, take))
            shortage += want - take

        if shortage > 0:
            # Fallback pool from all unused capacity
            flat_pool: list[RNNChunk] = []
            already_selected_ids = {id(chunk) for chunk in samples}

            for idx in warmed:
                for chunk in self._chunks_by_idx[idx]:
                    if id(chunk) not in already_selected_ids:
                        flat_pool.append(chunk)

            if len(flat_pool) < shortage:
                return None

            samples.extend(self._rng.sample(flat_pool, shortage))

        self._samples_delivered += 1

        if self._memory_not_balanced and self._samples_delivered % 500 == 0:
            self._log_chunk_distro()

        return samples

    def sample_transitions(
        self,
        batch_size: int,
    ) -> list[Transition] | None:
        """Sample random transitions for a linear model."""

        if len(self._memory) < max(MIN_FRAMES, batch_size):
            return None

        if self._memory_cold:
            self.log.debug("Memory has warmed up")
            self._memory_cold = False

        return self._rng.sample(self._memory, batch_size)
