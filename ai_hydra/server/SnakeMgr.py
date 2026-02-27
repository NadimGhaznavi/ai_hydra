# ai_hydra/server/SnakeMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DGame import DGameField


@dataclass
class SnakeSession:
    """
    Pure domain session state for one client identity.
    """

    seed: int
    rng: random.Random
    board: Any  # GameBoard, but keep Any here to avoid import cycles if needed
    done: bool = False
    step_n: int = 0
    score: int = 0
    episode_id: int = 0


class SnakeMgr:
    """
    Domain manager for Snake game sessions.

    Owns:
      - a master seed + seed RNG (for minting per-session seeds)
      - a per-client session map

    Does NOT:
      - talk to MQ
      - render
      - run neural nets (future)
    """

    def __init__(self, master_seed: int = DHydra.RANDOM_SEED) -> None:
        self.master_seed = int(master_seed)
        self.seed_rng = random.Random(self.master_seed)
        self.sessions: dict[str, SnakeSession] = {}

    # -------------------------
    # RNG API (Pattern B)
    # -------------------------

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

    def clone_rng(self, parent_rng: random.Random) -> random.Random:
        """
        Clone RNG state exactly (for future clone rollouts).
        """
        clone = random.Random()
        clone.setstate(parent_rng.getstate())
        return clone

    def get_rng_state(self, rng: random.Random) -> object:
        """
        Export RNG state (useful later for replay/persistence).
        """
        return rng.getstate()

    def set_rng_state(self, rng: random.Random, state: object) -> None:
        """
        Restore RNG state (useful later for replay/persistence).
        """
        rng.setstate(state)

    # -------------------------
    # Session lifecycle
    # -------------------------

    def has_session(self, client_id: str) -> bool:
        return client_id in self.sessions

    def reset_session(
        self, client_id: str, seed: Optional[int] = None
    ) -> SnakeSession:
        """
        Create or reset a client's session with an optional seed override.
        """
        # Lazy imports to keep SnakeMgr light and avoid circular deps
        from ai_hydra.game.GameBoard import GameBoard

        session_seed, rng = self.new_rng(seed)

        # You may need to adjust this constructor depending on your GameBoard API
        board = GameBoard.new(rng=rng)

        episode_id = rng.getrandbits(32)

        sess = SnakeSession(
            seed=session_seed,
            rng=rng,
            board=board,
            done=False,
            step_n=0,
            score=0,
            episode_id=episode_id,
        )
        self.sessions[client_id] = sess
        return sess

    def get_session(self, client_id: str) -> SnakeSession:
        """
        Fetch a session. Auto-creates one if missing (seed minted).
        """
        sess = self.sessions.get(client_id)
        if sess is None:
            sess = self.reset_session(client_id, seed=None)
        return sess

    # -------------------------
    # Game operations (Phase 1)
    # -------------------------

    def step(self, client_id: str, action: int) -> dict[str, Any]:
        """
        Step the client's session forward by one action.

        Returns a JSON-friendly dict suitable for putting into HydraMsg.payload.
        """
        sess = self.get_session(client_id)

        if sess.done:
            return {
                DGameField.OK: False,
                DGameField.ERROR: DGameField.EPISODE_DONE,
                DGameField.INFO: {
                    DGameField.EPISODE_ID: sess.episode_id,
                    DGameField.STEP_N: sess.step_n,
                    DGameField.SCORE: sess.score,
                },
            }

        # Lazy import: GameLogic can evolve without tangling SnakeMgr import time
        from ai_hydra.game.GameLogic import GameLogic

        # Contract we’re standardizing on:
        #   new_board, result = GameLogic.step(board, action, rng)
        new_board, result = GameLogic.step(sess.board, int(action), sess.rng)

        sess.board = new_board
        sess.step_n += 1

        # These fields depend on your MoveResult; keep it defensive
        reward = float(getattr(result, DGameField.SCORE, 0.0))
        done = bool(getattr(result, DGameField.DONE, False))
        reason = getattr(result, DGameField.REASON, None)
        score_delta = int(getattr(result, DGameField.SCORE_DELTA, 0))

        sess.score += score_delta
        sess.done = done

        return {
            DGameField.OK: True,
            DGameField.REWARD: reward,
            DGameField.DONE: sess.done,
            DGameField.SNAPSHOT: self.snapshot(client_id),
            DGameField.INFO: {
                DGameField.EPISODE_ID: sess.episode_id,
                DGameField.STEP_N: sess.step_n,
                DGameField.SCORE: sess.score,
                DGameField.REASON: reason,
                DGameField.SEED: sess.seed,
            },
        }

    def snapshot(self, client_id: str) -> dict[str, Any]:
        """
        Return a JSON-friendly snapshot of session state.
        """
        sess = self.get_session(client_id)

        # You’ll want GameBoard.to_dict() for this (or similar)
        board_dict = (
            sess.board.to_dict()
            if hasattr(sess.board, "to_dict")
            else {"repr": repr(sess.board)}
        )

        return {
            DGameField.BOARD: board_dict,
            DGameField.DONE: sess.done,
            DGameField.STEP_N: sess.step_n,
            DGameField.SCORE: sess.score,
            DGameField.EPISODE_ID: sess.episode_id,
            DGameField.SEED: sess.seed,
        }

    def reset(
        self, client_id: str, seed: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Reset and return snapshot.
        """
        self.reset_session(client_id, seed=seed)
        return self.snapshot(client_id)
