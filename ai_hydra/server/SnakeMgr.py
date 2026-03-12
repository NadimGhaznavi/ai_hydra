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
from datetime import datetime

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DGame import DGameField
from ai_hydra.constants.DNNet import DNetField

from ai_hydra.game.GameLogic import GameLogic
from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.utils.SimCfg import SimCfg


@dataclass
class SnakeSession:
    """
    Pure domain session state for one client identity.
    """

    board: Any  # GameBoard, but keep Any here to avoid import cycles if needed
    seed: int
    rng: random.Random
    # Attribs with defaults
    done: bool = False
    episode_id: int = 0
    epoch: int = 0
    highscore: int = 0  # Actual highscore
    highscore_lh: int = 0  # Lookahead highscore
    highscore_nlh: int = 0  # No lookahead highscore
    lookahead_on: bool = False
    reward_total: float = 0.0
    score: int = 0
    step_n: int = 0


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

    def __init__(
        self,
        cfg: SimCfg,
        master_seed: int = DHydra.RANDOM_SEED,
    ) -> None:
        if cfg is None:
            raise TypeError("SnakeMgr requires cfg (SimCfg)")
        self.cfg = cfg
        self.master_seed = int(master_seed)

        self.seed_rng = random.Random(self.master_seed)
        self.sessions: dict[str, SnakeSession] = {}
        self.policy: Optional[HydraPolicy] = None
        self._start_time = datetime.now()

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

    # -------------------------
    # Game operations (Phase 1)
    # -------------------------

    def reset_session(
        self, client_id: str, seed: Optional[int] = None
    ) -> SnakeSession:
        from ai_hydra.game.GameBoard import GameBoard

        prev = self.sessions.get(client_id)
        prev_highscore = getattr(prev, "highscore", 0)
        prev_highscore_lh = getattr(prev, "highscore_lh", 0)
        prev_highscore_nlh = getattr(prev, "highscore_nlh", 0)
        prev_epoch = getattr(prev, "epoch", 0)
        prev_lookahead_on = getattr(prev, "lookahead_on", False)

        session_seed, rng = self.new_rng(seed)
        episode_id = rng.getrandbits(32)

        board = GameBoard.new_game(
            rng=rng,
            seed=session_seed,
            episode_id=episode_id,
        )

        sess = SnakeSession(
            seed=session_seed,
            rng=rng,
            board=board,
            done=False,
            step_n=0,
            score=0,
            episode_id=episode_id,
            # carry-over stats
            highscore=prev_highscore,
            highscore_lh=prev_highscore_lh,
            highscore_nlh=prev_highscore_nlh,
            epoch=prev_epoch,
            lookahead_on=prev_lookahead_on,
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

    def reset(self, client_id: str, seed: Optional[int] = None) -> None:
        """
        Reset and return snapshot.
        """
        self.reset_session(client_id, seed=seed)

    def start_time(self, start_time=None):
        if start_time is not None:
            self._start_time = start_time
        return self._start_time

    def step(self, client_id: str, action: int) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """
        Step the client's session forward by one action.

        Returns:
        - state_dict
        """
        # The previous session....
        sess = self.get_session(client_id)

        # NN-ready state vector (pre-step)
        state = sess.board.get_state()

        # Apply action
        result = GameLogic.step(sess.board, int(action), sess.rng)

        sess.board = result.new_board
        sess.step_n += 1

        reward = int(result.reward)
        done = bool(result.is_terminal)
        outcome = result.outcome  # use as "reason" for now

        # Score delta (Phase 1): +1 on FOOD, else 0
        if outcome == DGameField.FOOD:
            sess.score += 1

        elapsed_secs = (datetime.now() - self.start_time()).total_seconds()
        elapsed_str = minutes_to_uptime(int(elapsed_secs))

        sess.done = done
        sess.reward_total += reward

        # NN-ready next state (post-step)
        next_state = sess.board.get_state()

        # ----- State information ---
        state_dict = {
            DGameField.DONE: done,
            DNetField.STATE: state,
            DNetField.NEXT_STATE: next_state,
            DGameField.REWARD: reward,
            DGameField.REWARD_TOTAL: sess.reward_total,
            DGameField.EPISODE_ID: sess.episode_id,
        }

        if sess.score > sess.highscore:
            sess.highscore = sess.score

        # ----- The payload for the ZeroMQ "scores" topic ---
        scores_payload = {
            DGameField.SCORE: sess.score,
            DGameField.HIGHSCORE: sess.highscore,
            DNetField.LOOKAHEAD_ON: sess.lookahead_on,
        }
        # Create a "final score" field for the TUI
        if done:
            scores_payload[DNetField.FINAL_SCORE] = sess.score
        # Add a "lookahead highscore" event
        if sess.score > sess.highscore_lh and sess.lookahead_on:
            sess.highscore_lh = sess.score
            scores_payload[DGameField.HIGHSCORE_EVENT_LH] = [
                sess.epoch,
                sess.highscore_lh,
                elapsed_str,
            ]

        # Add a new "no lookahead highscore" event
        if sess.score > sess.highscore_nlh and not sess.lookahead_on:
            sess.highscore_nlh = sess.score
            scores_payload[DGameField.HIGHSCORE_EVENT_NLH] = [
                sess.epoch,
                sess.highscore_nlh,
                elapsed_str,
            ]

        # ----- The payload for the ZeroMQ "per step" topic ---
        step_payload = None
        if self.cfg.get(DNetField.PER_STEP):
            step_payload = {DGameField.BOARD: sess.board.to_dict()}

        # ----- The payload for the ZeroMQ "per episode" topic ----
        ep_payload = None
        if sess.done:
            ep_payload: dict[str, Any] = {
                DGameField.EPOCH: sess.epoch,
                DGameField.REASON: outcome,
                DGameField.STEP_N: sess.step_n,
                DGameField.ELAPSED_TIME: elapsed_str,
            }

        return state_dict, scores_payload, step_payload, ep_payload


# Helper function
def minutes_to_uptime(seconds: int):
    # Return a string like:
    # 45s
    # 7h 23m
    # 1d 7h 32m
    days, minutes = divmod(int(seconds), 86400)
    hours, minutes = divmod(minutes, 3600)
    minutes, seconds = divmod(minutes, 60)

    if days > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            minutes = f" {minutes}"
        if hours < 10:
            hours = f" {hours}"
        return f"{days}d {hours}h {minutes}m"

    elif hours > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            minutes = f" {minutes}"
        return f"{hours}h {minutes}m"

    elif minutes > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            return f" {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"

    else:
        return f"{seconds}s"
