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
import pprint
import traceback

from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DGame import DGameField
from ai_hydra.constants.DNNet import DNetField

from ai_hydra.game.GameLogic import GameLogic
from ai_hydra.game.GameHelper import RewardCfg
from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.HydraRng import HydraRng
from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.utils.HydraHelper import minutes_to_uptime


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
    reward_total: float = 0.0
    score: int = 0
    step_n: int = 0


class SnakeMgr:
    """
    Domain manager for Snake game sessions.
    """

    def __init__(
        self,
        cfg: SimCfg,
        log_level: DHydraLog,
        hydra_rng: HydraRng,
        reward_cfg: RewardCfg,
        mmm: int,  # Max-Moves-Multiplier
    ) -> None:
        if cfg is None:
            raise TypeError("SnakeMgr requires cfg (SimCfg)")
        self.cfg = cfg
        self.hydra_rng = hydra_rng
        self._reward_cfg = reward_cfg
        self._mmm = mmm

        self.log = HydraLog(
            client_id="SnakeMgr",
            log_level=log_level,
            to_console=True,
        )

        self.log.info(
            f"Reward Configuration:\n{pprint.pformat(self._reward_cfg.values)}"
        )

        self.sessions: dict[str, SnakeSession] = {}
        self.policy: Optional[HydraPolicy] = None
        self._start_time = datetime.now()

        self._mcts_enabled = False

    # -------------------------
    # Session lifecycle
    # -------------------------

    def has_session(self, client_id: str) -> bool:
        return client_id in self.sessions

    def mcts_enabled(self, flag: bool) -> None:
        self._mcts_enabled = flag

    # -------------------------
    # Game operations (Phase 1)
    # -------------------------

    def reset_session(self, client_id: str) -> SnakeSession:
        from ai_hydra.game.GameBoard import GameBoard

        prev = self.sessions.get(client_id)
        prev_highscore = getattr(prev, "highscore", 0)
        prev_epoch = getattr(prev, "epoch", 0)
        next_step_n = getattr(prev, "step_n", 0) + 1

        session_seed, rng = self.hydra_rng.new_rng()
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
            score=0,
            episode_id=episode_id,
            # carry-over stats
            highscore=prev_highscore,
            epoch=prev_epoch,
            step_n=next_step_n,
        )
        self.sessions[client_id] = sess
        return sess

    def get_session(self, client_id: str) -> SnakeSession:
        """
        Fetch a session. Auto-creates one if missing.
        """
        sess = self.sessions.get(client_id)
        if sess is None:
            sess = self.reset_session(client_id)
        return sess

    def reset(self, client_id: str) -> None:
        """
        Reset and return snapshot.
        """
        self.reset_session(client_id)

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
        try:
            # The previous session....
            sess = self.get_session(client_id)

            # NN-ready state vector (pre-step)
            state = sess.board.get_state()

            # Apply action
            result = GameLogic.step(
                board=sess.board,
                action=int(action),
                rng=sess.rng,
                reward_cfg=self._reward_cfg,
                mmm=self._mmm,
            )

            sess.board = result.new_board

            reward = result.reward
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

            # ----- The payload for the ZeroMQ "scores" topic ---
            scores_payload = {
                DGameField.EPOCH: sess.epoch,
                DGameField.CUR_SCORE: sess.score,
            }
            # Create a "final score" field for the TUI
            if done:
                if self._mcts_enabled:
                    scores_payload[DNetField.FINAL_MCTS_SCORE] = sess.score
                    self.log.debug(f"Final MCTS Score: {sess.score}")
                else:
                    scores_payload[DNetField.FINAL_SCORE] = sess.score

            # Add a "highscore" event
            if sess.score > sess.highscore:
                sess.highscore = sess.score
                scores_payload[DGameField.HIGHSCORE] = sess.highscore
                self.log.info(
                    f"Epoch: {sess.epoch} - New High Score: {sess.highscore}"
                )

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
                    DGameField.ELAPSED_TIME: elapsed_str,
                }

            return state_dict, scores_payload, step_payload, ep_payload

        except Exception as e:
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"STACKTRACE: {traceback.format_exc()}")
