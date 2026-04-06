# ai_hydra/nnet/Policy/BehaviourPolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Sequence
import random

from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DEvent import EV_STATUS

from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.EpsilonNiceAlgo import EpsilonNiceAlgo
from ai_hydra.game.GameBoard import GameBoard
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import EventMsg, HydraEventMQ
from ai_hydra.mcts.Node import Node
from ai_hydra.mcts.MCTSConfig import MCTSConfig


class BehaviourPolicy(HydraPolicy):
    def __init__(
        self,
        log_level: DHydraLog,
        base_policy: HydraPolicy,
        epsilon_n: EpsilonNiceAlgo,
        nice_p_value: float,
        nice_steps: int,
        nice_rng: random.Random,
        mcts_cfg: MCTSConfig,
        pub_func,
    ):
        self.event = HydraEventMQ(
            client_id=DModule.EPSILON_NICE_POLICY,
            pub_func=pub_func,
        )
        self.log = HydraLog(
            client_id=DModule.EPSILON_NICE_POLICY,
            log_level=log_level,
            to_console=True,
        )

        self.base_policy = base_policy
        self.epsilon_n = epsilon_n
        self._nice_rng = nice_rng
        self._nice_p_value = nice_p_value
        self._nice_steps = nice_steps
        self._mcts_cfg = mcts_cfg

        self.log.info(f"Nice P-Value set: {nice_p_value}")
        self.log.info(f"Nice steps: {nice_steps}")
        self.log.info(f"Monte Carlo Search Depth {mcts_cfg.search_depth}")
        self.log.info(f"Monte Carlo Frequency {mcts_cfg.frequency}")
        self.log.info(
            f"Monte Carlo Exploration P-Value {mcts_cfg.explore_p_value}"
        )
        self.log.info(f"Monte Carlo Iterations {mcts_cfg.iterations}")

        self._nice_enabled = False
        self._nice_steps_remaining = 0

        self._mcts_enabled = False

    def cur_epsilon(self):
        return self.base_policy.cur_epsilon()

    async def disable_nice(self):
        # Globally disabled
        if self._nice_p_value == 0:
            return

        # Already disabled
        if not self._nice_enabled:
            return

        self._nice_enabled = False
        self._nice_steps_remaining = 0
        msg = "Disabling EpsilonNice"
        await self.event.publish(
            EventMsg(
                level=EV_STATUS.INFO,
                message=msg,
            )
        )
        self.log.info(msg)

    def disable_mcts(self):
        """
        Disable Monte Carlo Tree Search control
        """
        # MCTS burst is already enabled
        self._mcts_enabled = False

    def enable_mcts(self):
        """
        Enable a Monte Carlo Tree Search control for an episode
        """
        # MCTS burst is already enabled
        self._mcts_enabled = True

    async def enable_nice(self):
        # Globally disabled
        if self._nice_p_value == 0:
            return

        # Already enabled
        if self._nice_enabled:
            return

        self._nice_enabled = True
        msg = "Enabling EpsilonNice"
        await self.event.publish(
            EventMsg(
                level=EV_STATUS.INFO,
                message=msg,
            )
        )
        self.log.info(msg)

    async def played_game(self):
        await self.base_policy.played_game()
        await self.epsilon_n.played_game()

    def _select_mcts_action(self, board: GameBoard):
        mcts_cfg = self._mcts_cfg
        root = Node(
            board=board,
            parent=None,
            action_index=None,
            cfg=mcts_cfg,
            is_terminal=False,
            immediate_reward=0.0,
        )

        for _ in range(self._mcts_cfg.iterations):
            root.explore()

        _, action = root.next()
        return action

    def select_action(self, state: Sequence[float], board: GameBoard) -> int:

        suggested = self.base_policy.select_action(state, board)

        # Normal Epsilon is still active, do nothing
        if self.base_policy.cur_epsilon() > 0.0001:
            self._nice_steps_remaining = 0
            return suggested

        # Check if Monte Carlo Tree Search is enabled
        if self._mcts_enabled:
            return self._select_mcts_action(board=board)

        # Check if Nice is enabled
        if self._nice_enabled:

            self.epsilon_n.incr_calls()

            # We're on a "Nice" detour, lets explore!!! :)
            if self._nice_steps_remaining > 0:
                self._nice_steps_remaining -= 1
                return self.epsilon_n.override_action(
                    suggested_action=suggested,
                    board=board,
                )

            # Determine whether or not to activate EpsilonNice (on the next step)
            if self._nice_rng.random() < self._nice_p_value:
                self._nice_steps_remaining = self._nice_steps

        return suggested
