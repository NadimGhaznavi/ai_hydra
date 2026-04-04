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
from math import sqrt, log
import random
from typing import Optional

from ai_hydra.game.GameBoard import GameBoard
from ai_hydra.game.GameLogic import GameLogic
from ai_hydra.game.GameHelper import RewardCfg

GAME_ACTIONS = [0, 1, 2]
P_EXPLORE_VALUE = 0.9
SEARCH_DEPTH = 10


@dataclass
class MCTSConfig:
    reward_cfg: RewardCfg
    mmm: int
    food_ends_episode: bool = False
    search_depth: int = SEARCH_DEPTH
    explore_c: float = P_EXPLORE_VALUE


class Node:
    def __init__(
        self,
        board: GameBoard,
        parent: Optional["Node"],
        action_index: Optional[int],
        rng: random.Random,
        cfg: MCTSConfig,
        *,
        is_terminal: bool = False,
        immediate_reward: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.action_index = action_index
        self.rng = rng
        self.cfg = cfg

        self.done = is_terminal
        self.immediate_reward = immediate_reward

        self.child: Optional[dict[int, Node]] = None

        # accumulated rollout value
        self.T = 0.0

        # visit count
        self.N = 0

    def detach_parent(self) -> None:
        self.parent = None

    def get_ucb_score(self) -> float:
        if self.N == 0:
            return float("inf")

        if self.parent is None or self.parent.N == 0:
            return self.T / self.N

        return (self.T / self.N) + self.cfg.explore_c * sqrt(
            log(self.parent.N) / self.N
        )

    def create_child(self) -> None:
        if self.done:
            return

        child: dict[int, Node] = {}

        for action in GAME_ACTIONS:
            result = GameLogic.step(
                board=self.board,
                action=action,
                rng=self.rng,
                reward_cfg=self.cfg.reward_cfg,
                mmm=self.cfg.mmm,
                food_ends_episode=self.cfg.food_ends_episode,
            )

            child[action] = Node(
                board=result.new_board,
                parent=self,
                action_index=action,
                rng=self.rng,
                cfg=self.cfg,
                is_terminal=result.is_terminal,
                immediate_reward=result.reward,
            )

        self.child = child

    def explore(self) -> None:
        current = self

        # 1. Selection
        while current.child:
            max_ucb = max(c.get_ucb_score() for c in current.child.values())
            best_actions = [
                action
                for action, child in current.child.items()
                if child.get_ucb_score() == max_ucb
            ]
            chosen_action = current.rng.choice(best_actions)
            current = current.child[chosen_action]

        # 2. Expansion + Simulation
        if current.N == 0:
            rollout_value = current.rollout()
        else:
            current.create_child()
            if current.child:
                chosen_child = current.rng.choice(list(current.child.values()))
                current = chosen_child
            rollout_value = current.rollout()

        # 3. Backpropagation
        node = current
        while node is not None:
            node.N += 1
            node.T += rollout_value
            node = node.parent

    def rollout(self) -> float:
        """
        Random playout from this board for a bounded depth.
        Returns cumulative reward starting from this node.
        """
        if self.done:
            return self.immediate_reward

        total = self.immediate_reward
        board = self.board
        depth = 0
        terminal = False

        while not terminal and depth < self.cfg.rollout_depth:
            action = self.rng.choice(GAME_ACTIONS)
            result = GameLogic.step(
                board=board,
                action=action,
                rng=self.rng,
                reward_cfg=self.cfg.reward_cfg,
                mmm=self.cfg.mmm,
                food_ends_episode=self.cfg.food_ends_episode,
            )
            total += result.reward
            board = result.new_board
            terminal = result.is_terminal
            depth += 1

        return total

    def next(self) -> tuple["Node", int]:
        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError("no children found and game hasn't ended")

        max_n = max(node.N for node in self.child.values())
        max_children = [
            child for child in self.child.values() if child.N == max_n
        ]
        max_child = self.rng.choice(max_children)
        return max_child, max_child.action_index
