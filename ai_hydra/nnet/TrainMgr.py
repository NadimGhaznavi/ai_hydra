from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.constants.DHydra import DModule

from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.LinearTrainer import LinearTrainer
from ai_hydra.nnet.RNNTrainer import RNNTrainer
from ai_hydra.nnet.SimpleReplayMemory import SimpleReplayMemory
from ai_hydra.nnet.ATH.ATHMemory import ATHMemory
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.models.LinearModel import LinearModel
from ai_hydra.nnet.models.RNNModel import RNNModel


@dataclass(frozen=True)
class EpisodeStats:
    episode_id: int
    steps: int
    score: int
    reward_total: int
    losses: int
    loss_avg: float


@dataclass(frozen=True)
class TrainRunStats:
    episodes: int
    avg_score: float
    avg_steps: float
    avg_reward_total: float
    avg_loss: float


MAX_STAGNANT_EPISODES = 600

MAX_HARD_RESET_EPISODES = MAX_STAGNANT_EPISODES * 3


class TrainMgr:
    """
    Training orchestrator.

    Owns:
      - policy (possibly epsilon-greedy wrapper)
      - trainer (torch + optimizer)
      - replay memory

    Uses:
      - SnakeMgr as the environment/session API

    Does NOT:
      - talk to MQ
      - render
    """

    def __init__(
        self,
        *,
        snake_mgr: SnakeMgr,
        policy: HydraPolicy,
        trainer: LinearTrainer | RNNTrainer,
        replay: SimpleReplayMemory | ATHMemory,
        client_id: str = DModule.TRAIN_MGR,
        model: LinearModel | RNNModel,
    ) -> None:
        self.snake_mgr = snake_mgr
        self.policy = policy
        self.trainer = trainer
        self.replay = replay
        self.client_id = client_id
        self.model = model

        self._stag_ep_count = 0
        self._hard_reset_ep_count = 0
        self._cur_highscore = 0

    async def handle_stagnation(self, final_score):
        if final_score > self._cur_highscore:
            self._cur_highscore = final_score
            self._stag_ep_count = 0
            self._hard_reset_ep_count = 0
            await self.replay.stagnation_cleared()

        else:
            self._stag_ep_count += 1
            self._hard_reset_ep_count += 1

        if self._stag_ep_count >= MAX_STAGNANT_EPISODES:
            self._stag_ep_count = 0
            self.replay.stagnation_warning()

        if self._hard_reset_ep_count >= MAX_HARD_RESET_EPISODES:
            self._hard_reset_ep_count = 0
            await self.replay.hard_reset()
