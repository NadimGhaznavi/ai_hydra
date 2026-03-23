from __future__ import annotations

from dataclasses import dataclass

from ai_hydra.constants.DHydra import DModule, DHydraLog
from ai_hydra.constants.DReplayMemory import DMemDef

from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.nnet.ATH.ATHMemory import ATHMemory
from ai_hydra.nnet.LinearTrainer import LinearTrainer
from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.RecurrentTrainer import RecurrentTrainer
from ai_hydra.nnet.SimpleReplayMemory import SimpleReplayMemory
from ai_hydra.nnet.models.LinearModel import LinearModel
from ai_hydra.nnet.models.RNNModel import RNNModel
from ai_hydra.nnet.models.GRUModel import GRUModel
from ai_hydra.server.SnakeMgr import SnakeMgr


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


MAX_STAGNANT_EPISODES = DMemDef.MAX_STAGNANT_EPISODES
MAX_HARD_RESET_EPISODES = DMemDef.MAX_HARD_RESET_EPISODES


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
        trainer: LinearTrainer | RecurrentTrainer,
        replay: SimpleReplayMemory | ATHMemory,
        client_id: str = DModule.TRAIN_MGR,
        model: LinearModel | RNNModel | GRUModel,
        log_level: DHydraLog,
    ) -> None:

        self.log = HydraLog(
            client_id=DModule.TRAIN_MGR,
            log_level=log_level,
            to_console=True,
        )
        self.snake_mgr = snake_mgr
        self.policy = policy
        self.trainer = trainer
        self.replay = replay
        self.client_id = client_id
        self.model = model

        self._stag_ep_count = 0
        self._hard_reset_ep_count = 0
        self._hard_reset_count = 0
        self._cur_highscore = 0
        self._max_hard_reset_episodes = MAX_HARD_RESET_EPISODES

    # Called at the end of each episode
    async def handle_stagnation(self, final_score):
        self.replay.gearbox.incr_epoch()

        if final_score > self._cur_highscore:
            self._cur_highscore = final_score
            self._stag_ep_count = 0
            self._hard_reset_ep_count = 0
            await self.replay.stagnation_cleared()

        else:
            self._stag_ep_count += 1
            self._hard_reset_ep_count += 1

        if self._stag_ep_count >= MAX_STAGNANT_EPISODES:
            self.log.debug(
                "Stagnation event detected, notifying the replay memory"
            )
            self._stag_ep_count = 0
            self.replay.stagnation_warning()

        if self._hard_reset_ep_count >= self._max_hard_reset_episodes:
            self._hard_reset_count += 1
            if self._hard_reset_count > 1:
                self._max_hard_reset_episodes = (
                    self._max_hard_reset_episodes * self._hard_reset_count
                )
                self.log.debug(
                    f"Critical stagnation alert({self._hard_reset_count}): "
                    f"Exceeded {self._hard_reset_count} episodes without "
                    "improvement, notifying replay memory"
                )
            self._hard_reset_ep_count = 0
            self._stag_ep_count = 0
            await self.replay.hard_reset(self._hard_reset_count)
