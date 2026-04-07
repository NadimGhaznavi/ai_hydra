from __future__ import annotations

from dataclasses import dataclass

from ai_hydra.constants.DHydra import DModule, DHydraLog
from ai_hydra.constants.DReplayMemory import DMemDef
from ai_hydra.constants.DEvent import EV_STATUS, EV_TYPE


from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.nnet.ATH.ATHMemory import ATHMemory
from ai_hydra.nnet.LinearTrainer import LinearTrainer
from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.Policy.EpsilonNicePolicy import EpsilonNicePolicy
from ai_hydra.nnet.RecurrentTrainer import RecurrentTrainer
from ai_hydra.nnet.SimpleReplayMemory import SimpleReplayMemory
from ai_hydra.nnet.models.LinearModel import LinearModel
from ai_hydra.nnet.models.RNNModel import RNNModel
from ai_hydra.nnet.models.GRUModel import GRUModel
from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.zmq.HydraEventMQ import EventMsg, HydraEventMQ


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
        policy: EpsilonNicePolicy,
        trainer: LinearTrainer | RecurrentTrainer,
        replay: SimpleReplayMemory | ATHMemory,
        client_id: str = DModule.TRAIN_MGR,
        model: LinearModel | RNNModel | GRUModel,
        log_level: DHydraLog,
        pub_func,
        stag_thresh: int,
        crit_stag_thresh: int,
    ) -> None:
        self.event = HydraEventMQ(
            client_id=DModule.TRAIN_MGR,
            pub_func=pub_func,
        )
        self.log = HydraLog(
            client_id=DModule.TRAIN_MGR,
            log_level=log_level,
            to_console=True,
        )

        self.log.info(f"Set stagnation threshold: {stag_thresh}")
        self.log.info(f"Set critical stagnation threshold: {crit_stag_thresh}")
        self.snake_mgr = snake_mgr
        self.policy = policy
        self.trainer = trainer
        self.replay = replay
        self.client_id = client_id
        self.model = model
        self._stag_thresh = stag_thresh
        self._base_crit_stag_thresh = crit_stag_thresh
        self._crit_stag_thresh = crit_stag_thresh

        self._stag_alert_status = EV_TYPE.CLEARED
        self._stag_ep_count = 0
        self._hard_reset_ep_count = 0
        self._hard_reset_count = 0
        self._cur_highscore = 0

    # Called at the end of each episode
    async def handle_stagnation(self, final_score):
        self.replay.gearbox.incr_epoch()

        if final_score > self._cur_highscore:
            self._cur_highscore = final_score
            self._stag_ep_count = 0
            self._hard_reset_ep_count = 0
            if self._stag_alert_status != EV_TYPE.CLEARED:
                msg = (
                    f"Stagnation alert cleared (new highscore: {final_score})"
                )
                self.log.debug(msg)
                await self.event.publish(
                    EventMsg(level=EV_STATUS.INFO, message=msg)
                )
                await self.policy.disable_nice()
            await self.replay.gearbox.stagnation_cleared()
            # await self.replay.gearbox.reset_cooldown(
            #    highscore=self._cur_highscore
            # )
            self._stag_alert_status = EV_TYPE.CLEARED

        else:
            self._stag_ep_count += 1
            self._hard_reset_ep_count += 1

        if self._stag_ep_count >= self._stag_thresh:
            if self._stag_alert_status != EV_TYPE.SET:
                msg = f"Stagnation alert raised"
                self.log.debug(msg)
                await self.event.publish(
                    EventMsg(level=EV_STATUS.WARN, message=msg)
                )
                self._stag_ep_count = 0
                self.replay.gearbox.stagnation_warning()
                await self.policy.enable_nice()
            self._stag_alert_status = EV_TYPE.SET

        if self._hard_reset_ep_count >= self._crit_stag_thresh:
            self._hard_reset_count += 1
            self._crit_stag_thresh = self._base_crit_stag_thresh * (
                self._hard_reset_count + 1
            )
            msg = (
                f"Critical stagnation alert({self._hard_reset_count}). "
                f"New threshold set: {self._crit_stag_thresh}"
            )

            self.log.debug(msg)
            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.BAD,
                    message=(msg),
                )
            )
            self._hard_reset_ep_count = 0
            self._stag_ep_count = 0
            await self.replay.gearbox.hard_reset(self._hard_reset_count)
            if self._stag_alert_status != EV_TYPE.SET:
                await self.policy.enable_nice()
            self._stag_alert_status = EV_TYPE.SET
