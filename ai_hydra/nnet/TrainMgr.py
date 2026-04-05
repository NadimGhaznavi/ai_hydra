from __future__ import annotations

from dataclasses import dataclass

from ai_hydra.constants.DHydra import DModule, DHydraLog
from ai_hydra.constants.DReplayMemory import DMemDef
from ai_hydra.constants.DEvent import EV_STATUS, EV_TYPE
from ai_hydra.constants.DHydraTui import DField


from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.nnet.ATH.ATHMemory import ATHMemory
from ai_hydra.nnet.LinearTrainer import LinearTrainer
from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.Policy.BehaviourPolicy import BehaviourPolicy
from ai_hydra.nnet.RecurrentTrainer import RecurrentTrainer
from ai_hydra.nnet.SimpleReplayMemory import SimpleReplayMemory
from ai_hydra.nnet.models.LinearModel import LinearModel
from ai_hydra.nnet.models.RNNModel import RNNModel
from ai_hydra.nnet.models.GRUModel import GRUModel
from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.zmq.HydraEventMQ import EventMsg, HydraEventMQ
from ai_hydra.game.GameHelper import RewardCfg
from ai_hydra.game.GameBoard import GameBoard
from ai_hydra.mcts.Node import MCTSConfig


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
        policy: BehaviourPolicy,
        trainer: LinearTrainer | RecurrentTrainer,
        replay: SimpleReplayMemory | ATHMemory,
        client_id: str = DModule.TRAIN_MGR,
        model: LinearModel | RNNModel | GRUModel,
        log_level: DHydraLog,
        pub_func,
        stag_thresh: int,
        crit_stag_thresh: int,
        reward_cfg: RewardCfg,
        mcts_cfg: MCTSConfig,
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
        self.log.info(
            f"Set Monte Carlo score threshold: {mcts_cfg.score_thresh}"
        )
        self.log.info(
            f"Set Monte Carlo gating p-value {mcts_cfg.gate_p_value}"
        )
        self.snake_mgr = snake_mgr
        self.policy = policy
        self.trainer = trainer
        self.replay = replay
        self.client_id = client_id
        self.model = model
        self._stag_thresh = stag_thresh
        self._base_crit_stag_thresh = crit_stag_thresh
        self._crit_stag_thresh = crit_stag_thresh
        self.reward_cfg = reward_cfg
        self.mcts_cfg = mcts_cfg

        self._stag_alert_status = EV_TYPE.CLEARED
        self._stag_ep_count = 0
        self._hard_reset_ep_count = 0
        self._hard_reset_count = 0
        self._cur_highscore = 0

        self._reset_window()
        self._epoch = 0

    def get_stats(self) -> dict[str, int | float]:
        trigger_rate = self._triggered / self._calls if self._calls else 0.0
        return {
            DField.MCTS_CALLS: self._calls,
            DField.MCTS_TRIGGERED: self._triggered,
            DField.MCTS_TRIGGER_RATE: round(trigger_rate, 6),
        }

    # Called at the end of each episode
    async def handle_stagnation(self, final_score):
        self.replay.gearbox.incr_epoch()
        self._epoch += 1

        # New high score
        if final_score > self._cur_highscore:
            self._cur_highscore = final_score
            self._stag_ep_count = 0
            self._hard_reset_ep_count = 0

            # Clear stagnation alert (if it's set)
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
            self._stag_alert_status = EV_TYPE.CLEARED

        # Business as usual, increment the counters
        else:
            self._stag_ep_count += 1
            self._hard_reset_ep_count += 1

        # Stagnation threshold reached
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

        # Critical stagnation threshold reached
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

        if self._epoch % 100 == 0:
            payload = self.get_stats()
            payload[DField.STATS_WINDOW] = f"{self._epoch-99}-{self._epoch}"
            await self.event.publish(
                EventMsg(level=DHydraLog.INFO, payload=payload)
            )
            self._reset_window()

    async def maybe_trigger_mcts_burst(
        self, board: GameBoard, score: int
    ) -> None:
        self._calls += 1
        if (
            score >= self.mcts_cfg.score_threshold
            and self.mcts_cfg.rng.random() < self.mcts_cfg.gate_p_value
        ):
            self._triggered += 1
            await self.policy.enable_mcts_burst()

    def _reset_window(self) -> None:
        """
        Reset rolling window counters
        """
        self._calls = 0
        self._triggered = 0
