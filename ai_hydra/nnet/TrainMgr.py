from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.constants.DGame import DGameField

from ai_hydra.nnet.HydraPolicy import HydraPolicy
from ai_hydra.nnet.Trainer import Trainer
from ai_hydra.nnet.ReplayMemory import ReplayMemory
from ai_hydra.nnet.Transition import Transition


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
        policy: HydraPolicy,
        trainer: Trainer,
        replay: ReplayMemory,
        client_id: str = "trainer",
    ) -> None:
        self.snake_mgr = snake_mgr
        self.policy = policy
        self.trainer = trainer
        self.replay = replay
        self.client_id = client_id

    def train_episode(
        self,
        *,
        batch_size: int = 64,
        train_every: int = 1,
        max_steps: int = 10_000,
    ) -> EpisodeStats:
        sess = self.snake_mgr.reset_session(self.client_id, seed=None)
        episode_id = int(sess.episode_id)

        steps = 0
        score = 0
        reward_total = 0

        loss_sum = 0.0
        loss_n = 0

        done = False
        while not done and steps < max_steps:
            # Pre-step
            board = self.snake_mgr.get_session(self.client_id).board
            state = board.get_state()

            action = int(self.policy.select_action(state))

            # Step env
            result = self.snake_mgr.step(self.client_id, action)

            reward = float(result.get(DGameField.REWARD, 0))
            done = bool(result.get(DGameField.DONE, False))

            # Post-step
            next_board = self.snake_mgr.get_session(self.client_id).board
            next_state = next_board.get_state()

            info = result.get(DGameField.INFO, {})
            score = int(info.get(DGameField.SCORE, score))
            reward_total = int(info.get(DGameField.REWARD_TOTAL, reward_total))

            self.replay.add(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            # Train periodically
            if (steps % train_every) == 0:
                loss = self.trainer.train_step(batch_size=batch_size)
                if loss is not None:
                    loss_sum += float(loss)
                    loss_n += 1

            steps += 1

        return EpisodeStats(
            episode_id=episode_id,
            steps=steps,
            score=score,
            reward_total=reward_total,
            losses=loss_n,
            loss_avg=(loss_sum / loss_n) if loss_n else 0.0,
        )

    def train_n_episodes(
        self,
        *,
        n_episodes: int,
        batch_size: int = 64,
        train_every: int = 1,
        max_steps: int = 10_000,
        print_every: int = 1,
    ) -> tuple[list[EpisodeStats], TrainRunStats]:
        if n_episodes <= 0:
            raise ValueError("n_episodes must be > 0")

        stats_list: list[EpisodeStats] = []

        score_sum = 0
        steps_sum = 0
        reward_sum = 0
        loss_sum = 0.0
        loss_n = 0

        for ep in range(1, n_episodes + 1):
            st = self.train_episode(
                batch_size=batch_size,
                train_every=train_every,
                max_steps=max_steps,
            )
            stats_list.append(st)

            score_sum += st.score
            steps_sum += st.steps
            reward_sum += st.reward_total

            if st.losses:
                loss_sum += st.loss_avg
                loss_n += 1

            if print_every > 0 and (ep % print_every) == 0:
                eps_val = getattr(self.policy, "epsilon", None)
                eps_txt = f"{eps_val():.3f}" if callable(eps_val) else "-"
                print(
                    f"ep {ep:4d} | score {st.score:3d} | steps {st.steps:4d} "
                    f"| R {st.reward_total:6d} | loss {st.loss_avg:8.5f} "
                    f"| eps {eps_txt} | mem {len(self.replay):6d}"
                )

        run = TrainRunStats(
            episodes=n_episodes,
            avg_score=score_sum / n_episodes,
            avg_steps=steps_sum / n_episodes,
            avg_reward_total=reward_sum / n_episodes,
            avg_loss=(loss_sum / loss_n) if loss_n else 0.0,
        )

        return stats_list, run
