# ai_hydra/server/HydraMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from __future__ import annotations
from typing import Any
import asyncio
import argparse
import traceback
from datetime import datetime

from ai_hydra.constants.DGame import DGameMethod, DGameField
from ai_hydra.constants.DHydra import (
    DHydraLog,
    DModule,
    DHydraLogDef,
    DHydraRouterDef,
    DHydraServerDef,
)
from ai_hydra.constants.DNNet import DNetField, DLookaheadDef
from ai_hydra.constants.DSimCfg import Phase

from ai_hydra.server.HydraServer import HydraServer
from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.Policy.LookaheadPolicy import LookaheadPolicy
from ai_hydra.utils.SimCfg import SimCfg, _UNSET


class HydraMgr(HydraServer):
    def __init__(
        self,
        address: str = "*",
        port: int = DHydraServerDef.PORT,
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        identity: str = DModule.HYDRA_MGR,
        log_level: DHydraLog = DHydraLogDef.DEFAULT_LOG_LEVEL,
        debug: bool = False,
    ) -> None:
        super().__init__(
            address=address,
            port=port,
            router_address=router_address,
            router_port=router_port,
            identity=identity,
            log_level=log_level,
        )

        self.debug = bool(debug)
        self.cfg = SimCfg.init_server()
        self.snake: SnakeMgr | None = None
        self._train_mgr = None
        self._methods.update(
            {
                DGameMethod.RESET_GAME: self.reset_game,
                DGameMethod.START_RUN: self.start_run,
                DGameMethod.STOP_RUN: self.stop_run,
                DGameMethod.PUB_TYPE: self.set_per_step_topic,
            }
        )
        if self.debug:
            self._methods[DGameMethod.GAME_STEP] = self.game_step

        self._runs: dict[str, asyncio.Task[None]] = {}
        self._client_id = None

    def _ensure_train_mgr(self):
        """
        Lazily construct TrainMgr and NN components.

        This keeps HydraMgr import graph clean for headless server use,
        and avoids circular imports until training is actually invoked.
        """
        if self._train_mgr is not None:
            return self._train_mgr

        # Late imports to avoid circularities
        import torch

        from ai_hydra.nnet.TrainMgr import TrainMgr
        from ai_hydra.nnet.ReplayMemory import ReplayMemory
        from ai_hydra.nnet.Trainer import Trainer
        from ai_hydra.nnet.Policy.LinearPolicy import LinearPolicy
        from ai_hydra.nnet.models.LinearModel import LinearModel
        from ai_hydra.nnet.EpsilonAlgo import EpsilonAlgo
        from ai_hydra.nnet.Policy.EpsilonPolicy import EpsilonPolicy

        # Hydra-style RNG streams (minted from the existing SnakeMgr)
        _, policy_rng = self.snake.new_rng()
        _, replay_rng = self.snake.new_rng()

        # Replay memory (RAM now, SQLite later)
        replay = ReplayMemory(capacity=50_000, rng=replay_rng)

        # Model + trainer
        device = torch.device("cpu")  # keep simple; GPU can be passed later
        model = LinearModel()
        trainer = Trainer(model=model, replay=replay, device=device, gamma=0.9)

        # Policy stack
        nnet_policy = LinearPolicy(model=model, device=device)
        epsilon_schedule = EpsilonAlgo(rng=policy_rng)
        behaviour_policy = EpsilonPolicy(
            base_policy=nnet_policy, epsilon=epsilon_schedule
        )
        policy = LookaheadPolicy(base_policy=behaviour_policy)

        self._train_mgr = TrainMgr(
            snake_mgr=self.snake,
            policy=policy,
            trainer=trainer,
            replay=replay,
            client_id="trainer",
        )
        return self._train_mgr

    async def game_step(self, msg: HydraMsg) -> None:
        """
        Apply one action to the sender's session and reply with step result.
        """
        try:
            action = msg.payload[DGameField.ACTION]
            payload = self.snake.step(msg.sender, int(action))
        except KeyError:
            payload: dict[str, Any] = {
                DGameField.OK: False,
                DGameField.ERROR: DGameField.MISSING_ACTION,
            }
        except Exception as e:
            payload = {
                DGameField.OK: False,
                DGameField.ERROR: DGameField.INVALID_ACTION,
                DGameField.INFO: {DGameField.REASON: str(e)},
            }

        reply = HydraMsg(
            sender=self.identity,
            target=msg.sender,
            method=DGameMethod.STEP,
            payload=payload,
        )
        if self.mq is not None:
            await self.mq.send(reply)

    async def reset_game(self, msg: HydraMsg) -> None:
        """
        Reset (or create) the sender's snake session and reply with snapshot.
        """
        seed = msg.payload.get(DGameField.SEED)
        payload = self.snake.reset(
            msg.sender, seed=int(seed) if seed is not None else None
        )

        reply = HydraMsg(
            sender=self.identity,
            target=msg.sender,
            method=DGameMethod.RESET_GAME,
            payload=payload,
        )
        if self.mq is not None:
            await self.mq.send(reply)

    async def _run_loop(self, client_id: str) -> None:
        """
        Runs as fast as possible.
        """
        try:
            snake = self.snake = SnakeMgr(cfg=self.cfg)
            mq = self.mq
            train_mgr = self._ensure_train_mgr()

            sess = snake.get_session(client_id)
            snake.start_time(datetime.now())

            # Training vars: Should be moved out of this loop
            train_start = 1_000
            train_every = 4
            grad_steps = 1
            batch_size = 64
            count = 0

            # Lookahead setting
            lookahead_p = DLookaheadDef.PROBABILITY
            sess.lookahead_on = sess.rng.random() < lookahead_p

            while True:
                # Auto-reset when done
                if sess.done:
                    snake.reset_session(client_id, seed=None)
                    sess = snake.get_session(client_id)

                # Decide if we will publish per_step *this tick*
                # If NO BOARD, we don't even build the step payload.
                want_step = (mq is not None) and self.cfg.get(
                    DNetField.PER_STEP
                )

                # Choose action (server-side)
                state = sess.board.get_state()
                action = train_mgr.policy.select_action(
                    state,
                    board=sess.board,
                    lookahead_on=sess.lookahead_on,
                )

                # Advance sim
                ep_payload, step_payload = snake.step(
                    client_id=client_id,
                    action=action,
                )

                done = ep_payload[DGameField.DONE]

                # Episode-end bookkeeping
                if done:
                    sess.epoch += 1
                    count += 1

                    if count % 50 == 0:
                        self.log.info(
                            f"Epoch: {count} - Highscore: {sess.highscore}"
                        )

                    info = ep_payload.setdefault(DGameField.INFO, {})

                    # Epsilon
                    train_mgr.policy.played_game()
                    info[DNetField.CUR_EPSILON] = (
                        train_mgr.policy.cur_epsilon()
                    )

                    # Lookahead coinflip
                    if sess.epoch % 10 == 0:
                        sess.lookahead_on = sess.rng.random() < lookahead_p
                    info[DNetField.LOOKAHEAD_ON] = sess.lookahead_on

                    # Loss snapshot
                    loss_snap = train_mgr.trainer.get_loss()
                    if loss_snap is not None:
                        info[DNetField.LOSS] = loss_snap

                # Build/store transition (unchanged for now)
                t = Transition(
                    state=ep_payload[DNetField.STATE],
                    action=action,
                    reward=ep_payload[DGameField.REWARD],
                    next_state=ep_payload[DNetField.NEXT_STATE],
                    done=done,
                )
                train_mgr.replay.add(t)

                # Train periodically once replay is warm
                if len(train_mgr.replay) >= train_start and (
                    sess.step_n % train_every == 0
                ):
                    for _ in range(grad_steps):
                        await asyncio.to_thread(
                            train_mgr.trainer.train_step, batch_size
                        )

                # Publish
                if mq is not None:
                    await mq.publish_per_episode(ep_payload)
                    # publish per_step only if enabled and payload exists
                    if want_step and step_payload:
                        await mq.publish_per_step(step_payload)

                # Yield to event loop so MQ IO stays healthy
                delay = (
                    0.0
                    if not want_step
                    else self.cfg.get(DNetField.MOVE_DELAY)
                )
                await asyncio.sleep(delay)

        except asyncio.CancelledError:
            return
        except Exception as e:
            err = HydraMsg(
                sender=self.identity,
                target=client_id,
                method=DGameMethod.UPDATE,
                payload={
                    DGameField.OK: False,
                    DGameField.ERROR: "run_loop_failed",
                    DGameField.INFO: {DGameField.REASON: str(e)},
                },
            )
            if self.mq is not None:
                await self.mq.send(err)
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"TRACEBACK: {traceback.format_exc()}")

    async def set_move_delay(self, msg: HydraMsg) -> None:
        delay = msg.payload[DNetField.MOVE_DELAY]
        self.cfg.apply(
            payload={DNetField.MOVE_DELAY: delay}, phase=Phase.RUNTIME
        )
        self.log.debug(f"Move delay is: {delay}")

    async def set_per_step_topic(self, msg: HydraMsg) -> None:
        """
        Set the PUB/SUB type to PER_STEP or PER_EPISODE.
        """
        enabled = msg.payload[DNetField.PER_STEP]
        self.cfg.apply(
            payload={DNetField.PER_STEP: enabled}, phase=Phase.RUNTIME
        )
        self.log.debug(f"Enable per step telemetry: {enabled}")

    async def start_run(self, msg: HydraMsg) -> None:
        client_id = msg.sender
        self._client_id = client_id

        await self.set_per_step_topic(msg)
        await self.set_move_delay(msg)

        # already running?
        if client_id in self._runs and not self._runs[client_id].done():
            payload = {
                DGameField.OK: True,
                DGameField.INFO: {"status": "already_running"},
            }
        else:
            task = asyncio.create_task(self._run_loop(client_id))
            self._runs[client_id] = task
            payload = {
                DGameField.OK: True,
                DGameField.INFO: {"status": "started"},
            }

        reply = HydraMsg(
            sender=self.identity,
            target=client_id,
            method=DGameMethod.START_RUN,
            payload=payload,
        )
        if self.mq is not None:
            await self.mq.send(reply)

    async def stop_run(self, msg: HydraMsg) -> None:
        client_id = msg.sender

        task = self._runs.get(client_id)
        if task and not task.done():
            task.cancel()
            payload = {
                DGameField.OK: True,
                DGameField.INFO: {"status": "stopping"},
            }
        else:
            payload = {
                DGameField.OK: True,
                DGameField.INFO: {"status": "not_running"},
            }

        reply = HydraMsg(
            sender=self.identity,
            target=client_id,
            method=DGameMethod.STOP_RUN,
            payload=payload,
        )
        if self.mq is not None:
            await self.mq.send(reply)

    def train_n_episodes(self, *, n_episodes: int, **kwargs):
        """
        Local training entrypoint. Does not touch MQ.
        """
        tm = self._ensure_train_mgr()
        return tm.train_n_episodes(n_episodes=n_episodes, **kwargs)


async def amain() -> None:
    p = argparse.ArgumentParser(description="AI Hydra Manager")
    p.add_argument("--address", default="*", help="Bind address")
    p.add_argument("--port", type=int, default=DHydraServerDef.PORT)
    p.add_argument("--router-address", default=DHydraRouterDef.HOSTNAME)
    p.add_argument("--router-port", type=int, default=DHydraRouterDef.PORT)
    p.add_argument("--identity", default=DModule.HYDRA_MGR)
    p.add_argument("--log-level", default=DHydraLogDef.DEFAULT_LOG_LEVEL)
    args = p.parse_args()

    server = HydraMgr(
        address=args.address,
        port=args.port,
        router_address=args.router_address,
        router_port=args.router_port,
        identity=args.identity,
        log_level=args.log_level,
    )

    await server.run()


def main() -> None:
    try:
        asyncio.run(amain())
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
