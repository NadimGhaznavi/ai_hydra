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
    DHydra,
)
from ai_hydra.constants.DNNet import DNetField, DRNN, DLinear
from ai_hydra.constants.DHydra import DMethod
from ai_hydra.constants.DHydraTui import DField

from ai_hydra.server.HydraServer import HydraServer
from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.zmq.HydraMsg import HydraMsg
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.Policy.LookaheadPolicy import LookaheadPolicy
from ai_hydra.utils.SimCfg import SimCfg


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
        self.cfg = SimCfg()
        self.snake: SnakeMgr | None = None
        self._train_mgr = None
        self._methods.update(
            {
                DMethod.HANDSHAKE: self.handshake,
                DGameMethod.RESET_GAME: self.reset_game,
                DGameMethod.START_RUN: self.start_run,
                DGameMethod.STOP_RUN: self.stop_run,
                DGameMethod.UPDATE_CONFIG: self.update_config,
            }
        )
        if self.debug:
            self._methods[DGameMethod.GAME_STEP] = self.game_step

        self._runs: dict[str, asyncio.Task[None]] = {}
        self._client_id = None
        self._sim_running = False
        self._db_mgr = None

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
        from ai_hydra.nnet.LinearTrainer import LinearTrainer

        # from ai_hydra.nnet.UNUSED_RNNTrainer import RNNTrainer
        from ai_hydra.nnet.RNNTrainer import RNNTrainer
        from ai_hydra.nnet.models.LinearModel import LinearModel
        from ai_hydra.nnet.Policy.LinearPolicy import LinearPolicy

        # from ai_hydra.nnet.models.UNUSED_RNNModel import RNNModel
        from ai_hydra.nnet.models.RNNModel import RNNModel
        from ai_hydra.nnet.Policy.RNNPolicy import RNNPolicy
        from ai_hydra.nnet.EpsilonAlgo import EpsilonAlgo
        from ai_hydra.nnet.Policy.EpsilonPolicy import EpsilonPolicy

        from ai_hydra.constants.DNNet import DNetField

        # Hydra-style RNG streams (minted from the existing SnakeMgr)
        _, policy_rng = self.snake.new_rng()
        _, replay_rng = self.snake.new_rng()

        device = torch.device("cpu")  # keep simple; GPU can be passed later

        # NN Model being used
        model_type = self.cfg.get(DNetField.MODEL_TYPE)

        if model_type == DField.LINEAR:
            self.log.debug("Using Linear Model")
            replay = ReplayMemory(rng=replay_rng, log_level=self.log_level)
            model = LinearModel()
            nnet_policy = LinearPolicy(model=model, device=device)
            trainer = LinearTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                gamma=0.9,
                log_level=self.log_level,
            )

        elif model_type == DField.RNN:
            self.log.debug("Using RNN Model")
            replay = ReplayMemory(
                rng=replay_rng, log_level=self.log_level, rnn=True
            )
            model = RNNModel()
            nnet_policy = RNNPolicy(model=model, device=device)
            trainer = RNNTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                gamma=0.9,
                log_level=self.log_level,
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Model + trainer

        # Policy stack
        epsilon_algo = EpsilonAlgo(rng=policy_rng, log_level=self.log_level)
        epsilon_algo.initial_epsilon(self.cfg.get(DNetField.INITIAL_EPSILON))
        epsilon_algo.min_epsilon(self.cfg.get(DNetField.MIN_EPSILON))
        epsilon_algo.decay_rate(self.cfg.get(DNetField.EPSILON_DECAY))

        behaviour_policy = EpsilonPolicy(
            base_policy=nnet_policy, epsilon=epsilon_algo
        )
        policy = LookaheadPolicy(base_policy=behaviour_policy)

        self._train_mgr = TrainMgr(
            snake_mgr=self.snake,
            policy=policy,
            trainer=trainer,
            replay=replay,
            client_id="TrainMgr",
        )
        return self._train_mgr

    async def handshake(self, msg: HydraMsg) -> None:
        """
        When a HydraClient starts, it sends a "handshake".
        """
        try:
            payload = self.cfg.to_dict()
            if self._sim_running:
                payload[DGameField.SIM_RUNNING] = True
            else:
                payload[DGameField.SIM_RUNNING] = False

            reply = HydraMsg(
                sender=self.identity,
                target=msg.sender,
                method=DMethod.HANDSHAKE_REPLY,
                payload=payload,
            )

            if self.mq is not None:
                await self.mq.send(reply)
        except Exception as e:
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"TRACEBACK: {traceback.format_exc()}")

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
        self.log.debug("Starting simulation run")
        self.log.debug(f"Using random seed: {DHydra.RANDOM_SEED}")
        try:
            model_type = self.cfg.get(DNetField.MODEL_TYPE)

            snake = self.snake = SnakeMgr(cfg=self.cfg)
            mq = self.mq
            train_mgr = self._ensure_train_mgr()

            sess = snake.get_session(client_id)
            snake.start_time(datetime.now())

            count = 0

            # Lookahead setting
            lookahead_p = self.cfg.get(DNetField.LOOKAHEAD_P_VAL)
            self.log.debug(f"Setting look ahead p-value to: {lookahead_p}")
            if model_type == DField.LINEAR:
                self.log.debug(
                    f"Look ahead replay memory sample p-value: {DLinear.LOOKAHEAD_SAMPLE_P_VALUE}"
                )
            elif model_type == DField.RNN:
                self.log.debug(
                    f"Look ahead replay memory sample p-value: {DRNN.LOOKAHEAD_SAMPLE_P_VALUE}"
                )
            sess.lookahead_on = sess.rng.random() < lookahead_p

            train_every = 4
            grad_steps = 1
            batch_size = 64

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
                old_state = sess.board.get_state()
                action = train_mgr.policy.select_action(
                    old_state,
                    board=sess.board,
                    lookahead_on=sess.lookahead_on,
                )

                # Advance sim
                state_dict, scores_payload, step_payload, ep_payload = (
                    snake.step(
                        client_id=client_id,
                        action=action,
                    )
                )

                done = state_dict[DGameField.DONE]

                # Episode-end bookkeeping
                if done:
                    sess.epoch += 1
                    count += 1

                    if count % 50 == 0:
                        self.log.info(
                            f"Epoch: {count} - Highscore: {sess.highscore}"
                        )

                    # Epsilon
                    train_mgr.policy.played_game()
                    ep_payload[DNetField.CUR_EPSILON] = (
                        train_mgr.policy.cur_epsilon()
                    )

                    # Lookahead coinflip
                    sess.lookahead_on = sess.rng.random() < lookahead_p
                    ep_payload[DNetField.LOOKAHEAD_ON] = sess.lookahead_on

                    if model_type == DField.RNN:
                        train_mgr.trainer.train_long_memory()

                    # Loss, if available, is loaded into the telemetry here
                    ep_loss = train_mgr.trainer.get_per_ep_loss()
                    if ep_loss is not None:
                        ep_payload[DNetField.EP_LOSS] = ep_loss
                    step_loss = train_mgr.trainer.get_avg_per_step_loss()
                    if step_loss is not None:
                        ep_payload[DNetField.STEP_LOSS] = step_loss

                reward = state_dict[DGameField.REWARD]
                new_state = state_dict[DNetField.NEXT_STATE]

                # Build/store transition (unchanged for now)
                t = Transition(
                    old_state=tuple(old_state),
                    action=int(action),
                    reward=float(reward),
                    new_state=tuple(new_state),
                    done=bool(done),
                )

                train_mgr.replay.append(t=t, lookahead=sess.lookahead_on)

                if model_type != DField.RNN:
                    if sess.step_n % train_every == 0:
                        for _ in range(grad_steps):
                            train_mgr.trainer.train_long_memory(batch_size)

                # Publish
                if mq is not None:
                    # Publish scores at every step
                    await mq.publish_scores(scores_payload)

                    # Publish per epsisode if the episode is "done"
                    if done:
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
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"STACKTRACE: {traceback.format_exc()}")

    async def set_move_delay(self, msg: HydraMsg) -> None:
        delay = msg.payload[DNetField.MOVE_DELAY]
        self.cfg.set(DNetField.MOVE_DELAY, delay)
        self.log.debug(f"Move delay set to: {delay}")

    async def set_per_step_topic(self, msg: HydraMsg) -> None:
        """
        Set the PUB/SUB type to PER_STEP or PER_EPISODE.
        """
        enabled = msg.payload[DNetField.PER_STEP]
        self.cfg.set(DNetField.PER_STEP, enabled)
        self.log.debug(f"Enable per step telemetry: {enabled}")

    async def start_run(self, msg: HydraMsg) -> None:
        self._sim_running = True
        client_id = msg.sender
        self._client_id = client_id
        self.log.debug(f"Received START_RUN, starting the simulation")

        try:

            # Get runtime settings
            self.cfg = SimCfg.from_dict(msg.payload)

            # Setup the PUB/SUB topic publications
            await self.set_per_step_topic(msg)
            await self.set_move_delay(msg)

            # already running?
            if client_id in self._runs and not self._runs[client_id].done():
                payload = {
                    DGameField.OK: False,
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

        except Exception as e:
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"STACKTRACE: {traceback.format_exc()}")

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

    async def update_config(self, msg: HydraMsg) -> None:
        """
        Update settings while a simulation is running
        """
        self.log.debug(f"Received config update: {msg.payload}")
        await self.set_move_delay(msg)
        await self.set_per_step_topic(msg)


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
