# ai_hydra/server/HydraMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations
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
from ai_hydra.constants.DNNet import DNetField
from ai_hydra.constants.DHydra import DMethod
from ai_hydra.constants.DHydraTui import DField

from ai_hydra.server.HydraServer import HydraServer
from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.zmq.HydraMsg import HydraMsg
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.HydraRng import HydraRng
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
        self.hydra_rng: HydraRng | None = None
        self.snake: SnakeMgr | None = None
        self._train_mgr = None
        self._train_mgr_model_type: str | None = None
        self._methods.update(
            {
                DMethod.HANDSHAKE: self.handshake,
                DGameMethod.PAUSE_RUN: self._pause_run,
                DGameMethod.RESET_RUN: self._reset_run,
                DGameMethod.RESUME_RUN: self._resume_run,
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
        self._sim_paused = False

        self._run_loop_task: asyncio.Task[None] | None = None
        self._run_loop_pause_event = asyncio.Event()
        self._run_loop_stop_event = asyncio.Event()

    def _ensure_train_mgr(self):
        """
        Lazily construct TrainMgr and NN components.

        This keeps HydraMgr import graph clean for headless server use,
        and avoids circular imports until training is actually invoked.
        """

        # Late imports to avoid circularities
        import torch

        from ai_hydra.nnet.TrainMgr import TrainMgr

        from ai_hydra.nnet.ATH.ATHMemory import ATHMemory
        from ai_hydra.nnet.SimpleReplayMemory import SimpleReplayMemory

        from ai_hydra.nnet.LinearTrainer import LinearTrainer
        from ai_hydra.nnet.RecurrentTrainer import RecurrentTrainer

        from ai_hydra.nnet.models.GRUModel import GRUModel
        from ai_hydra.nnet.models.LinearModel import LinearModel
        from ai_hydra.nnet.models.RNNModel import RNNModel

        from ai_hydra.nnet.Policy.LinearPolicy import LinearPolicy
        from ai_hydra.nnet.Policy.RecurrentPolicy import RecurrentPolicy
        from ai_hydra.nnet.Policy.EpsilonPolicy import EpsilonPolicy
        from ai_hydra.nnet.Policy.EpsilonNicePolicy import EpsilonNicePolicy

        from ai_hydra.nnet.EpsilonAlgo import EpsilonAlgo
        from ai_hydra.nnet.EpsilonNiceAlgo import EpsilonNiceAlgo

        from ai_hydra.constants.DNNet import DNetField, DLinear, DRNN, DGRU

        model_type = self.cfg.get(DNetField.MODEL_TYPE)
        master_seed = self.cfg.get(DNetField.RANDOM_SEED)

        self.hydra_rng = HydraRng(
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
            master_seed=master_seed,
        )
        master_seed = self.hydra_rng.get_seed()

        # Hydra-style RNG streams
        _, policy_rng = self.hydra_rng.new_rng()
        _, replay_rng = self.hydra_rng.new_rng()

        device = torch.device(
            DField.CPU
        )  # keep simple; GPU can be passed later

        if model_type == DField.LINEAR:
            self.log.debug("Using Linear Model")
            epsilon_nice_p_val = DLinear.NICE_P_VALUE
            replay = SimpleReplayMemory(
                rng=replay_rng,
                log_level=self.log_level,
                pub_func=self.mq.publish_events,
            )
            model = LinearModel(log_level=self.log_level, seed=master_seed)
            model.set_params(
                hidden_size=self.cfg.get(DNetField.HIDDEN_SIZE),
                dropout_p=self.cfg.get(DNetField.DROPOUT_P),
            )
            nnet_policy = LinearPolicy(model=model, device=device)
            trainer = LinearTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                log_level=self.log_level,
            )
            trainer.set_params(
                gamma=self.cfg.get(DNetField.GAMMA),
                batch_size=self.cfg.get(DNetField.BATCH_SIZE),
            )

        elif model_type == DField.RNN:
            self.log.debug("Using RNN Model")
            epsilon_nice_p_val = DRNN.NICE_P_VALUE
            epsilon_nice_steps = DRNN.NICE_STEPS
            model = RNNModel(log_level=self.log_level, seed=master_seed)
            model.set_params(
                hidden_size=self.cfg.get(DNetField.HIDDEN_SIZE),
                dropout_p=self.cfg.get(DNetField.DROPOUT_P),
                rnn_layers=self.cfg.get(DNetField.LAYERS),
            )
            nnet_policy = RecurrentPolicy(model=model, device=device)
            replay = ATHMemory(
                rng=replay_rng,
                log_level=self.log_level,
                pub_func=self.mq.publish_events,
            )
            trainer = RecurrentTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                log_level=self.log_level,
            )
            trainer.set_params(
                tau=self.cfg.get(DNetField.TAU),
                gamma=self.cfg.get(DNetField.GAMMA),
            )

        elif model_type == DField.GRU:
            self.log.debug("Using GRU Model")
            epsilon_nice_p_val = DGRU.NICE_P_VALUE
            epsilon_nice_steps = DGRU.NICE_STEPS
            model = GRUModel(log_level=self.log_level, seed=master_seed)
            model.set_params(
                hidden_size=self.cfg.get(DNetField.HIDDEN_SIZE),
                dropout_p=self.cfg.get(DNetField.DROPOUT_P),
                gru_layers=self.cfg.get(DNetField.LAYERS),
            )
            nnet_policy = RecurrentPolicy(model=model, device=device)
            replay = ATHMemory(
                rng=replay_rng,
                log_level=self.log_level,
                pub_func=self.mq.publish_events,
            )
            trainer = RecurrentTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                log_level=self.log_level,
            )
            trainer.set_params(
                tau=self.cfg.get(DNetField.TAU),
                gamma=self.cfg.get(DNetField.GAMMA),
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.log.debug(f"Model details:")
        self.log.debug(str(model))

        # Policy stack
        epsilon_algo = EpsilonAlgo(
            rng=policy_rng,
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
        )
        epsilon_algo.initial_epsilon(self.cfg.get(DNetField.INITIAL_EPSILON))
        epsilon_algo.min_epsilon(self.cfg.get(DNetField.MIN_EPSILON))
        epsilon_algo.decay_rate(self.cfg.get(DNetField.EPSILON_DECAY))

        epsilon_nice = EpsilonNiceAlgo(
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
            rng=policy_rng,
        )
        epsilon_policy = EpsilonPolicy(
            base_policy=nnet_policy, epsilon=epsilon_algo
        )

        behaviour_policy = EpsilonNicePolicy(
            base_policy=epsilon_policy,
            epsilon_n=epsilon_nice,
            p_value=epsilon_nice_p_val,
            rng=policy_rng,
            steps=epsilon_nice_steps,
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
        )

        self._train_mgr = TrainMgr(
            snake_mgr=self.snake,
            policy=behaviour_policy,
            trainer=trainer,
            replay=replay,
            client_id=DModule.TRAIN_MGR,
            model=model,
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
        )
        self._train_mgr_model_type = model_type
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

            if self._sim_paused:
                payload[DGameField.SIM_PAUSED] = True
            else:
                payload[DGameField.SIM_PAUSED] = False

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

    async def _pause_run(self, msg: HydraMsg) -> None:
        """
        Pause a running simulation.
        """
        self._run_loop_pause_event.set()
        self._sim_paused = True
        reply = HydraMsg(
            sender=self.identity,
            target=msg.sender,
            method=DGameMethod.PAUSE_RUN_REPLY,
        )
        if self.mq is not None:
            await self.mq.send(reply)

    async def _resume_run(self, msg: HydraMsg) -> None:
        """
        Resume a paused simulation.
        """
        self._sim_paused = False
        self._run_loop_pause_event.clear()
        reply = HydraMsg(
            sender=self.identity,
            target=msg.sender,
            method=DGameMethod.RESUME_RUN_REPLY,
        )
        if self.mq is not None:
            await self.mq.send(reply)

    async def _reset_run(self, msg: HydraMsg) -> None:
        """
        Stop the running simulation.
        """
        self._run_loop_stop_event.set()

        reply = HydraMsg(
            sender=self.identity,
            target=msg.sender,
            method=DGameMethod.RESET_RUN_REPLY,
        )
        if self.mq is not None:
            await self.mq.send(reply)

        # Wait for the simulation to end
        while self._sim_running:
            await asyncio.sleep(0.1)

        # Reset the model's weights
        # self._train_mgr.model.reset_weights()
        # Reset the optimizer
        # self._train_mgr.trainer.reset()

        self._sim_paused = False
        self._runs = {}
        self._run_loop_pause_event.clear()
        self._run_loop_stop_event.clear()
        self.log.info("Stopping the simulation and resetting the environment")

    async def _run_loop(self, client_id: str) -> None:
        """
        Runs as fast as possible.
        """
        self.log.debug("Starting simulation run")
        self.log.debug(
            f"Using random seed: {self.cfg.get(DNetField.RANDOM_SEED)}"
        )
        try:
            model_type = self.cfg.get(DNetField.MODEL_TYPE)

            train_mgr = self._ensure_train_mgr()
            snake = self.snake = SnakeMgr(
                cfg=self.cfg,
                log_level=self.log_level,
                hydra_rng=self.hydra_rng,
            )
            mq = self.mq
            train_mgr.policy.reset_episode()

            sess = snake.get_session(client_id)
            snake.start_time(datetime.now())

            count = 0
            train_every = 4
            grad_steps = 1
            batch_size = 64

            try:
                while not self._run_loop_stop_event.is_set():

                    while self._run_loop_pause_event.is_set():
                        await asyncio.sleep(0.1)

                    # Auto-reset when done
                    if sess.done:
                        snake.reset_session(client_id)
                        sess = snake.get_session(client_id)
                        train_mgr.policy.reset_episode()

                    # Decide if we will publish per_step *this tick*
                    want_step = self.cfg.get(DNetField.PER_STEP)

                    # Choose action (server-side)
                    old_state = sess.board.get_state()

                    # EpsilonNice needs the board
                    action = train_mgr.policy.select_action(
                        old_state, sess.board
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

                        if count % 100 == 0:
                            self.log.info(f"Epoch: {count}")

                        # Epsilon
                        await train_mgr.policy.played_game()
                        ep_payload[DNetField.CUR_EPSILON] = (
                            train_mgr.policy.cur_epsilon()
                        )

                        if (
                            model_type == DField.RNN
                            or model_type == DField.GRU
                        ):
                            # train_mgr.replay.append(t=t, final_score=sess.score)
                            await train_mgr.trainer.train_long_memory()
                            # Anti-Stagnation strategy
                            if DNetField.FINAL_SCORE in scores_payload:
                                await train_mgr.handle_stagnation(
                                    scores_payload[DNetField.FINAL_SCORE]
                                )

                        # Loss, if available, is loaded into the telemetry here
                        loss = train_mgr.trainer.get_avg_loss()
                        if loss is not None:
                            ep_payload[DNetField.LOSS] = loss

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

                    if model_type == DField.RNN or model_type == DField.GRU:
                        await train_mgr.replay.append(
                            t=t, final_score=sess.score
                        )
                    else:
                        await train_mgr.replay.append(t=t)

                    if model_type == DField.LINEAR:
                        if sess.step_n % train_every == 0:
                            for _ in range(grad_steps):
                                await train_mgr.trainer.train_long_memory()

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

                self._sim_running = False

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.log.critical(f"ERROR: {e}")
                self.log.critical(f"STACKTRACE: {traceback.format_exc()}")

        except asyncio.CancelledError:
            return
        except Exception as e:
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"STACKTRACE: {traceback.format_exc()}")

    async def start_run(self, msg: HydraMsg) -> None:
        self._sim_running = True
        client_id = msg.sender
        self._client_id = client_id
        self.log.debug(f"Received START_RUN {msg.payload}")

        try:

            # Get runtime settings
            self.cfg = SimCfg.from_dict(msg.payload)

            # already running?
            if client_id in self._runs and not self._runs[client_id].done():
                payload = {
                    DGameField.OK: False,
                    DGameField.INFO: {"status": "already_running"},
                }
            else:
                self._run_loop_task = asyncio.create_task(
                    self._run_loop(client_id)
                )
                self._runs[client_id] = self._run_loop_task
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
        self.log.debug(f"Received config update")
        # Currently only per_step (on/off) and move_delay are settable
        # from the TUI when a sim is running. Also, the TUI can only
        # send the update when the sim is running.
        self.cfg.set(
            key=DNetField.PER_STEP, value=msg.payload[DNetField.PER_STEP]
        )
        self.cfg.set(
            key=DNetField.MOVE_DELAY, value=msg.payload[DNetField.MOVE_DELAY]
        )


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
