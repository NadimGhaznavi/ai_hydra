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
from ai_hydra.game.GameHelper import RewardCfg


class HydraMgr(HydraServer):
    def __init__(
        self,
        address: str = "*",
        port: int = DHydraServerDef.PORT,
        pub_port: int = DHydraServerDef.PUB_PORT,
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        router_hb_port: int = DHydraRouterDef.HEARTBEAT_PORT,
        identity: str = DModule.HYDRA_MGR,
        log_level: DHydraLog = DHydraLogDef.DEFAULT_LOG_LEVEL,
    ) -> None:
        super().__init__(
            address=address,
            port=port,
            pub_port=pub_port,
            router_address=router_address,
            router_port=router_port,
            router_hb_port=router_hb_port,
            identity=identity,
            log_level=log_level,
        )

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
        from ai_hydra.nnet.Policy.BehaviourPolicy import BehaviourPolicy

        from ai_hydra.nnet.EpsilonAlgo import EpsilonAlgo
        from ai_hydra.nnet.EpsilonNiceAlgo import EpsilonNiceAlgo

        from ai_hydra.mcts.Node import MCTSConfig

        from ai_hydra.constants.DNNet import DNetField

        model_type = self.cfg.get(DNetField.MODEL_TYPE)
        master_seed = self.cfg.get(DNetField.RANDOM_SEED)

        self.hydra_rng = HydraRng(
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
            master_seed=master_seed,
        )
        master_seed = self.hydra_rng.get_seed()

        # Hydra-style RNG streams
        _, epsilon_rng = self.hydra_rng.new_rng()
        _, nice_rng = self.hydra_rng.new_rng()
        _, mcts_rng = self.hydra_rng.new_rng()
        _, replay_rng = self.hydra_rng.new_rng()

        if torch.cuda.is_available():
            device = torch.device(DField.CUDA)
        elif torch.backends.mps.is_available():
            device = torch.device(DField.MPS)
        else:
            device = torch.device(DField.CPU)

        replay = ATHMemory(
            rng=replay_rng,
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
            max_buckets=self.cfg.get(DNetField.MAX_BUCKETS),
            max_gear=self.cfg.get(DNetField.MAX_GEAR),
            max_training_frames=self.cfg.get(DNetField.MAX_TRAINING_FRAMES),
            max_frames=self.cfg.get(DNetField.MAX_FRAMES),
            upshift_count_thresh=self.cfg.get(
                DNetField.UPSHIFT_COUNT_THRESHOLD
            ),
            downshift_count_thresh=self.cfg.get(
                DNetField.DOWNSHIFT_COUNT_THRESHOLD
            ),
            num_cooldown_eps=self.cfg.get(DNetField.NUM_COOLDOWN_EPISODES),
        )

        # Linear Model
        if model_type == DField.LINEAR:
            self.log.info("Using Linear Model")
            model = LinearModel(log_level=self.log_level, seed=master_seed)
            model.set_params(
                hidden_size=self.cfg.get(DNetField.HIDDEN_SIZE),
                dropout_p=self.cfg.get(DNetField.DROPOUT_P),
                layers=self.cfg.get(DNetField.LAYERS),
            )

            trainer = LinearTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                log_level=self.log_level,
            )
            nnet_policy = LinearPolicy(model=model, device=device)

        # RNN Model
        elif model_type == DField.RNN:
            self.log.info("Using RNN Model")
            model = RNNModel(log_level=self.log_level, seed=master_seed)

        # GRU Model
        elif model_type == DField.GRU:
            self.log.info("Using GRU Model")
            model = GRUModel(log_level=self.log_level, seed=master_seed)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Recurrent (GRU or RNN) model
        if model_type == DField.RNN or model_type == DField.GRU:
            model.set_params(
                hidden_size=self.cfg.get(DNetField.HIDDEN_SIZE),
                dropout_p=self.cfg.get(DNetField.DROPOUT_P),
                layers=self.cfg.get(DNetField.LAYERS),
            )

            trainer = RecurrentTrainer(
                model=model,
                replay=replay,
                lr=self.cfg.get(DNetField.LEARNING_RATE),
                device=device,
                log_level=self.log_level,
            )
            nnet_policy = RecurrentPolicy(model=model, device=device)

        trainer.set_params(
            tau=self.cfg.get(DNetField.TAU),
            gamma=self.cfg.get(DNetField.GAMMA),
        )
        self.log.debug(f"Model details:")
        self.log.debug(str(model))

        # Policy stack
        epsilon_algo = EpsilonAlgo(
            rng=epsilon_rng,
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
        )
        epsilon_algo.initial_epsilon(self.cfg.get(DNetField.INITIAL_EPSILON))
        epsilon_algo.min_epsilon(self.cfg.get(DNetField.MIN_EPSILON))
        epsilon_algo.decay_rate(self.cfg.get(DNetField.EPSILON_DECAY))

        epsilon_nice = EpsilonNiceAlgo(
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
            rng=nice_rng,
        )
        epsilon_policy = EpsilonPolicy(
            base_policy=nnet_policy, epsilon=epsilon_algo
        )

        reward_cfg = RewardCfg(
            values={
                DGameField.EMPTY: self.cfg.get(DNetField.EMPTY_MOVE_REWARD),
                DGameField.FOOD: self.cfg.get(DNetField.FOOD_REWARD),
                DGameField.WALL: self.cfg.get(DNetField.COLLISION_PENALTY),
                DGameField.SNAKE: self.cfg.get(DNetField.COLLISION_PENALTY),
                DGameField.MAX_MOVES: self.cfg.get(
                    DNetField.MAX_MOVES_PENALTY
                ),
                DGameField.CLOSER_TO_FOOD: self.cfg.get(
                    DNetField.CLOSER_TO_FOOD
                ),
                DGameField.FURTHER_FROM_FOOD: self.cfg.get(
                    DNetField.FURTHER_FROM_FOOD
                ),
            }
        )

        # Monte Carlo Tree Search config
        mcts_cfg = MCTSConfig(
            reward_cfg=reward_cfg,
            mmm=self.cfg.get(DNetField.MAX_MOVES_MULTIPLIER),
            food_ends_episode=False,
            search_depth=self.cfg.get(DNetField.MCTS_DEPTH),
            explore_p_value=self.cfg.get(DNetField.MCTS_EXPLORE_P_VALUE),
            gate_p_value=self.cfg.get(DNetField.MCTS_GATE_P_VALUE),
            iterations=self.cfg.get(DNetField.MCTS_ITER),
            rng=mcts_rng,
            score_threshold=self.cfg.get(DNetField.MCTS_SCORE_THRESH),
            steps=self.cfg.get(DNetField.MCTS_STEPS),
        )

        behaviour_policy = BehaviourPolicy(
            base_policy=epsilon_policy,
            epsilon_n=epsilon_nice,
            nice_p_value=self.cfg.get(DNetField.NICE_P_VALUE),
            nice_rng=nice_rng,
            nice_steps=self.cfg.get(DNetField.NICE_STEPS),
            log_level=self.log_level,
            pub_func=self.mq.publish_events,
            mcts_cfg=mcts_cfg,
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
            stag_thresh=self.cfg.get(DNetField.MAX_STAGNANT_EPISODES),
            crit_stag_thresh=self.cfg.get(DNetField.MAX_HARD_RESET_EPISODES),
            reward_cfg=reward_cfg,
            mcts_cfg=mcts_cfg,
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
            train_mgr = self._ensure_train_mgr()

            snake = self.snake = SnakeMgr(
                cfg=self.cfg,
                log_level=self.log_level,
                hydra_rng=self.hydra_rng,
                reward_cfg=train_mgr.reward_cfg,
                mmm=self.cfg.get(DNetField.MAX_MOVES_MULTIPLIER),
            )
            mq = self.mq
            train_mgr.policy.reset_episode()

            sess = snake.get_session(client_id)
            snake.start_time(datetime.now())

            count = 0

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

                    # See if we're engaging Monte Carlo tree search
                    await train_mgr.maybe_trigger_mcts_burst(
                        board=sess.board, score=sess.score
                    )

                    # EpsilonNice and Monte Carlo tree search needs the board
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

                    await train_mgr.replay.append(t=t, final_score=sess.score)

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


async def amain(
    address,
    port,
    pub_port,
    router_address,
    router_port,
    router_hb_port,
    identity,
    log_level,
) -> None:

    server = HydraMgr(
        address=address,
        port=port,
        pub_port=pub_port,
        router_address=router_address,
        router_port=router_port,
        router_hb_port=router_hb_port,
        identity=identity,
        log_level=log_level,
    )

    await server.run()


def main() -> None:
    p = argparse.ArgumentParser(description="AI Hydra Manager")
    p.add_argument("--address", default="*", help="Bind address")
    p.add_argument(
        "--port",
        type=int,
        default=DHydraServerDef.PORT,
        help=f"Control port ({DHydraServerDef.PORT})",
    )
    p.add_argument(
        "--pub-port",
        type=int,
        default=DHydraServerDef.PUB_PORT,
        help=f"Control port ({DHydraServerDef.PUB_PORT})",
    )
    p.add_argument(
        "--router-address",
        default=DHydraRouterDef.HOSTNAME,
        help=f"Router hostname/IP address ({DHydraRouterDef.HOSTNAME})",
    )
    p.add_argument(
        "--router-port",
        type=int,
        default=DHydraRouterDef.PORT,
        help=f"Router port ({DHydraRouterDef.PORT})",
    )
    p.add_argument(
        "--router-hb-port",
        type=int,
        default=DHydraRouterDef.HEARTBEAT_PORT,
        help=f"Router heartbeat port ({DHydraRouterDef.HEARTBEAT_PORT})",
    )
    p.add_argument(
        "--identity",
        default=DModule.HYDRA_MGR,
        help=f"ZeroMQ identity ({DModule.HYDRA_MGR})",
    )
    p.add_argument(
        "--log-level",
        default=DHydraLogDef.DEFAULT_LOG_LEVEL,
        help=f"Default log level ({str(DHydraLogDef.DEFAULT_LOG_LEVEL)})",
    )
    args = p.parse_args()
    try:
        asyncio.run(
            amain(
                address=args.address,
                port=args.port,
                pub_port=args.pub_port,
                router_address=args.router_address,
                router_port=args.router_port,
                router_hb_port=args.router_hb_port,
                identity=args.identity,
                log_level=args.log_level,
            )
        )
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
