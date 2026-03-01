# ai_hydra/server/HydraMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

# ai_hydra/server/HydraMgr.py

from __future__ import annotations
from typing import Any
import asyncio
import argparse
import traceback

from ai_hydra.server.HydraServer import HydraServer
from ai_hydra.server.SnakeMgr import SnakeMgr
from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.constants.DGame import DGameMethod, DGameField
from ai_hydra.constants.DHydra import (
    DHydraLog,
    DModule,
    DHydraLogDef,
    DHydraRouterDef,
    DHydraServerDef,
)
from ai_hydra.constants.DNet import DNetField
from ai_hydra.nnet.Transition import Transition


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
        self.snake = SnakeMgr()
        self._train_mgr = None
        self._methods.update(
            {
                DGameMethod.RESET_GAME: self.reset_game,
                DGameMethod.START_RUN: self.start_run,
                DGameMethod.STOP_RUN: self.stop_run,
            }
        )
        if self.debug:
            self._methods[DGameMethod.GAME_STEP] = self.game_step

        self._runs: dict[str, asyncio.Task[None]] = {}

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
        from ai_hydra.nnet.LinearPolicy import LinearModel, LinearPolicy
        from ai_hydra.nnet.EpsilonAlgo import EpsilonAlgo
        from ai_hydra.nnet.EpsilonPolicy import EpsilonPolicy

        # Hydra-style RNG streams (minted from the existing SnakeMgr)
        _, policy_rng = self.snake.new_rng()
        _, replay_rng = self.snake.new_rng()

        # Replay memory (RAM now, SQLite later)
        replay = ReplayMemory(capacity=50_000, rng=replay_rng)

        # Model + trainer
        device = torch.device("cpu")  # keep simple; GPU can be passed later
        model = LinearModel()
        trainer = Trainer(model, replay, device=device, gamma=0.9)

        # Policy stack
        greedy = LinearPolicy(model=model, device=device)
        eps = EpsilonAlgo(rng=policy_rng)
        policy = EpsilonPolicy(policy=greedy, epsilon=eps)

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
            sess = self.snake.get_session(client_id)
            train_mgr = self._ensure_train_mgr()

            train_start = 1_000
            train_every = 4
            grad_steps = 1
            batch_size = 64

            while True:
                # If episode is done, reset automatically (viewer stays continuous)
                if sess.done:
                    self.snake.reset_session(client_id, seed=None)
                    sess = self.snake.get_session(client_id)
                    sess.epoch += 1

                # Choose action (server-side)
                state = sess.board.get_state()
                action = train_mgr.policy.select_action(state)

                payload = self.snake.step(client_id, action)

                # Game over...
                if payload[DGameField.DONE]:
                    # Currently policy is EpsilonPolicy which calls EpsilonAlgo...
                    train_mgr.policy.played_game()
                    payload.setdefault(DGameField.INFO, {})[
                        DNetField.CUR_EPSILON
                    ] = train_mgr.policy.epsilon()

                # Build/store transition
                t = Transition(
                    state=payload[DNetField.STATE],
                    action=action,
                    reward=payload[DGameField.REWARD],
                    next_state=payload[DNetField.NEXT_STATE],
                    done=payload[DGameField.DONE],
                )
                train_mgr.replay.add(t)

                # Train periodically once replay is "warm"
                loss = None
                if len(train_mgr.replay) >= train_start and (
                    sess.step_n % train_every == 0
                ):
                    for _ in range(grad_steps):
                        loss = await asyncio.to_thread(
                            train_mgr.trainer.train_step, batch_size
                        )

                # TODO: Attach training stats to telemetry

                if self.mq is not None:
                    await self.mq.publish("snake.trainer.snapshot", payload)

                # Yield to event loop so MQ IO stays healthy
                await asyncio.sleep(0.01)

                sess = self.snake.get_session(client_id)

        except asyncio.CancelledError:
            # Normal stop
            return
        except Exception as e:
            # Send one error update, then stop
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

    async def start_run(self, msg: HydraMsg) -> None:
        self._ensure_train_mgr()
        client_id = msg.sender

        # already running?
        if client_id in self._runs and not self._runs[client_id].done():
            payload = {
                DGameField.OK: True,
                DGameField.INFO: {"status": "already_running"},
            }
        else:
            # ensure session exists (or reset if you prefer)
            self.snake.get_session(client_id)

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
