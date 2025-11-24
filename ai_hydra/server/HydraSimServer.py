# ai_hydra/server/HydraSimServer.py
#
# AI Snake Lab – HydraSimServer
# Author: Nadim-Daniel Ghaznavi
# Copyright: (c) 2025 Nadim-Daniel Ghaznavi
# License: GPL 3.0
#

import asyncio
from datetime import datetime

from ai_hydra.env.Env import Env
from ai_hydra.policy.Policy import Policy
from ai_hydra.model.Model import Model
from ai_hydra.misc.DataLoader import DataLoader
from ai_hydra.network.MQClient import MQClient

from ai_hydra.constants.DMQ import DMQ
from ai_hydra.constants.DSim import DSim


class HydraSimServer:
    """
    Hydra-based simulation server replacement.
    Fully async. Uses MQClient for messaging.
    Clean separation between server logic and messaging.
    """

    def __init__(self, router_addr=None):
        # Messaging client
        self.mq = MQClient(router=router_addr)

        # Environment & agent setup
        self.env = Env(size=DSim.BOARD_SIZE, light_decay=0.9)
        self.policy = Policy()
        self.model = Model()
        self.data_loader = DataLoader()

        # Simulation control
        self.running = False
        self.pause_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.sim_task = None
        self.move_delay = 0.1
        self.highscore = 0
        self.epoch = 0

        # Configuration flags
        self.config = self._default_config()

    def _default_config(self):
        return {
            DMQ.MOVE_DELAY: False,
            DMQ.MODEL_TYPE: False,
            DMQ.LEARNING_RATE: False,
            DMQ.MEM_TYPE: False,
            DMQ.DYNAMIC_TRAINING: False,
            DMQ.EXPLORE_TYPE: False,
            DMQ.EPSILON_INITIAL: False,
            DMQ.EPSILON_DECAY: False,
            DMQ.EPSILON_MIN: False,
        }

    async def handle_requests(self):
        """Continuously receive MQ messages and handle them."""
        while True:
            msg = await self.mq.recv_multipart()
            if not msg:
                await asyncio.sleep(0.01)
                continue

            # Extract DMQ message fields
            elem = msg.get(DMQ.ELEM)
            data = msg.get(DMQ.DATA, {})

            # Command dispatch
            if elem == DMQ.CMD:
                await self._handle_cmd(data)
            elif elem == DMQ.MOVE_DELAY:
                self.move_delay = float(data)
                await self.mq.send_ack()
            elif elem == DMQ.HEARTBEAT:
                continue  # ignore heartbeat for now
            else:
                await self.mq.send_error(f"Unknown command: {elem}")

    async def _handle_cmd(self, cmd):
        """Handle start/pause/resume/stop/reset commands."""
        if cmd == DMQ.START:
            if not self.running:
                self.running = True
                self.pause_event.clear()
                self.stop_event.clear()
                self.sim_task = asyncio.create_task(self.run_simulation())
        elif cmd == DMQ.PAUSE:
            self.pause_event.set()
        elif cmd == DMQ.RESUME:
            self.pause_event.clear()
        elif cmd == DMQ.STOP:
            self.running = False
            self.stop_event.set()
            if self.sim_task:
                self.sim_task.cancel()
        elif cmd == DMQ.RESET:
            await self._reset_sim()
        await self.mq.send_ack()

    async def _reset_sim(self):
        """Reset environment and agent state."""
        self.running = False
        self.stop_event.set()
        self.pause_event.clear()
        self.epoch = 0
        self.highscore = 0
        # Reset environment/agent/model
        self.env = Env(size=DSim.BOARD_SIZE, light_decay=0.9)
        self.policy = Policy()
        self.model.reset_parameters()
        self.data_loader.clear_runtime_data()
        self.config = self._default_config()
        await self.mq.send_ack()

    async def run_simulation(self):
        """Main async simulation loop."""
        start_time = datetime.now()

        while not self.stop_event.is_set():
            # Pause handling
            if self.pause_event.is_set():
                await asyncio.sleep(0.05)
                continue

            # Simulation step
            state = self.env.get_state()
            action = self.policy.get_action(state)
            reward, done, score = self.env.step(action)
            self.model.train_short_memory(
                state, action, reward, self.env.get_state(), done
            )

            # Update highscore
            if score > self.highscore:
                self.highscore = score
                await self.mq.send_msg(
                    DMQ.HIGHSCORE_EVENT, {"epoch": self.epoch, "score": score}
                )

            # End of game handling
            if done:
                self.epoch += 1
                self.env.reset()
                self.policy.end_game(score)
                # Optionally send summary info
                await self.mq.send_msg(DMQ.SCORE, {"epoch": self.epoch, "score": score})

            await asyncio.sleep(self.move_delay)

    async def quit(self):
        """Clean shutdown."""
        self.stop_event.set()
        if self.sim_task:
            self.sim_task.cancel()
            try:
                await self.sim_task
            except asyncio.CancelledError:
                pass
        await self.mq.quit()


async def main_async(router_addr=None):
    server = HydraSimServer(router_addr=router_addr)
    await server.handle_requests()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--router", type=str, default=None, help="Router address")
    args = parser.parse_args()
    asyncio.run(main_async(router_addr=args.router))


if __name__ == "__main__":
    main()
