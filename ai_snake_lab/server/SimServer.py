"""
ai_snake_lab/server/SimServer.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import asyncio
import zmq
import zmq.asyncio
import numpy as np
import sys
import argparse
from datetime import datetime


from ai_snake_lab.game.ServerGameBoard import ServerGameBoard
from ai_snake_lab.game.SnakeGame import SnakeGame
from ai_snake_lab.ai.AIAgent import AIAgent

from ai_snake_lab.utils.MQHelper import mq_srv_msg
from ai_snake_lab.utils.LabLogger import LabLogger
from ai_snake_lab.utils.DBMgr import DBMgr

from ai_snake_lab.constants.DSim import DSim
from ai_snake_lab.constants.DMQ import DMQ, DMQ_Label
from ai_snake_lab.constants.DReplayMemory import MEM_TYPE
from ai_snake_lab.constants.DDef import DDef
from ai_snake_lab.constants.DAIAgent import DAIAgent
from ai_snake_lab.constants.DLabLogger import DLog


class SimServer:
    """Runs the simulation, sends updates to SimRouter, and receives commands."""

    def __init__(self, router_addr=None):
        # Process command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-r",
            "--router",
            type=str,
            default=DSim.HOST,
            help="IP or hostname of the AI Snake Lab router",
        )
        parser.add_argument("-v", "--verbose", help="Show additional information")
        args = parser.parse_args()
        if args.verbose:
            loglevel = DLog.DEBUG
        else:
            loglevel = DLog.INFO
        # Seed value for random, numpy and pytorch
        seed = DSim.RANDOM_SEED

        # Logging object
        self.log = LabLogger(client_id=DMQ.SIM_SERVER)
        self.log.loglevel(loglevel)

        # SQLite database manager
        self.db_mgr = DBMgr(seed=seed)

        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)  # DEALER talks to ROUTER
        # Set the random seed value *AFTER* we generate a unique ID, otherwise all instances
        # will end up with the same ID!!
        self.identity = f"{DMQ.SIM_SERVER}-{np.random.randint(0,10000)}".encode()
        np.random.seed(DSim.RANDOM_SEED)
        self.socket.setsockopt(zmq.IDENTITY, self.identity)

        self.router_addr = (
            router_addr or f"{DSim.PROTOCOL}://{args.router}:{DSim.MQ_PORT}"
        )
        self.socket.connect(self.router_addr)
        self.log.info(
            f"{DMQ.SIM_SERVER} {DMQ_Label.CONNECTED_TO_ROUTER} {self.router_addr}"
        )

        # Handy shortcut
        self.send_mq = self.socket.send_json

        # Server side game board
        self.game_board = ServerGameBoard(DSim.BOARD_SIZE)

        # The Snake Game
        self.snake_game = SnakeGame(game_board=self.game_board)

        # The AI Agent
        self.agent = AIAgent(db_mgr=self.db_mgr, seed=seed)

        # Set the initial state of the simulation
        self.running = DSim.STOPPED

        # Prepare to run the main training loop as an asyncio task.
        self.stop_event = asyncio.Event()  # set() means "stop"
        self.pause_event = asyncio.Event()  # set() means "paused"
        self.simulator_task = None
        # A stop event for the heartbeat
        self.heartbeat_stop_event = asyncio.Event()
        # Start sending heartbeat messages
        self.heartbeat_task = asyncio.create_task(self.send_heartbeat())

        # Move delay
        self.move_delay = DDef.MOVE_DELAY

        # Current highscore
        self.highscore = 0

        # Full throttle mode. This is engaged when there are no SimClients attached
        # to the router. The move delay is set to zero and game state info is not
        # sent to the router.

        # Configuration needs to be completed before the simulation can start
        self.config = {}
        self.reset_config()

        self.full_throttle = False

    async def handle_requests(self):
        """Continuously listen for commands from the router."""
        count = 0
        while True:
            try:
                frames = await self.socket.recv_multipart()

                if len(frames) == 1:
                    msg_bytes = frames[0]

                elif len(frames) != 2:
                    _, msg_bytes = frames

                else:
                    self.log.error(f"Malformed message: {frames}")

                msg = zmq.utils.jsonapi.loads(msg_bytes)

            except asyncio.CancelledError:
                self.log.info("SimServer shutting down...")
                break

            except Exception as e:
                self.log.error(f"{DMQ_Label.RECEIVE_ERROR}: {e}")
                await asyncio.sleep(1)
                continue

            elem = msg.get(DMQ.ELEM)
            data = msg.get(DMQ.DATA, {})

            # self.log.debug(f"{count} MQ Message: {msg}")
            count += 1

            if elem == DMQ.STATUS:
                # DON'T ack acks!!! Causes an endless zmq echo chamber.
                continue

            elif elem == DMQ.CUR_NUM_CLIENTS:
                if data == 0:
                    if not self.full_throttle:
                        self.log.info("Enabling full throttle mode")
                    self.full_throttle = True
                else:
                    if self.full_throttle:
                        self.log.info("Disabling full throttle mode")
                    self.full_throttle = False

            elif elem == DMQ.DYNAMIC_TRAINING:
                self.agent.dynamic_training(enable_flag=data)
                self.config[DMQ.DYNAMIC_TRAINING] = True
                await self.send_ack()

            elif elem == DMQ.EPSILON_DECAY:
                self.agent.explore.epsilon_decay(epsilon_decay=data)
                self.config[DMQ.EPSILON_DECAY] = True
                await self.send_ack()

            elif elem == DMQ.EPSILON_INITIAL:
                self.agent.explore.initial_epsilon(initial_epsilon=data)
                self.config[DMQ.EPSILON_INITIAL] = True
                await self.send_ack()

            elif elem == DMQ.EPSILON_MIN:
                self.agent.explore.epsilon_min(epsilon_min=data)
                self.config[DMQ.EPSILON_MIN] = True
                await self.send_ack()

            elif elem == DMQ.EXPLORE_TYPE:
                self.agent.set_explore(explore_type=data)
                self.config[DMQ.EXPLORE_TYPE] = True
                await self.send_ack()

            elif elem == DMQ.GET_AVG_LOSS_DATA:
                sim_client = data
                await self.send_mq(
                    mq_srv_msg(
                        DMQ.AVG_LOSS_DATA,
                        [sim_client, self.db_mgr.get_avg_loss_data()],
                    )
                )

            elif elem == DMQ.GET_CUR_HIGHSCORE:
                sim_client = data
                await self.send_mq(
                    mq_srv_msg(DMQ.CUR_HIGHSCORE, [sim_client, self.highscore])
                )

            elif elem == DMQ.GET_GAME_SCORE_DATA:
                sim_client = data
                await self.send_mq(
                    mq_srv_msg(
                        DMQ.OLD_GAME_SCORE_DATA,
                        [sim_client, self.db_mgr.get_game_score_data()],
                    )
                )

            elif elem == DMQ.GET_HIGHSCORE_EVENTS:
                sim_client = data
                await self.send_mq(
                    mq_srv_msg(
                        DMQ.OLD_HIGHSCORE_EVENTS,
                        [sim_client, self.db_mgr.get_highscore_events()],
                    )
                )

            elif elem == DMQ.GET_SIM_STATE:
                # A client is asking for the current state of the simulation
                sim_client = data
                await self.send_mq(
                    mq_srv_msg(DMQ.CUR_SIM_STATE, [sim_client, self.running])
                )

            elif elem == DMQ.LEARNING_RATE:
                self.agent.trainer.set_learning_rate(learning_rate=data)
                self.config[DMQ.LEARNING_RATE] = True
                await self.send_ack()

            elif elem == DMQ.MODEL_TYPE:
                self.agent.model_type(model_type=data)
                self.config[DMQ.MODEL_TYPE] = True
                await self.send_ack()

            elif elem == DMQ.MEM_TYPE:
                self.agent.memory.mem_type(mem_type=data)
                self.config[DMQ.MEM_TYPE] = True
                await self.send_ack()

            elif elem == DMQ.MOVE_DELAY:
                self.config[DMQ.MOVE_DELAY] = True
                self.move_delay = float(data)

            elif elem == DMQ.CMD:
                if data == DMQ.START:
                    if self.running == DSim.STOPPED:
                        self.running = DSim.RUNNING
                        self.stop_event.clear()
                        self.pause_event.clear()
                        if self.simulator_task is None or self.simulator_task.done():
                            self.simulator_task = asyncio.create_task(
                                self.run_simulation()
                            )
                    elif self.running == DSim.PAUSED:
                        self.running = DSim.RUNNING
                        self.pause_event.clear()
                    await self.send_ack()

                elif data == DMQ.PAUSE:
                    if self.running == DSim.RUNNING:
                        self.running = DSim.PAUSED
                        self.pause_event.set()
                    await self.send_ack()

                elif data == DMQ.RESET:
                    # stop simulation task if running
                    self.running = DSim.STOPPED
                    self.stop_event.set()
                    self.pause_event.clear()
                    if self.simulator_task is not None:
                        self.simulator_task.cancel()
                        try:
                            await self.simulator_task
                        except asyncio.CancelledError:
                            pass
                        self.simulator_task = None

                    self.snake_game.reset()
                    self.epoch = 0
                    model = self.agent.model()
                    model.reset_parameters()
                    self.db_mgr.clear_runtime_data()
                    self.reset_config()
                    await self.send_ack()

                elif data == DMQ.RESUME:
                    self.running = DSim.RUNNING
                    self.pause_event.clear()

                elif data == DMQ.STOP:
                    self.running = DSim.STOPPED
                    self.stop_event.set()
                    if self.simulator_task is not None:
                        self.simulator_task.cancel()
                        try:
                            await self.simulator_task
                        except asyncio.CancelledError:
                            pass
                        self.simulator_task = None
                    await self.send_ack()

            else:
                await self.send_error(f"{DMQ_Label.UNKNOWN_COMMAND}: {msg}")

    async def quit(self):
        if self.simulator_task is not None:
            self.simulator_task.cancel()
            try:
                await self.simulator_task
            except asyncio.CancelledError:
                pass
            self.simulator_task = None
        self.heartbeat_stop_event.clear()
        self.stop_event.set()
        self.pause_event.clear()
        self.running = DSim.STOPPED
        await self.socket.disconnect(self.router_addr)
        await self.socket.close()
        await asyncio.sleep(1)
        sys.exit(0)

    def reset_config(self):
        self.config = {
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

    async def send_ack(self):
        msg = {DMQ.SENDER: DMQ.SIM_SERVER, DMQ.ELEM: DMQ.STATUS, DMQ.DATA: DMQ.OK}
        await self.send_mq(msg)

    async def send_error(self, msg):
        await self.send_mq(mq_srv_msg(DMQ.ERROR, msg))

    async def send_heartbeat(self):
        """Periodic heartbeat to let the SimRouter know this client is alive."""
        while not self.heartbeat_stop_event.is_set():
            self.log.debug(
                f"Sending heartbeat: {DMQ.HEARTBEAT}/{self.identity.decode()}"
            )
            await self.send_mq(mq_srv_msg(DMQ.HEARTBEAT, self.identity.decode()))
            await asyncio.sleep(DSim.HEARTBEAT_INTERVAL)

    async def run_simulation(self):
        """Async simulation loop (called as an asyncio.Task)."""

        # Wait for config readiness with a timeout (avoid infinite hang)
        try:
            await self._wait_for_config(timeout=10.0)
        except asyncio.TimeoutError:
            # send an error back and abort
            await self.send_error(
                "Configuration not completed within timeout; aborting start"
            )
            self.running = DSim.STOPPED
            return

        # For convenience in this function
        game_board = self.game_board
        agent = self.agent
        snake_game = self.snake_game

        # Reset the score, highscore and epoch
        score = 0
        highscore = 0
        epoch = 0

        # We use this to calculate the simulation runtime
        start_time = datetime.now()

        # Send initial snake, food and epoch data to the SimRouter
        mq_loc_data = game_board.location_data()
        await self.send_mq(mq_srv_msg(DMQ.SNAKE_HEAD, mq_loc_data[DMQ.SNAKE_HEAD]))
        await self.send_mq(mq_srv_msg(DMQ.SNAKE_BODY, mq_loc_data[DMQ.SNAKE_BODY]))
        await self.send_mq(mq_srv_msg(DMQ.FOOD, mq_loc_data[DMQ.FOOD]))

        # The main training loop
        try:

            while not self.stop_event.is_set():

                # Pause handling
                if self.pause_event.is_set():
                    await asyncio.sleep(0.1)
                    continue

                # RL step
                old_state = game_board.get_state()
                move = agent.get_move(old_state, score)
                reward, game_over, score = snake_game.play_step(move)

                # New highscore!
                if score > highscore:
                    self.highscore = highscore = score
                    # Send the EpsilonN a signal to instantiate a new EpsilonAlgo.
                    # This call is accepted, but ignored by the vanilla EpsilonAlog
                    agent.explore.new_highscore(score=score)
                    elapsed_secs = (datetime.now() - start_time).total_seconds()
                    runtime_str = minutes_to_uptime(elapsed_secs)
                    if not self.full_throttle:
                        await self.send_mq(
                            mq_srv_msg(
                                DMQ.HIGHSCORE_EVENT,
                                [str(epoch), str(score), runtime_str],
                            )
                        )
                    self.db_mgr.add_highscore_event(
                        epoch=epoch, score=score, runtime=runtime_str
                    )
                    self.log.info(f"New highscore at game {epoch}: {highscore}")

                # We're still playing the game...
                if not game_over:
                    # Next RL steps
                    new_state = game_board.get_state()
                    agent.train_short_memory(
                        old_state, move, reward, new_state, game_over
                    )
                    agent.memory.append((old_state, move, reward, new_state, game_over))

                    # Send updated board data to the SimRouter
                    if not self.full_throttle:
                        mq_loc_data = game_board.location_data()
                        await self.send_mq(
                            mq_srv_msg(DMQ.SNAKE_HEAD, mq_loc_data[DMQ.SNAKE_HEAD])
                        )
                        await self.send_mq(
                            mq_srv_msg(DMQ.SNAKE_BODY, mq_loc_data[DMQ.SNAKE_BODY])
                        )
                        await self.send_mq(mq_srv_msg(DMQ.FOOD, mq_loc_data[DMQ.FOOD]))

                        # Send the score
                        await self.send_mq(mq_srv_msg(DMQ.SCORE, str(score)))

                        # Repect the configurable move delay
                        await asyncio.sleep(self.move_delay)

                # Game is over
                else:
                    epoch += 1
                    if epoch % 50 == 0:
                        self.log.info(f"Epoch: {epoch}")

                    ## RL steps
                    # Save the last move to memory
                    agent.memory.append(
                        (old_state, move, reward, new_state, game_over),
                        final_score=score,
                    )
                    # Load the training data from ReplayMemory
                    agent.load_training_data()
                    # Pass the epoch to the AI AGent to support "dynamic training"
                    agent.epoch(epoch)
                    # Execute the long training phase
                    agent.train_long_memory()
                    # Reset the game
                    snake_game.reset()
                    # The EpsilonAlgo needs to know that the game ended to decay epsilon.
                    # Additionally, the EpsilonAlgoN needs the current game score.
                    agent.explore.played_game(score=score)

                    if not self.full_throttle:
                        # Send final location data
                        mq_loc_data = game_board.location_data()
                        await self.send_mq(
                            mq_srv_msg(DMQ.SNAKE_HEAD, mq_loc_data[DMQ.SNAKE_HEAD])
                        )
                        await self.send_mq(
                            mq_srv_msg(DMQ.SNAKE_BODY, mq_loc_data[DMQ.SNAKE_BODY])
                        )
                        await self.send_mq(mq_srv_msg(DMQ.FOOD, mq_loc_data[DMQ.FOOD]))

                        # Send the score
                        await self.send_mq(mq_srv_msg(DMQ.SCORE, str(score)))

                        # Send training metadata to the SimRouter
                        mem_type = agent.memory.mem_type()
                        if mem_type == MEM_TYPE.SHUFFLE:
                            await self.send_mq(
                                mq_srv_msg(DMQ.NUM_FRAMES, str(agent.num_frames()))
                            )
                        elif mem_type == MEM_TYPE.RANDOM_GAME:
                            await self.send_mq(
                                mq_srv_msg(DMQ.GAME_ID, str(agent.game_id()))
                            )

                        # Get the current epsilon value and send it to the SimRouter
                        cur_epsilon_value = agent.explore.epsilon(score=score)
                        if cur_epsilon_value < 0.0001:
                            cur_epsilon = "0.0000"
                        else:
                            cur_epsilon = str(round(cur_epsilon_value, 4))
                        await self.send_mq(
                            mq_srv_msg(DMQ.CUR_EPSILON, str(cur_epsilon))
                        )

                        # Stored memory counter
                        stored_games = self.db_mgr.get_num_games()
                        await self.send_mq(
                            mq_srv_msg(DMQ.STORED_GAMES, str(stored_games))
                        )

                        # Simulation runtime
                        elapsed_secs = (datetime.now() - start_time).total_seconds()
                        runtime_str = minutes_to_uptime(elapsed_secs)
                        await self.send_mq(mq_srv_msg(DMQ.RUNTIME, runtime_str))

                        # Current epoch
                        await self.send_mq(mq_srv_msg(DMQ.EPOCH, str(epoch)))

                        # Average loss per epoch
                        await self.send_mq(
                            mq_srv_msg(
                                DMQ.AVG_EPOCH_LOSS, str(agent.trainer.get_epoch_loss())
                            )
                        )

                        # Model type
                        await self.send_mq(
                            mq_srv_msg(DMQ.MODEL_TYPE, agent.model_type_name())
                        )

                        # Move delay
                        await self.send_mq(mq_srv_msg(DMQ.MOVE_DELAY, self.move_delay))

                        # Memory type
                        await self.send_mq(
                            mq_srv_msg(DMQ.MEM_TYPE, self.agent.memory.mem_type())
                        )

                        # Training loops
                        if self.agent.dynamic_training():
                            loops = max(
                                1,
                                min(epoch // 250, DAIAgent.MAX_DYNAMIC_TRAINING_LOOPS),
                            )
                        else:
                            loops = 1
                        await self.send_mq(mq_srv_msg(DMQ.TRAINING_LOOPS, loops))

                        # Current epsilon
                        await self.send_mq(
                            mq_srv_msg(
                                DMQ.CUR_EPSILON, agent.explore.epsilon(score=score)
                            )
                        )

                        # Runtime
                        await self.send_mq(mq_srv_msg(DMQ.RUNTIME, runtime_str))

                    # Add game score data to the DB
                    self.db_mgr.add_game_score(epoch, score)

                    # Store the average epoch loss in the DB
                    self.db_mgr.add_avg_loss(
                        epoch=epoch, avg_loss=agent.trainer.get_epoch_loss()
                    )
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            # task cancelled (stop/reset) — swallow and exit cleanly
            return

        finally:
            # Ensure we mark running stopped
            self.running = DSim.STOPPED
            self.pause_event.clear()
            self.stop_event.clear()

    async def _wait_for_config(self, timeout: float = 10.0):
        """Wait until all config flags are truthy or timeout."""

        async def all_ready():
            return all(bool(v) for v in self.config.values())

        # poll with timeout
        start = asyncio.get_event_loop().time()
        while True:
            if await all_ready():
                return
            if (asyncio.get_event_loop().time() - start) > timeout:
                raise asyncio.TimeoutError()
            await asyncio.sleep(0.05)


# Helper function
def minutes_to_uptime(seconds: int):
    # Return a string like:
    # 45s
    # 7h 23m
    # 1d 7h 32m
    days, minutes = divmod(int(seconds), 86400)
    hours, minutes = divmod(minutes, 3600)
    minutes, seconds = divmod(minutes, 60)

    if days > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            minutes = f" {minutes}"
        if hours < 10:
            hours = f" {hours}"
        return f"{days}d {hours}h {minutes}m"

    elif hours > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            minutes = f" {minutes}"
        return f"{hours}h {minutes}m"

    elif minutes > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            return f" {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"

    else:
        return f"{seconds}s"


async def main_async():
    sim_server = SimServer()
    await sim_server.handle_requests()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
