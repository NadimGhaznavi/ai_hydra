"""
ai_snake_lab/ui/SimClient.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import argparse
import sys, os
import asyncio
import zmq
import zmq.asyncio
import numpy as np
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Label, Input, Button, Static, Log, Select, Checkbox
from textual.containers import Vertical, Horizontal
from textual.theme import Theme
from textual.reactive import var

from ai_snake_lab.constants.DDef import DDef
from ai_snake_lab.constants.DEpsilon import DEpsilon
from ai_snake_lab.constants.DFile import DFile
from ai_snake_lab.constants.DLayout import DLayout
from ai_snake_lab.constants.DLabels import DLabel
from ai_snake_lab.constants.DReplayMemory import MEM_TYPE, MEM
from ai_snake_lab.constants.DSim import DSim
from ai_snake_lab.constants.DModelL import DModelL
from ai_snake_lab.constants.DModelLRNN import DModelRNN
from ai_snake_lab.constants.DMQ import DMQ, DMQ_Label
from ai_snake_lab.constants.DDir import DDir
from ai_snake_lab.constants.DLabLogger import DLog

from ai_snake_lab.game.ClientGameBoard import ClientGameBoard
from ai_snake_lab.utils.MQHelper import mq_cli_msg
from ai_snake_lab.utils.LabLogger import LabLogger

from ai_snake_lab.ui.TabbedPlots import TabbedPlots


SNAKE_LAB_THEME = Theme(
    name=DLayout.SNAKE_LAB_THEME,
    primary="#88C0D0",
    secondary="#1f6a83ff",
    accent="#B48EAD",
    foreground="#31b8e6",
    background="black",
    success="#A3BE8C",
    warning="#EBCB8B",
    error="#BF616A",
    surface="#111111",
    panel="#000000",
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "footer-key-foreground": "#88C0D0",
        "input-selection-background": "#81a1c1 35%",
    },
)

# A list of tuples, for the TUI's model selection drop down menu (Select widget).
MODEL_TYPES: list = [
    (DLabel.LINEAR_MODEL, DModelL.MODEL),
    (DLabel.RNN_MODEL, DModelRNN.MODEL),
]

# A list of tuples, for the TUI's exploration selection drop down menu (Select widget).
EXPLORATION_TYPES: list = [
    (DLabel.EPSILON_N, DEpsilon.EPSILON_N),
    (DLabel.EPSILON, DEpsilon.EPSILON),
]

# Model type lookup table
MODEL_TYPE_TABLE: dict = {
    DModelL.MODEL: DLabel.LINEAR_MODEL,
    DModelRNN.MODEL: DLabel.RNN_MODEL,
    DLabel.N_SLASH_A: DLabel.N_SLASH_A,
    None: DLabel.N_SLASH_A,
}


class SimClient(App):
    """A Textual app that has an AI Agent playing the Snake Game."""

    TITLE = DLabel.APP_TITLE
    CSS_PATH = DFile.CSS_FILE

    # Simulation data
    avg_epoch_loss = var(0)
    cur_epsilon = var(DLabel.N_SLASH_A)
    epoch = var(DLabel.N_SLASH_A)
    game_id = var(DLabel.N_SLASH_A)
    highscore = var(DLabel.N_SLASH_A)
    highscore_event = var([])
    mem_type = var(DLabel.N_SLASH_A)
    model_type = var(DLabel.N_SLASH_A)
    move_delay = var(DLabel.N_SLASH_A)
    num_frames = var(DLabel.N_SLASH_A)
    runtime = var(DLabel.N_SLASH_A)
    score = var(DLabel.N_SLASH_A)
    stored_games = var(DLabel.N_SLASH_A)
    training_loops = var(DLabel.N_SLASH_A)

    def __init__(self, router, loglevel, logfile) -> None:
        """Constructor"""
        super().__init__()

        # Setup logging
        self.lablog = LabLogger(
            client_id=DMQ.SIM_CLIENT, log_file=logfile, to_console=False
        )
        self.lablog.loglevel(loglevel)

        # The game board, game, agent and epsilon algorithm object
        self.game_board = ClientGameBoard(20, id=DLayout.GAME_BOARD)

        # A dictionary to hold runtime statistics
        self.stats = {
            DSim.GAME_SCORE: {
                DSim.GAME_NUM: [],
                DSim.GAME_SCORE: [],
            }
        }

        # Setup the connection to the SimRouter
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.identity_str = f"{DMQ.SIM_CLIENT}-{np.random.randint(0,10000)}"
        self.identity = self.identity_str.encode()
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        self.router_addr = f"{DSim.PROTOCOL}://{router}:{DSim.MQ_PORT}"
        self.stop_event = asyncio.Event()  # To control the simulation loop
        self.heartbeat_stop_event = asyncio.Event()  # To control the heartbeat loop
        self.socket.connect(self.router_addr)

        # Handy alias
        self.send_mq = self.socket.send_json

        # Flag
        self.running = True

        # Current simulation state
        self.cur_sim_state = None

    async def action_quit(self) -> None:
        """Quit the application."""
        await super().action_quit()

    async def check_sim_state(self):
        # Get the current simulation state
        await self.send_mq(mq_cli_msg(DMQ.GET_SIM_STATE, self.identity_str))

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""

        # Title bar
        yield Label(DLabel.APP_TITLE, id=DLayout.TITLE)

        # Configuration Settings
        yield Vertical(
            Horizontal(
                Label(
                    f"{DLabel.EPSILON_INITIAL}",
                    classes=DLayout.LABEL_SETTINGS,
                ),
                Input(
                    restrict=f"0.[0-9]*",
                    compact=True,
                    id=DLayout.EPSILON_INITIAL,
                    classes=DLayout.INPUT_10,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.EPSILON_DECAY}",
                    classes=DLayout.LABEL_SETTINGS,
                ),
                Input(
                    restrict=f"0.[0-9]*",
                    compact=True,
                    id=DLayout.EPSILON_DECAY,
                    classes=DLayout.INPUT_10,
                ),
            ),
            Horizontal(
                Label(f"{DLabel.EPSILON_MIN}", classes=DLayout.LABEL_SETTINGS),
                Input(
                    restrict=f"0.[0-9]*",
                    compact=True,
                    id=DLayout.EPSILON_MIN,
                    classes=DLayout.INPUT_10,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.MOVE_DELAY}",
                    classes=DLayout.LABEL_SETTINGS,
                ),
                Input(
                    restrict=f"[0-9]*.[0-9]*",
                    compact=True,
                    id=DLayout.MOVE_DELAY,
                    classes=DLayout.INPUT_10,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.LEARNING_RATE}",
                    classes=DLayout.LABEL_SETTINGS,
                ),
                Input(
                    restrict=f"[0-9]*.[0-9]*",
                    compact=True,
                    id=DLayout.LEARNING_RATE,
                    classes=DLayout.INPUT_10,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.DYNAMIC_TRAINING}",
                    classes=DLayout.LABEL_SETTINGS,
                ),
                Checkbox(
                    id=DLayout.DYNAMIC_TRAINING,
                    classes=DLayout.INPUT_10,
                    compact=True,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.MODEL_TYPE}",
                    classes=DLayout.LABEL_SETTINGS_19,
                ),
                Select(
                    MODEL_TYPES,
                    compact=True,
                    id=DLayout.MODEL_TYPE,
                    allow_blank=False,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.MEM_TYPE}",
                    classes=DLayout.LABEL_SETTINGS_12,
                ),
                Select(
                    MEM_TYPE.MEMORY_TYPES,
                    compact=True,
                    id=DLayout.MEM_TYPE,
                    allow_blank=False,
                ),
            ),
            Horizontal(
                Label(
                    f"{DLabel.EXPLORATION}",
                    classes=DLayout.LABEL_SETTINGS_12,
                ),
                Select(
                    EXPLORATION_TYPES,
                    compact=True,
                    id=DLayout.EXPLORATION,
                    allow_blank=False,
                ),
            ),
            id=DLayout.SETTINGS_BOX,
        )

        # The Snake Game
        yield Vertical(
            self.game_board,
            id=DLayout.GAME_BOX,
        )

        # Runtime values
        yield Vertical(
            Horizontal(
                Label(f"{DLabel.MODEL_TYPE}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.CUR_MODEL_TYPE),
            ),
            Horizontal(
                Label(f"{DLabel.MOVE_DELAY}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.CUR_MOVE_DELAY),
            ),
            Horizontal(
                Label(f"{DLabel.MEM_TYPE}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.CUR_MEM_TYPE),
            ),
            Horizontal(
                Label(f"{DLabel.STORED_GAMES}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.STORED_GAMES),
            ),
            Horizontal(
                Label(f"{DLabel.TRAINING_LOOPS}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.TRAINING_LOOPS),
            ),
            Horizontal(
                Label(
                    f"{DLabel.TRAINING_GAME_ID}",
                    classes=DLayout.LABEL,
                    id=DLayout.TRAINING_ID_LABEL,
                ),
                Label(DLabel.N_SLASH_A, id=DLayout.CUR_TRAINING_GAME_ID),
            ),
            Horizontal(
                Label(f"{DLabel.CUR_EPSILON}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.CUR_EPSILON),
            ),
            Horizontal(
                Label(f"{DLabel.RUNTIME}", classes=DLayout.LABEL),
                Label(DLabel.N_SLASH_A, id=DLayout.RUNTIME),
            ),
            id=DLayout.RUNTIME_BOX,
        )

        # Buttons
        yield Vertical(
            Horizontal(
                Button(label=DLabel.START, id=DLayout.BUTTON_START, compact=True),
                Button(label=DLabel.PAUSE, id=DLayout.BUTTON_PAUSE, compact=True),
                Button(label=DLabel.RESUME, id=DLayout.BUTTON_RESUME, compact=True),
                Button(label=DLabel.STOP, id=DLayout.BUTTON_STOP, compact=True),
                classes=DLayout.BUTTON_ROW,
            ),
            Horizontal(
                Button(label=DLabel.DEFAULTS, id=DLayout.BUTTON_DEFAULTS, compact=True),
                Button(label=DLabel.UPDATE, id=DLayout.BUTTON_UPDATE, compact=True),
                classes=DLayout.BUTTON_ROW,
            ),
            Horizontal(
                Button(label=DLabel.RESET, id=DLayout.BUTTON_RESET, compact=True),
                Button(label=DLabel.QUIT, id=DLayout.BUTTON_QUIT, compact=True),
                classes=DLayout.BUTTON_ROW,
            ),
        )

        # Highscores
        yield Vertical(
            Label(
                f"[b #3e99af]{DLabel.GAME:>7s}{DLabel.SCORE:>7s}{DLabel.TIME:>13s}[/]"
            ),
            Log(highlight=False, auto_scroll=True, id=DLayout.HIGHSCORES),
            id=DLayout.HIGHSCORES_BOX,
        )

        # Empty placeholders for the grid layout
        yield Static(id=DLayout.FILLER_2)

        # The game score plot
        yield TabbedPlots(id=DLayout.TABBED_PLOTS)

    async def handle_requests(self):
        """Continuously listen for simulation state data from the router"""
        while self.running:
            try:
                msg_bytes = await self.socket.recv()
                msg = zmq.utils.jsonapi.loads(msg_bytes)
            except Exception as e:
                self.lablog.error(f"{DMQ_Label.RECEIVE_ERROR}: {e}")
                await asyncio.sleep(1)
                continue

            elem = msg.get(DMQ.ELEM)
            data = msg.get(DMQ.DATA, {})
            ## Uncomment to log *every* MQ message: Very verbose!!!
            # self.lablog.debug(f"MQ message: {elem}/{data}")
            if elem == DMQ.AVG_EPOCH_LOSS:
                self.avg_epoch_loss = data
            elif elem == DMQ.AVG_LOSS_DATA:
                tabbed_plots = self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots)
                for row in data:
                    epoch = row[0]
                    loss = row[1]
                    tabbed_plots.add_loss_data(row[0], row[1], plot=False)
            elif elem == DMQ.CUR_EPSILON:
                self.cur_epsilon = data
            elif elem == DMQ.CUR_HIGHSCORE:
                self.highscore = data
            elif elem == DMQ.CUR_SIM_STATE:
                if data == DSim.PAUSED:
                    self.remove_class(DSim.RUNNING)
                    self.add_class(DSim.PAUSED)
                    self.cur_sim_state = DSim.PAUSED
                elif data == DSim.STOPPED:
                    self.remove_class(DSim.RUNNING)
                    self.remove_class(DSim.PAUSED)
                    self.add_class(DSim.STOPPED)
                    self.cur_sim_state = DSim.STOPPED
                elif data == DSim.RUNNING:
                    self.remove_class(DSim.STOPPED)
                    self.remove_class(DSim.PAUSED)
                    self.add_class(DSim.RUNNING)
                    self.cur_sim_state = DSim.RUNNING
                    if self.highscore == DLabel.N_SLASH_A:
                        await self.send_mq(
                            mq_cli_msg(DMQ.GET_CUR_HIGHSCORE, self.identity_str)
                        )
            elif elem == DMQ.FOOD:
                self.game_board.update_food(data)
            elif elem == DMQ.GAME_ID:
                self.game_id = data
            elif elem == DMQ.EPOCH:
                self.epoch = data
            elif elem == DMQ.HIGHSCORE_EVENT:
                self.highscore_event = data
            elif elem == DMQ.MEM_TYPE:
                self.mem_type = data
            elif elem == DMQ.MODEL_TYPE:
                self.model_type = data
            elif elem == DMQ.MOVE_DELAY:
                self.move_delay = data
            elif elem == DMQ.NUM_FRAMES:
                self.num_frames = data
            elif elem == DMQ.OLD_GAME_SCORE_DATA:
                tabbed_plots = self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots)
                for event in data:
                    epoch = event[0]
                    score = event[1]
                    tabbed_plots.add_game_score_data(
                        epoch=epoch, score=score, plot=False
                    )
            elif elem == DMQ.OLD_HIGHSCORE_EVENTS:
                tabbed_plots = self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots)
                highscores_widget = self.query_one(f"#{DLayout.HIGHSCORES}", Log)
                for event in data:
                    epoch = event[0]
                    score = event[1]
                    runtime = event[2]
                    highscores_widget.write_line(f"{epoch:7,d}{score:7d}{runtime:>13s}")
                    tabbed_plots.add_highscore_data(epoch=epoch, score=score)
            elif elem == DMQ.RUNTIME:
                self.runtime = data
            elif elem == DMQ.SCORE:
                self.score = data
            elif elem == DMQ.SNAKE_HEAD:
                self.game_board.update_snake_head(data)
            elif elem == DMQ.SNAKE_BODY:
                self.game_board.update_snake(data)
            elif elem == DMQ.STORED_GAMES:
                self.stored_games = data
            elif elem == DMQ.TRAINING_LOOPS:
                self.training_loops = data

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        # Pause button was pressed
        if button_id == DLayout.BUTTON_PAUSE:
            self.sim_state = DSim.PAUSED
            self.remove_class(DSim.RUNNING)
            self.remove_class(DSim.STOPPED)
            self.add_class(DSim.PAUSED)
            await self.socket.send_json(mq_cli_msg(DMQ.CMD, DMQ.PAUSE))

        # Restart button was pressed
        elif button_id == DLayout.BUTTON_RESET:
            self.running = DSim.STOPPED
            self.remove_class(DSim.PAUSED)
            self.remove_class(DSim.RUNNING)
            self.add_class(DSim.STOPPED)

            # We display the game number, highscore and score here, so clear it.
            game_box = self.query_one(f"#{DLayout.GAME_BOX}", Vertical)
            game_box.border_title = ""
            game_box.border_subtitle = ""

            # The highscores (a Log widget ) should be cleared
            self.query_one(f"#{DLayout.HIGHSCORES}", Log).clear()

            # Clear the plot data
            self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots).clear_data()

            # Set the current stored games back to zero
            self.query_one(f"#{DLayout.STORED_GAMES}", Label).update("0")

            # Reset the training game id
            self.query_one(f"#{DLayout.CUR_TRAINING_GAME_ID}", Label).update(
                DLabel.N_SLASH_A
            )

            # Reset things on the SimServer side too
            await self.send_mq(mq_cli_msg(DMQ.CMD, DMQ.RESET))

        # Start button was pressed
        elif button_id == DLayout.BUTTON_START:
            # Get the configuration settings, put them into the runtime widgets and
            # pass the values to the actual backend objects
            self.running = DSim.RUNNING
            await self.update_settings()
            self.remove_class(DSim.STOPPED)
            self.add_class(DSim.RUNNING)
            await self.update_settings()
            await self.send_mq(mq_cli_msg(DMQ.CMD, DMQ.START))

        # Stop button was pressed
        elif button_id == DLayout.BUTTON_STOP:
            self.running = DSim.STOPPED
            self.remove_class(DSim.PAUSED)
            self.remove_class(DSim.RUNNING)
            self.add_class(DSim.STOPPED)
            await self.send_mq(mq_cli_msg(DMQ.CMD, DMQ.STOP))

        # Resume button was pressed
        elif button_id == DLayout.BUTTON_RESUME:
            self.running = DSim.RUNNING
            self.remove_class(DSim.STOPPED)
            self.remove_class(DSim.PAUSED)
            self.add_class(DSim.RUNNING)
            await self.send_mq(mq_cli_msg(DMQ.CMD, DMQ.RESUME))

        # Defaults button was pressed
        elif button_id == DLayout.BUTTON_DEFAULTS:
            self.set_defaults()

        # Quit button was pressed
        elif button_id == DLayout.BUTTON_QUIT:
            await self.on_quit()

        # Update button was pressed
        elif button_id == DLayout.BUTTON_UPDATE:
            self.update_settings()

    async def on_mount(self):

        # Configuration defaults
        self.set_defaults()

        # Set "Settings" box border to "Configuration Settings"
        self.query_one(f"#{DLayout.SETTINGS_BOX}", Vertical).border_title = (
            DLabel.SETTINGS
        )

        # Set "Highscores" box border to "Highscores"
        self.query_one(f"#{DLayout.HIGHSCORES_BOX}", Vertical).border_title = (
            DLabel.HIGHSCORES
        )

        # Set "Runtime" box border to "Runtime Values"
        self.query_one(f"#{DLayout.RUNTIME_BOX}", Vertical).border_title = (
            DLabel.RUNTIME_VALUES
        )

        # Add a starting point of (0,0) to the highscores plot
        self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots).add_highscore_data(0, 0)

        # Initial state is that the app is stopped
        self.add_class(DSim.STOPPED)

        # Register and set the theme
        self.register_theme(SNAKE_LAB_THEME)
        self.theme = DLayout.SNAKE_LAB_THEME

        ## NOTE: asyncio.gather breaks Textual...
        # Start listening for ZMQ messages
        self.handle_requests_task = asyncio.create_task(self.handle_requests())
        self.send_heartbeat_task = asyncio.create_task(self.send_heartbeat())

        # Get the current simulation state
        await self.send_mq(mq_cli_msg(DMQ.GET_SIM_STATE, self.identity_str))

        # Get the older highscore events
        await self.send_mq(mq_cli_msg(DMQ.GET_HIGHSCORE_EVENTS, self.identity_str))

        # Get the older average loss per epoch data
        await self.send_mq(mq_cli_msg(DMQ.GET_AVG_LOSS_DATA, self.identity_str))

        # Get the older game score data
        await self.send_mq(mq_cli_msg(DMQ.GET_GAME_SCORE_DATA, self.identity_str))

        # Start an async process to check current simulation state. This is to
        # cover the case where a 2nd SimClient connects the running simulation and
        # changes the simulation state.
        self.check_sim_state_task = asyncio.create_task(self.check_sim_state())

    async def on_quit(self):
        for task in [
            self.handle_requests_task,
            self.send_heartbeat_task,
            self.check_sim_state_task,
        ]:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # expected on cancellation

        sys.exit(0)

    async def send_heartbeat(self):
        """Periodic heartbeat to let the SimRouter know this client is alive."""
        while not self.heartbeat_stop_event.is_set():
            await self.send_mq(mq_cli_msg(DMQ.HEARTBEAT, self.identity.decode()))
            await asyncio.sleep(DSim.HEARTBEAT_INTERVAL)

    def set_defaults(self):
        """Load default values into the TUI widgets. Don't send anything to the SimRouter yet."""

        # The score and highscore are reactive variables
        self.score = 0
        self.highscore = 0

        self.query_one(f"#{DLayout.EPSILON_DECAY}", Input).value = str(
            DEpsilon.EPSILON_DECAY
        )
        self.query_one(f"#{DLayout.EPSILON_INITIAL}", Input).value = str(
            DEpsilon.EPSILON_INITIAL
        )
        self.query_one(f"#{DLayout.EPSILON_MIN}", Input).value = str(
            DEpsilon.EPSILON_MIN
        )
        self.query_one(f"#{DLayout.MEM_TYPE}", Select).value = MEM_TYPE.RANDOM_GAME

        # There is no "default" value for the model (Linear, RNN etc.). Get the current selection.
        cur_model_type = self.query_one(f"#{DLayout.MODEL_TYPE}", Select).value
        # Set the learning rate based on the current model selection.
        if cur_model_type == DModelL.MODEL:
            learning_rate_value = f"{DModelL.LEARNING_RATE:.6f}"
        elif cur_model_type == DModelRNN.MODEL:
            learning_rate_value = f"{DModelRNN.LEARNING_RATE:.6f}"
        self.query_one(f"#{DLayout.LEARNING_RATE}", Input).value = str(
            learning_rate_value
        )

        # Adaptive Memory
        self.query_one(f"#{DLayout.DYNAMIC_TRAINING}", Checkbox).value = True

        # Move delay
        self.query_one(f"#{DLayout.MOVE_DELAY}", Input).value = str(DDef.MOVE_DELAY)

    async def update_settings(self):
        # Get the move delay from the settings, put it into the runtime widget and send it to the SimRouter
        move_delay = self.query_one(f"#{DLayout.MOVE_DELAY}", Input).value
        self.query_one(f"#{DLayout.CUR_MOVE_DELAY}", Label).update(move_delay)
        await self.send_mq(mq_cli_msg(DMQ.MOVE_DELAY, move_delay))

        ##----------------------------------------------------
        ## Changing these next settings on-the-fly is like swapping out your carburetor while
        ## you're in the middle of a race. But, this is a sandbox, so let the user play.

        # Get the model type from the settings, put it into the runtime, send to SimRouter
        model_type = self.query_one(f"#{DLayout.MODEL_TYPE}", Select).value
        self.query_one(f"#{DLayout.CUR_MODEL_TYPE}", Label).update(
            MODEL_TYPE_TABLE[model_type]
        )
        await self.send_mq(mq_cli_msg(DMQ.MODEL_TYPE, model_type))

        # Now that we've set the model, we can pass in the learning rate
        learning_rate = self.query_one(f"#{DLayout.LEARNING_RATE}", Input).value
        await self.send_mq(mq_cli_msg(DMQ.LEARNING_RATE, float(learning_rate)))

        # Get the memory type from the settings, put it into the runtime
        memory_type = self.query_one(f"#{DLayout.MEM_TYPE}", Select).value
        self.query_one(f"#{DLayout.CUR_MEM_TYPE}", Label).update(
            MEM_TYPE.MEM_TYPE_TABLE[memory_type]
        )
        # Also pass the selected memory type to the ReplayMemory object
        await self.send_mq(mq_cli_msg(DMQ.MEM_TYPE, memory_type))

        # Dynamic training
        dyn_train_flag = self.query_one(f"#{DLayout.DYNAMIC_TRAINING}", Checkbox).value
        await self.send_mq(mq_cli_msg(DMQ.DYNAMIC_TRAINING, dyn_train_flag))

        # Set the widget label to "Random Frames" if memory type is "Random Frames"
        if memory_type == MEM_TYPE.SHUFFLE:
            self.query_one(f"#{DLayout.TRAINING_ID_LABEL}", Label).update(
                DLabel.RANDOM_FRAMES
            )
        # Set the widget label to "Game ID" if the memory type is "Random Game"
        elif memory_type == MEM_TYPE.RANDOM_GAME:
            self.query_one(f"#{DLayout.TRAINING_ID_LABEL}", Label).update(
                DLabel.TRAINING_GAME_ID
            )

        # Set the exploration type (Epsilon or EpsilonN)
        explore_type = self.query_one(f"#{DLayout.EXPLORATION}", Select).value
        await self.send_mq(mq_cli_msg(DMQ.EXPLORE_TYPE, explore_type))

        # Get the epsilon settings and send them to SimRouter
        epsilon_min = self.query_one(f"#{DLayout.EPSILON_MIN}", Input).value
        await self.socket.send_json(mq_cli_msg(DMQ.EPSILON_MIN, float(epsilon_min)))
        epsilon_initial = self.query_one(f"#{DLayout.EPSILON_INITIAL}", Input).value
        await self.send_mq(mq_cli_msg(DMQ.EPSILON_INITIAL, float(epsilon_initial)))
        epsilon_decay = self.query_one(f"#{DLayout.EPSILON_DECAY}", Input).value
        await self.send_mq(mq_cli_msg(DMQ.EPSILON_DECAY, float(epsilon_decay)))

    def watch_avg_epoch_loss(self, avg_epoch_loss_value: str):
        if self.epoch != DLabel.N_SLASH_A:
            self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots).add_loss_data(
                int(self.epoch), float(self.avg_epoch_loss)
            )

    def watch_cur_epsilon(self, cur_epsilon_value: str):
        self.query_one(f"#{DLayout.CUR_EPSILON}", Label).update(str(cur_epsilon_value))

    def watch_epoch(self, epoch_value: str):
        self.query_one(f"#{DLayout.GAME_BOX}", Vertical).border_title = (
            f"{DLabel.GAME} {self.epoch}"
        )
        if self.epoch != DLabel.N_SLASH_A and self.score != DLabel.N_SLASH_A:
            self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots).add_game_score_data(
                int(self.epoch), float(self.score)
            )

    def watch_game_id(self, game_id_value):
        if game_id_value == str(MEM.NO_DATA):
            game_id_value = DLabel.N_SLASH_A
        self.query_one(f"#{DLayout.CUR_TRAINING_GAME_ID}", Label).update(game_id_value)

    def watch_highscore_event(self, highscore_event: list):
        if len(highscore_event) == 0:
            return

        epoch = int(highscore_event[0])
        self.highscore = int(highscore_event[1])
        event_time = highscore_event[2]

        self.query_one(f"#{DLayout.HIGHSCORES}", Log).write_line(
            f"{epoch:7,d}{self.highscore:7d}{event_time:>13s}"
        )
        self.query_one(f"#{DLayout.TABBED_PLOTS}", TabbedPlots).add_highscore_data(
            epoch, self.highscore
        )

    def watch_highscore(self, highscore_value: str):
        self.query_one(f"#{DLayout.GAME_BOX}", Vertical).border_subtitle = (
            f"{DLabel.HIGHSCORE}: {self.highscore}, {DLabel.SCORE}: {self.score}"
        )

    def watch_mem_type(self, mem_type_value: str):
        self.query_one(f"#{DLayout.CUR_MEM_TYPE}", Label).update(
            MEM_TYPE.MEM_TYPE_TABLE[mem_type_value]
        )

    def watch_model_type(self, model_type_value: str):
        self.query_one(f"#{DLayout.CUR_MODEL_TYPE}", Label).update(model_type_value)

    def watch_move_delay(self, move_delay_value: str):
        self.query_one(f"#{DLayout.CUR_MOVE_DELAY}", Label).update(
            str(move_delay_value)
        )

    def watch_num_frames(self, num_frames_value):
        if num_frames_value == int(MEM.NO_DATA):
            num_frames_value = DLabel.N_SLASH_A
        self.query_one(f"#{DLayout.CUR_TRAINING_GAME_ID}", Label).update(
            num_frames_value
        )

    def watch_runtime(self, runtime_value):
        self.query_one(f"#{DLayout.RUNTIME}", Label).update(runtime_value)

    def watch_score(self, score_value: str):
        self.query_one(f"#{DLayout.GAME_BOX}", Vertical).border_subtitle = (
            f"{DLabel.HIGHSCORE}: {self.highscore}, {DLabel.SCORE}: {score_value}"
        )

    def watch_stored_games(self, stored_games_value: str):
        self.query_one(f"#{DLayout.STORED_GAMES}", Label).update(stored_games_value)

    def watch_training_loops(self, training_loops_value: str):
        self.query_one(f"#{DLayout.TRAINING_LOOPS}", Label).update(
            str(training_loops_value)
        )


# Helper function
def minutes_to_uptime(seconds: int):
    # Return a string like:
    # 0h 0m 45s
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


# This is for the pyPI ai-snake-lab entry point to work....
def main():
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--router",
        type=str,
        default=DSim.HOST,
        help="IP or hostname of the AI Snake Lab router",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show additional information"
    )
    args = parser.parse_args()
    if args.verbose:
        loglevel = DLog.DEBUG
    else:
        loglevel = DLog.INFO

    # The SimClient log file
    snake_dir = os.path.join(Path.home(), DDir.DOT + DDir.AI_SNAKE_LAB)
    if not os.path.exists(snake_dir):
        os.mkdir(snake_dir)
    logfile = os.path.join(snake_dir, DFile.CLIENT_LOG)

    print(f"Connecting to SimRouter: {args.router}")
    print(f"Log file: {logfile}")
    if args.verbose:
        print("Verbose logging enabled")
    time.sleep(3)

    app = SimClient(router=args.router, loglevel=loglevel, logfile=logfile)
    app.run()


if __name__ == "__main__":
    main()
