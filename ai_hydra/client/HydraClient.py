# ai_hydra/client/HydraClient.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import asyncio
import sys
import time
from pathlib import Path
import os
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import Button, Label, Input, Checkbox, Select
from textual.message import Message

from ai_hydra.zmq.HydraClientMQ import HydraClientMQ
from ai_hydra.zmq.HydraMsg import HydraMsg
from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.utils.HydraMetrics import HydraMetrics
from ai_hydra.client.ClientGameBoard import ClientGameBoard
from ai_hydra.client.TabbedScores import TabbedScores
from ai_hydra.client.TabbedPlots import TabbedPlots

from ai_hydra.constants.DHydra import (
    DHydra,
    DHydraRouterDef,
    DMethod,
    DModule,
    DHydraMQDef,
    DHydraMQ,
)
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.constants.DGame import DGameField, DGameMethod
from ai_hydra.constants.DNNet import (
    DNetField,
    DNetDef,
    DLinear,
    DRNN,
    MODEL_TYPE_TABLE,
    MODEL_TYPES,
)

HYDRA_THEME = Theme(
    name="hydra_theme",
    primary="#0E191C",
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
        "footer-key-foreground": "#FF0000",
        "input-selection-background": "#81a1c1 35%",
    },
)


class TelemetryReceived(Message):
    def __init__(self, topic: str, payload: dict) -> None:
        super().__init__()
        self.topic = topic
        self.payload = payload


class HydraClientTui(App):
    """A Textual interface to the HydraServer"""

    TITLE = DLabel.CLIENT_TITLE
    CSS_PATH = DFile.CLIENT_CSS

    def __init__(
        self,
        address: str = DHydraRouterDef.HOSTNAME,
        port: int = DHydraRouterDef.PORT,
    ) -> None:
        """Constructor"""
        super().__init__()

        self._address: str = address
        self._port = port
        self._identity = DModule.HYDRA_CLIENT
        self._connected_msg = DStatus.BAD + " " + DLabel.DISCONNECTED
        self.mq: HydraClientMQ | None = None

        self.game_board = ClientGameBoard(20, id=DGameField.BOARD)
        self.cfg = SimCfg()
        self._running = False
        self._wgt = None

        # Metrics to hold config and runtime sim data
        self.metrics = HydraMetrics()
        self._cur_epsilon = None

        # Ahem...
        self._not_first_time_kludge = False

        # ZeroMQ batching support at this app layer it checks for lost
        # messages. PUB/SUB does not guarantee delivery, batching  aims to
        # mitigate that.
        self._prev_scores_batch_num = None
        self._cur_scores_batch_num = None
        self._first_scores_batch = True

        self._prev_per_ep_batch_num = None
        self._cur_per_ep_batch_num = None
        self._first_per_ep_batch = True

        # AI Hydra stores temporary and persistent files in this directory
        self._hydra_dir = os.path.join(Path.home(), DHydra.HYDRA_DIR)
        if not os.path.exists(self._hydra_dir):
            os.mkdir(self._hydra_dir)

    @work(exclusive=True)
    async def check_heartbeat(self) -> None:
        while True:
            if self.mq.connected():
                status = DStatus.GOOD
            else:
                status = DStatus.BAD

            self.query_one(f"#{DField.ROUTER_HB}", Label).update(
                f"{DLabel.ROUTER}: {status}"
            )

            await asyncio.sleep(DHydra.HEARTBEAT_INTERVAL)

    def compose(self) -> ComposeResult:
        """The TUI is created here"""

        ## We're using a Textual "grid" layout...

        # ----- Title --- 1 row x 4 cols
        yield Label(DLabel.CLIENT_TITLE, id=DField.TITLE)

        # ----- Buttons --- 2x1
        yield Vertical(
            Button(label=DLabel.HANDSHAKE, id=DField.HANDSHAKE, compact=True),
            Button(label=DLabel.START, id=DField.START_RUN, compact=True),
            Button(
                label=DLabel.UPDATE_CONFIG,
                id=DField.UPDATE_RUNTIME_CONFIG,
                compact=True,
            ),
            Label(),
            Button(label=DLabel.SNAPSHOT, id=DField.SNAPSHOT, compact=True),
            Label(),
            Button(label=DLabel.QUIT, id=DMethod.QUIT, compact=True),
            id=DField.BUTTONS,
        )

        # ------ The Snake Game --- 2x1
        yield Vertical(self.game_board, id=DField.BOARD_BOX)

        # ----- Settings ---
        yield Vertical(
            # Random seed
            Horizontal(
                Label(f"{DLabel.RANDOM_SEED:>11s}: "),
                Input(
                    type=DField.INTEGER,
                    compact=True,
                    valid_empty=False,
                    value=str(DHydra.RANDOM_SEED),
                    id=DField.RANDOM_SEED_INPUT,
                ),
                Label(str(DHydra.RANDOM_SEED), id=DField.RANDOM_SEED_LABEL),
                classes=DField.INPUT_FIELD,
            ),
            # Move delay
            Horizontal(
                Label(f"{DLabel.MOVE_DELAY:>11s}: "),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DNetDef.MOVE_DELAY),
                    id=DField.MOVE_DELAY_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Turbo mode
            Horizontal(
                Label(f"{DLabel.TURBO_MODE:>11s}: "),
                Checkbox(value=False, id=DField.TURBO_MODE, compact=True),
                classes=DField.INPUT_FIELD,
            ),
            id=DField.SETTINGS,
        )

        # ----- Highscores widget ---
        yield TabbedScores(id=DField.TABBED_SCORES)

        # ----- Epsilon ---
        yield Vertical(
            # Initial epsilon
            Horizontal(
                Label(f"{DLabel.INITIAL_EPSILON:>15s}: "),
                Label(
                    str(DLinear.INITIAL_EPSILON),
                    id=DField.INITIAL_EPSILON_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DLinear.INITIAL_EPSILON),
                    id=DField.INITIAL_EPSILON_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Minimum epsilon
            Horizontal(
                Label(f"{DLabel.MIN_EPSILON:>15s}: "),
                Label(
                    str(DLinear.MINIMUM_EPSILON),
                    id=DField.MIN_EPSILON_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DLinear.MINIMUM_EPSILON),
                    id=DField.MIN_EPSILON_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Epsilon decay
            Horizontal(
                Label(f"{DLabel.EPSILON_DECAY:>15s}: "),
                Label(
                    str(DLinear.EPSILON_DECAY_RATE),
                    id=DField.EPSILON_DECAY_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DLinear.EPSILON_DECAY_RATE),
                    id=DField.EPSILON_DECAY_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Current epsilon
            Horizontal(
                Label(f"{DLabel.CUR_EPSILON:>15s}: "),
                Label(id=DField.CUR_EPSILON),
                classes=DField.INPUT_FIELD,
            ),
            id=DField.EPSILON,
        )

        # ------ Network ---
        yield Vertical(
            Label(f"{DLabel.TARGET_HOST}: {self._address}"),
            Label(f"{DLabel.TARGET_PORT}: {self._port}"),
            Label(f"{DLabel.ROUTER}", id=DField.ROUTER_HB),
            id=DField.NETWORK,
        )

        # ----- Model ---
        yield Vertical(
            # Model type
            Horizontal(
                Label(f"{DLabel.NN_MODEL:>15s}: "),
                Label(DLabel.LINEAR, id=DField.MODEL_TYPE_LABEL),
                Select(
                    MODEL_TYPES,
                    compact=True,
                    id=DField.MODEL_TYPE_SELECT,
                    allow_blank=False,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Hidden Size
            Horizontal(
                Label(f"{DLabel.HIDDEN_SIZE:>15s}: "),
                Label(
                    f"{DLinear.HIDDEN_SIZE:.7f}",
                    id=DField.HIDDEN_SIZE_LABEL,
                ),
                Input(
                    type=DField.INTEGER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DLinear.HIDDEN_SIZE:.7f}",
                    id=DField.HIDDEN_SIZE_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Learning rate
            Horizontal(
                Label(f"{DLabel.LEARNING_RATE:>15s}: "),
                Label(
                    f"{DLinear.LEARNING_RATE:.7f}",
                    id=DField.LEARNING_RATE_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DLinear.LEARNING_RATE:.7f}",
                    id=DField.LEARNING_RATE_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Dropout p-value
            Horizontal(
                Label(f"{DLabel.DROPOUT_P_VAL:>15s}: "),
                Label(
                    f"{DLinear.DROPOUT_P:.2f}",
                    id=DField.DROPOUT_P_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DLinear.DROPOUT_P:.2f}",
                    id=DField.DROPOUT_P_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # RNN Layers
            Horizontal(
                Label(f"{DLabel.RNN_LAYERS:>15s}: ", id=DField.RNN_LAYERS_OPT),
                Label(
                    f"{DRNN.RNN_LAYERS}",
                    id=DField.RNN_LAYERS_LABEL,
                ),
                Input(
                    type=DField.INTEGER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DRNN.RNN_LAYERS}",
                    id=DField.RNN_LAYERS_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            id=DField.MODEL,
        )

        # Plots
        yield TabbedPlots(id=DField.TABBED_PLOTS)

        # Console
        yield Vertical(Label(id=DField.CONSOLE_SCREEN), id=DField.CONSOLE_BOX)

        # Focus widget: This is hidden, but it allows me to move focus away
        # from the selected button when a button is clicked.
        yield Checkbox(id=DField.HIDDEN_WIDGET)

    def console_msg(self, value: str) -> None:
        self._w_console_label.update(str(value))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if self.mq is None:
            raise TypeError("self.mq is None!!!")

        elif button_id == DField.HANDSHAKE:
            await self._send_handshake()

        elif button_id == DMethod.QUIT:
            await self.on_quit()

        elif button_id == DGameMethod.RESET_GAME:
            await self._send_reset()

        elif button_id == DField.SNAPSHOT:
            await self._take_snapshot()

        elif button_id == DGameMethod.START_RUN:
            self._update_tui_labels()
            self.mq.enable_per_episode_sub()
            self.mq.enable_scores_sub()
            if self.cfg.get(DNetField.PER_STEP):
                self.mq.enable_per_step_sub()
            else:
                self.mq.disable_per_step_sub()
            await self._send_start_run()

        elif button_id == DGameMethod.STOP_RUN:
            await self._send_stop_run()

        elif button_id == DField.UPDATE_RUNTIME_CONFIG:
            self._update_tui_labels()
            await self._send_update_config()
        self._w_hidden_widget.focus()

    def on_mount(self) -> None:
        self.add_class(DField.BAD_HANDSHAKE)
        self.add_class(DField.LINEAR)

        # Create references to TUI elements that are being updated
        self._w_board_box = self.query_one(f"#{DField.BOARD_BOX}", Vertical)
        self._w_console_box = self.query_one(
            f"#{DField.CONSOLE_BOX}", Vertical
        )
        self._w_console_label = self.query_one(
            f"#{DField.CONSOLE_SCREEN}", Label
        )
        self._w_cur_epsilon_label = self.query_one(
            f"#{DField.CUR_EPSILON}", Label
        )
        self._w_dropout_p_input = self.query_one(
            f"#{DField.DROPOUT_P_INPUT}", Input
        )
        self._w_dropout_p_label = self.query_one(
            f"#{DField.DROPOUT_P_LABEL}", Label
        )
        self._w_epsilon_decay_input = self.query_one(
            f"#{DField.EPSILON_DECAY_INPUT}", Input
        )
        self._w_epsilon_decay_label = self.query_one(
            f"#{DField.EPSILON_DECAY_LABEL}", Label
        )
        self._w_hidden_size_input = self.query_one(
            f"#{DField.HIDDEN_SIZE_INPUT}", Input
        )
        self._w_hidden_size_label = self.query_one(
            f"#{DField.HIDDEN_SIZE_LABEL}", Label
        )
        self._w_hidden_widget = self.query_one(
            f"#{DField.HIDDEN_WIDGET}", Checkbox
        )
        self._w_initial_epsilon_input = self.query_one(
            f"#{DField.INITIAL_EPSILON_INPUT}", Input
        )
        self._w_initial_epsilon_label = self.query_one(
            f"#{DField.INITIAL_EPSILON_LABEL}", Label
        )
        self._w_learning_rate_input = self.query_one(
            f"#{DField.LEARNING_RATE_INPUT}", Input
        )
        self._w_learning_rate_label = self.query_one(
            f"#{DField.LEARNING_RATE_LABEL}", Label
        )
        self._w_min_epsilon_input = self.query_one(
            f"#{DField.MIN_EPSILON_INPUT}", Input
        )
        self._w_min_epsilon_label = self.query_one(
            f"#{DField.MIN_EPSILON_LABEL}", Label
        )
        self._w_model_type_label = self.query_one(
            f"#{DField.MODEL_TYPE_LABEL}", Label
        )
        self._w_model_type_select = self.query_one(
            f"#{DField.MODEL_TYPE_SELECT}", Select
        )
        self._w_move_delay_input = self.query_one(
            f"#{DField.MOVE_DELAY_INPUT}", Input
        )
        self._w_random_seed_input = self.query_one(
            f"#{DField.RANDOM_SEED_INPUT}", Input
        )
        self._w_random_seed_label = self.query_one(
            f"#{DField.RANDOM_SEED_LABEL}", Label
        )
        self._w_rnn_layers_input = self.query_one(
            f"#{DField.RNN_LAYERS_INPUT}", Input
        )
        self._w_rnn_layers_label = self.query_one(
            f"#{DField.RNN_LAYERS_LABEL}", Label
        )
        self._w_tabbed_plots = self.query_one(
            f"#{DField.TABBED_PLOTS}", TabbedPlots
        )
        self._w_tabbed_scores = self.query_one(
            f"#{DField.TABBED_SCORES}", TabbedScores
        )
        self._w_turbo_mode = self.query_one(f"#{DField.TURBO_MODE}", Checkbox)

        # Network
        self.query_one(f"#{DField.ROUTER_HB}", Label).update(
            f"{DLabel.ROUTER:>11s}: {DStatus.UNKNOWN}"
        )

        # Monitor the connection to the server in the background
        self.check_heartbeat()

        ## Add some text to the borders around the widgets
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.SETTINGS}").border_subtitle = DLabel.SETTINGS
        self.query_one(f"#{DField.NETWORK}").border_subtitle = DLabel.NETWORK
        self.query_one(f"#{DField.TABBED_SCORES}").border_subtitle = (
            DLabel.HIGHSCORES
        )
        self.query_one(f"#{DField.BUTTONS}").border_subtitle = DLabel.ACTIONS
        self._w_console_box.border_subtitle = DLabel.CONSOLE
        self._w_tabbed_plots.border_subtitle = DLabel.VISUALIZATIONS
        self.query_one(f"#{DField.MODEL}").border_subtitle = DLabel.MODEL
        self.query_one(f"#{DField.EPSILON}").border_subtitle = DLabel.EPSILON

        # Switch focus to a hidden widget
        self._w_hidden_widget.focus()

        self.mq = HydraClientMQ(
            router_address=self._address,
            router_port=self._port,
            identity=self._identity,
            srv_host=self._address,
        )
        self.mq.sub_methods = {
            self.mq.topic(DHydraMQDef.PER_STEP_TOPIC): self.on_per_step,
            self.mq.topic(DHydraMQDef.PER_EPISODE_TOPIC): self.on_per_episode,
            self.mq.topic(DHydraMQDef.SCORES_TOPIC): self.on_scores,
        }
        self.mq.start()
        self.mq.disable_per_episode_sub()
        self.mq.disable_per_step_sub()
        self.mq.disable_scores_sub()

        self.console_msg("Initialized...")

    def on_per_episode(self, topic: str, payload: dict) -> None:
        batch_msg = HydraMsg.from_dict(payload)
        messages = batch_msg.payload[DHydraMQ.MSGS]
        self._cur_per_ep_batch_num = messages[0]["payload"]["count"]
        if self._first_per_ep_batch:
            self._first_per_ep_batch_num = False
            self._prev_per_ep_batch_num = self._cur_per_ep_batch_num - 1

        if (self._cur_per_ep_batch_num - self._prev_per_ep_batch_num) != 1:
            self.console_msg("WARNING: Per episode data is being dropped!!!")

        self._prev_per_ep_batch_num = self._cur_per_ep_batch_num

        for payload in messages[1:]:
            # Epoch
            epoch = payload[DGameField.EPOCH]
            self._w_board_box.border_title = f"{DLabel.GAME}: {epoch}"
            self.metrics.add_cur_epoch(epoch)

            # Elapsed time
            elapsed_time = payload[DGameField.ELAPSED_TIME]
            self._w_console_box.border_subtitle = elapsed_time
            self.metrics.add_elapsed_time(elapsed_time)

            # Current epsilon value
            epsilon = payload.get(DNetField.CUR_EPSILON)
            if epsilon is not None:
                self._w_cur_epsilon_label.update(str(round(epsilon, 4)))
                self._cur_epsilon = epsilon

            # Loss
            if DNetField.LOSS in payload:
                self._w_tabbed_plots.add_loss(
                    epoch=epoch, loss=payload[DNetField.LOSS]
                )

    def on_per_step(self, topic: str, payload: dict) -> None:
        board = payload.get(DGameField.BOARD)
        self.game_board.apply_board_dict(board)

    async def on_quit(self) -> None:
        ###
        ### The code below (clean shutdown) causes the client to hang
        ### requiring a `kill <PID>``, followed by a `tput reset` to get
        ### your terminal back into a sane state. So....
        ###
        ### "If in doubt, nuke it from orbit"
        ###
        if self.mq is not None:
            await self.mq.quit()
            self.mq = None
        self.exit()

    def on_scores(self, topic: str, payload) -> None:
        batch_msg = HydraMsg.from_dict(payload)
        messages = batch_msg.payload[DHydraMQ.MSGS]
        self._cur_scores_batch_num = messages[0]["payload"]["count"]
        if self._first_scores_batch:
            self._first_scores_batch_num = False
            self._prev_scores_batch_num = self._cur_scores_batch_num - 1

        if (self._cur_scores_batch_num - self._prev_scores_batch_num) != 1:
            self.console_msg("WARNING: Score data is being dropped!!!")

        self._prev_scores_batch_num = self._cur_scores_batch_num
        cur_epsilon = self._cur_epsilon or self._w_initial_epsilon_input.value

        for payload in messages[1:]:

            # Current highscore
            highscore = payload[DGameField.HIGHSCORE]
            self._w_tabbed_scores.border_subtitle = (
                f"{DLabel.HIGHSCORE}: {highscore}"
            )

            # current score
            score = payload[DGameField.SCORE]
            if self.cfg.get(DNetField.PER_STEP):
                self._w_board_box.border_subtitle = (
                    f"{DLabel.SCORE}: {score:<2}"
                )
            else:
                self._w_board_box.border_subtitle = ""

            # Highscore event
            if DGameField.HIGHSCORE_EVENT in payload:
                hs_event = payload[DGameField.HIGHSCORE_EVENT]
                self._w_tabbed_scores.add_highscore(
                    epoch=hs_event[0],
                    highscore=hs_event[1],
                    event_time=hs_event[2],
                )
                self._w_tabbed_plots.add_scatter_score(
                    scores=(hs_event[0], hs_event[1])
                )
                self.console_msg(f"🎉 New highscore : {hs_event[1]}")
                self.metrics.add_highscore_event(
                    episode=hs_event[0],
                    highscore=hs_event[1],
                    event_time=hs_event[2],
                    cur_ep=cur_epsilon,
                )

            # Final score
            if DNetField.FINAL_SCORE in payload:
                self._w_tabbed_plots.add_score(
                    cur_score=payload[DNetField.FINAL_SCORE],
                    cur_epoch=payload[DGameField.EPOCH],
                )

    async def on_shutdown_request(self) -> None:
        if self.mq is not None:
            await self.mq.quit()
            self.mq = None

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.control.id != DField.MODEL_TYPE_SELECT:
            return

        model_type = event.value
        # Linear model defaults
        if model_type == DField.LINEAR:
            lr = f"{DLinear.LEARNING_RATE:.5f}"
            initial_epsilon = DLinear.INITIAL_EPSILON
            min_epsilon = DLinear.MINIMUM_EPSILON
            epsilon_decay = DLinear.EPSILON_DECAY_RATE
            hidden_size = DLinear.HIDDEN_SIZE
            dropout_p = DLinear.DROPOUT_P
            self.remove_class(DField.RNN)
            self.add_class(DField.LINEAR)

        # RNN model defaults
        elif model_type == DField.RNN:
            lr = f"{DRNN.LEARNING_RATE:.4f}"
            initial_epsilon = DRNN.INITIAL_EPSILON
            min_epsilon = DRNN.MINIMUM_EPSILON
            epsilon_decay = DRNN.EPSILON_DECAY_RATE
            hidden_size = DRNN.HIDDEN_SIZE
            dropout_p = DRNN.DROPOUT_P_VALUE
            self.remove_class(DField.LINEAR)
            self.add_class(DField.RNN)

        else:
            raise ValueError(f"EFFOR: Unrecognized model type {model_type}")

        # Update widgets with model specific defaults
        self._w_learning_rate_input.value = lr
        self._w_initial_epsilon_input.value = str(initial_epsilon)
        self._w_min_epsilon_input.value = str(min_epsilon)
        self._w_epsilon_decay_input.value = str(epsilon_decay)
        self._w_hidden_size_input.value = str(hidden_size)
        self._w_dropout_p_input.value = str(dropout_p)

        if self._not_first_time_kludge:
            self.console_msg("Updated defaults setting...")
        self._not_first_time_kludge = True

    async def _send_handshake(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DMethod.HANDSHAKE,
        )
        await self.mq.send(msg)

        try:
            reply = await self.mq.recv()
            sim_running = reply.payload.pop(DGameField.SIM_RUNNING)
            if sim_running:
                # Always receive per-episode updates
                self.mq.enable_per_episode_sub()
                # Always receive score updates
                self.mq.enable_scores_sub()
                self.add_class(DField.SIM_RUNNING)
                self.cfg = SimCfg.from_dict(reply.payload)
                if self.cfg.get(DNetField.PER_STEP):
                    self.mq.enable_per_step_sub()
                else:
                    self.mq.disable_per_step_sub()
                # Load configurable TUI settings from the running sim config
                self._w_initial_epsilon_input.value = str(
                    self.cfg.get(DNetField.INITIAL_EPSILON)
                )
                self._w_min_epsilon_input.value = str(
                    self.cfg.get(DNetField.MIN_EPSILON)
                )
                self._w_move_delay_input.value = str(
                    self.cfg.get(DNetField.MOVE_DELAY)
                )
                self._w_model_type_select.value = self.cfg.get(
                    DNetField.MODEL_TYPE
                )
                self._w_hidden_size_input.value = str(
                    self.cfg.get(DNetField.HIDDEN_SIZE)
                )
                self._update_tui_labels()
                self.console_msg("Connecting to running simulation...")

            else:
                self.add_class(DField.SIM_STOPPED)
                self.console_msg("Connected to simulation server...")
            self.remove_class(DField.BAD_HANDSHAKE)

        except asyncio.TimeoutError:
            self.console_msg("Unable to connect to simulation server...")

    async def _send_reset(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.RESET_GAME,
        )
        await self.mq.send(msg)

        try:
            reply = await self.mq.recv()
        except asyncio.TimeoutError:
            pass

    async def _send_start_run(self):
        # Send a START_RUN message to the HydraMgr
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.START_RUN,
            payload=self.cfg.to_dict(),
        )
        await self.mq.send(msg)

        # Check what the HydraMgr sent back:
        try:
            reply = await self.mq.recv()
            if reply.payload[DGameField.OK]:
                # Perfect, sim wasn't running now it is.
                self.remove_class(DField.SIM_STOPPED)
                self.add_class(DField.SIM_RUNNING)
            else:
                # Sim was already running
                self.remove_class(DField.SIM_STOPPED)
                self.add_class(DField.SIM_RUNNING)
            self.console_msg("Simulation started...")

        except asyncio.TimeoutError:
            pass

    async def _send_stop_run(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.STOP_RUN,
        )
        await self.mq.send(msg)
        try:
            reply = await self.mq.recv()

        except asyncio.TimeoutError:
            pass

    async def _send_update_config(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.UPDATE_CONFIG,
            payload=self.cfg.to_dict(),
        )
        await self.mq.send(msg)
        if self._w_turbo_mode.value:
            self.mq.disable_per_step_sub()
        else:
            self.mq.enable_per_step_sub()

    async def _take_snapshot(self):
        # Snapshot file
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M.txt")
        snapshot_file = DFile.BASE_SNAPSHOT + timestamp
        fq_snapshot = os.path.join(self._hydra_dir, snapshot_file)
        self.metrics.create_snapshot(snap_file=fq_snapshot, cfg=self.cfg)

        self.console_msg(f"📸 Created snapshot file: {fq_snapshot}")

    def _update_tui_labels(self):
        """
        Update the SimCfg settings and the TUI labels
        """
        # Random Seed
        random_seed = self._w_random_seed_input.value
        self._w_random_seed_label.update(random_seed)
        # Epsilon values
        epsilon_decay = self._w_epsilon_decay_input.value
        initial_epsilon = self._w_initial_epsilon_input.value
        min_epsilon = self._w_min_epsilon_input.value
        self._w_epsilon_decay_label.update(str(epsilon_decay))
        self._w_initial_epsilon_label.update(str(initial_epsilon))
        self._w_min_epsilon_label.update(str(min_epsilon))
        # Turbo mode == Not per_step
        per_step = not self._w_turbo_mode.value
        # Move delay
        move_delay = self._w_move_delay_input.value
        # Model type
        model_type = self._w_model_type_select.value
        model_type_label = MODEL_TYPE_TABLE[model_type]
        self._w_model_type_label.update(model_type_label)
        # Hidden size
        hidden_size = self._w_hidden_size_input.value
        self._w_hidden_size_label.update(str(hidden_size))
        # Learning rate
        learning_rate = self._w_learning_rate_input.value
        self._w_learning_rate_label.update(learning_rate)
        # Dropout p-value
        dropout_p = self._w_dropout_p_input.value
        self._w_dropout_p_label.update(dropout_p)
        # RNN layers
        rnn_layers = self._w_rnn_layers_input.value
        self._w_rnn_layers_label.update(rnn_layers)

        self.cfg.apply(
            {
                DNetField.DROPOUT_P: dropout_p,
                DNetField.EPSILON_DECAY: epsilon_decay,
                DNetField.HIDDEN_SIZE: hidden_size,
                DNetField.INITIAL_EPSILON: initial_epsilon,
                DNetField.LEARNING_RATE: float(learning_rate),
                DNetField.MIN_EPSILON: min_epsilon,
                DNetField.PER_STEP: per_step,
                DNetField.MODEL_TYPE: model_type,
                DNetField.MOVE_DELAY: move_delay,
                DNetField.RANDOM_SEED: int(random_seed),
                DNetField.RNN_LAYERS: int(rnn_layers),
            }
        )


def main() -> None:
    router = HydraClientTui()
    router.run()


if __name__ == "__main__":
    main()
