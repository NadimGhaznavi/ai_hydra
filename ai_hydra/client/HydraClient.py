# ai_hydra/client/HydraClient.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

# General Python Modules
import asyncio
import sys
import time
from pathlib import Path
import os
from datetime import datetime

# Textual Modules
from textual import work, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import (
    Button,
    Label,
    Input,
    Checkbox,
    Select,
    Switch,
)
from textual.message import Message

# AI Hydra Constants
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

# AI Hydra Modules
from ai_hydra.zmq.HydraClientMQ import HydraClientMQ
from ai_hydra.zmq.HydraMsg import HydraMsg
from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.utils.HydraSnapshot import HydraSnapshot
from ai_hydra.utils.HydraMetrics import HydraMetrics
from ai_hydra.client.ClientGameBoard import ClientGameBoard
from ai_hydra.client.HighScoresLog import HighScoresLog
from ai_hydra.client.HydraTelemetry import HydraTelemetry


HYDRA_THEME = Theme(
    name="hydra_theme",
    primary="#0E191C",
    secondary="#2aa5ceff",
    accent="#B48EAD",
    foreground="#31b8e6",
    background="black",
    success="#A3BE8C",
    warning="#EBCB8B",
    error="#BF616A",
    surface="#2aa5ce",
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

        # Metrics to telemetry data
        self.metrics = HydraMetrics(initial_epsilon=DLinear.INITIAL_EPSILON)

        # Snapshot class to create a snapshot file.
        self.snapshot = HydraSnapshot(metrics=self.metrics)

        # Telemetric Widget
        self.telemtry: HydraTelemetry | None = None

        self._cur_epsilon = None

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

        # ----- Title ---
        yield Label(DLabel.CLIENT_TITLE, id=DField.TITLE)

        # ----- Buttons ---
        yield Vertical(
            Button(label=DLabel.HANDSHAKE, id=DField.HANDSHAKE, compact=True),
            Button(label=DLabel.START, id=DField.START_RUN, compact=True),
            Label(id=DField.START_RUN_DIVIDER),
            Button(label=DLabel.SNAPSHOT, id=DField.SNAPSHOT, compact=True),
            Label(),
            Button(label=DLabel.QUIT, id=DMethod.QUIT, compact=True),
            id=DField.BUTTONS,
        )

        # ------ The Snake Game ---
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
                Label(f"0.0", id=DField.MOVE_DELAY_LABEL),
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
                # Checkbox(value=False, id=DField.TURBO_MODE, compact=True),
                Switch(value=False, id=DField.TURBO_MODE),
                classes=DField.INPUT_FIELD,
            ),
            id=DField.SETTINGS,
        )

        # ----- Highscores widget ---
        yield HighScoresLog(id=DField.HIGHSCORES_LOG)

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

        # ------ Network ---
        yield Vertical(
            Label(f"{DLabel.TARGET_HOST}: {self._address}"),
            Label(f"{DLabel.TARGET_PORT}: {self._port}"),
            Label(f"{DLabel.ROUTER}", id=DField.ROUTER_HB),
            id=DField.NETWORK,
        )

        # ----- Training ---
        yield Vertical(
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
            # Discount/Gamma
            Horizontal(
                Label(f"{DLabel.GAMMA:>15s}: "),
                Label(
                    f"{DLinear.GAMMA:.2f}",
                    id=DField.GAMMA_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DLinear.GAMMA:.2f}",
                    id=DField.GAMMA_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Batch size
            Horizontal(
                Label(f"{DLabel.BATCH_SIZE:>15s}: "),
                Label(
                    f"{DLinear.BATCH_SIZE}",
                    id=DField.BATCH_SIZE_LABEL,
                ),
                Input(
                    type=DField.INTEGER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DLinear.BATCH_SIZE}",
                    id=DField.BATCH_SIZE_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # RNN Sequence Length
            Horizontal(
                Label(
                    f"{DLabel.SEQUENCE_LENGTH:>15s}: ",
                    id=DField.SEQ_LENGTH_OPT,
                ),
                Label(
                    f"{DRNN.SEQ_LENGTH}",
                    id=DField.SEQ_LENGTH_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DRNN.SEQ_LENGTH}",
                    id=DField.SEQ_LENGTH_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # RNN Trainer Tau
            Horizontal(
                Label(f"{DLabel.RNN_TAU:>15s}: ", id=DField.RNN_TAU_OPT),
                Label(
                    f"{DRNN.TAU}",
                    id=DField.RNN_TAU_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=f"{DRNN.TAU}",
                    id=DField.RNN_TAU_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            id=DField.TRAINING_BOX,
        )

        # Realtime Telemetry (plots and event log)
        yield HydraTelemetry(metrics=self.metrics, id=DField.HYDRA_TELEMETRY)

        # Focus widget: This is hidden, but it allows me to move focus away
        # from the selected button when a button is clicked.
        yield Checkbox(id=DField.HIDDEN_WIDGET)

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

    async def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id
        if input_id != DField.MOVE_DELAY_INPUT:
            return

        self._update_tui_labels()
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.UPDATE_CONFIG,
            payload=self.cfg.to_dict(),
        )
        await self.mq.send(msg)

    def on_mount(self) -> None:
        self.add_class(DField.BAD_HANDSHAKE)
        self.add_class(DField.LINEAR)
        self.add_class(DField.TURBO_OFF)

        # Create references to TUI elements that are being updated
        self.telemtry = self.query_one(
            f"#{DField.HYDRA_TELEMETRY}", HydraTelemetry
        )
        self.event_log = self.telemtry.event_log
        self._w_batch_size_input = self.query_one(
            f"#{DField.BATCH_SIZE_INPUT}", Input
        )
        self._w_batch_size_label = self.query_one(
            f"#{DField.BATCH_SIZE_LABEL}", Label
        )
        self._w_board_box = self.query_one(f"#{DField.BOARD_BOX}", Vertical)
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
        self._w_gamma_input = self.query_one(f"#{DField.GAMMA_INPUT}", Input)
        self._w_gamma_label = self.query_one(f"#{DField.GAMMA_LABEL}", Label)
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
        self._w_rnn_tau_input = self.query_one(
            f"#{DField.RNN_TAU_INPUT}", Input
        )
        self._w_rnn_tau_label = self.query_one(
            f"#{DField.RNN_TAU_LABEL}", Label
        )
        self._w_seq_length_input = self.query_one(
            f"#{DField.SEQ_LENGTH_INPUT}", Input
        )
        self._w_seq_length_label = self.query_one(
            f"#{DField.SEQ_LENGTH_LABEL}", Label
        )
        self._w_highscores_log = self.query_one(
            f"#{DField.HIGHSCORES_LOG}", HighScoresLog
        )

        # Network
        self.query_one(f"#{DField.ROUTER_HB}", Label).update(
            f"{DLabel.ROUTER:>11s}: {DStatus.UNKNOWN}"
        )

        # Monitor the connection to the server in the background
        self.check_heartbeat()

        # ----- Add some text to the borders around the widgets
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.SETTINGS}").border_subtitle = DLabel.SETTINGS
        self.query_one(f"#{DField.NETWORK}").border_subtitle = DLabel.NETWORK
        self._w_highscores_log.border_subtitle = DLabel.HIGHSCORES
        self.query_one(f"#{DField.BUTTONS}").border_subtitle = DLabel.ACTIONS
        self.query_one(f"#{DField.MODEL}").border_subtitle = DLabel.MODEL
        self.query_one(f"#{DField.EPSILON}").border_subtitle = DLabel.EPSILON
        self.query_one(f"#{DField.TRAINING_BOX}").border_subtitle = (
            DLabel.TRAINING
        )
        # --------------------------------------------------------------

        # Switch focus to a hidden widget
        self._w_hidden_widget.focus()

        # Initialize ZeroMQ
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

        self.event_log.add_event(ev_type=DField.TUI, event="Initialized...")

    def on_per_episode(self, topic: str, payload: dict) -> None:
        # Unbatch messages
        batch_msg = HydraMsg.from_dict(payload)
        messages = batch_msg.payload[DHydraMQ.MSGS]
        self._cur_per_ep_batch_num = messages[0]["payload"]["count"]
        if self._first_per_ep_batch:
            self._first_per_ep_batch_num = False
            self._prev_per_ep_batch_num = self._cur_per_ep_batch_num - 1
        if (self._cur_per_ep_batch_num - self._prev_per_ep_batch_num) != 1:
            self.event_log.add_event(
                ev_type=DField.WARNING,
                event="Per episode data is being dropped",
            )
        self._prev_per_ep_batch_num = self._cur_per_ep_batch_num

        metrics = self.metrics

        # Process messages
        for payload in messages[1:]:

            # Get the data from the payload
            epoch = payload.get(DGameField.EPOCH)
            elapsed_time = payload.get(DGameField.ELAPSED_TIME)
            cur_epsilon = payload.get(DNetField.CUR_EPSILON)
            cur_loss = payload.get(DNetField.LOSS, None)

            # Load the data into the metrics object
            metrics.add_cur_epoch(epoch)
            metrics.add_elapsed_time(elapsed_time)
            metrics.add_cur_epsilon(cur_epsilon)
            is_new_data = metrics.add_cur_loss(cur_loss)
            if is_new_data:
                self.telemtry.loss_plot.plot_all()

            # Update the TUI
            self._w_board_box.border_title = f"{DLabel.GAME}: {epoch}"
            self.query_one(f"#{DField.HYDRA_TELEMETRY}").border_subtitle = (
                elapsed_time
            )
            self._w_cur_epsilon_label.update(str(round(cur_epsilon, 4)))

    def on_per_step(self, topic: str, payload: dict) -> None:
        # These messages are not batched (there's a MOVE_DELAY)
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
        # Unpack message batch
        batch_msg = HydraMsg.from_dict(payload)
        messages = batch_msg.payload[DHydraMQ.MSGS]
        self._cur_scores_batch_num = messages[0]["payload"]["count"]
        if self._first_scores_batch:
            self._first_scores_batch_num = False
            self._prev_scores_batch_num = self._cur_scores_batch_num - 1
        if (self._cur_scores_batch_num - self._prev_scores_batch_num) != 1:
            self.event_log.add_event(
                ev_type=DField.WARNING, event="Score data is being dropped"
            )
        self._prev_scores_batch_num = self._cur_scores_batch_num

        metrics = self.metrics

        for payload in messages[1:]:

            # Get the data from the payload
            cur_epoch = payload.get(DGameField.EPOCH)
            cur_score = payload.get(DGameField.CUR_SCORE)
            highscore = payload.get(DGameField.HIGHSCORE)
            final_score = payload.get(DNetField.FINAL_SCORE)

            # Load the data into the metrics object
            metrics.add_cur_epoch(cur_epoch)
            metrics.add_cur_score(cur_score)
            is_new_highscore = metrics.add_highscore(highscore)
            if final_score is not None:
                is_new_final_score = metrics.add_final_score(final_score)
                if is_new_final_score:
                    self.telemtry.game_score_plot.plot_cur_scores()
                    self.telemtry.scores_dist_plot.plot_all()

            ## Update the TUI

            # Only update if the highscore is new
            if is_new_highscore:
                highscore_log = self.query_one(
                    f"#{DField.HIGHSCORES_LOG}", HighScoresLog
                )
                highscore_log.border_subtitle = (
                    f"{DLabel.HIGHSCORE}: {highscore}"
                )
                highscore_event = metrics.get_last_highscore_event()
                highscore_log.add_highscore(highscore_event)
                self.event_log.add_event(
                    ev_type=DField.HIGHSCORE,
                    event=f"🎉 New highscore: {highscore}",
                )
                self.telemtry.game_score_plot.plot_highscores()

            # Only update the score, if "per_step" is enabled.
            if self.cfg.get(DNetField.PER_STEP):
                self._w_board_box.border_subtitle = (
                    f"{DLabel.SCORE}: {cur_score:<2}"
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
            batch_size = DLinear.BATCH_SIZE
            dropout_p = DLinear.DROPOUT_P
            epsilon_decay = DLinear.EPSILON_DECAY_RATE
            gamma = DLinear.GAMMA
            hidden_size = DLinear.HIDDEN_SIZE
            initial_epsilon = DLinear.INITIAL_EPSILON
            lr = f"{DLinear.LEARNING_RATE:.5f}"
            min_epsilon = DLinear.MINIMUM_EPSILON
            self.remove_class(DField.RNN)
            self.add_class(DField.LINEAR)

        # RNN model defaults
        elif model_type == DField.RNN:
            batch_size = DRNN.BATCH_SIZE
            dropout_p = DRNN.DROPOUT_P_VALUE
            epsilon_decay = DRNN.EPSILON_DECAY_RATE
            gamma = DRNN.GAMMA
            hidden_size = DRNN.HIDDEN_SIZE
            initial_epsilon = DRNN.INITIAL_EPSILON
            lr = f"{DRNN.LEARNING_RATE:.4f}"
            min_epsilon = DRNN.MINIMUM_EPSILON
            self.remove_class(DField.LINEAR)
            self.add_class(DField.RNN)

        else:
            raise ValueError(f"EFFOR: Unrecognized model type {model_type}")

        # Update widgets with model specific defaults
        self._w_batch_size_input.value = str(batch_size)
        self._w_dropout_p_input.value = str(dropout_p)
        self._w_epsilon_decay_input.value = str(epsilon_decay)
        self._w_gamma_input.value = str(gamma)
        self._w_hidden_size_input.value = str(hidden_size)
        self._w_initial_epsilon_input.value = str(initial_epsilon)
        self._w_learning_rate_input.value = lr
        self._w_min_epsilon_input.value = str(min_epsilon)

        # Update HydraMetrics
        self.metrics.set_initial_epsilon(initial_epsilon)

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.control.id != DField.TURBO_MODE:
            return

        # Turbo enabled, replaces the Input with a Label showing "0.0"
        if event.value:
            self.remove_class(DField.TURBO_OFF)
            self.add_class(DField.TURBO_ON)
            self.event_log.add_event(
                ev_type=DField.SIM_LOOP,
                event="Turbo mode enabled: Game rendering disabled, move delay set to 0.0",
            )
            self.mq.disable_per_step_sub()
        # Turbo disabled
        else:
            self.remove_class(DField.TURBO_ON)
            self.add_class(DField.TURBO_OFF)
            move_delay = self.query_one(f"#{DField.MOVE_DELAY_INPUT}").value
            self.event_log.add_event(
                ev_type=DField.SIM_LOOP,
                event=f"Turbo mode disabled: Game rendering enabled, move delay set to {move_delay}",
            )
            self.mq.enable_per_step_sub()

        self._update_tui_labels()
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.UPDATE_CONFIG,
            payload=self.cfg.to_dict(),
        )
        await self.mq.send(msg)

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

            # ----- Simulation is running ---
            if sim_running:
                # Always receive per-episode updates
                self.mq.enable_per_episode_sub()

                # Always receive score updates
                self.mq.enable_scores_sub()
                self.add_class(DField.SIM_RUNNING)
                self.cfg = SimCfg.from_dict(reply.payload)

                # Turbo mode disabled
                if self.cfg.get(DNetField.PER_STEP):
                    self.mq.enable_per_step_sub()
                    self.remove_class(DField.TURBO_ON)
                    self.add_class(DField.TURBO_OFF)
                    self.query_one(f"#{DField.TURBO_MODE}", Switch).value = (
                        False
                    )

                # Turbo mode enabled
                else:
                    self.mq.disable_per_step_sub()
                    self.remove_class(DField.TURBO_OFF)
                    self.add_class(DField.TURBO_ON)
                    self.query_one(f"#{DField.TURBO_MODE}", Switch).value = (
                        True
                    )

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
                self.event_log.add_event(
                    ev_type=DField.SIM_LOOP,
                    event="Connecting to running simulation",
                )

            # ----- Simulation is not running ---
            else:
                self.add_class(DField.SIM_STOPPED)
                self.event_log.add_event(
                    ev_type=DField.SIM_LOOP,
                    event="Connected to simulation server",
                )
            self.remove_class(DField.BAD_HANDSHAKE)

        # ----- Cannot connect to Simulation Server (HydraMgr)
        except asyncio.TimeoutError:
            self.event_log.add_event(
                ev_type=DField.SIM_LOOP,
                event="Unable to connect to simulation server",
            )

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
            self.event_log.add_event(
                ev_type=DField.SIM_LOOP, event="Simulation started"
            )

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
        if self.query_one(f"#{DField.TURBO_MODE}", Switch).value:
            self.mq.disable_per_step_sub()
        else:
            self.mq.enable_per_step_sub()

    async def _take_snapshot(self):
        # Snapshot file
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M.txt")
        snapshot_file = DFile.BASE_SNAPSHOT + timestamp
        fq_snapshot = os.path.join(self._hydra_dir, snapshot_file)
        self.snapshot.create_snapshot(snap_file=fq_snapshot, cfg=self.cfg)

        self.event_log.add_event(
            ev_type=DField.SNAPSHOT,
            event=f"Created snapshot file: {fq_snapshot}",
        )

    def _update_tui_labels(self):
        """
        Update the SimCfg settings and the TUI labels
        """
        # Batch size
        batch_size = self._w_batch_size_input.value
        self._w_batch_size_label.update(batch_size)
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
        per_step = not self.query_one(f"#{DField.TURBO_MODE}", Switch).value
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
        # Discount/Gamma
        gamma = self._w_gamma_input.value
        self._w_gamma_label.update(gamma)
        # Dropout p-value
        dropout_p = self._w_dropout_p_input.value
        self._w_dropout_p_label.update(dropout_p)
        # RNN layers
        rnn_layers = self._w_rnn_layers_input.value
        self._w_rnn_layers_label.update(rnn_layers)
        # RNN Tau
        rnn_tau = self._w_rnn_tau_input.value
        self._w_rnn_tau_label.update(rnn_tau)
        # Sequence length
        seq_length = self._w_seq_length_input.value
        self._w_seq_length_label.update(seq_length)

        self.cfg.apply(
            {
                DNetField.BATCH_SIZE: batch_size,
                DNetField.DROPOUT_P: dropout_p,
                DNetField.EPSILON_DECAY: epsilon_decay,
                DNetField.GAMMA: gamma,
                DNetField.HIDDEN_SIZE: hidden_size,
                DNetField.INITIAL_EPSILON: initial_epsilon,
                DNetField.LEARNING_RATE: learning_rate,
                DNetField.MIN_EPSILON: min_epsilon,
                DNetField.PER_STEP: per_step,
                DNetField.MODEL_TYPE: model_type,
                DNetField.MOVE_DELAY: move_delay,
                DNetField.RANDOM_SEED: random_seed,
                DNetField.RNN_LAYERS: rnn_layers,
                DNetField.RNN_TAU: rnn_tau,
                DNetField.SEQ_LENGTH: seq_length,
            }
        )


def main() -> None:
    router = HydraClientTui()
    router.run()


if __name__ == "__main__":
    main()
