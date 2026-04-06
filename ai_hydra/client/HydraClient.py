# ai_hydra/client/HydraClient.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

# General Python Modules
import argparse
import asyncio
from pathlib import Path
import os
from datetime import datetime
import time

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
    DHydraServerDef,
    DMethod,
    DModule,
)
from ai_hydra.constants.DEvent import EV_TYPE, EV_STATUS
from ai_hydra.constants.DHydraMQ import DHydraMQDef, DHydraMQ, DEvent
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.constants.DGame import DGameField, DGameMethod
from ai_hydra.constants.DNNet import (
    DNetField,
    DNetDef,
    DLinear,
    DRNN,
    DGRU,
    MODEL_TYPE_TABLE,
)
from ai_hydra.constants.DEpsilonNice import DEpsilonNice
from ai_hydra.constants.DReplayMemory import DMemDef

# AI Hydra Modules
from ai_hydra.zmq.HydraClientMQ import HydraClientMQ
from ai_hydra.zmq.HydraMsg import HydraMsg
from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.utils.HydraSnapshot import HydraSnapshot
from ai_hydra.utils.HydraMetrics import HydraMetrics
from ai_hydra.client.ClientGameBoard import ClientGameBoard
from ai_hydra.client.HighScoresLog import HighScoresLog
from ai_hydra.client.HydraTelemetry import HydraTelemetry
from ai_hydra.client.ATHMemory import ATHMemory
from ai_hydra.client.TabbedSettings import TabbedSettings


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
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        router_hb_port: int = DHydraRouterDef.HEARTBEAT_PORT,
        server_address: str = DHydraServerDef.HOSTNAME,
        server_pub_port: int = DHydraServerDef.PUB_PORT,
    ) -> None:
        """Constructor"""
        super().__init__()

        self._router_address: str = router_address
        self._router_port = router_port
        self._router_hb_port = router_hb_port
        self._server_address = server_address
        self._server_pub_port = server_pub_port
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
            if self.mq is None:
                await asyncio.sleep(0.1)
                continue

            if self.mq.connected():
                status = DStatus.GOOD
            else:
                status = DStatus.BAD

            self.settings.router_hb_status.update(status)
            await asyncio.sleep(DHydra.HEARTBEAT_INTERVAL)

    def compose(self) -> ComposeResult:
        """The TUI is created here"""

        ## We're using a Textual "grid" layout...

        # ----- Title ---
        yield Label(DLabel.CLIENT_TITLE, id=DField.TITLE)

        # ----- Buttons ---
        yield Vertical(
            Button(label=DLabel.HANDSHAKE, id=DField.HANDSHAKE, compact=True),
            Label(id=DField.HANDSHAKE_DIVIDER),
            Button(label=DLabel.START, id=DField.START_RUN, compact=True),
            Label(id=DField.START_RUN_DIVIDER),
            Button(label=DLabel.PAUSE, id=DField.PAUSE_RUN, compact=True),
            Label(id=DField.PAUSE_RUN_DIVIDER),
            Button(label=DLabel.RESUME, id=DField.RESUME_RUN, compact=True),
            Label(id=DField.RESUME_RUN_DIVIDER),
            Button(label=DLabel.SNAPSHOT, id=DField.SNAPSHOT, compact=True),
            Label(id=DField.SNAPSHOT_DIVIDER),
            Button(label=DLabel.RESET, id=DField.RESET_RUN, compact=True),
            Label(id=DField.RESET_RUN_DIVIDER),
            Button(label=DLabel.QUIT, id=DMethod.QUIT, compact=True),
            id=DField.BUTTONS,
        )

        # ----- Tabbed Settings ---
        yield TabbedSettings(id=DField.TABBED_SETTINGS)

        # ------ The Snake Game ---
        yield Vertical(self.game_board, id=DField.BOARD_BOX)

        # ----- Highscores widget ---
        yield HighScoresLog(id=DField.HIGHSCORES_LOG)

        # Random seed
        yield Horizontal(
            Input(
                type=DField.INTEGER,
                compact=True,
                valid_empty=False,
                value=str(DHydra.RANDOM_SEED),
                id=DField.RANDOM_SEED_INPUT,
            ),
            Label(str(DHydra.RANDOM_SEED), id=DField.RANDOM_SEED_LABEL),
            id=DField.RANDOM_SEED_BOX,
        )

        # Move delay
        yield Horizontal(
            Label(f"0.0", id=DField.MOVE_DELAY_LABEL),
            Input(
                type=DField.NUMBER,
                compact=True,
                valid_empty=False,
                value=str(DNetDef.MOVE_DELAY),
                id=DField.MOVE_DELAY_INPUT,
            ),
            id=DField.MOVE_DELAY_BOX,
        )

        # Turbo mode
        yield Vertical(
            Switch(value=False, id=DField.TURBO_MODE),
            classes=DField.INPUT_FIELD,
            id=DField.TURBO_BOX,
        )

        # Blank Label to make a nice black filler
        yield Label(" " * 44, id=DField.BLANK)

        # Realtime Telemetry (plots and event log)
        yield HydraTelemetry(metrics=self.metrics, id=DField.HYDRA_TELEMETRY)

        # ATH Memory widget
        yield ATHMemory(
            metrics=self.metrics,
            max_buckets=DMemDef.MAX_BUCKETS,
            mem_id=DField.REPLAY_MEM,
            mem_label=DLabel.MEMORY,
            id=DField.ATH_Memory,
        )

        # ATH Memory widget
        yield ATHMemory(
            metrics=self.metrics,
            max_buckets=DMemDef.MAX_BUCKETS,
            mem_id=DField.MCTS_REPLAY_MEM,
            mem_label=DLabel.MCTS_MEMORY,
            id=DField.MCTS_MEMORY,
        )

        # Focus widget: This is hidden, but it allows me to move focus away
        # from the selected button when a button is clicked.
        yield Checkbox(id=DField.HIDDEN_WIDGET)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if self.mq is None:
            raise TypeError("self.mq is None!!!")

        elif button_id == DField.HANDSHAKE:
            await self._send_handshake()

        elif button_id == DField.PAUSE_RUN:
            await self._send_pause_run()

        elif button_id == DMethod.QUIT:
            await self.on_quit()

        elif button_id == DGameMethod.RESET_RUN:
            await self._send_reset()

        elif button_id == DGameMethod.RESUME_RUN:
            await self._send_resume_run()

        elif button_id == DField.SNAPSHOT:
            await self._take_snapshot()

        elif button_id == DGameMethod.START_RUN:
            self._update_tui_labels()
            self.mq.enable_per_episode_sub()
            self.mq.enable_scores_sub()
            self.mq.enable_events_sub()
            if self.cfg.get(DNetField.PER_STEP):
                self.mq.enable_per_step_sub()
            else:
                self.mq.disable_per_step_sub()
            await self._send_start_run()
            mcts_freq = self.settings.mcts_frequency_input.value
            self.metrics.set_mcts_freq(frequency=mcts_freq)

        elif button_id == DGameMethod.STOP_RUN:
            await self._send_stop_run()

        elif button_id == DField.UPDATE_RUNTIME_CONFIG:
            self._update_tui_labels()
            await self._send_update_config()
        # Remove focus from the clicked button (looks nicer)
        self.query_one(f"#{DField.HIDDEN_WIDGET}", Checkbox).focus()

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

        # Handy references
        self.telemtry = self.query_one(
            f"#{DField.HYDRA_TELEMETRY}", HydraTelemetry
        )
        self.event_log = self.telemtry.event_log
        self.settings = self.query_one(
            f"#{DField.TABBED_SETTINGS}", TabbedSettings
        )

        # Show the network config in the TUI
        self.settings.router_address.update(self._router_address)
        self.settings.router_port.update(str(self._router_port))
        self.settings.router_hb_port.update(str(self._router_hb_port))
        self.settings.server_address.update(self._server_address)
        self.settings.server_pub_port.update(str(self._server_pub_port))

        # Seed the highscores with a zero for plotting
        self.metrics.add_highscore(0)
        self.telemtry.game_score_plot.plot_highscores()

        # Create references to some TUI element that are frequently updated
        # in the ZeroMQ listening loop.
        self._w_board_box = self.query_one(f"#{DField.BOARD_BOX}", Vertical)

        # Network

        # Monitor the connection to the server in the background
        self.check_heartbeat()

        # ----- Add some text to the borders around the widgets
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(
            f"#{DField.HIGHSCORES_LOG}", HighScoresLog
        ).border_subtitle = DLabel.HIGHSCORES
        self.query_one(f"#{DField.BUTTONS}").border_subtitle = DLabel.ACTIONS
        self.query_one(f"#{DField.TURBO_BOX}").border_subtitle = (
            DLabel.TURBO_MODE
        )
        self.query_one(f"#{DField.RANDOM_SEED_BOX}").border_subtitle = (
            DLabel.RANDOM_SEED
        )
        self.query_one(f"#{DField.MOVE_DELAY_BOX}").border_subtitle = (
            DLabel.MOVE_DELAY
        )
        self.query_one(f"#{DField.TABBED_SETTINGS}").border_title = (
            DLabel.SETTINGS
        )
        # --------------------------------------------------------------

        # Switch focus to a hidden widget
        self.query_one(f"#{DField.HIDDEN_WIDGET}", Checkbox).focus()

        # Initialize ZeroMQ
        self.mq = HydraClientMQ(
            router_address=self._router_address,
            router_port=self._router_port,
            router_hb_port=self._router_hb_port,
            identity=self._identity,
            server_address=self._server_address,
            server_pub_port=self._server_pub_port,
        )
        self.mq.sub_methods = {
            self.mq.topic(DHydraMQDef.EVENTS_TOPIC): self._on_sim_event,
            self.mq.topic(DHydraMQDef.PER_STEP_TOPIC): self.on_per_step,
            self.mq.topic(DHydraMQDef.PER_EPISODE_TOPIC): self.on_per_episode,
            self.mq.topic(DHydraMQDef.SCORES_TOPIC): self.on_scores,
        }
        self.mq.start()
        self.mq.disable_events_sub()
        self.mq.disable_per_episode_sub()
        self.mq.disable_per_step_sub()
        self.mq.disable_scores_sub()

        self.event_log.add_event(
            ev_type=DField.TUI, status=EV_STATUS.INFO, event="Initialized..."
        )

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
                status=EV_STATUS.BAD,
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
            self.settings.cur_epsilon.update(str(round(cur_epsilon, 4)))

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
                ev_type=DField.WARNING,
                status=EV_STATUS.BAD,
                event="Score data is being dropped",
            )
        self._prev_scores_batch_num = self._cur_scores_batch_num

        metrics = self.metrics

        for payload in messages[1:]:

            # Get the data from the payload
            cur_epoch = payload.get(DGameField.EPOCH)
            cur_score = payload.get(DGameField.CUR_SCORE)
            highscore = payload.get(DGameField.HIGHSCORE)
            final_score = payload.get(DNetField.FINAL_SCORE)
            final_mcts_score = payload.get(DNetField.FINAL_MCTS_SCORE)

            # Load the data into the metrics object
            metrics.add_cur_epoch(cur_epoch)
            metrics.add_cur_score(cur_score)
            is_new_highscore = metrics.add_highscore(highscore)
            if final_score is not None:
                is_new_final_score = metrics.add_final_score(final_score)
                if is_new_final_score:
                    self.telemtry.game_score_plot.plot_cur_scores()
                    self.telemtry.scores_dist_plot.plot_all()
            if final_mcts_score is not None:
                is_new_final_score = metrics.add_final_mcts_score(
                    final_mcts_score
                )
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
            closer_to_food = DLinear.CLOSER_TO_FOOD
            downshift_thresh = DLinear.DOWNSHIFT_COUNT_THRESHOLD
            dropout_p = DLinear.DROPOUT_P
            empty_move_reward = DLinear.EMPTY_MOVE_REWARD
            epsilon_decay = DLinear.EPSILON_DECAY_RATE
            further_from_food = DLinear.FURTHER_FROM_FOOD
            gamma = DLinear.GAMMA
            hidden_size = DLinear.HIDDEN_SIZE
            initial_epsilon = DLinear.INITIAL_EPSILON
            layers = DLinear.LINEAR_LAYERS
            lr = f"{DLinear.LEARNING_RATE:.5f}"
            min_epsilon = DLinear.MINIMUM_EPSILON
            max_frames = DLinear.MAX_FRAMES
            max_gear = DLinear.MAX_GEAR
            max_crit_stag_eps = DLinear.MAX_HARD_RESET_EPISODES
            max_moves_multiplier = DLinear.MAX_MOVES_MULTIPLIER
            max_moves_penalty = DLinear.MAX_MOVES_PENALTY
            max_stag_eps = DLinear.MAX_STAGNANT_EPISODES
            max_training_frames = DLinear.MAX_TRAINING_FRAMES
            mcts_depth = DLinear.MCTS_DEPTH
            mcts_iter = DLinear.MCTS_ITER
            mcts_explore_p_value = DLinear.MCTS_EXPLORE_P_VALUE
            mcts_frequency = DLinear.MCTS_FREQUENCY
            nice_p_value = DLinear.NICE_P_VALUE
            nice_steps = DLinear.NICE_STEPS
            num_cooldown_eps = DLinear.NUM_COOLDOWN_EPISODES
            upshift_thresh = DLinear.UPSHIFT_COUNT_THRESHOLD
            self.remove_class(DField.RNN)
            self.remove_class(DField.GRU)
            self.add_class(DField.LINEAR)

        # RNN model defaults
        elif model_type == DField.RNN:
            closer_to_food = DRNN.CLOSER_TO_FOOD
            downshift_thresh = DRNN.DOWNSHIFT_COUNT_THRESHOLD
            dropout_p = DRNN.DROPOUT_P_VALUE
            empty_move_reward = DRNN.EMPTY_MOVE_REWARD
            epsilon_decay = DRNN.EPSILON_DECAY_RATE
            further_from_food = DRNN.FURTHER_FROM_FOOD
            gamma = DRNN.GAMMA
            hidden_size = DRNN.HIDDEN_SIZE
            initial_epsilon = DRNN.INITIAL_EPSILON
            layers = DRNN.RNN_LAYERS
            lr = f"{DRNN.LEARNING_RATE:.5f}"
            min_epsilon = DRNN.MINIMUM_EPSILON
            max_frames = DRNN.MAX_FRAMES
            max_gear = DRNN.MAX_GEAR
            max_crit_stag_eps = DRNN.MAX_HARD_RESET_EPISODES
            max_moves_multiplier = DRNN.MAX_MOVES_MULTIPLIER
            max_moves_penalty = DRNN.MAX_MOVES_PENALTY
            max_stag_eps = DRNN.MAX_STAGNANT_EPISODES
            max_training_frames = DRNN.MAX_TRAINING_FRAMES
            mcts_depth = DRNN.MCTS_DEPTH
            mcts_iter = DRNN.MCTS_ITER
            mcts_explore_p_value = DRNN.MCTS_EXPLORE_P_VALUE
            mcts_frequency = DRNN.MCTS_FREQUENCY
            nice_p_value = DRNN.NICE_P_VALUE
            nice_steps = DRNN.NICE_STEPS
            num_cooldown_eps = DRNN.NUM_COOLDOWN_EPISODES
            upshift_thresh = DRNN.UPSHIFT_COUNT_THRESHOLD
            self.remove_class(DField.LINEAR)
            self.remove_class(DField.GRU)
            self.add_class(DField.RNN)

        # GRU model defaults
        elif model_type == DField.GRU:
            closer_to_food = DGRU.CLOSER_TO_FOOD
            downshift_thresh = DGRU.DOWNSHIFT_COUNT_THRESHOLD
            dropout_p = DGRU.DROPOUT_P_VALUE
            empty_move_reward = DGRU.EMPTY_MOVE_REWARD
            epsilon_decay = DGRU.EPSILON_DECAY_RATE
            further_from_food = DGRU.FURTHER_FROM_FOOD
            gamma = DGRU.GAMMA
            hidden_size = DGRU.HIDDEN_SIZE
            initial_epsilon = DGRU.INITIAL_EPSILON
            layers = DGRU.GRU_LAYERS
            lr = f"{DGRU.LEARNING_RATE:.5f}"
            min_epsilon = DGRU.MINIMUM_EPSILON
            max_frames = DGRU.MAX_FRAMES
            max_gear = DGRU.MAX_GEAR
            max_crit_stag_eps = DGRU.MAX_HARD_RESET_EPISODES
            max_moves_multiplier = DGRU.MAX_MOVES_MULTIPLIER
            max_moves_penalty = DGRU.MAX_MOVES_PENALTY
            max_stag_eps = DGRU.MAX_STAGNANT_EPISODES
            max_training_frames = DGRU.MAX_TRAINING_FRAMES
            mcts_depth = DGRU.MCTS_DEPTH
            mcts_iter = DGRU.MCTS_ITER
            mcts_explore_p_value = DGRU.MCTS_EXPLORE_P_VALUE
            mcts_frequency = DGRU.MCTS_FREQUENCY
            nice_p_value = DGRU.NICE_P_VALUE
            nice_steps = DGRU.NICE_STEPS
            num_cooldown_eps = DGRU.NUM_COOLDOWN_EPISODES
            upshift_thresh = DGRU.UPSHIFT_COUNT_THRESHOLD
            self.remove_class(DField.LINEAR)
            self.remove_class(DField.RNN)
            self.add_class(DField.GRU)

        else:
            raise ValueError(f"ERROR: Unrecognized model type {model_type}")

        # Update widgets with model specific defaults
        self.query_one(f"#{DField.DROPOUT_P_INPUT}", Input).value = str(
            dropout_p
        )
        self.query_one(f"#{DField.EPSILON_DECAY_INPUT}", Input).value = str(
            epsilon_decay
        )
        self.query_one(f"#{DField.GAMMA_INPUT}", Input).value = str(gamma)
        self.query_one(f"#{DField.HIDDEN_SIZE_INPUT}", Input).value = str(
            hidden_size
        )
        self.query_one(f"#{DField.INITIAL_EPSILON_INPUT}", Input).value = str(
            initial_epsilon
        )
        self.query_one(f"#{DField.LEARNING_RATE_INPUT}", Input).value = lr
        self.query_one(f"#{DField.MIN_EPSILON_INPUT}", Input).value = str(
            min_epsilon
        )
        self.query_one(f"#{DField.RNN_LAYERS_INPUT}", Input).value = str(
            layers
        )
        self.settings.downshift_count_threshold_input.value = str(
            downshift_thresh
        )
        self.settings.max_frames_input.value = str(max_frames)
        self.settings.max_gear_input.value = str(max_gear)
        self.settings.max_crit_stag_eps_input.value = str(max_crit_stag_eps)
        self.settings.max_stag_eps_input.value = str(max_stag_eps)
        self.settings.max_training_frames_input.value = str(
            max_training_frames
        )
        # Nice
        self.settings.nice_p_value_input.value = str(nice_p_value)
        self.settings.nice_steps_input.value = str(nice_steps)
        # Monte Carlo Tree Search
        self.settings.mcts_depth_input.value = str(mcts_depth)
        self.settings.mcts_explore_p_value_input.value = str(
            mcts_explore_p_value
        )
        self.settings.mcts_frequency_input.value = str(mcts_frequency)
        self.settings.mcts_iter_input.value = str(mcts_iter)
        # ATH Memeory
        self.settings.num_cooldown_eps_input.value = str(num_cooldown_eps)
        self.settings.upshift_count_threshold_input.value = str(upshift_thresh)
        # Rewards
        self.settings.max_moves_multiplier_input.value = str(
            max_moves_multiplier
        )
        self.settings.closer_to_food_input.value = str(closer_to_food)
        self.settings.further_from_food_input.value = str(further_from_food)
        self.settings.max_moves_penalty_input.value = str(max_moves_penalty)
        self.settings.empty_move_reward_input.value = str(empty_move_reward)

        # Update HydraMetrics
        self.metrics.set_initial_epsilon(initial_epsilon)
        self._update_tui_labels()

    async def _on_sim_event(self, topic: str, payload) -> None:
        sender = payload.get(DEvent.SENDER)
        message = payload.get(DEvent.MESSAGE)
        ev_type = payload.get(DEvent.EV_TYPE)
        ev_payload = payload.get(DEvent.PAYLOAD)
        status = payload.get(DEvent.LEVEL)

        # Send event to be displayed in the EventLog widget
        if message is not None:
            cur_epoch = self.metrics.get_cur_epoch()
            self.event_log.add_event(
                ev_type=sender, status=status, event=message, epoch=cur_epoch
            )

        ## Check for additional data

        if sender == DModule.ATH_GEARBOX:

            # Shifting gears
            if ev_type == EV_TYPE.SHIFTING:
                batch_size = ev_payload[DField.BATCH_SIZE]
                seq_length = ev_payload[DField.SEQ_LENGTH]
                gear = ev_payload[DField.GEAR]
                is_mcts = ev_payload[DField.MCTS_MEMORY]

                # We only display the main memory seq_length and batch_size
                # in the TUI, not the MCTS values (they can be seen in the
                # event log widget)

                if not is_mcts:
                    self.query_one(
                        f"#{DField.RNN_BATCH_SIZE_LABEL}", Label
                    ).update(str(batch_size))
                    self.query_one(
                        f"#{DField.SEQ_LENGTH_LABEL}", Label
                    ).update(str(seq_length))

                    self.metrics.add_shift_event(
                        gear=gear, seq_length=seq_length, batch_size=batch_size
                    )

        if sender == DModule.ATH_DATA_MGR:

            # Bucket status
            if ev_type == EV_TYPE.BUCKETS_STATUS:
                # The numeric dictionary keys are turned into string by JSON
                # Convert them back to ints.
                raw_bucket_counts = ev_payload[EV_TYPE.BUCKET_COUNTS]
                cur_gear = ev_payload[EV_TYPE.CUR_GEAR]
                bucket_counts = {
                    int(k): v for k, v in raw_bucket_counts.items()
                }
                # Add the data to the metric object
                self.metrics.add_bucket_stats(
                    bucket_counts=bucket_counts, gear=cur_gear
                )
                # Let the TUI widget know there's new data
                self.query_one(
                    f"#{DField.ATH_Memory}", ATHMemory
                ).refresh_data()

            # Bucket status
            if ev_type == EV_TYPE.MCTS_BUCKETS_STATUS:
                # The numeric dictionary keys are turned into string by JSON
                # Convert them back to ints.
                raw_bucket_counts = ev_payload[EV_TYPE.BUCKET_COUNTS]
                cur_gear = ev_payload[EV_TYPE.CUR_GEAR]
                bucket_counts = {
                    int(k): v for k, v in raw_bucket_counts.items()
                }
                # Add the data to the metric object
                self.metrics.add_mctp_bucket_stats(
                    bucket_counts=bucket_counts, gear=cur_gear
                )
                # Let the TUI widget know there's new data
                self.query_one(
                    f"#{DField.MCTS_MEMORY}", ATHMemory
                ).refresh_data()

        if sender == DModule.EPSILON_NICE_ALGO:
            window = ev_payload[DEpsilonNice.WINDOW]
            epoch = ev_payload[DEpsilonNice.EPOCH]
            calls = ev_payload[DEpsilonNice.CALLS]
            triggered = ev_payload[DEpsilonNice.TRIGGERED]
            fatal_suggested = ev_payload[DEpsilonNice.FATAL_SUGGESTED]
            overrides = ev_payload[DEpsilonNice.OVERRIDES]
            no_safe_alternatives = ev_payload[DEpsilonNice.NO_SAFE_ALTERNATIVE]
            trigger_rate = ev_payload[DEpsilonNice.TRIGGER_RATE]
            override_rate = ev_payload[DEpsilonNice.OVERRIDE_RATE]
            self.metrics.add_nice_event(
                window=window,
                epoch=epoch,
                calls=calls,
                triggered=triggered,
                fatal_suggested=fatal_suggested,
                overrides=overrides,
                no_safe_alternative=no_safe_alternatives,
                trigger_rate=trigger_rate,
                override_rate=override_rate,
            )

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.control.id != DField.TURBO_MODE:
            return

        # Turbo enabled, replaces the Input with a Label showing "0.0"
        if event.value:
            self.remove_class(DField.TURBO_OFF)
            self.add_class(DField.TURBO_ON)
            self.mq.disable_per_step_sub()
            self._w_board_box.border_subtitle = ""
        # Turbo disabled
        else:
            self.remove_class(DField.TURBO_ON)
            self.add_class(DField.TURBO_OFF)
            move_delay = self.query_one(f"#{DField.MOVE_DELAY_INPUT}").value
            self.mq.enable_per_step_sub()

        self._update_tui_labels()
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.UPDATE_CONFIG,
            payload=self.cfg.to_dict(),
        )
        await self.mq.send(msg)

    def reset_tui(self):
        # Reset metrics data
        initial_epsilon = self.query_one(
            f"#{DField.INITIAL_EPSILON_INPUT}", Input
        )
        self.metrics.clear(initial_epsilon=initial_epsilon)
        # Clear widgets
        self.query_one(f"#{DField.HIGHSCORES_LOG}", HighScoresLog).clear()
        self.query_one(f"#{DField.HYDRA_TELEMETRY}", HydraTelemetry).clear()
        self._w_board_box.border_title = ""
        self.query_one(f"#{DField.HYDRA_TELEMETRY}").border_subtitle = ""
        self.query_one(
            f"#{DField.HIGHSCORES_LOG}", HighScoresLog
        ).border_subtitle = ""
        self.query_one(f"#{DField.CUR_EPSILON}", Label).update("")
        self.query_one(f"#{DField.RNN_BATCH_SIZE_LABEL}").update("")
        self.query_one(f"#{DField.SEQ_LENGTH_LABEL}").update("")

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
                # Always receive per-episode and event updates
                self.mq.enable_per_episode_sub()
                self.mq.enable_events_sub()

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

                # Simulation is paused
                if self.cfg.get(DNetField.SIM_PAUSED):
                    print("Is paused")
                    self.add_class(DGameField.SIM_PAUSED)
                    self.remove_class(DGameField.SIM_RESUMED)
                else:
                    print("Not paused")
                    self.add_class(DGameField.SIM_RESUMED)
                    self.remove_class(DGameField.SIM_PAUSED)

                # Load configurable TUI settings from the running sim config
                self.query_one(
                    f"#{DField.INITIAL_EPSILON_INPUT}", Input
                ).value = str(self.cfg.get(DNetField.INITIAL_EPSILON))
                self.query_one(f"#{DField.MIN_EPSILON_INPUT}", Input).value = (
                    str(self.cfg.get(DNetField.MIN_EPSILON))
                )
                self.query_one(f"#{DField.MOVE_DELAY_INPUT}", Input).value = (
                    str(self.cfg.get(DNetField.MOVE_DELAY))
                )
                self.query_one(
                    f"#{DField.MODEL_TYPE_SELECT}", Select
                ).value = self.cfg.get(DNetField.MODEL_TYPE)
                self.query_one(f"#{DField.HIDDEN_SIZE_INPUT}", Input).value = (
                    str(self.cfg.get(DNetField.HIDDEN_SIZE))
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

    async def _send_pause_run(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.PAUSE_RUN,
        )
        await self.mq.send(msg)

        try:
            reply = await self.mq.recv()
            if reply.method == DGameMethod.PAUSE_RUN_REPLY:
                self.add_class(DGameField.SIM_PAUSED)
                self.remove_class(DGameField.SIM_RESUMED)
        except asyncio.TimeoutError:
            pass

    async def _send_resume_run(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.RESUME_RUN,
        )
        await self.mq.send(msg)

        try:
            reply = await self.mq.recv()
            if reply.method == DGameMethod.RESUME_RUN_REPLY:
                self.remove_class(DGameField.SIM_PAUSED)
                self.add_class(DGameField.SIM_RESUMED)
        except asyncio.TimeoutError:
            pass

    async def _send_reset(self):
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DGameMethod.RESET_RUN,
        )
        await self.mq.send(msg)

        try:
            reply = await self.mq.recv()
            if reply.method == DGameMethod.RESET_RUN_REPLY:
                self.mq.disable_events_sub()
                self.mq.disable_per_episode_sub()
                self.mq.disable_per_step_sub()
                self.mq.disable_scores_sub()
                self.remove_class(DGameField.SIM_PAUSED)
                self.remove_class(DGameField.SIM_RUNNING)
                self.add_class(DField.SIM_STOPPED)
                self.reset_tui()
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
                self.remove_class(DGameField.SIM_PAUSED)
                self.add_class(DField.SIM_RUNNING)
                self.add_class(DGameField.SIM_RESUMED)
            else:
                # Sim was already running
                self.remove_class(DField.SIM_STOPPED)
                self.remove_class(DGameField.SIM_RESUMED)
                self.add_class(DField.SIM_RUNNING)
                self.add_class(DGameField.SIM_PAUSED)
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
        model_type = self.cfg.get(DNetField.MODEL_TYPE)
        snapshot_file = DFile.BASE_SNAPSHOT + f"{model_type}_" + timestamp
        fq_snapshot = os.path.join(self._hydra_dir, snapshot_file)
        self.snapshot.create_snapshot(snap_file=fq_snapshot, cfg=self.cfg)

    def _update_tui_labels(self):
        """
        Update the SimCfg settings and the TUI labels
        """
        # Random Seed
        random_seed = self.query_one(
            f"#{DField.RANDOM_SEED_INPUT}", Input
        ).value
        self.query_one(f"#{DField.RANDOM_SEED_LABEL}", Label).update(
            random_seed
        )

        # Epsilon values
        epsilon_decay = self.query_one(f"#{DField.EPSILON_DECAY_INPUT}").value
        initial_epsilon = self.query_one(
            f"#{DField.INITIAL_EPSILON_INPUT}", Input
        ).value
        min_epsilon = self.query_one(f"#{DField.MIN_EPSILON_INPUT}").value
        self.query_one(f"#{DField.EPSILON_DECAY_LABEL}", Label).update(
            str(epsilon_decay)
        )
        self.query_one(f"#{DField.INITIAL_EPSILON_LABEL}", Label).update(
            str(initial_epsilon)
        )
        self.query_one(f"#{DField.MIN_EPSILON_LABEL}", Label).update(
            str(min_epsilon)
        )

        # Turbo mode == Not per_step
        per_step = not self.query_one(f"#{DField.TURBO_MODE}", Switch).value
        # Move delay
        move_delay = self.query_one(f"#{DField.MOVE_DELAY_INPUT}", Input).value
        # Model type
        model_type = self.query_one(
            f"#{DField.MODEL_TYPE_SELECT}", Select
        ).value
        model_type_label = MODEL_TYPE_TABLE[model_type]
        self.query_one(f"#{DField.MODEL_TYPE_LABEL}", Label).update(
            model_type_label
        )

        # Hidden size
        hidden_size = self.query_one(
            f"#{DField.HIDDEN_SIZE_INPUT}", Input
        ).value
        self.query_one(f"#{DField.HIDDEN_SIZE_LABEL}", Label).update(
            str(hidden_size)
        )

        # Learning rate
        learning_rate = self.query_one(
            f"#{DField.LEARNING_RATE_INPUT}", Input
        ).value
        self.query_one(f"#{DField.LEARNING_RATE_LABEL}", Label).update(
            learning_rate
        )

        # Discount/Gamma
        gamma = self.query_one(f"#{DField.GAMMA_INPUT}", Input).value
        self.query_one(f"#{DField.GAMMA_LABEL}", Label).update(gamma)
        # Dropout p-value
        dropout_p = self.query_one(f"#{DField.DROPOUT_P_INPUT}", Input).value
        self.query_one(f"#{DField.DROPOUT_P_LABEL}", Label).update(dropout_p)
        # RNN/GRU layers
        layers = self.query_one(f"#{DField.RNN_LAYERS_INPUT}", Input).value
        self.query_one(f"#{DField.RNN_LAYERS_LABEL}", Label).update(layers)
        # RNN Tau
        tau = self.query_one(f"#{DField.RNN_TAU_INPUT}", Input).value
        self.query_one(f"#{DField.RNN_TAU_LABEL}", Label).update(tau)

        # ----- Tabbed Settings ---
        # MAX_TRAINING_FRAMES
        max_training_frames = self.settings.max_training_frames_input.value
        self.settings.max_training_frames_label.update(
            str(max_training_frames)
        )
        # MAX_GEAR
        max_gear = self.settings.max_gear_input.value
        self.settings.max_gear_label.update(max_gear)
        # MAX_FRAMES
        max_frames = self.settings.max_frames_input.value
        self.settings.max_frames_label.update(max_frames)
        # MAX_BUCKETS
        max_buckets = self.settings.max_buckets_input.value
        self.settings.max_buckets_label.update(max_buckets)
        # NUM_COOLDOWN_EPISODES
        num_cooldown_eps = self.settings.num_cooldown_eps_input.value
        self.settings.num_cooldown_eps_label.update(num_cooldown_eps)
        # UPSHIFT_COUNT_THRESHOLD
        upshift_count_thresh = (
            self.settings.upshift_count_threshold_input.value
        )
        self.settings.upshift_count_threshold_label.update(
            upshift_count_thresh
        )
        # DOWNSHIFT_COUNT_THRESHOLD
        downshift_count_thresh = (
            self.settings.downshift_count_threshold_input.value
        )
        self.settings.downshift_count_threshold_label.update(
            downshift_count_thresh
        )
        # MAX_STAGNANT_EPISODES
        max_stag_eps = self.settings.max_stag_eps_input.value
        self.settings.max_stag_eps_label.update(max_stag_eps)
        # MAX_HARD_RESET_EPISODES
        max_crit_stag_eps = self.settings.max_crit_stag_eps_input.value
        self.settings.max_crit_stag_eps_label.update(max_crit_stag_eps)

        # NICE_P_VALUE
        nice_p_value = self.settings.nice_p_value_input.value
        self.settings.nice_p_value_label.update(nice_p_value)
        # NICE_STEPS
        nice_steps = self.settings.nice_steps_input.value
        self.settings.nice_steps_label.update(nice_steps)

        # MCTS_DEPTH
        mcts_depth = self.settings.mcts_depth_input.value
        self.settings.mcts_depth_label.update(mcts_depth)
        # MCTS_ITER
        mcts_iter = self.settings.mcts_iter_input.value
        self.settings.mcts_iter_label.update(mcts_iter)
        # MCTS_EXPLORE_P_VALUE
        mcts_explore_p_value = self.settings.mcts_explore_p_value_input.value
        self.settings.mcts_explore_p_value_label.update(mcts_explore_p_value)
        # MCTS_FREQUENCY
        mcts_frequency = self.settings.mcts_frequency_input.value
        self.settings.mcts_frequency_label.update(mcts_frequency)

        # FOOD_REWARD
        food_reward = self.settings.food_rewards_input.value
        self.settings.food_rewards_label.update(food_reward)
        # COLLISION_PENALTY
        collision_penalty = self.settings.collision_penalty_input.value
        self.settings.collision_penalty_label.update(collision_penalty)
        # MAX_MOVES_PENALTY
        max_moves_penalty = self.settings.max_moves_penalty_input.value
        self.settings.max_moves_penalty_label.update(max_moves_penalty)
        # EMPTY_MOVE_REWARD
        empty_move_reward = self.settings.empty_move_reward_input.value
        self.settings.empty_move_reward_label.update(empty_move_reward)
        # CLOSER_TO_FOOD
        closer_to_food = self.settings.closer_to_food_input.value
        self.settings.closer_to_food_label.update(closer_to_food)
        # FURTHER_FROM_FOOD
        further_from_food = self.settings.further_from_food_input.value
        self.settings.further_from_food_label.update(further_from_food)
        # MAX_MOVES_MULTIPLIER
        max_moves_multiplier = self.settings.max_moves_multiplier_input.value
        self.settings.max_moves_multiplier_label.update(max_moves_multiplier)

        cfg_dict = {
            DNetField.CLOSER_TO_FOOD: closer_to_food,
            DNetField.COLLISION_PENALTY: collision_penalty,
            DNetField.DOWNSHIFT_COUNT_THRESHOLD: downshift_count_thresh,
            DNetField.DROPOUT_P: dropout_p,
            DNetField.EMPTY_MOVE_REWARD: empty_move_reward,
            DNetField.EPSILON_DECAY: epsilon_decay,
            DNetField.FOOD_REWARD: food_reward,
            DNetField.FURTHER_FROM_FOOD: further_from_food,
            DNetField.GAMMA: gamma,
            DNetField.HIDDEN_SIZE: hidden_size,
            DNetField.INITIAL_EPSILON: initial_epsilon,
            DNetField.LEARNING_RATE: learning_rate,
            DNetField.MAX_BUCKETS: max_buckets,
            DNetField.MAX_FRAMES: max_frames,
            DNetField.MAX_GEAR: max_gear,
            DNetField.MAX_HARD_RESET_EPISODES: max_crit_stag_eps,
            DNetField.MAX_MOVES_MULTIPLIER: max_moves_multiplier,
            DNetField.MAX_MOVES_PENALTY: max_moves_penalty,
            DNetField.MAX_STAGNANT_EPISODES: max_stag_eps,
            DNetField.MCTS_DEPTH: mcts_depth,
            DNetField.MCTS_EXPLORE_P_VALUE: mcts_explore_p_value,
            DNetField.MCTS_FREQUENCY: mcts_frequency,
            DNetField.MCTS_ITER: mcts_iter,
            DNetField.MIN_EPSILON: min_epsilon,
            DNetField.NICE_P_VALUE: nice_p_value,
            DNetField.NICE_STEPS: nice_steps,
            DNetField.NUM_COOLDOWN_EPISODES: num_cooldown_eps,
            DNetField.PER_STEP: per_step,
            DNetField.MAX_TRAINING_FRAMES: max_training_frames,
            DNetField.MODEL_TYPE: model_type,
            DNetField.MOVE_DELAY: move_delay,
            DNetField.RANDOM_SEED: random_seed,
            DNetField.LAYERS: layers,
            DNetField.TAU: tau,
            DNetField.UPSHIFT_COUNT_THRESHOLD: upshift_count_thresh,
        }
        # For debugging
        # print(cfg_dict)

        self.cfg.apply(cfg_dict)


def main() -> None:

    p = argparse.ArgumentParser(description="AI Hydra Client")
    p.add_argument(
        "--router-address",
        default=DHydraRouterDef.HOSTNAME,
        help=f"Router hostname/IP address ({DHydraRouterDef.HOSTNAME})",
    )
    p.add_argument(
        "--router-port",
        default=DHydraRouterDef.PORT,
        help=f"Router port ({DHydraRouterDef.PORT})",
    )
    p.add_argument(
        "--router-hb-port",
        default=DHydraRouterDef.HEARTBEAT_PORT,
        help=f"Router heartbeat port ({DHydraRouterDef.HEARTBEAT_PORT})",
    )
    p.add_argument(
        "--server-address",
        default=DHydraServerDef.HOSTNAME,
        help=f"Server hostname/IP address ({DHydraServerDef.HOSTNAME})",
    )
    p.add_argument(
        "--server-pub-port",
        default=DHydraServerDef.PUB_PORT,
        help=f"Server PUB port ({DHydraServerDef.PUB_PORT})",
    )

    args = p.parse_args()

    print("Using network configuration:")
    print(f"         Router address: {args.router_address}")
    print(f"            Router port: {args.router_port}")
    print(f"  Router heartbeat port: {args.router_hb_port}")
    print(f"         Server address: {args.server_address}")
    print(f"        Server PUB port: {args.server_pub_port}")

    tui = HydraClientTui(
        router_address=args.router_address,
        router_port=args.router_port,
        router_hb_port=args.router_hb_port,
        server_address=args.server_address,
        server_pub_port=args.server_pub_port,
    )
    tui.run()


if __name__ == "__main__":
    main()
