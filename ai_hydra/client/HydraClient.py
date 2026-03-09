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

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import Button, Label, Input, Checkbox, Select
from textual.message import Message

from ai_hydra.zmq.HydraClientMQ import HydraClientMQ
from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.client.ClientGameBoard import ClientGameBoard
from ai_hydra.client.TabbedScores import TabbedScores
from ai_hydra.client.TabbedPlots import TabbedPlots

from ai_hydra.constants.DHydra import (
    DHydra,
    DHydraRouterDef,
    DMethod,
    DModule,
    DHydraMQDef,
)
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.constants.DGame import DGameField, DGameMethod
from ai_hydra.constants.DNNet import (
    DNetField,
    DEpsilonDef,
    DLookaheadDef,
    DNetDef,
    DLinear,
    DRNN,
    MODEL_TYPE_TABLE,
    MODEL_TYPES,
)
from ai_hydra.constants.DSimCfg import Phase

HYDRA_THEME = Theme(
    name="hydra_theme",
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
        self._cur_lookahead = None
        self.cfg = SimCfg()
        self._running = False
        self._wgt = None

    @work(exclusive=True)
    async def check_heartbeat(self) -> None:
        while True:
            if self.mq.connected():
                status = DStatus.GOOD
            else:
                status = DStatus.BAD

            self.query_one(f"#{DField.ROUTER_HB}", Label).update(
                f"{DLabel.ROUTER:>11s}: {status}"
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
            Button(label=DLabel.QUIT, id=DMethod.QUIT, compact=True),
            id=DField.BUTTONS,
        )

        # ------ The Snake Game --- 2x1
        yield Vertical(self.game_board, id=DField.BOARD_BOX)

        # ------ Network --- 1x1
        yield Vertical(
            Label(f"{DLabel.TARGET_HOST:>11s}: {self._address}"),
            Label(f"{DLabel.TARGET_PORT:>11s}: {self._port}"),
            Label(f"{DLabel.ROUTER:>11s}", id=DField.ROUTER_HB),
            id=DField.NETWORK,
        )

        # ----- Highscores widget --- 2x1
        yield TabbedScores(id=DField.TABBED_SCORES)

        # ----- Runtime Settings ---1x1
        yield Vertical(
            # Random seed
            Horizontal(
                Label(f"{DLabel.RANDOM_SEED:>11s}: "),
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
            Label(),
            # Model selection
            Horizontal(
                Label(f"{DLabel.NN_MODEL}: "),
                Label(DLabel.LINEAR, id=DField.MODEL_TYPE_LABEL),
                Select(
                    MODEL_TYPES,
                    compact=True,
                    id=DField.MODEL_TYPE_SELECT,
                    allow_blank=False,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Learning rate
            Horizontal(
                Label(f"{DLabel.LEARNING_RATE}: "),
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
            Label(),
            # Initial epsilon
            Horizontal(
                Label(f"{DLabel.INITIAL_EPSILON:>15s}: "),
                Label(
                    str(DEpsilonDef.INITIAL),
                    id=DField.INITIAL_EPSILON_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DEpsilonDef.INITIAL),
                    id=DField.INITIAL_EPSILON_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Minimum epsilon
            Horizontal(
                Label(f"{DLabel.MIN_EPSILON:>15s}: "),
                Label(
                    str(DEpsilonDef.MINIMUM),
                    id=DField.MIN_EPSILON_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DEpsilonDef.MINIMUM),
                    id=DField.MIN_EPSILON_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Epsilon decay
            Horizontal(
                Label(f"{DLabel.EPSILON_DECAY:>15s}: "),
                Label(
                    str(DEpsilonDef.DECAY_RATE),
                    id=DField.EPSILON_DECAY_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DEpsilonDef.DECAY_RATE),
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
            Label(),
            # Lookahead p-value
            Horizontal(
                Label(f"{DLabel.LOOKAHEAD_P_VAL:>18}: "),
                Label(
                    str(DLookaheadDef.PROBABILITY),
                    id=DField.LOOKAHEAD_P_VAL_LABEL,
                ),
                Input(
                    type=DField.NUMBER,
                    compact=True,
                    valid_empty=False,
                    value=str(DLookaheadDef.PROBABILITY),
                    id=DField.LOOKAHEAD_P_VAL_INPUT,
                ),
                classes=DField.INPUT_FIELD,
            ),
            # Lookahead enabled
            Label(
                f"{DLabel.LOOKAHEAD_STATUS:>18}: {DStatus.UNKNOWN}",
                id=DField.LOOKAHEAD_STATUS,
            ),
            id=DField.SETTINGS,
        )

        # Plots
        yield TabbedPlots(id=DField.TABBED_PLOTS)

        # Consolr
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

        elif button_id == DGameMethod.START_RUN:
            self._update_tui_labels()
            self.mq.enable_per_episode_sub()
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

        # Create references to TUI elements that are being updated
        self._w_board_box = self.query_one(f"#{DField.BOARD_BOX}", Vertical)
        self._w_console_label = self.query_one(
            f"#{DField.CONSOLE_SCREEN}", Label
        )
        self._w_cur_epsilon_label = self.query_one(
            f"#{DField.CUR_EPSILON}", Label
        )
        self._w_epsilon_decay_input = self.query_one(
            f"#{DField.EPSILON_DECAY_INPUT}", Input
        )
        self._w_epsilon_decay_label = self.query_one(
            f"#{DField.EPSILON_DECAY_LABEL}", Label
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
        self._w_lookahead_enabled = self.query_one(
            f"#{DField.LOOKAHEAD_STATUS}", Label
        )
        self._w_lookahead_p_val_input = self.query_one(
            f"#{DField.LOOKAHEAD_P_VAL_INPUT}", Input
        )
        self._w_lookahead_p_val_label = self.query_one(
            f"#{DField.LOOKAHEAD_P_VAL_LABEL}", Label
        )
        self._w_move_delay_input = self.query_one(
            f"#{DField.MOVE_DELAY_INPUT}", Input
        )
        self._w_tabbed_plots = self.query_one(
            f"#{DField.TABBED_PLOTS}", TabbedPlots
        )
        self._w_tabbed_scores = self.query_one(
            f"#{DField.TABBED_SCORES}", TabbedScores
        )
        self._w_turbo_mode = self.query_one(f"#{DField.TURBO_MODE}", Checkbox)

        self.mq = HydraClientMQ(
            router_address=self._address,
            router_port=self._port,
            identity=self._identity,
            srv_host=self._address,
        )
        self.mq.sub_methods = {
            self.mq.topic(DHydraMQDef.PER_STEP_TOPIC): self.on_per_step,
            self.mq.topic(DHydraMQDef.PER_EPISODE_TOPIC): self.on_per_episode,
        }
        self.mq.start()
        self.mq.disable_per_episode_sub()
        self.mq.disable_per_step_sub()

        # Network
        self.query_one(f"#{DField.ROUTER_HB}", Label).update(
            f"{DLabel.ROUTER:>11s}: {DStatus.UNKNOWN}"
        )

        # Monitor the connection to the server in the background
        self.check_heartbeat()

        # Add some text to the borders around the widgets
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.SETTINGS}").border_subtitle = DLabel.SETTINGS
        self.query_one(f"#{DField.NETWORK}").border_subtitle = DLabel.NETWORK
        self.query_one(f"#{DField.TABBED_SCORES}").border_subtitle = (
            DLabel.HIGHSCORES
        )
        self.query_one(f"#{DField.BUTTONS}").border_subtitle = DLabel.ACTIONS
        self.query_one(f"#{DField.CONSOLE_BOX}", Vertical).border_subtitle = (
            DLabel.CONSOLE
        )
        self._w_tabbed_plots.border_subtitle = DLabel.VISUALIZATIONS
        self._w_hidden_widget.focus()
        self.console_msg("Initialized...")

    def on_per_episode(self, topic: str, payload: dict) -> None:
        info = payload.get(DGameField.INFO, {})

        # Epoch
        epoch = info.get(DGameField.EPOCH)
        self._w_board_box.border_title = f"{DLabel.GAME}: {epoch}"

        # Current score and highscore
        highscore = info.get(DGameField.HIGHSCORE)
        self._w_tabbed_scores.border_subtitle = (
            f"{DLabel.HIGHSCORE}: {highscore}"
        )

        # Lookahead Highscore event
        if DGameField.HIGHSCORE_EVENT_LH in info:
            highscore_event_lh = info[DGameField.HIGHSCORE_EVENT_LH]
            if highscore_event_lh[2]:
                self._w_tabbed_scores.add_highscore_lh(
                    epoch=highscore_event_lh[0],
                    highscore=highscore_event_lh[1],
                    event_time=highscore_event_lh[2],
                )
                self.console_msg(
                    f"🎉 New highscore with look ahead enabled: {highscore_event_lh[1]}"
                )

        # Lookahead Highscore event
        if DGameField.HIGHSCORE_EVENT_NLH in info:
            highscore_event_nlh = info[DGameField.HIGHSCORE_EVENT_NLH]
            if not highscore_event_lh[2]:
                self._w_tabbed_scores.add_highscore_nlh(
                    epoch=highscore_event_nlh[0],
                    highscore=highscore_event_nlh[1],
                    event_time=highscore_event_nlh[2],
                )
                self.console_msg(
                    f"🎉 New highscore without look ahead enabled: {highscore_event_lh[1]}"
                )

        # Current epsilon value
        epsilon = info.get(DNetField.CUR_EPSILON)
        if epsilon is not None:
            self._w_cur_epsilon_label.update(str(round(epsilon, 4)))

        # Lookahead status
        if DNetField.LOOKAHEAD_ON in info:
            if info[DNetField.LOOKAHEAD_ON]:
                cur_lookahead = DStatus.GOOD
            else:
                cur_lookahead = DStatus.BAD
            self._cur_lookahead = cur_lookahead
            self._w_lookahead_enabled.update(
                f"{DLabel.LOOKAHEAD_STATUS:>18}: {cur_lookahead}"
            )

        # Loss
        if DNetField.LOSS in info:
            self._w_tabbed_plots.add_loss(
                epoch,
                info[DNetField.LOSS],
            )

        # Final score
        if DNetField.FINAL_SCORE in info:
            self._w_tabbed_plots.add_score(
                cur_score=info[DNetField.FINAL_SCORE],
                lookahead=info[DNetField.LOOKAHEAD_ON],
            )

    def on_per_step(self, topic: str, payload: dict) -> None:
        score = "N/A"
        score = payload.get(DGameField.SCORE)
        board = payload.get(DGameField.BOARD)
        self.game_board.apply_board_dict(board)
        self._w_board_box.border_subtitle = f"{DLabel.SCORE}: {score:<2}"

    async def on_shutdown_request(self) -> None:
        if self.mq is not None:
            await self.mq.quit()
            self.mq = None

    async def on_quit(self) -> None:
        sys.exit(0)

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

    def on_select_changed(self, event: Select.Changed) -> None:
        # We want to change the default learning rate if the user changes the
        # model.
        if event.control.id == DField.MODEL_TYPE_SELECT:
            if event.value == DField.LINEAR:
                lr = f"{DLinear.LEARNING_RATE:.5f}"  # 0.00005
                self._w_learning_rate_input.value = lr
            elif event.value == DField.RNN:
                lr = f"{DRNN.LEARNING_RATE:.5f}"  # 0.00009
                self._w_learning_rate_input.value = lr
            self.console_msg(f"Updated default learning rate to: {lr}...")

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

    def _update_tui_labels(self):
        """
        Update the SimCfg settings and the TUI labels
        """
        # Epsilon values
        epsilon_decay = self._w_epsilon_decay_input.value
        initial_epsilon = self._w_initial_epsilon_input.value
        min_epsilon = self._w_min_epsilon_input.value
        self._w_cur_epsilon_label.update(str(epsilon_decay))
        self._w_initial_epsilon_label.update(str(initial_epsilon))
        self._w_min_epsilon_label.update(str(min_epsilon))
        # Turbo mode == Not per_step
        per_step = not self._w_turbo_mode.value
        # Lookahead p-value
        lookahead_p_value = self._w_lookahead_p_val_input.value
        self._w_lookahead_p_val_label.update(str(lookahead_p_value))
        # Move delay
        move_delay = self._w_move_delay_input.value
        # Model type
        model_type = self._w_model_type_select.value
        model_type_label = MODEL_TYPE_TABLE[model_type]
        self._w_model_type_label.update(model_type_label)
        # Learning rate
        learning_rate = self._w_learning_rate_input.value
        self._w_learning_rate_label.update(learning_rate)

        self.cfg.apply(
            {
                DNetField.EPSILON_DECAY: epsilon_decay,
                DNetField.INITIAL_EPSILON: initial_epsilon,
                DNetField.LEARNING_RATE: float(learning_rate),
                DNetField.LOOKAHEAD_P_VAL: lookahead_p_value,
                DNetField.MIN_EPSILON: min_epsilon,
                DNetField.PER_STEP: per_step,
                DNetField.MODEL_TYPE: model_type,
                DNetField.MOVE_DELAY: move_delay,
            }
        )


def main() -> None:
    router = HydraClientTui()
    router.run()


if __name__ == "__main__":
    main()
