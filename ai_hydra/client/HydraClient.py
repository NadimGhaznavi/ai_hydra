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
from textual.widgets import Button, Label, Log
from textual.message import Message

from ai_hydra.utils.HydraMQ import HydraMQ
from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.client.ClientGameBoard import ClientGameBoard

from ai_hydra.constants.DHydra import DHydra, DHydraRouterDef, DMethod, DModule
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.constants.DGame import DGameField, DGameMethod
from ai_hydra.constants.DNet import (
    DNetField,
    DEpsilonDef,
    DLookahead,
    DLookaheadDef,
)

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
    CSS_PATH = DFile.CLIENT_CSS_PATH

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
        self.mq: HydraMQ | None = None
        self._telemetry_prefix = "telemetry.snake.trainer"

        self.game_board = ClientGameBoard(20, id=DGameField.BOARD)

    @work(exclusive=True)
    async def check_connection_bg(self) -> None:
        while True:
            mq = self.mq
            if mq is None:
                raise TypeError("self.mq is None!!!")

            if mq.connected():
                self._connected_msg = DStatus.GOOD + " " + DLabel.CONNECTED
            else:
                self._connected_msg = DStatus.BAD + " " + DLabel.DISCONNECTED

            self.query_one(f"#{DField.CONNECTED}", Label).update(
                self._connected_msg
            )

            await asyncio.sleep(DHydra.HEARTBEAT_INTERVAL + 1)

    def compose(self) -> ComposeResult:
        """The TUI is created here"""

        # Title
        yield Label(DLabel.CLIENT_TITLE, id=DField.TITLE)

        # Configuration
        yield Vertical(
            Label(f"{DLabel.TARGET_HOST}: {self._address}"),
            Label(f"{DLabel.TARGET_PORT}: {self._port}"),
            id=DField.CONFIG,
        )

        # Runtime status
        yield Vertical(
            Label(f"{self._connected_msg}", id=DField.CONNECTED),
            id=DField.STATUS,
        )

        # Buttons
        yield Horizontal(
            Button(
                label=DLabel.PING_ROUTER,
                id=DMethod.PING_ROUTER,
                compact=True,
            ),
            Label(" "),
            Button(label=DLabel.QUIT, id=DMethod.QUIT, compact=True),
            Label(" "),
            Button(label=DLabel.START, id=DGameMethod.START_RUN, compact=True),
            Label(" "),
            Button(label=DLabel.STOP, id=DGameMethod.STOP_RUN, compact=True),
            Label(" "),
            Button(
                label=DLabel.RESET, id=DGameMethod.RESET_GAME, compact=True
            ),
            id=DField.BUTTONS,
        )

        # The Snake Game
        yield Horizontal(
            Vertical(self.game_board, id=DField.BOARD_BOX),
            Vertical(
                Label(f"{DLabel.INITIAL_EPSILON:>15s}: {DEpsilonDef.INITIAL}"),
                Label(f"{DLabel.MIN_EPSILON:>15s}: {DEpsilonDef.MINIMUM}"),
                Label(
                    f"{DLabel.EPSILON_DECAY:>15s}: {DEpsilonDef.DECAY_RATE}"
                ),
                Label(f"{DLabel.CUR_EPSILON:>15s}:", id=DField.CUR_EPSILON),
                Label(""),
                Label(
                    f"{DLabel.LOOKAHEAD_P_VAL:>18}: {DLookaheadDef.PROBABILITY}"
                ),
                Label(
                    f"{DLabel.LOOKAHEAD_ENABLED:>18}: {DLookahead.UNKNOWN}",
                    id=DField.LOOKAHEAD_ENABLED,
                ),
                id=DField.RUNTIME_VALUES,
            ),
        )

        # Console
        # yield Log(highlight=True, auto_scroll=True, id=DField.CONSOLE)

    def console_msg(self, msg: str) -> None:
        console = self.query_one(Log)
        console.write_line(str(msg))
        if isinstance(msg, HydraMsg):
            console.write_line(str(msg.payload))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == DMethod.QUIT:
            await self.on_quit()

        mq = self.mq
        if mq is None:
            raise TypeError("self.mq is None!!!")

        if button_id == DMethod.PING_ROUTER:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_ROUTER,
                method=DMethod.PING,
            )
            # self.console_msg("Sending ping to router...")
            await mq.send(msg)

            try:
                reply = await mq.recv()
                if reply.method == DMethod.PONG:
                    pass
                    # self.console_msg("Received pong")
            except asyncio.TimeoutError:
                pass
                # self.console_msg("Ping timed out.")

        elif button_id == DGameMethod.RESET_GAME:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.RESET_GAME,
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
                self.console_msg(reply)

            except asyncio.TimeoutError:
                self.console_msg("Reset timed out.")

        elif button_id == DGameMethod.START_RUN:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.START_RUN,
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
                # self.console_msg(reply)

            except asyncio.TimeoutError:
                pass
                # self.console_msg("Start run timed out.")

        elif button_id == DGameMethod.STOP_RUN:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.STOP_RUN,
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
                # self.console_msg(reply)

            except asyncio.TimeoutError:
                pass
                # self.console_msg("Stop run timed out.")

    def on_mount(self) -> None:
        self.mq = HydraMQ(
            router_address=self._address,
            router_port=self._port,
            identity=self._identity,
            # --- Telemetry SUB ---
            srv_host=self._address,
            cli_sub_prefixes=[self._telemetry_prefix],
            cli_sub_methods={
                self._telemetry_prefix: self._on_telemetry,
            },
        )
        self.mq.start()
        self.check_connection_bg()
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.CONFIG}").border_subtitle = DLabel.CONFIG
        self.query_one(f"#{DField.STATUS}").border_subtitle = DLabel.STATUS
        # self.query_one(f"#{DField.CONSOLE}", Log).write_line(
        #    "Initialization complete"
        # )

    def on_telemetry_received(self, msg: TelemetryReceived) -> None:
        payload = msg.payload

        snap = payload.get(DGameField.SNAPSHOT, {})
        board = snap.get(DGameField.BOARD)

        if board is not None:
            self.game_board.apply_snapshot(snap)

        info = payload.get(DGameField.INFO, {})
        score = info.get(DGameField.SCORE)
        highscore = info.get(DGameField.HIGHSCORE)
        epoch = info.get(DGameField.EPOCH)
        steps = info.get(DGameField.STEP_N)
        board_box = self.query_one(f"#{DField.BOARD_BOX}", Vertical)
        board_box.border_title = f"{DLabel.GAME}: {epoch}"
        board_box.border_subtitle = (
            f"{DLabel.HIGHSCORE}: {highscore}  {DLabel.SCORE}: {score}"
        )

        # Current epsilon value
        epsilon = info.get(DNetField.CUR_EPSILON)
        if epsilon:
            self.query_one(f"#{DField.CUR_EPSILON}", Label).update(
                f"{DLabel.CUR_EPSILON:>15s}: {round(epsilon, 5)}"
            )

        # Lookahead status
        if DNetField.LOOKAHEAD_ON in info:
            if info[DNetField.LOOKAHEAD_ON]:
                cur_lookahead = DLookahead.ON
            else:
                cur_lookahead = DLookahead.OFF
            self.query_one(f"#{DField.LOOKAHEAD_ENABLED}", Label).update(
                f"{DLabel.LOOKAHEAD_ENABLED:>18}: {cur_lookahead}"
            )

    def _on_telemetry(self, topic: str, payload: dict) -> None:
        self.post_message(TelemetryReceived(topic, payload))

    async def on_quit(self) -> None:
        sys.exit(0)


def main() -> None:
    router = HydraClientTui()
    router.run()


if __name__ == "__main__":
    main()
