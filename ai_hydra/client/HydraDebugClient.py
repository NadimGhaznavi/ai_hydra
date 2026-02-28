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

from ai_hydra.utils.HydraMQ import HydraMQ
from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.constants.DHydra import DHydra, DHydraRouterDef, DMethod, DModule
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.constants.DGame import DGameField
from ai_hydra.game.GameBoard import MoveAction
from ai_hydra.client.ClientGameBoard import ClientGameBoard

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

MOVE_MAP = {
    DMethod.ACTION_LEFT: 0,
    DMethod.ACTION_STRAIGHT: 1,
    DMethod.ACTION_RIGHT: 2,
}


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
        yield Vertical(
            Horizontal(
                Button(
                    label=DLabel.PING_ROUTER,
                    id=DMethod.PING_ROUTER,
                    compact=True,
                ),
                Label(" "),
                Button(label="Quit", id="quit", compact=True),
            ),
            Horizontal(
                Button(
                    label=DLabel.RESET, id=DMethod.RESET_GAME, compact=True
                ),
                Label(" "),
                Button(
                    label=DLabel.LEFT, id=DMethod.ACTION_LEFT, compact=True
                ),
                Label(" "),
                Button(
                    label=DLabel.STRAIGHT,
                    id=DMethod.ACTION_STRAIGHT,
                    compact=True,
                ),
                Label(" "),
                Button(
                    label=DLabel.RIGHT, id=DMethod.ACTION_RIGHT, compact=True
                ),
            ),
            id="buttons",
        )

        # Console
        yield Log(highlight=True, auto_scroll=True, id=DField.CONSOLE)

        # The Snake Game
        yield Vertical(
            self.game_board,
        )

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
            self.console_msg("Sending ping to router...")
            await mq.send(msg)

            try:
                reply = await mq.recv()
                if reply.method == DMethod.PONG:
                    self.console_msg("Received pong")
            except asyncio.TimeoutError:
                self.console_msg("Ping timed out.")

        elif button_id == DMethod.PING_SERVER:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DMethod.PING,
            )
            self.console_msg("Sending ping to Hydra manager")
            await mq.send(msg)

            try:
                reply = await mq.recv()
                if reply.method == DMethod.PONG:
                    self.console_msg("Received pong")
            except asyncio.TimeoutError:
                self.console_msg("Ping timed out...")

        elif button_id == DMethod.RESET_GAME:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DMethod.RESET_GAME,
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
                self.console_msg(reply)
                self.game_board.apply_snapshot(reply.payload)

            except asyncio.TimeoutError:
                self.console_msg("Reset timed out.")

        elif button_id == DMethod.ACTION_LEFT:
            await self.send_action(DMethod.ACTION_LEFT)
        elif button_id == DMethod.ACTION_STRAIGHT:
            await self.send_action(DMethod.ACTION_STRAIGHT)
        elif button_id == DMethod.ACTION_RIGHT:
            await self.send_action(DMethod.ACTION_RIGHT)

    def on_mount(self) -> None:
        self.mq = HydraMQ(
            router_address=self._address,
            router_port=self._port,
            identity=self._identity,
        )
        self.mq.start()
        self.check_connection_bg()
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.CONFIG}").border_subtitle = DLabel.CONFIG
        self.query_one(f"#{DField.STATUS}").border_subtitle = DLabel.STATUS
        self.query_one(f"#{DField.CONSOLE}", Log).write_line(
            "Initialization complete"
        )

    async def on_quit(self) -> None:
        sys.exit(0)

    async def send_action(self, action: DMethod):
        self.console_msg(f"Sending action: {action} > {MOVE_MAP[action]}")
        msg = HydraMsg(
            sender=DModule.HYDRA_CLIENT,
            target=DModule.HYDRA_MGR,
            method=DMethod.GAME_STEP,
            payload={DGameField.ACTION: str(MOVE_MAP[action])},
        )
        await self.mq.send(msg)
        try:
            reply = await self.mq.recv()
            self.console_msg(reply)
            self.game_board.apply_snapshot(
                reply.payload.get(DGameField.SNAPSHOT)
            )
        except asyncio.TimeoutError:
            self.console_msg("Action timed out.")


def main() -> None:
    router = HydraClientTui()
    router.run()


if __name__ == "__main__":
    main()
