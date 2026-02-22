import asyncio
import sys
import time

import zmq
import zmq.asyncio
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import Button, Label, Log

from ai_hydra.constants.DHydra import DHydra, DHydraRouterDef, DMethod, DModule
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.utils.HydraMsg import HydraMsg

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


class HydraRouter(App):
    """A Textual interface to the Hydra Router"""

    TITLE = DLabel.ROUTER_TITLE
    CSS_PATH = DFile.ROUTER_CSS_PATH

    def __init__(
        self,
        address: str = "*",
        port: int = DHydraRouterDef.PORT,
        heartbeat_port: int = DHydraRouterDef.HEARTBEAT_PORT,
    ) -> None:
        """Constructor"""
        super().__init__()

        self._address = address
        self._port = port
        self._hb_port = heartbeat_port
        self._listening = False
        self._num_msgs = 0
        self.socket: zmq.asyncio.Socket | None = None
        self.hb_socket: zmq.asyncio.Socket | None = None
        self._init_socket()
        self._clients: dict[bytes, float] = {}

    def _init_socket(self) -> None:
        try:
            bind_address = f"tcp://{self._address}:{self._port}"
            hb_bind_address = f"tcp://{self._address}:{self._hb_port}"
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(bind_address)
            self.hb_socket = self.context.socket(zmq.ROUTER)
            self.hb_socket.bind(hb_bind_address)
        except Exception as e:
            print(f"{DLabel.ERROR}: {e}")
            exit(1)

    @work(group="hb", exclusive=True)
    async def bg_hb_listen(self) -> None:
        if self.hb_socket is None:
            self._init_socket()

        try:
            while True:
                if self.hb_socket is not None:
                    try:
                        frames = await asyncio.wait_for(
                            self.hb_socket.recv_multipart(),
                            timeout=DHydra.NETWORK_TIMEOUT,
                        )

                        routing_id, message_data, route = self._split_router_frames(
                            frames
                        )
                        self._clients[routing_id] = time.time()

                        # Deserialize to HydraMsg
                        hydra_msg = HydraMsg.from_json(message_data)

                        # Handle the message
                        await self.handle_hb(routing_id, hydra_msg)

                    except asyncio.TimeoutError:
                        # No message received, continue
                        pass

                else:
                    raise RuntimeError("Socket is not initialized")

        except Exception as e:
            self.query_one(f"#{DField.CONSOLE_SCREEN}", Log).write_line(f"ERROR: {e}")
            exit(1)

    @work(group="main", exclusive=True)
    async def bg_listen(self) -> None:
        if self.socket is None:
            self._init_socket()

        try:
            while True:
                if self.socket is not None:
                    try:
                        frames = await asyncio.wait_for(
                            self.socket.recv_multipart(), timeout=DHydra.NETWORK_TIMEOUT
                        )

                        routing_id, message_data, route = self._split_router_frames(
                            frames
                        )
                        self._clients[routing_id] = time.time()

                        # Deserialize to HydraMsg
                        hydra_msg = HydraMsg.from_json(message_data)

                        # Handle the message
                        await self.handle_message(routing_id, hydra_msg)

                    except asyncio.TimeoutError:
                        # No message received, continue
                        pass

                else:
                    raise RuntimeError("Socket is not initialized")

        except Exception as e:
            self.query_one(f"#{DField.CONSOLE_SCREEN}", Log).write_line(f"ERROR: {e}")
            exit(1)

    def compose(self) -> ComposeResult:
        """The TUI is created here"""

        # Title
        yield Label(DLabel.ROUTER_TITLE, id=DField.TITLE)

        # Configuration
        yield Vertical(Label(f"{DLabel.LISTEN_PORT}: {self._port}"), id=DField.CONFIG)

        # Buttons
        yield Horizontal(
            Button(label=DLabel.START, id=DMethod.START, compact=True),
            Label(DLabel.SPACE),
            Button(label=DLabel.QUIT, id=DField.QUIT, compact=True),
            id=DField.BUTTONS,
        )

        # Console
        yield Vertical(
            Label(f"[b]   # {'Sender':>12s} > {'Target':>12s} : {'Method':<10s}[/]"),
            Log(highlight=True, auto_scroll=True, id=DField.CONSOLE_SCREEN),
            id=DField.CONSOLE,
        )

        # Clients
        yield Vertical(
            Label(f"[b]{'Client':>12s} : {'Status'}"),
            Log(highlight=True, auto_scroll=True, id=DField.CLIENTS_SCREEN),
            id=DField.CLIENTS,
        )

    def console_msg(self, msg: HydraMsg):
        self._num_msgs += 1
        line = (
            f"{self._num_msgs:>4d} {msg.sender:>12s} > {msg.target:>12s} : "
            + f"{msg.method:<10s}"
        )
        self.query_one(f"#{DField.CONSOLE_SCREEN}", Log).write_line(line)

    async def handle_hb(self, routing_id: bytes, msg: HydraMsg) -> None:
        # Display the message
        self.console_msg(msg=msg)

        if msg.target == DModule.HYDRA_ROUTER:
            if msg.method == DMethod.HEARTBEAT:
                # Create and send reply
                reply_msg = HydraMsg(
                    sender=DModule.HYDRA_ROUTER,
                    target=msg.sender,
                    method=DMethod.HEARTBEAT_REPLY,
                )

                # Send reply using ROUTER multipart format
                sock = self.hb_socket
                if sock is None:
                    raise RuntimeError("Heartbeat socket not initialized")

                await sock.send_multipart([routing_id, reply_msg.to_json()])

    async def handle_message(self, routing_id: bytes, msg: HydraMsg) -> None:
        # Display in log
        self.console_msg(msg=msg)

        if msg.target == DModule.HYDRA_ROUTER:
            if msg.method == DMethod.PING:
                # Create and send reply
                reply_msg = HydraMsg(
                    sender=DModule.HYDRA_ROUTER,
                    target=msg.sender,
                    method=DMethod.PONG,
                )

                # Send reply using ROUTER multipart format
                sock = self.socket
                if sock is None:
                    raise RuntimeError("Socket is not initialized")
                await sock.send_multipart([routing_id, reply_msg.to_json()])

                self.console_msg(msg=reply_msg)
            return

        if not msg.target:
            self.query_one(f"#{DField.CONSOLE_SCREEN}", Log).write_line(
                "ERROR: message missing target"
            )
            return

        # Generic routing: route to whichever identity matches msg.target.
        sock = self.socket
        if sock is None:
            raise RuntimeError("Socket is not initialized")

        await sock.send_multipart([msg.target.encode(), msg.to_json()])

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == DMethod.START:
            self.bg_hb_listen()
            self.bg_listen()
            self.update_client_table()
            self.add_class(DField.RUNNING)

        elif button_id == DField.QUIT:
            await self.on_quit()

    def on_mount(self):
        self.query_one(f"#{DField.TITLE}", Label).border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.CONFIG}", Vertical).border_subtitle = DLabel.CONFIG
        self.query_one(f"#{DField.CLIENTS}", Vertical).border_subtitle = DLabel.CLIENTS

    async def on_quit(self):
        sys.exit(0)

    def _split_router_frames(
        self, frames: list[bytes]
    ) -> tuple[bytes, bytes, list[bytes]]:
        """
        Returns (sender, payload, routing_prefix)
        routing_prefix is what you should echo back before payload.
        """
        if len(frames) < 2:
            raise ValueError(f"Expected >=2 frames, got {len(frames)}")

        routing_id = frames[0]

        if len(frames) >= 3 and frames[1] == b"":
            return routing_id, frames[-1], [routing_id, b""]
        else:
            return routing_id, frames[-1], [routing_id]

    @work(group="clients", exclusive=True)
    async def update_client_table(self) -> None:
        while True:
            now = time.time()
            screen = self.query_one(f"#{DField.CLIENTS_SCREEN}", Log)
            screen.clear()
            for routing_id in self._clients.keys():
                interval = now - self._clients[routing_id]
                client_str = f"{routing_id.decode():>12s}"
                if interval > (3 * DHydra.HEARTBEAT_INTERVAL):
                    screen.write_line(f"{client_str} : {DStatus.BAD}")
                elif interval > (2 * DHydra.HEARTBEAT_INTERVAL):
                    screen.write_line(f"{client_str} : {DStatus.OK}")
                else:
                    screen.write_line(f"{client_str} : {DStatus.GOOD}")

            await asyncio.sleep(DHydra.HEARTBEAT_INTERVAL + 1)


def main():
    router = HydraRouter()
    router.run()


if __name__ == "__main__":
    main()
