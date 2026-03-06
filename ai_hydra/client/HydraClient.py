# ai_hydra/client/HydraClient.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import asyncio

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import Button, Label
from textual.message import Message

from ai_hydra.utils.HydraMQ import HydraMQ
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
    DHydraServer,
)
from ai_hydra.constants.DHydraTui import DField, DFile, DLabel, DStatus
from ai_hydra.constants.DGame import DGameField, DGameMethod
from ai_hydra.constants.DNNet import (
    DNetField,
    DEpsilonDef,
    DLookahead,
    DLookaheadDef,
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
        self._cur_lookahead = None
        self.cfg = SimCfg.init_client()

    @work(exclusive=True)
    async def check_connection_bg(self) -> None:
        while True:
            mq = self.mq
            if mq is None:
                raise TypeError("self.mq is None!!!")

            if mq.connected():
                self._connected_msg = DStatus.GOOD
            else:
                self._connected_msg = DStatus.BAD

            self.query_one(f"#{DField.CONNECTED}", Label).update(
                f"{DLabel.CONNECTED:>11s}: {self._connected_msg}"
            )

            await asyncio.sleep(DHydra.HEARTBEAT_INTERVAL + 1)

    def compose(self) -> ComposeResult:
        """The TUI is created here"""

        # Title - 1x4
        yield Label(DLabel.CLIENT_TITLE, id=DField.TITLE)

        # Buttons - 2x1
        yield Vertical(
            Button(label=DLabel.START, id=DGameMethod.START_RUN, compact=True),
            Label(),
            Button(label=DLabel.STOP, id=DGameMethod.STOP_RUN, compact=True),
            Label(),
            Button(
                label=DLabel.RESET, id=DGameMethod.RESET_GAME, compact=True
            ),
            Label(),
            Button(label=DLabel.BOARD, id=DField.SHOW_BOARD, compact=True),
            Button(label=DLabel.NO_BOARD, id=DField.NO_BOARD, compact=True),
            Label(),
            Button(label=DLabel.QUIT, id=DMethod.QUIT, compact=True),
            id=DField.BUTTONS,
        )

        # The Snake Game - 2x1
        yield Vertical(self.game_board, id=DField.BOARD_BOX)

        # Network - 1x1
        yield Vertical(
            Label(f"{DLabel.TARGET_HOST:>11s}: {self._address}"),
            Label(f"{DLabel.TARGET_PORT:>11s}: {self._port}"),
            Label(f"{DLabel.CONNECTED:>11s}:", id=DField.CONNECTED),
            id=DField.NETWORK,
        )

        # Highscores - 1x1
        yield TabbedScores(id=DField.TABBED_SCORES)

        # Runtime Settings
        yield Vertical(
            Label(f"{DLabel.INITIAL_EPSILON:>15s}: {DEpsilonDef.INITIAL}"),
            Label(f"{DLabel.MIN_EPSILON:>15s}: {DEpsilonDef.MINIMUM}"),
            Label(f"{DLabel.EPSILON_DECAY:>15s}: {DEpsilonDef.DECAY_RATE}"),
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
        )

        # Plots
        yield TabbedPlots(id=DField.TABBED_PLOTS)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == DMethod.QUIT:
            await self.on_quit()
            return

        mq = self.mq
        if mq is None:
            raise TypeError("self.mq is None!!!")

        elif button_id == DGameMethod.RESET_GAME:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.RESET_GAME,
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
            except asyncio.TimeoutError:
                pass

        elif button_id == DGameMethod.START_RUN:
            # Make sure we have the latest TUI choice
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.START_RUN,
                payload=self.cfg.to_start_payload(),
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
            except asyncio.TimeoutError:
                pass

        elif button_id == DGameMethod.STOP_RUN:
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.STOP_RUN,
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
            except asyncio.TimeoutError:
                pass

        elif button_id == DField.NO_BOARD:
            # No board == Turbo mode
            self._set_per_step(False)
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.PUB_TYPE,
                payload=self.cfg.to_runtime_payload(),
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
            except asyncio.TimeoutError:
                pass

        elif button_id == DField.SHOW_BOARD:
            # Show board == Normal mode
            self._set_per_step(True)
            msg = HydraMsg(
                sender=DModule.HYDRA_CLIENT,
                target=DModule.HYDRA_MGR,
                method=DGameMethod.PUB_TYPE,
                payload=self.cfg.to_runtime_payload(),
            )
            await mq.send(msg)

            try:
                reply = await mq.recv()
            except asyncio.TimeoutError:
                pass

    def on_mount(self) -> None:
        self.mq = HydraMQ(
            router_address=self._address,
            router_port=self._port,
            identity=self._identity,
            # --- Telemetry SUB ---
            srv_host=self._address,
            cli_sub_methods={"*": self._on_telemetry},
            # --- Default PUB_TYPE is "BOARD" (show the board moves)
        )
        self.mq.start()

        # Always receive per-episode updates
        self.mq.ensure_per_episode_sub()

        self.check_connection_bg()
        self.query_one(f"#{DField.TITLE}").border_subtitle = (
            DLabel.VERSION + " " + DHydra.VERSION
        )
        self.query_one(f"#{DField.RUNTIME_VALUES}").border_subtitle = (
            DLabel.RUNTIME_VALS
        )
        self.query_one(f"#{DField.NETWORK}").border_subtitle = DLabel.NETWORK
        self.query_one(f"#{DField.TABBED_SCORES}").border_subtitle = (
            DLabel.HIGHSCORES
        )
        self.query_one(f"#{DField.BUTTONS}").border_subtitle = DLabel.ACTIONS
        self._set_per_step(self.cfg.get(DNetField.PER_STEP))

    async def on_shutdown_request(self) -> None:
        if self.mq is not None:
            await self.mq.quit()
            self.mq = None

    def on_telemetry_received(self, msg: TelemetryReceived) -> None:
        topic = msg.topic
        payload = msg.payload
        score = "N/A"

        # print(f"TOPIC: {topic} - PAYLOAD: {payload}")

        # per_step: board telemetry + score
        if topic.endswith(f".{DHydraServer.PER_STEP_TOPIC}"):
            score = payload.get(DGameField.SCORE)
            board = payload.get(DGameField.BOARD)
            self.game_board.apply_board_dict(board)
            self.query_one(
                f"#{DField.BOARD_BOX}", Vertical
            ).border_subtitle = f"{DLabel.SCORE}: {score:<2}"
            # update score display if you want here
            return

        # Ignore anything that isn't per episode
        if not topic.endswith(f".{DHydraServer.PER_EPISODE_TOPIC}"):
            return

        # per_episode: training + episode stats
        info = payload.get(DGameField.INFO, {})

        # Current game score and highscore
        highscore = info.get(DGameField.HIGHSCORE)
        epoch = info.get(DGameField.EPOCH)

        self.query_one(f"#{DField.BOARD_BOX}", Vertical).border_title = (
            f"{DLabel.GAME}: {epoch}"
        )

        self.query_one(
            f"#{DField.TABBED_SCORES}", TabbedScores
        ).border_subtitle = f"{DLabel.HIGHSCORE}: {highscore}"

        # Highscore event
        if DGameField.HIGHSCORE_EVENT_NLH in info:
            highscore_event = info[DGameField.HIGHSCORE_EVENT_NLH]
            if highscore_event[2]:
                self.query_one(
                    f"#{DField.TABBED_SCORES}", TabbedScores
                ).add_highscore_nlh(
                    epoch=highscore_event[0],
                    highscore=highscore_event[1],
                    event_time=highscore_event[2],
                )

        # Lookahead Highscore event
        if DGameField.HIGHSCORE_EVENT_LH in info:
            highscore_event_lh = info[DGameField.HIGHSCORE_EVENT_LH]
            if highscore_event_lh[2]:
                self.query_one(
                    f"#{DField.TABBED_SCORES}", TabbedScores
                ).add_highscore_lh(
                    epoch=highscore_event_lh[0],
                    highscore=highscore_event_lh[1],
                    event_time=highscore_event_lh[2],
                )

        # Current epsilon value
        epsilon = info.get(DNetField.CUR_EPSILON)
        if epsilon is not None:
            self.query_one(f"#{DField.CUR_EPSILON}", Label).update(
                f"{DLabel.CUR_EPSILON:>15s}: {round(epsilon, 4)}"
            )

        # Lookahead status
        if DNetField.LOOKAHEAD_ON in info:
            if info[DNetField.LOOKAHEAD_ON]:
                cur_lookahead = DLookahead.ON
            else:
                cur_lookahead = DLookahead.OFF
            self._cur_lookahead = cur_lookahead
            self.query_one(f"#{DField.LOOKAHEAD_ENABLED}", Label).update(
                f"{DLabel.LOOKAHEAD_ENABLED:>18}: {cur_lookahead}"
            )

        # Loss
        if DNetField.LOSS in info:
            self.query_one(f"#{DField.TABBED_PLOTS}", TabbedPlots).add_loss(
                epoch,
                info[DNetField.LOSS],
            )

        # Final score
        if DNetField.FINAL_SCORE in info:
            self.query_one(f"#{DField.TABBED_PLOTS}", TabbedPlots).add_score(
                cur_score=info[DNetField.FINAL_SCORE],
                lookahead=info[DNetField.LOOKAHEAD_ON],
            )

    def _on_telemetry(self, topic: str, payload: dict) -> None:
        self.post_message(TelemetryReceived(topic, payload))

    async def on_quit(self) -> None:
        # Stop background tasks and close ZMQ sockets cleanly
        if self.mq is not None:
            await self.mq.quit()
            self.mq = None
        self.exit()

    def _set_move_delay(self, delay: float) -> None:
        self.cfg.apply(
            payload={DNetField.MOVE_DELAY: delay}, phase=Phase.RUNTIME
        )

    def _set_per_step(self, enabled: bool) -> None:
        mq = self.mq
        if mq is None:
            raise TypeError("self.mq is None!!!")
        self.cfg.apply(
            payload={DNetField.PER_STEP: enabled}, phase=Phase.RUNTIME
        )

        if enabled:
            mq.enable_per_step_sub()
            self.remove_class(DField.NO_BOARD)
            self.add_class(DField.SHOW_BOARD)
        else:
            mq.disable_per_step_sub()
            self.remove_class(DField.SHOW_BOARD)
            self.add_class(DField.NO_BOARD)


def main() -> None:
    router = HydraClientTui()
    router.run()


if __name__ == "__main__":
    main()
