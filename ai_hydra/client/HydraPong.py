from textual.theme import Theme
from textual.app import App, ComposeResult
from textual.widgets import Label
from textual.containers import Vertical

from ai_hydra.constants.DPong import DPLabel, DPField, DPFile

from ai_hydra.game.PongBoard import PongBoard

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


class HydraPong(App):

    TITLE = DPLabel.TITLE
    CSS_PATH = DPFile.PONG_CSS

    def __init__(self):
        super().__init__()

        self.pong_board = PongBoard(20, id=DPField.PONG_BOARD)

    def compose(self) -> ComposeResult:

        # ----- Title ---
        yield Label(DPLabel.TITLE, id=DPField.TITLE, classes=DPField.BOX)

        # ----- Pong Board ---
        yield Vertical(
            self.pong_board, id=DPField.PONG_BOX, classes=DPField.BOX
        )


def main() -> None:
    pong = HydraPong()
    pong.run()


if __name__ == "__main__":
    main()
