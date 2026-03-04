# ai_hydra/client/TabbedScores.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from textual.containers import Vertical
from textual.widgets import TabbedContent, Label, Log
from textual.app import ComposeResult, Widget

from ai_hydra.constants.DHydraTui import DField, DLabel
from ai_hydra.constants.DNNet import DLookahead


class TabbedScores(Widget):

    def compose(self) -> ComposeResult:
        with TabbedContent(
            DLabel.HIGHSCORES, DLabel.HIGHSCORES_LH, DLabel.BOTH
        ):
            yield Vertical(
                Label(
                    f"[b #3e99af]{DLabel.GAME:>7s}{DLabel.SCORE:>7s}{DLabel.TIME:>13s}[/]"
                ),
                Log(highlight=False, auto_scroll=True, id=DField.HIGHSCORES),
                classes=DField.HIGHSCORES_BOX,
            )
            yield Vertical(
                Label(
                    f"[b #3e99af]{DLabel.GAME:>7s}{DLabel.SCORE:>7s}{DLabel.TIME:>13s}[/]"
                ),
                Log(
                    highlight=False, auto_scroll=True, id=DField.HIGHSCORES_LH
                ),
                classes=DField.HIGHSCORES_BOX,
            )
            yield Vertical(
                Label(
                    f"[b #3e99af]{DLabel.GAME:>7s}{DLabel.SCORE:>7s}{DLabel.LH:>4s}{DLabel.TIME:>13s}[/]"
                ),
                Log(highlight=False, auto_scroll=True, id=DField.BOTH),
                classes=DField.HIGHSCORES_BOX,
            )

    def add_highscore_nlh(self, epoch, highscore, event_time):
        self.query_one(f"#{DField.HIGHSCORES}", Log).write_line(
            f"{epoch:7d}{highscore:7d}{event_time:>13s}"
        )
        self.query_one(f"#{DField.BOTH}", Log).write_line(
            f"{epoch:7d}{highscore:7d}{DLookahead.OFF:>3s}{event_time:>13s}"
        )

    def add_highscore_lh(self, epoch, highscore, event_time):
        self.query_one(f"#{DField.HIGHSCORES_LH}", Log).write_line(
            f"{epoch:7d}{highscore:7d}{event_time:>13s}"
        )
        self.query_one(f"#{DField.BOTH}", Log).write_line(
            f"{epoch:7d}{highscore:7d}{DLookahead.ON:>3s}{event_time:>13s}"
        )
