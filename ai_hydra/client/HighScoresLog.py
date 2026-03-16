# ai_hydra/client/HighScoresLog.py
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

from ai_hydra.constants.DHydraTui import DField, DLabel, DStatus
from ai_hydra.utils.HighscoreEvent import HighscoreEvent


class HighScoresLog(Widget):

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(
                f"[b #3e99af]{DLabel.GAME:>7s}{DLabel.SCORE:>7s}{DLabel.TIME:>13s}[/]"
            ),
            Log(highlight=False, auto_scroll=True, id=DField.HIGHSCORES),
            classes=DField.HIGHSCORES_BOX,
        )

    def add_highscore(self, event: HighscoreEvent):
        self.query_one(f"#{DField.HIGHSCORES}", Log).write_line(
            f"{event.epoch:7d}{event.highscore:7d}{event.elapsed_time:>13s}"
        )
