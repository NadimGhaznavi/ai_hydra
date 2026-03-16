# ai_hydra/client/plots/GameScorePlot.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from textual.app import ComposeResult, Widget
from textual.containers import Horizontal
from textual.widgets import Static, Label
from textual_plot import HiResMode, LegendLocation, PlotWidget


from ai_hydra.constants.DHydraTui import DField

from ai_hydra.utils.HydraMetrics import HydraMetrics


class GameScorePlot(Widget):

    def __init__(self, metrics: HydraMetrics, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def compose(self) -> ComposeResult:
        yield Horizontal(
            PlotWidget(id=DField.PLOT_HIGHSCORES),
            PlotWidget(id=DField.PLOT_CUR_SCORE),
        )

    def plot_all(self):
        self._plot_highscores()

    def _plot_highscores(self):
        pass
