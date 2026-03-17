# ai_hydra/client/HydraTelemetry.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from textual.app import ComposeResult, Widget
from textual.widgets import TabbedContent, Label


from ai_hydra.constants.DHydraTui import DField, DLabel

from ai_hydra.utils.HydraMetrics import HydraMetrics
from ai_hydra.client.plots.GameScorePlot import GameScorePlot
from ai_hydra.client.plots.ScoresDistPlot import ScoresDistPlot


class HydraTelemetry(Widget):
    """Tabbed plot widget for game scores, loss, and scatter visualizations."""

    def __init__(self, metrics: HydraMetrics, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.game_score_plot = GameScorePlot(
            id=DField.GAME_SCORE_PLOT, metrics=self.metrics
        )
        self.scores_dist = ScoresDistPlot(
            id=DField.SCORES_DIST_PLOT, metrics=self.metrics
        )

    def compose(self) -> ComposeResult:
        with TabbedContent(DLabel.GAME_SCORES, DLabel.SCORE_DISTRIBUTION):
            yield self.game_score_plot
            yield self.scores_dist
