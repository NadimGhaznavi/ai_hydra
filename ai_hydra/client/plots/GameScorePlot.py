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
from textual_plot import HiResMode, LegendLocation, PlotWidget

from ai_hydra.constants.DHydraTui import DField, DLabel, DColor
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

    def clear(self):
        self.query_one(f"#{DField.PLOT_HIGHSCORES}", PlotWidget).clear()
        self.query_one(f"#{DField.PLOT_CUR_SCORE}", PlotWidget).clear()

    def plot_cur_scores(self):
        score_events = self.metrics.get_cur_score_plot_points()
        if not score_events:
            return

        episodes, scores = zip(*score_events)

        plot = self.query_one(f"#{DField.PLOT_CUR_SCORE}", PlotWidget)
        plot.clear()

        plot.plot(
            x=episodes,
            y=scores,
            line_style=DColor.GREEN,
            hires_mode=HiResMode.BRAILLE,
            label=DLabel.SCORE,
        )

        avg_score_events = self.metrics.get_avg_cur_score_plot_points()
        if avg_score_events:
            avg_episodes, avg_scores = zip(*avg_score_events)
            plot.plot(
                x=avg_episodes,
                y=avg_scores,
                line_style=DColor.RED,
                hires_mode=HiResMode.BRAILLE,
                label=DLabel.AVERAGE,
            )

        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.CUR_SCORES)
        plot.show_legend(location=LegendLocation.TOPLEFT)

    def plot_highscores(self):
        plot_points = self.metrics.get_highscore_plot_points()
        if not plot_points:
            return
        episodes, scores = zip(*plot_points)

        plot = self.query_one(f"#{DField.PLOT_HIGHSCORES}", PlotWidget)
        plot.clear()

        plot.plot(
            x=episodes,
            y=scores,
            line_style=DColor.GREEN,
            hires_mode=HiResMode.BRAILLE,
            label=DLabel.HIGHSCORE,
        )
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.HIGHSCORES)
        plot.show_legend(location=LegendLocation.TOPLEFT)
