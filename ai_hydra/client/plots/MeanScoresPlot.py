# ai_hydra/client/plots/MeanScoresPlot.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import traceback

from textual.app import ComposeResult, Widget
from textual.containers import Horizontal
from textual_plot import HiResMode, PlotWidget

from ai_hydra.constants.DHydraTui import DField, DLabel, DColor, DPlotDef
from ai_hydra.client.plots.PlotUtils import thin_series
from ai_hydra.utils.HydraMetrics import HydraMetrics


class MeanScoresPlot(Widget):

    def __init__(self, metrics: HydraMetrics, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def compose(self) -> ComposeResult:
        yield Horizontal(
            PlotWidget(id=DField.PLOT_RECENT_MEAN),
            PlotWidget(id=DField.PLOT_MEAN),
        )

    def clear(self):
        self.query_one(f"#{DField.PLOT_RECENT_MEAN}", PlotWidget).clear()
        self.query_one(f"#{DField.PLOT_MEAN}", PlotWidget).clear()

    def plot_all(self):
        self.plot_mean()
        self.plot_recent_mean()

    def plot_mean(self):
        mean_events = self.metrics.get_means()
        if not mean_events:
            return

        episodes, means = zip(*mean_events)
        if len(means) > DPlotDef.MAX_MEAN_DATA_POINTS:
            episodes, means = thin_series(
                x=episodes,
                y=means,
                max_points=DPlotDef.MAX_MEAN_DATA_POINTS,
            )

        self._plot(tag=DField.ALL, episodes=episodes, means=means)

    def plot_recent_mean(self):
        recent_mean_events = self.metrics.get_recent_means()
        if not recent_mean_events:
            return

        episodes, means = zip(*recent_mean_events)
        if len(means) > DPlotDef.MAX_RECENT_MEAN_DATA_POINTS:
            episodes, means = thin_series(
                x=episodes,
                y=means,
                max_points=DPlotDef.MAX_RECENT_MEAN_DATA_POINTS,
            )

        self._plot(tag=DField.RECENT, episodes=episodes, means=means)

    def _plot(self, tag, episodes, means):

        if tag == DField.ALL:
            plot = self.query_one(f"#{DField.PLOT_MEAN}", PlotWidget)
            ylabel = DLabel.MEAN_SCORES
        elif tag == DField.RECENT:
            plot = self.query_one(f"#{DField.PLOT_RECENT_MEAN}", PlotWidget)
            ylabel = DLabel.RECENT_MEANS

        plot.clear()

        plot.plot(
            x=episodes,
            y=means,
            line_style=DColor.GREEN,
            hires_mode=HiResMode.BRAILLE,
            label=ylabel,
        )
        plot.set_ylimits()
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(ylabel)
