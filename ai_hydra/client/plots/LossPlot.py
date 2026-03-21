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
from textual.widgets import Label
from textual_plot import HiResMode, LegendLocation, PlotWidget

from ai_hydra.constants.DHydraTui import DField, DLabel, DColor, DPlotDef
from ai_hydra.client.plots.PlotUtils import thin_series
from ai_hydra.utils.HydraMetrics import HydraMetrics


class LossPlot(Widget):

    def __init__(self, metrics: HydraMetrics, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def clear(self):
        self.query_one(f"#{DField.PLOT_LOSS}", PlotWidget).clear()
        self.query_one(f"#{DField.PLOT_RECENT_LOSS}", PlotWidget).clear()
        self.plot_all()

    def compose(self) -> ComposeResult:
        yield Horizontal(
            PlotWidget(id=DField.PLOT_LOSS),
            PlotWidget(id=DField.PLOT_RECENT_LOSS),
        )

    def plot_all(self):
        self.plot_loss()
        self.plot_recent_loss()

    def plot_loss(self):
        loss_events = self.metrics.get_loss_plot_points()
        if not loss_events:
            return

        episodes, losses = zip(*loss_events)
        if len(losses) > DPlotDef.MAX_LOSS_DATA_POINTS:
            episodes, losses = thin_series(
                x=episodes,
                y=losses,
                max_points=DPlotDef.MAX_LOSS_DATA_POINTS,
            )

        self._plot(tag=DField.ALL, episodes=episodes, losses=losses)

    def plot_recent_loss(self):
        loss_events = self.metrics.get_recent_loss_plot_points()
        if not loss_events:
            return

        episodes, losses = zip(*loss_events)
        self._plot(tag=DField.RECENT, episodes=episodes, losses=losses)

    def _plot(self, tag, episodes, losses):

        if tag == DField.ALL:
            plot = self.query_one(f"#{DField.PLOT_LOSS}", PlotWidget)
            ylabel = DLabel.LOSS
        elif tag == DField.RECENT:
            plot = self.query_one(f"#{DField.PLOT_RECENT_LOSS}", PlotWidget)
            ylabel = DLabel.RECENT_LOSS

        plot.clear()

        plot.plot(
            x=episodes,
            y=losses,
            line_style=DColor.GREEN,
            hires_mode=HiResMode.BRAILLE,
            label=DLabel.LOSS,
        )
        plot.set_ylimits()
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(ylabel)
