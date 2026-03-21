# ai_hydra/client/plots/ScoreDistPlot.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from statistics import mean, median

from textual.app import ComposeResult, Widget
from textual.containers import Horizontal
from textual_plot import HiResMode, LegendLocation, PlotWidget

from ai_hydra.constants.DHydraTui import DField, DLabel, DColor, DPlotDef
from ai_hydra.utils.HydraMetrics import HydraMetrics


class ScoresDistPlot(Widget):

    def __init__(self, metrics: HydraMetrics, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def compose(self) -> ComposeResult:
        yield Horizontal(
            PlotWidget(id=DField.PLOT_SCORES_DIST),
            PlotWidget(id=DField.PLOT_RECENT_SCORES_DIST),
        )

    def clear(self):
        self.query_one(f"#{DField.PLOT_SCORES_DIST}").clear()
        self.query_one(f"#{DField.PLOT_RECENT_SCORES_DIST}").clear()
        self.plot_all()

    def plot_all(self):
        self.plot_scores_dist()
        self.plot_recent_scores_dist()

    def plot_scores_dist(self):
        scores, counts = self.metrics.get_scores_dist_plot_points()
        stats = self.metrics.get_scores_dist_stats()
        self._plot(tag=DField.ALL, scores=scores, counts=counts, stats=stats)

    def plot_recent_scores_dist(self):
        scores, counts = self.metrics.get_recent_scores_dist_plot_points()
        stats = self.metrics.get_recent_scores_dist_stats()
        self._plot(
            tag=DField.RECENT, scores=scores, counts=counts, stats=stats
        )

    def _plot(self, tag, scores, counts, stats):
        if not scores:
            return

        if tag == DField.ALL:
            plot = self.query_one(f"#{DField.PLOT_SCORES_DIST}", PlotWidget)
            ylabel = DLabel.COUNT
        elif tag == DField.RECENT:
            plot = self.query_one(
                f"#{DField.PLOT_RECENT_SCORES_DIST}", PlotWidget
            )
            ylabel = f"Recent ({DPlotDef.RECENT_SCORES_MAX}) Count"

        plot.clear()
        plot.bar(
            scores,
            counts,
            bar_style=DColor.BLUE,
            width=0.8,
            hires_mode=HiResMode.BRAILLE,
        )

        plot.set_ylimits()
        plot.set_xlabel(DLabel.SCORES)
        plot.set_ylabel(ylabel)
        plot.show_legend(location=LegendLocation.TOPRIGHT)

        if stats is None:
            return

        mean_score, median_score = stats

        plot.add_v_line(
            mean_score, DColor.GREEN, f"{DLabel.MEAN}  : {mean_score:.2f}"
        )
        plot.add_v_line(
            median_score, DColor.PURPLE, f"{DLabel.MEDIAN}: {median_score:.2f}"
        )

        cur_epoch = self.metrics.get_cur_epoch()

        if cur_epoch > 0 and cur_epoch % 500 == 0:
            if tag == DField.ALL:
                self.metrics.add_mean_median(
                    epoch=cur_epoch, mean=mean_score, median=median_score
                )
            elif tag == DField.RECENT:
                self.metrics.add_recent_mean_and_median(
                    epoch=cur_epoch, mean=mean_score, median=median_score
                )
