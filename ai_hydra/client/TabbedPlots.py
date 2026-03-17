# ai_hydra/client/TabbedPlots.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from collections import deque
from statistics import median, mean
from typing import Iterable

from textual import on
from textual.app import ComposeResult, Widget
from textual.widgets import TabbedContent
from textual.containers import Horizontal
from textual_plot import HiResMode, LegendLocation, PlotWidget

from ai_hydra.constants.DHydraTui import DColor, DField, DLabel, DPlotDef

from ai_hydra.client.plots.PlotUtils import (
    rolling_mean,
    thin_series,
    weighted_mean,
    weighted_median,
)


class TabbedPlots(Widget):
    """Tabbed plot widget for game scores, loss, and scatter visualizations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reset_data()

    def compose(self) -> ComposeResult:
        with TabbedContent(
            DLabel.GAME_SCORES,
            DLabel.SCORES,
            DLabel.LOSS,
        ):
            yield PlotWidget(id=DField.GAME_SCORES)
            yield PlotWidget(id=DField.SCORES_PLOT)
            yield Horizontal(
                PlotWidget(id=DField.LOSS_PLOT), PlotWidget(id=DField.CUR_LOSS)
            )

    def action_show_tab(self, tab: str) -> None:
        self.get_child_by_type(TabbedContent).active = tab

    @on(TabbedContent.TabActivated)
    def handle_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id
        self.call_later(lambda: self._replot_active_tab(pane_id))

    def add_loss(self, epoch: int, loss: float, plot: bool = True) -> None:
        self.loss_epochs.append(epoch)
        self.losses.append(loss)

        self.cur_losses.append(loss)
        self.cur_epochs.append(epoch)

        if plot:
            self._plot_loss()

    def add_score(
        self, cur_score: int | None, cur_epoch: int | None, plot: bool = True
    ) -> None:
        if cur_score is None or cur_epoch is None:
            return

        score = int(cur_score)
        epoch = int(cur_epoch)

        self.score_counts[score] = self.score_counts.get(score, 0) + 1
        self.game_scores.append(score)
        self.game_epochs.append(epoch)

        if plot:
            self._plot_scores()
            self._plot_game_scores()

    def add_scatter_score(
        self,
        scores: tuple[int, int] | list[tuple[int, int]],
        plot: bool = True,
    ) -> None:
        """Add one or more scatter points as (episode, score)."""
        pass

    def reset(self) -> None:
        self._reset_data()

        self.query_one(f"#{DField.GAME_SCORES}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.LOSS_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.CUR_LOSS}", PlotWidget).clear()

    def _reset_data(self) -> None:
        max_points = DPlotDef.MAX_CUR_DATA_POINTS
        max_loss_points = DPlotDef.MAX_CUR_LOSS_DATA_POINTS

        self.loss_epochs: list[int] = []
        self.losses: list[float] = []

        self.score_counts: dict[int, int] = {}

        self.game_scores: deque[int] = deque(maxlen=max_points)
        self.game_epochs: deque[int] = deque(maxlen=max_points)
        self.avg_game_score: list[float] = []

        self.cur_losses: deque[float] = deque(maxlen=max_loss_points)
        self.cur_epochs: deque[int] = deque(maxlen=max_loss_points)

    def _replot_active_tab(self, pane_id: str) -> None:
        if pane_id == "tab-1":
            self._plot_game_scores()
        elif pane_id == "tab-2":
            self._plot_scores()
        elif pane_id == "tab-3":
            self._plot_loss()
        else:
            raise ValueError(f"Unhandled tab: {pane_id}")

    def _plot_game_scores(self) -> None:
        plot = self.query_one(f"#{DField.GAME_SCORES}", PlotWidget)
        plot.clear()

        if not self.game_epochs or not self.game_scores:
            return

        epochs = list(self.game_epochs)
        scores = list(self.game_scores)

        # Thin only if the series gets too dense for the viewport.
        plot_epochs = epochs
        plot_scores = scores
        if len(scores) > DPlotDef.MAX_CUR_DATA_POINTS:
            plot_epochs, plot_scores = thin_series(
                x=epochs,
                y=scores,
                max_points=DPlotDef.MAX_CUR_DATA_POINTS,
            )

        # Current score as a line plot.
        plot.plot(
            x=plot_epochs,
            y=plot_scores,
            hires_mode=HiResMode.BRAILLE,
            line_style=DColor.GREEN,
            label=DLabel.CURRENT,
        )

        # Rolling average / trend line.
        window = max(8, len(scores) // DPlotDef.AVG_DIVISOR)
        if len(scores) >= window:
            smoothed_scores = rolling_mean(scores, window)
            smoothed_epochs = epochs[window - 1 :]

            plot.plot(
                x=smoothed_epochs,
                y=smoothed_scores,
                hires_mode=HiResMode.BRAILLE,
                line_style=DColor.RED,
                label=DLabel.AVERAGE,
            )

        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.SCORES)
        plot.set_ylimits()
        plot.show_legend(location=LegendLocation.TOPLEFT)

    def _plot_scores(self) -> None:
        plot = self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget)
        plot.clear()

        if not self.score_counts:
            return

        x_values = sorted(self.score_counts.keys())
        y_values = [self.score_counts[score] for score in x_values]

        plot.bar(
            x_values,
            y_values,
            bar_style=DColor.GREEN,
            width=0.8,
            hires_mode=HiResMode.BRAILLE,
        )

        plot.set_ylimits()
        plot.set_xlabel(DLabel.SCORES)
        plot.set_ylabel(DLabel.COUNT)
        plot.show_legend(location=LegendLocation.TOPRIGHT)

        mean_score = mean(self.score_counts)
        median_score = median(self.score_counts)

        plot.add_v_line(mean_score, DColor.GREEN, f"Mean  : {mean_score:.2f}")
        plot.add_v_line(median_score, "purple", f"Median: {median_score:.2f}")

    def _plot_loss(self) -> None:
        self._plot_full_loss()
        self._plot_current_loss()

    def _plot_full_loss(self) -> None:
        plot = self.query_one(f"#{DField.LOSS_PLOT}", PlotWidget)
        plot.clear()

        if not self.loss_epochs or not self.losses:
            return

        epochs = self.loss_epochs
        losses = self.losses

        if len(losses) > DPlotDef.MAX_LOSS_DATA_POINTS:
            epochs, losses = thin_series(
                x=epochs,
                y=losses,
                max_points=DPlotDef.MAX_LOSS_DATA_POINTS,
            )

        plot.plot(
            x=epochs,
            y=losses,
            hires_mode=HiResMode.BRAILLE,
            line_style=DColor.GREEN,
            label=DLabel.COMPLETE,
        )
        plot.set_ylimits()
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.LOSS)
        plot.show_legend(location=LegendLocation.TOPLEFT)

        # Add vertical lines to mark high score events
        # for event in self.scatter_scores:
        #    episode, score = event
        #    plot.add_v_line(episode, DColor.GREEN)

    def _plot_current_loss(self) -> None:
        plot = self.query_one(f"#{DField.CUR_LOSS}", PlotWidget)
        plot.clear()

        if not self.cur_epochs or not self.cur_losses:
            return

        epochs = list(self.cur_epochs)
        losses = list(self.cur_losses)

        plot.plot(
            x=epochs,
            y=losses,
            hires_mode=HiResMode.BRAILLE,
            line_style=DColor.GREEN,
            label=DLabel.CURRENT,
        )
        plot.set_ylimits()
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.LOSS)
        plot.show_legend(location=LegendLocation.BOTTOMLEFT)
