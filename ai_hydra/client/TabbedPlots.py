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
from statistics import median
from typing import Iterable

from textual import on
from textual.app import ComposeResult, Widget
from textual.widgets import TabbedContent
from textual_plot import HiResMode, LegendLocation, PlotWidget

from ai_hydra.constants.DHydraTui import DColor, DField, DLabel, DPlotDef


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
            DLabel.CUR_LOSS,
            DLabel.HIGHSCORES,
        ):
            yield PlotWidget(id=DField.GAME_SCORES)
            yield PlotWidget(id=DField.SCORES_PLOT)
            yield PlotWidget(id=DField.LOSS_PLOT)
            yield PlotWidget(id=DField.CUR_LOSS)
            yield PlotWidget(id=DField.SCORES_SCATTER_PLOT)

    def action_show_tab(self, tab: str) -> None:
        self.get_child_by_type(TabbedContent).active = tab

    @on(TabbedContent.TabActivated)
    def handle_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id
        self.call_later(lambda: self._replot_active_tab(pane_id))

    def add_loss(self, epoch: int, loss: float, plot: bool = True) -> None:
        self.loss_epochs.append(epoch)
        self.losses.append(loss)

        self.cur_epochs.append(epoch)
        self.cur_losses.append(loss)

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
        if isinstance(scores, tuple):
            self.scatter_scores.append(scores)
        else:
            self.scatter_scores.extend(scores)

        if plot:
            self._plot_scatter_scores()

    def reset(self) -> None:
        self._reset_data()

        self.query_one(f"#{DField.GAME_SCORES}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.LOSS_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.CUR_LOSS}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_SCATTER_PLOT}", PlotWidget).clear()

    def _reset_data(self) -> None:
        max_points = DPlotDef.MAX_CUR_DATA_POINTS
        max_loss_points = DPlotDef.MAX_CUR_LOSS_DATA_POINTS

        self.loss_epochs: list[int] = []
        self.losses: list[float] = []

        self.score_counts: dict[int, int] = {}
        self.scatter_scores: list[tuple[int, int]] = []

        self.game_scores: deque[int] = deque(maxlen=max_points)
        self.game_epochs: deque[int] = deque(maxlen=max_points)

        self.cur_losses: deque[float] = deque(maxlen=max_loss_points)
        self.cur_epochs: deque[int] = deque(maxlen=max_loss_points)

    def _replot_active_tab(self, pane_id: str) -> None:
        if pane_id == "tab-1":
            self._plot_game_scores()
        elif pane_id == "tab-2":
            self._plot_scores()
        elif pane_id == "tab-3":
            self._plot_loss()
        elif pane_id == "tab-4":
            self._plot_loss()
        elif pane_id == "tab-5":
            self._plot_scatter_scores()
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
            plot_epochs, plot_scores = self._thin_series(
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
            smoothed_scores = self._rolling_mean(scores, window)
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

        mean_score = self._weighted_mean(self.score_counts)
        median_score = self._weighted_median(self.score_counts)

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
            epochs, losses = self._thin_series(
                x=epochs,
                y=losses,
                max_points=DPlotDef.MAX_LOSS_DATA_POINTS,
            )

        plot.plot(
            x=epochs,
            y=losses,
            hires_mode=HiResMode.BRAILLE,
            line_style=DColor.GREEN,
            label=DLabel.LOSS,
        )
        plot.set_ylimits()
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.LOSS)
        plot.show_legend(location=LegendLocation.TOPRIGHT)

    def _plot_current_loss(self) -> None:
        plot = self.query_one(f"#{DField.CUR_LOSS}", PlotWidget)
        plot.clear()

        if not self.cur_epochs or not self.cur_losses:
            return

        plot.plot(
            x=list(self.cur_epochs),
            y=list(self.cur_losses),
            hires_mode=HiResMode.BRAILLE,
            line_style=DColor.GREEN,
            label=DLabel.LOSS,
        )
        plot.set_ylimits()
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.LOSS)
        plot.show_legend(location=LegendLocation.TOPRIGHT)

    def _plot_scatter_scores(self) -> None:
        plot = self.query_one(f"#{DField.SCORES_SCATTER_PLOT}", PlotWidget)
        plot.clear()

        if not self.scatter_scores:
            return

        episodes, scores = zip(*self.scatter_scores)

        plot.scatter(
            x=episodes,
            y=scores,
            marker_style=DColor.GREEN,
            hires_mode=HiResMode.BRAILLE,
            label=DLabel.SCORES,
        )
        plot.set_xlabel(DLabel.EPISODES)
        plot.set_ylabel(DLabel.SCORES)
        plot.show_legend(location=LegendLocation.TOPLEFT)

    @staticmethod
    def _get_smoothing_window(series_len: int) -> int:
        return max(5, series_len // DPlotDef.AVG_DIVISOR)

    @staticmethod
    def _rolling_mean(values: Iterable[float], window: int) -> list[float]:
        values_list = list(values)
        if window <= 1:
            return values_list

        return [
            sum(values_list[i : i + window]) / window
            for i in range(len(values_list) - window + 1)
        ]

    @staticmethod
    def _thin_series(
        x: list[int],
        y: list[float],
        max_points: int,
    ) -> tuple[list[int], list[float]]:
        if len(y) <= max_points:
            return x, y

        step = max(1, len(y) // max_points)
        thinned_x: list[int] = []
        thinned_y: list[float] = []

        for i in range(0, len(y), step):
            x_chunk = x[i : i + step]
            y_chunk = y[i : i + step]

            if not x_chunk or not y_chunk:
                continue

            # Keep first x, and average y within the chunk.
            thinned_x.append(x_chunk[0])
            thinned_y.append(sum(y_chunk) / len(y_chunk))

        return thinned_x, thinned_y

    @staticmethod
    def _weighted_mean(score_counts: dict[int, int]) -> float:
        total = sum(score_counts.values())
        if total == 0:
            return 0.0

        return (
            sum(score * count for score, count in score_counts.items()) / total
        )

    @staticmethod
    def _weighted_median(score_counts: dict[int, int]) -> float:
        total = sum(score_counts.values())
        if total == 0:
            return 0.0

        midpoint = total / 2
        running_total = 0

        for score in sorted(score_counts):
            running_total += score_counts[score]
            if running_total >= midpoint:
                return float(score)

        # Defensive fallback.
        return float(max(score_counts))
