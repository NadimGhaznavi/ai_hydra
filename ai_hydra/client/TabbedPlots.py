# ai_hydra/client/TabbedPlots.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

### ----- Version II ---

from statistics import mean, median

from textual import on
from textual.app import ComposeResult, Widget
from textual.widgets import TabbedContent
from textual_plot import PlotWidget, HiResMode, LegendLocation

from ai_hydra.constants.DHydraTui import DField, DLabel, DPlotDef, DColor
from ai_hydra.constants.DNNet import DNetField


class TabbedPlots(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_epochs = []
        self.losses = []
        self.scores = {}
        self.scatter_scores = []

    def compose(self) -> ComposeResult:
        with TabbedContent(
            DLabel.LOSS,
            DLabel.SCORES,
            DLabel.SCORES_SCATTER,
        ):
            yield PlotWidget(id=DField.LOSS_PLOT)
            yield PlotWidget(id=DField.SCORES_PLOT)
            yield PlotWidget(id=DField.SCORES_SCATTER_PLOT)

    def action_show_tab(self, tab: str) -> None:
        self.get_child_by_type(TabbedContent).active = tab

    @on(TabbedContent.TabActivated)
    def handle_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id
        self.call_later(lambda: self._replot_active_tab(pane_id))

    def _replot_active_tab(self, pane_id: str) -> None:
        if pane_id == "tab-1":
            self._plot_loss()
        elif pane_id == "tab-2":
            self._plot_scores(self.scores)
        elif pane_id == "tab-3":
            self._plot_scatter_scores()
        else:
            raise ValueError(f"Unhandled tab: {pane_id}")

    def add_score(self, cur_score, plot=True):
        self.scores[cur_score] = self.scores.get(cur_score, 0) + 1

        if plot:
            self._plot_scores(self.scores)

    def add_scatter_score(self, score: list[tuple[int, int]], plot=True):
        self.scatter_scores.append(score)

        if plot:
            self._plot_scatter_scores()

    def _plot_scatter_scores(self):
        scatter_plot = self.query_one(
            f"#{DField.SCORES_SCATTER_PLOT}", PlotWidget
        )
        scatter_plot.clear()
        if not self.scatter_scores:
            return

        episode, score = zip(*self.scatter_scores)
        scatter_plot.scatter(
            episode,
            score,
            marker_style=DColor.GREEN,
            hires_mode=HiResMode.BRAILLE,
        )
        scatter_plot.show_legend
        scatter_plot.set_xlabel(DLabel.EPISODES)
        scatter_plot.set_ylabel(DLabel.SCORES)
        scatter_plot.show_legend(location=LegendLocation.TOPLEFT)

    def _plot_scores(self, scores_dict):
        x = sorted(scores_dict.keys())
        y = [scores_dict[k] for k in x]

        scores_plot = self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget)
        scores_plot.clear()
        if x:
            scores_plot.bar(
                x,
                y,
                bar_style=DColor.BLUE,
                width=0.8,
                hires_mode=HiResMode.BRAILLE,
            )
            scores_plot.set_ylimits()
            scores_plot.set_xlabel(DLabel.SCORES)
            scores_plot.set_ylabel(DLabel.EPISODES)
            scores_plot.show_legend(location=LegendLocation.TOPRIGHT)
            cur_mean = mean(scores_dict)
            cur_median = median(scores_dict)
            scores_plot.add_v_line(
                cur_mean, DColor.GREEN, f"Mean  : {cur_mean:.2f}"
            )
            scores_plot.add_v_line(
                cur_median, "purple", f"Median: {cur_median:.2f}"
            )

    def add_loss(self, epoch, loss, plot=True):
        self.losses.append(loss)
        self.loss_epochs.append(epoch)
        if plot:
            self._plot_loss()

    def _plot_loss(self) -> None:
        line_style = DColor.GREEN
        losses = self.losses
        epochs = self.loss_epochs
        loss_plot = self.query_one(f"#{DField.LOSS_PLOT}", PlotWidget)

        if len(losses) > DPlotDef.MAX_LOSS_DATA_POINTS:
            step = max(1, len(epochs) // DPlotDef.MAX_LOSS_DATA_POINTS)
            thinned_epochs = []
            thinned_losses = []

            for i in range(0, len(losses), step):
                segment = losses[i : i + step]
                thinned_losses.append(sum(segment) / len(segment))
                thinned_epochs.append(epochs[i])

            losses = thinned_losses
            epochs = thinned_epochs

        loss_plot.clear()

        if epochs:
            loss_plot.plot(
                x=epochs,
                y=losses,
                hires_mode=HiResMode.BRAILLE,
                line_style=line_style,
                label=DLabel.LOSS,
            )
            loss_plot.set_ylimits()
            loss_plot.set_xlabel(DLabel.EPISODES)
            loss_plot.set_ylabel(DLabel.LOSS)
            loss_plot.show_legend(location=LegendLocation.TOPRIGHT)

    def reset(self) -> None:
        self.losses = []
        self.loss_epochs = []
        self.scores = {}
        self.scatter_scores = []

        self.query_one(f"#{DField.LOSS_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_SCATTER_PLOT}", PlotWidget).clear()
