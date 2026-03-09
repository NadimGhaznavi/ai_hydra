# ai_hydra/client/TabbedPlots.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

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
        self.ep_epochs = []
        self.ep_loss = []
        self.step_epochs = []
        self.step_loss = []
        self.scores = {}
        self.scores_lh = {}
        self.scores_nlh = {}

    def compose(self) -> ComposeResult:
        with TabbedContent(
            DLabel.EP_LOSS,
            DLabel.STEP_LOSS,
            DLabel.SCORES_ALL,
            DLabel.SCORES_LH,
            DLabel.SCORES_NLH,
        ):
            yield PlotWidget(id=DField.LOSS_EP_PLOT)
            yield PlotWidget(id=DField.LOSS_STEP_PLOT)
            yield PlotWidget(id=DField.SCORES_PLOT)
            yield PlotWidget(id=DField.SCORES_PLOT_LH)
            yield PlotWidget(id=DField.SCORES_PLOT_NLH)

    def action_show_tab(self, tab: str) -> None:
        self.get_child_by_type(TabbedContent).active = tab

    @on(TabbedContent.TabActivated)
    def handle_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id
        self.call_later(lambda: self._replot_active_tab(pane_id))

    def _replot_active_tab(self, pane_id: str) -> None:
        if pane_id == "tab-1":
            self._plot_loss(DNetField.EP_LOSS)
        elif pane_id == "tab-2":
            self._plot_loss(DNetField.STEP_LOSS)
        elif pane_id == "tab-3":
            self._plot_scores(self.scores, "all")
        elif pane_id == "tab-4":
            self._plot_scores(self.scores_lh, "lh")
        elif pane_id == "tab-5":
            self._plot_scores(self.scores_nlh, "nlh")
        else:
            raise ValueError(f"Unhandled tab: {pane_id}")

    def add_score(self, cur_score, lookahead, plot=True):
        self.scores[cur_score] = self.scores.get(cur_score, 0) + 1

        if lookahead:
            self.scores_lh[cur_score] = self.scores_lh.get(cur_score, 0) + 1
        else:
            self.scores_nlh[cur_score] = self.scores_nlh.get(cur_score, 0) + 1

        if plot:
            self._plot_scores(self.scores, "all")
            self._plot_scores(self.scores_lh, "lh")
            self._plot_scores(self.scores_nlh, "nlh")

    def _plot_scores(self, scores_dict, scores_type):
        x = sorted(scores_dict.keys())
        y = [scores_dict[k] for k in x]

        if scores_type == "all":
            scores_plot = self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget)
        elif scores_type == "lh":
            scores_plot = self.query_one(
                f"#{DField.SCORES_PLOT_LH}", PlotWidget
            )
        else:
            scores_plot = self.query_one(
                f"#{DField.SCORES_PLOT_NLH}", PlotWidget
            )

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

    def add_ep_loss(self, epoch, loss, plot=True):
        self.ep_loss.append(loss)
        self.ep_epochs.append(epoch)
        if plot:
            self._plot_loss(type=DNetField.EP_LOSS)

    def add_step_loss(self, epoch, loss, plot=True):
        self.step_loss.append(loss)
        self.step_epochs.append(epoch)
        if plot:
            self._plot_loss(type=DNetField.STEP_LOSS)

    def _plot_loss(self, type: str) -> None:
        if type == DNetField.EP_LOSS:
            line_style = DColor.GREEN
            losses = self.ep_loss
            plot_name = "Episode"
            epochs = self.ep_epochs
            loss_plot = self.query_one(f"#{DField.LOSS_EP_PLOT}", PlotWidget)
        else:
            line_style = DColor.RED
            losses = self.step_loss
            plot_name = "Step"
            epochs = self.step_epochs
            loss_plot = self.query_one(f"#{DField.LOSS_STEP_PLOT}", PlotWidget)

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
                label=plot_name,
            )
            loss_plot.set_ylimits()
            loss_plot.set_xlabel(DLabel.EPISODES)
            loss_plot.set_ylabel(DLabel.LOSS)
            loss_plot.show_legend(location=LegendLocation.TOPRIGHT)

    def reset(self) -> None:
        self.ep_loss = []
        self.ep_epochs = []
        self.step_loss = []
        self.step_epochs = []
        self.scores = {}
        self.scores_lh = {}
        self.scores_nlh = {}

        self.query_one(f"#{DField.LOSS_EP_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.LOSS_STEP_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_PLOT_LH}", PlotWidget).clear()
        self.query_one(f"#{DField.SCORES_PLOT_NLH}", PlotWidget).clear()
