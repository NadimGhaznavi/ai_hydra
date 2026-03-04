# ai_hydra/client/TabbedPlots.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from textual.containers import Vertical
from textual.widgets import TabbedContent, Label, Log
from textual.app import ComposeResult, Widget
from textual_plot import PlotWidget, HiResMode, LegendLocation

from ai_hydra.constants.DHydraTui import DField, DLabel, DPlotDef, DColor
from ai_hydra.constants.DNNet import DLookahead


class TabbedPlots(Widget):

    loss = []
    loss_epoch = []

    def compose(self) -> ComposeResult:
        with TabbedContent(DLabel.LOSS):
            yield PlotWidget(id=DField.LOSS)

    def add_loss(self, epoch, loss, plot=True):
        self.loss.append(loss)
        self.loss_epoch.append(epoch)

        if plot:
            losses = self.loss
            epochs = self.loss_epoch

            # We need to "thin" the data as the number of games/epochs rises otherwise
            # plot gets "blurry". For this kind of data, average binning makes sense.
            if len(losses) > DPlotDef.MAX_LOSS_DATA_POINTS:
                step = max(1, len(epochs) // DPlotDef.MAX_LOSS_DATA_POINTS)
                thinned_epochs = []
                thinned_losses = []
                for i in range(0, len(losses), step):
                    segment = losses[i : i + step]
                    thinned_losses.append(sum(segment) / len(segment))
                    thinned_epochs.append(epochs[i])  # midpoint of the bin
                losses = thinned_losses
                epochs = thinned_epochs

            # Clear the existing plot and plot the new data
            loss_plot = self.query_one(f"#{DField.LOSS}")
            loss_plot.clear()
            loss_plot.plot(
                x=epochs,
                y=losses,
                hires_mode=HiResMode.BRAILLE,
                line_style=DColor.GREEN,
            )

    def reset(self) -> None:
        self.loss = []
        self.loss_epoch = []
        self.query_one(f"#{DField.LOSS}", PlotWidget).clear()
