"""
ai_snake_lab/ui/TabbedPlots.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from collections import deque

from textual.widgets import TabbedContent, Static
from textual.app import ComposeResult, Widget
from textual_plot import PlotWidget, HiResMode, LegendLocation

from ai_snake_lab.constants.DLabels import DLabel
from ai_snake_lab.constants.DPlot import Plot
from ai_snake_lab.constants.DLayout import DLayout
from ai_snake_lab.constants.DColors import DColor


class TabbedPlots(Widget):

    game_score_epoch = deque(maxlen=Plot.MAX_GAMESCORE_DATA_POINTS)
    game_score_score = deque(maxlen=Plot.MAX_GAMESCORE_DATA_POINTS)
    highscores_game = []
    highscores_score = []
    loss = []
    loss_epoch = []

    def compose(self) -> ComposeResult:
        with TabbedContent(DLabel.GAME_SCORE, DLabel.HIGHSCORES, DLabel.LOSS):
            yield PlotWidget(id=DLayout.GAME_SCORE_PLOT)
            yield PlotWidget(id=DLayout.HIGHSCORES_PLOT)
            yield PlotWidget(id=DLayout.LOSS_PLOT)

    def action_show_tab(self, tab: str) -> None:
        """Switch to a new tab."""
        self.get_child_by_type(TabbedContent).active = tab

    def add_game_score_data(self, epoch, score, plot=True):
        game_score = self.query_one(f"#{DLayout.GAME_SCORE_PLOT}", PlotWidget)
        self.game_score_epoch.append(epoch)
        self.game_score_score.append(score)
        if plot:
            epochs = self.game_score_epoch
            scores = self.game_score_score

            # Clear the existing plot
            game_score.clear()

            # Current score plot
            game_score.plot(
                x=epochs,
                y=scores,
                hires_mode=HiResMode.BRAILLE,
                line_style=DColor.GREEN,
                label=DLabel.CURRENT,
            )

            # Add an average plot over 20 to wash out the spikes and identify when the
            # AI is maxing out.
            window = max(1, len(scores) // Plot.AVG_DIVISOR)
            # e.g., 5% smoothing window
            scores_list = list(scores)
            epochs_list = list(epochs)
            if len(scores) > window:
                smoothed = [
                    sum(scores_list[i : i + window]) / len(scores_list[i : i + window])
                    for i in range(len(scores_list) - window + 1)
                ]
                smoothed_epochs = epochs_list[window - 1 :]
                game_score.plot(
                    x=smoothed_epochs,
                    y=smoothed,
                    hires_mode=HiResMode.BRAILLE,
                    line_style=DColor.RED,  # distinct color for trend
                    label=DLabel.AVERAGE,
                )
            game_score.show_legend(location=LegendLocation.TOPLEFT)

    def add_highscore_data(self, epoch, score, plot=True):
        highscores = self.query_one(f"#{DLayout.HIGHSCORES_PLOT}", PlotWidget)
        self.highscores_game.append(int(epoch))
        self.highscores_score.append(int(score))

        if plot:
            games = self.highscores_game
            scores = self.highscores_score

            # Clear the existing plot
            highscores.clear()

            # Plot the data
            highscores.plot(
                x=games,
                y=scores,
                hires_mode=HiResMode.BRAILLE,
                line_style=DColor.GREEN,
            )

    def add_loss_data(self, epoch, loss, plot=True):
        loss_plot = self.query_one(f"#{DLayout.LOSS_PLOT}", PlotWidget)
        self.loss.append(loss)
        self.loss_epoch.append(epoch)

        if plot:
            losses = self.loss
            epochs = self.loss_epoch

            # We need to "thin" the data as the number of games/epochs rises otherwise
            # plot gets "blurry". For this kind of data, average binning makes sense.
            if len(losses) > Plot.MAX_LOSS_DATA_POINTS:
                step = max(1, len(epochs) // Plot.MAX_LOSS_DATA_POINTS)
                thinned_epochs = []
                thinned_losses = []
                for i in range(0, len(losses), step):
                    segment = losses[i : i + step]
                    thinned_losses.append(sum(segment) / len(segment))
                    thinned_epochs.append(epochs[i])  # midpoint of the bin
                losses = thinned_losses
                epochs = thinned_epochs

            # Clear the existing plot and plot the new data
            loss_plot.clear()
            loss_plot.plot(
                x=epochs,
                y=losses,
                hires_mode=HiResMode.BRAILLE,
                line_style=DColor.GREEN,
            )

    def clear_data(self):
        self.game_score_epoch = deque(maxlen=Plot.MAX_GAMESCORE_DATA_POINTS)
        self.game_score_score = deque(maxlen=Plot.MAX_GAMESCORE_DATA_POINTS)
        self.highscores_game = []
        self.highscores_score = []
        self.loss = []
        self.loss_epoch = []
        self.query_one(f"#{DLayout.HIGHSCORES_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DLayout.GAME_SCORE_PLOT}", PlotWidget).clear()
        self.query_one(f"#{DLayout.LOSS_PLOT}", PlotWidget).clear()

    def on_mount(self):
        # Game score plot
        game_score = self.query_one(f"#{DLayout.GAME_SCORE_PLOT}", PlotWidget)
        game_score.set_xlabel(DLabel.GAME_NUM)
        game_score.set_ylabel(DLabel.GAME_SCORE)
        # Highscores plot
        highscores = self.query_one(f"#{DLayout.HIGHSCORES_PLOT}", PlotWidget)
        highscores.set_xlabel(DLabel.GAME_NUM)
        highscores.set_ylabel(DLabel.SCORE)
        # Loss Plot
        losses = self.query_one(f"#{DLayout.LOSS_PLOT}", PlotWidget)
        losses.set_xlabel(DLabel.GAME_NUM)
        losses.set_ylabel(DLabel.LOSS)
