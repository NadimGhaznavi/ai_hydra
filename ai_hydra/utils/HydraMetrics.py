# ai_hydra/utils/HydraMetrics.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from collections import deque
from statistics import mean

from ai_hydra.constants.DHydraTui import DPlotDef
from ai_hydra.utils.MetricEvent import HighscoreEvent, ScoreEvent

MAX_CUR_SCORES = DPlotDef.MAX_CUR_SCORES
AVG_CUR_SCORES = MAX_CUR_SCORES // 5


class HydraMetrics:

    def __init__(self, initial_epsilon: float) -> None:
        self._cur_epsilon = initial_epsilon
        self._cur_epoch: int = 0
        self._cur_loss: float | None = None
        self._cur_score: int = 0
        self._elapsed_time: str = "0s"

        self._cur_scores: deque[ScoreEvent] = deque(maxlen=MAX_CUR_SCORES)
        self._avg_cur_scores: deque[ScoreEvent] = deque(maxlen=AVG_CUR_SCORES)
        self._avg_cur_scores_buf: list = []

        self._highscore_events: list[HighscoreEvent] = []

    def add_cur_epoch(self, epoch: int) -> None:
        self._cur_epoch = epoch

    def add_cur_epsilon(self, epsilon: float) -> None:
        self._cur_epsilon = epsilon

    def add_cur_loss(self, loss: float) -> None:
        if loss is not None:
            self._cur_loss = loss

    def add_cur_score(self, score: int) -> None:
        self._cur_score = score

    def add_elapsed_time(self, elapsed_time: str) -> None:
        self._elapsed_time = elapsed_time

    def add_final_score(self, score: int) -> None:
        cur_scores = self._cur_scores

        # Don't insert dupes
        if cur_scores and self._cur_epoch == cur_scores[-1].epoch:
            return False

        score_event = ScoreEvent(epoch=self._cur_epoch, score=score)
        cur_scores.append(score_event)

        avg_buf = self._avg_cur_scores_buf
        avg_buf.append(score_event)
        if len(avg_buf) == 5:
            avg_epoch = avg_buf[-1].epoch
            avg_score = mean(e.score for e in avg_buf)
            self._avg_cur_scores.append(
                ScoreEvent(epoch=avg_epoch, score=avg_score)
            )
            self._avg_cur_scores_buf = []

        return True

    def add_highscore(self, highscore: int) -> bool:
        """
        Add a highscore.

        Do not insert duplicate highscores. If the highscore parameter is a
        duplicate, return "False". Otherwise return "True".
        """
        if highscore is None:
            return False

        events = self._highscore_events
        # Don't insert dupes
        if events and highscore == events[-1].highscore:
            return False

        # Add new record
        events.append(
            HighscoreEvent(
                epoch=self._cur_epoch,
                highscore=highscore,
                epsilon=self._cur_epsilon,
                elapsed_time=self._elapsed_time,
            )
        )
        return True

    def get_avg_cur_score_plot_points(self) -> list[tuple[int, float]]:
        return [(e.epoch, e.score) for e in self._avg_cur_scores]

    def get_cur_epoch(self) -> int:
        return self._cur_epoch

    def get_cur_score(self) -> int:
        return self._cur_score

    def get_cur_score_plot_points(self) -> list[tuple[int, int]]:
        return [(e.epoch, e.score) for e in self._cur_scores]

    def get_elapsed_time(self) -> str:
        return self._elapsed_time

    def get_highscore_plot_points(self) -> list[tuple[int, int]]:
        return [(e.epoch, e.highscore) for e in self._highscore_events]

    def get_last_highscore_event(self) -> HighscoreEvent:
        if self._highscore_events:
            return self._highscore_events[-1]

    def get_highscore_snapshot_rows(self) -> list[tuple[int, int, float, str]]:
        return [
            (e.epoch, e.highscore, e.epsilon, e.elapsed_time)
            for e in self._highscore_events
        ]

    def set_initial_epsilon(self, value: float) -> None:
        self._cur_epsilon = value
