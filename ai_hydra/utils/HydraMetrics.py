# ai_hydra/utils/HydraMetrics.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from ai_hydra.utils.HighscoreEvent import HighscoreEvent


class HydraMetrics:

    def __init__(self, initial_epsilon: float) -> None:
        self._cur_epsilon = initial_epsilon
        self._cur_epoch: int | None = None
        self._cur_loss: float | None = None
        self._cur_score: int | None = None
        self._elapsed_time: str | None = None

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

    def get_cur_score(self) -> int:
        return self._cur_score

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
        self._initial_epsilon = value
