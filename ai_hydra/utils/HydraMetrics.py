# ai_hydra/utils/HydraMetrics.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from collections import deque
from statistics import mean, median

from ai_hydra.constants.DNNet import DRNN
from ai_hydra.constants.DHydraTui import DPlotDef
from ai_hydra.utils.MetricEvent import (
    HighscoreEvent,
    ScoreEvent,
    LossEvent,
    ShiftEvent,
    MemEvent,
    NiceEvent,
)

MAX_CUR_SCORES = DPlotDef.MAX_CUR_SCORES
RECENT_SCORES_MAX = DPlotDef.RECENT_SCORES_MAX
RECENT_LOSS_MAX = DPlotDef.RECENT_LOSS_MAX

AVG_DIVISOR = 5
AVG_CUR_SCORES = MAX_CUR_SCORES // AVG_DIVISOR


class HydraMetrics:

    def __init__(self, initial_epsilon: float) -> None:
        self._cur_epsilon = initial_epsilon
        self._init_data()

    def _init_data(self):
        self._cur_epoch: int = 0
        self._cur_loss: float | None = None
        self._cur_score: int = 0
        self._elapsed_time: str = "0s"
        self._next_bucket_snapshot_epoch = 500

        # Current scores data
        self._cur_scores: deque[ScoreEvent] = deque(maxlen=MAX_CUR_SCORES)
        self._avg_cur_scores: deque[ScoreEvent] = deque(maxlen=AVG_CUR_SCORES)
        self._avg_cur_scores_buf: list = []

        # Highscore events
        self._highscore_events: list[HighscoreEvent] = []

        # Score distribution
        self._scores_dist: dict[int, int] = {}
        self._recent_scores_dist: dict[int, int] = {}
        self._scores_dist_list: list[int] = []
        self._recent_scores_dist_list: list[int] = []

        # Loss events
        self._losses: list[LossEvent] = []
        self._recent_losses: deque[LossEvent] = deque(maxlen=RECENT_LOSS_MAX)

        # EventLog events
        self._eventlog_msgs = []

        # Epoch, mean and median values
        self._mean_and_median: list[tuple[int, float, float]] = []
        self._recent_mean_and_median: list[tuple[int, float, float]] = []

        # Gear shift events
        self._shift_events: list[ShiftEvent] = []

        # ATH Memory events
        self._memory_events: list[MemEvent] = []
        self._latest_memory_event: MemEvent | None = None

        # EpsilonNice events
        self._epsilon_nice_events: list[NiceEvent] = []

        # Seed the shift events list...
        self.add_shift_event(
            gear=0, seq_length=DRNN.SEQ_LENGTH, batch_size=DRNN.BATCH_SIZE
        )

    def clear(self, initial_epsilon: float) -> None:
        self._cur_epsilon = initial_epsilon
        self._init_data()

    def add_bucket_stats(self, bucket_counts) -> None:

        ordered_counts = tuple(
            count for _, count in sorted(bucket_counts.items())
        )

        mem_event = MemEvent(
            epoch=self.get_cur_epoch(),
            bucket_counts=ordered_counts,
        )

        self._latest_memory_event = mem_event

        if self._cur_epoch < self._next_bucket_snapshot_epoch:
            return

        mem_event = MemEvent(
            epoch=self._next_bucket_snapshot_epoch,
            bucket_counts=ordered_counts,
        )

        self._memory_events.append(mem_event)
        self._next_bucket_snapshot_epoch += 500

    def add_cur_epoch(self, epoch: int) -> None:
        self._cur_epoch = epoch

    def add_cur_epsilon(self, epsilon: float) -> None:
        self._cur_epsilon = epsilon

    def add_cur_loss(self, loss: float) -> None:
        if loss is None:
            return False

        # Don't insert dupes
        if self._losses and self._cur_epoch == self._losses[-1].epoch:
            return False

        self._cur_loss = loss
        self._losses.append(LossEvent(epoch=self._cur_epoch, loss=loss))
        self._recent_losses.append(LossEvent(epoch=self._cur_epoch, loss=loss))
        return True

    def add_cur_score(self, score: int) -> None:
        self._cur_score = score

    def add_elapsed_time(self, elapsed_time: str) -> None:
        self._elapsed_time = elapsed_time

    def add_event_log_msg(self, msg):
        self._eventlog_msgs.append(msg)

    def add_final_score(self, score: int) -> None:
        cur_scores = self._cur_scores

        # Don't insert dupes
        if cur_scores and self._cur_epoch == cur_scores[-1].epoch:
            return False

        # Current Scores plot data
        score_event = ScoreEvent(epoch=self._cur_epoch, score=score)
        cur_scores.append(score_event)

        avg_buf = self._avg_cur_scores_buf
        avg_buf.append(score_event)
        if len(avg_buf) == AVG_DIVISOR:
            avg_epoch = avg_buf[-1].epoch
            avg_score = mean(e.score for e in avg_buf)
            self._avg_cur_scores.append(
                ScoreEvent(epoch=avg_epoch, score=avg_score)
            )
            self._avg_cur_scores_buf = []

        # All time score distribution data
        self._scores_dist[score] = self._scores_dist.get(score, 0) + 1
        self._scores_dist_list.append(score)

        # Recent time score distribution data
        self._recent_scores_dist[score] = (
            self._recent_scores_dist.get(score, 0) + 1
        )
        self._recent_scores_dist_list.append(score)

        if len(self._recent_scores_dist_list) > RECENT_SCORES_MAX:
            old_score = self._recent_scores_dist_list.pop(0)
            self._recent_scores_dist[old_score] -= 1
            if self._recent_scores_dist[old_score] == 0:
                del self._recent_scores_dist[old_score]

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

    def add_mean_median(self, epoch, mean, median):
        self._mean_and_median.append((epoch, mean, median))

    def add_nice_event(
        self,
        window,
        epoch,
        calls,
        fatal_suggested,
        triggered,
        overrides,
        no_safe_alternative,
        trigger_rate,
        override_rate,
        rescue_rate,
    ):
        self._epsilon_nice_events.append(
            NiceEvent(
                window=window,
                epoch=epoch,
                calls=calls,
                triggered=triggered,
                fatal_suggested=fatal_suggested,
                overrides=overrides,
                no_safe_alternative=no_safe_alternative,
                trigger_rate=trigger_rate,
                override_rate=override_rate,
                rescue_rate=rescue_rate,
            )
        )

    def add_recent_mean_and_median(self, epoch, mean, median):
        self._recent_mean_and_median.append((epoch, mean, median))

    def add_shift_event(
        self,
        gear: int,
        seq_length: int,
        batch_size: int,
    ) -> None:
        self._shift_events.append(
            ShiftEvent(
                epoch=self._cur_epoch,
                gear=gear,
                seq_length=seq_length,
                batch_size=batch_size,
            )
        )

    # ----- Get Ops ------------------

    def get_avg_cur_score_plot_points(self) -> list[tuple[int, float]]:
        return [(e.epoch, e.score) for e in self._avg_cur_scores]

    def get_bucket_snapshot_rows(self) -> list[tuple[int, ...]]:
        rows = []
        for e in self._memory_events:
            rows.append((e.epoch, *e.bucket_counts))
        return rows

    def get_bucket_snaphot(self) -> MemEvent:
        return self._latest_memory_event

    def get_cur_epoch(self) -> int:
        return self._cur_epoch

    def get_cur_score(self) -> int:
        return self._cur_score

    def get_cur_score_plot_points(self) -> list[tuple[int, int]]:
        return [(e.epoch, e.score) for e in self._cur_scores]

    def get_elapsed_time(self) -> str:
        return self._elapsed_time

    def get_eventlog_msgs(self) -> list:
        return self._eventlog_msgs

    def get_highscore_plot_points(self) -> list[tuple[int, int]]:
        return [(e.epoch, e.highscore) for e in self._highscore_events]

    def get_highscore_snapshot_rows(self) -> list[tuple[int, int, float, str]]:
        return [
            (e.epoch, e.highscore, e.epsilon, e.elapsed_time)
            for e in self._highscore_events
        ]

    def get_last_highscore_event(self) -> HighscoreEvent:
        if self._highscore_events:
            return self._highscore_events[-1]

    def get_loss_plot_points(self) -> list[tuple[int, float]]:
        return [(e.epoch, e.loss) for e in self._losses]

    def get_epsilon_nice_events(
        self,
    ) -> list[tuple[str, int, int, int, int, int, int, float, float, float]]:
        return [
            (
                e.window,
                e.epoch,
                e.calls,
                e.triggered,
                e.fatal_suggested,
                e.overrides,
                e.no_safe_alternative,
                e.trigger_rate,
                e.override_rate,
                e.rescue_rate,
            )
            for e in self._epsilon_nice_events
        ]

    def get_recent_loss_plot_points(self) -> list[tuple[int, float]]:
        return [(e.epoch, e.loss) for e in self._recent_losses]

    def get_recent_scores_dist_plot_points(
        self,
    ) -> tuple[list[int], list[int]]:
        scores = sorted(self._recent_scores_dist.keys())
        counts = [self._recent_scores_dist[score] for score in scores]
        return scores, counts

    def get_recent_scores_dist_stats(self) -> tuple[float, float] | None:
        if not self._recent_scores_dist_list:
            return None
        return mean(self._recent_scores_dist_list), median(
            self._recent_scores_dist_list
        )

    def get_scores_dist_plot_points(self) -> tuple[list[int], list[int]]:
        scores = sorted(self._scores_dist.keys())
        counts = [self._scores_dist[score] for score in scores]
        return scores, counts

    def get_scores_dist_stats(self) -> tuple[float, float] | None:
        if not self._scores_dist_list:
            return None
        return mean(self._scores_dist_list), median(self._scores_dist_list)

    def get_shift_mean_median_snapshot_rows(
        self,
    ) -> list[tuple[int, int, int, int, float, float]]:
        rows = []
        shifts = sorted(self._shift_events, key=lambda e: e.epoch)

        for epoch, mean_score, median_score in self._mean_and_median:
            active_shift = None
            for shift in shifts:
                if shift.epoch <= epoch:
                    active_shift = shift
                else:
                    break

            if active_shift is None:
                rows.append(
                    (epoch, None, None, None, mean_score, median_score)
                )
            else:
                rows.append(
                    (
                        epoch,
                        active_shift.gear,
                        active_shift.seq_length,
                        active_shift.batch_size,
                        mean_score,
                        median_score,
                    )
                )
        return rows

    def set_initial_epsilon(self, value: float) -> None:
        self._cur_epsilon = value
