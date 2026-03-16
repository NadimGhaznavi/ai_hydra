# ai_hydra/client/plots/PlotUtils.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Iterable

from ai_hydra.constants.DHydraTui import DPlotDef


@staticmethod
def _get_smoothing_window(series_len: int) -> int:
    return max(5, series_len // DPlotDef.AVG_DIVISOR)


def rolling_mean(values: Iterable[float], window: int) -> list[float]:
    values_list = list(values)
    if window <= 1:
        return values_list

    return [
        sum(values_list[i : i + window]) / window
        for i in range(len(values_list) - window + 1)
    ]


def thin_series(
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


def weighted_mean(score_counts: dict[int, int]) -> float:
    total = sum(score_counts.values())
    if total == 0:
        return 0.0

    return sum(score * count for score, count in score_counts.items()) / total


def weighted_median(score_counts: dict[int, int]) -> float:
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
