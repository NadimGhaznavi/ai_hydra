from textual.app import ComposeResult, Widget
from textual.containers import Horizontal
from textual.widgets import Label
from textual.color import Color

from ai_hydra.utils.HydraMetrics import HydraMetrics
from ai_hydra.constants.DHydraTui import DField

BUCKET = "bucket"
BASE_COLOR = "#0f1e10"


def percent_to_hex(p: float) -> str:
    """
    Convert a float in [0.0, 1.0] to a hex string "00" - "ff".
    """
    p = max(0.0, min(1.0, p))
    value = int(round(p * 255))
    return f"{value:02x}"


class ATHMemory(Widget):

    def __init__(
        self,
        metrics: HydraMetrics,
        max_buckets: int,
        mem_id: str,
        mem_label: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self._max_buckets = max_buckets
        self._mem_id = mem_id
        self._mem_label = mem_label

    def clear(self):
        for count in range(self._max_buckets):
            label_id = f"b{count}"
            self.query_one(f"#{label_id}", Label).update("")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(self._mem_label, id=self._mem_id)
            for i in range(self._max_buckets):
                yield Label(id=f"b{i}", classes=BUCKET)

    def refresh_data(self) -> None:

        if self._mem_id == DField.REPLAY_MEM:
            mem_event = self.metrics.get_bucket_snaphot()
        elif self._mem_id == DField.MCTS_REPLAY_MEM:
            mem_event = self.metrics.get_mcts_bucket_snaphot()

        if mem_event is None:
            return

        bucket_counts = mem_event.bucket_counts
        max_size = max(bucket_counts)
        min_size = min(bucket_counts)
        span = max_size - min_size

        bucket_idx = 0
        for count in bucket_counts:
            label_id = f"b{bucket_idx}"
            cur_label = self.query_one(f"#{label_id}", Label).update(
                str(count)
            )

            pct = 1.0 if span == 0 else (count - min_size) / span
            alpha_hex = percent_to_hex(pct)
            cur_label.styles.background = Color.parse(
                f"{BASE_COLOR}{alpha_hex}"
            )
            bucket_idx += 1

    def set_max_buckets(self, max_buckets: int):
        self._max_buckets = max_buckets
        self.refresh()
