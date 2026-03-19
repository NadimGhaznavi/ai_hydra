from textual.app import ComposeResult, Widget
from textual.containers import Horizontal
from textual.widgets import Label
from textual.color import Color

from ai_hydra.constants.DHydraTui import DField

BUCKET = "bucket"
NUM_BUCKETS = 20
BASE_COLOR = "#0f1e10"


def percent_to_hex(p: float) -> str:
    """
    Convert a float in [0.0, 1.0] to a hex string "00" - "ff".
    """
    p = max(0.0, min(1.0, p))
    value = int(round(p * 255))
    return f"{value:02x}"


class ATHMemory(Widget):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("Memory", id=DField.REPLAY_MEM)
            for i in range(NUM_BUCKETS):
                yield Label(id=f"b{i}", classes=BUCKET)

    def update_stats(self, stats: dict[str | int, int]) -> None:
        if not stats:
            return

        max_size = max(stats.values())
        min_size = min(stats.values())
        span = max_size - min_size

        for bucket, count in stats.items():
            label_id = f"b{bucket}"
            cur_label = self.query_one(f"#{label_id}", Label)
            cur_label.update(str(count))

            pct = 1.0 if span == 0 else (count - min_size) / span
            alpha_hex = percent_to_hex(pct)
            cur_label.styles.background = Color.parse(
                f"{BASE_COLOR}{alpha_hex}"
            )
