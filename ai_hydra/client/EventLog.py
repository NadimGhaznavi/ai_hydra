# ai_hydra/client/EventLog.py
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

from ai_hydra.constants.DHydraTui import DField, DLabel
from ai_hydra.constants.DEvent import EVENT_MAP

from ai_hydra.utils.HydraMetrics import HydraMetrics


class EventLog(Widget):

    def __init__(self, metrics: HydraMetrics, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(
                f"[b #3e99af]{'Epoch':>7s}{DLabel.TIME:>9s}    {'Source':<19s}{DLabel.EVENT}[/]"
            ),
            Log(highlight=True, auto_scroll=True, id=DField.EVENT_LOG_LOG),
        )

    def clear(self):
        self.query_one(f"#{DField.EVENT_LOG_LOG}", Log).clear()

    def add_event(self, ev_type: str, event: str, epoch: str = None):
        if epoch:
            epoch = str(epoch)
        else:
            epoch = "-"

        elap_time = self.metrics.get_elapsed_time()
        ev_icon, ev_type_str = EVENT_MAP[ev_type]

        # Unicode 2 byte vs. 1 byte workaround
        if ev_icon == "⚙️":
            ev_icon = " ⚙️"

        msg = f"{epoch:>7s}{elap_time:>9s}  {ev_icon:>3s} {ev_type_str:<16s}{event}"

        self.query_one(f"#{DField.EVENT_LOG_LOG}", Log).write_line(msg)

        self.metrics.add_event_log_msg(msg)
