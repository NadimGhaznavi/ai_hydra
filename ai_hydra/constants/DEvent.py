# ai_hydra/constants/DEvent.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final

from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DHydra import DModule


EVENT_MAP = {
    DModule.ATH_REPLAY_MEMORY: ("💾", "ATH Memory"),
    DModule.ATH_DATA_MGR: ("💾", "ATH Data Mgr"),
    DModule.ATH_GEARBOX: ("🧠", "ATH Gearbox"),
    DModule.EPSILON_ALGO: ("🧭", "Epsilon"),
    DModule.HYDRA_RNG: ("🎲", "Random"),
    DModule.SIMPLE_REPLAY_MEMORY: ("💾", "Replay Memory"),
    DField.SIM_LOOP: ("⚡", "Simulation"),
    DField.SNAPSHOT: ("📸", "Snapshot"),
    DField.TUI: ("💻", "Hydra Client"),
    DField.WARNING: ("⚠️", "Warning"),
}


class EV_TYPE:
    BUCKET_CAPACITIES: Final[str] = "bucket_capacities"
    BUCKET_COUNTS: Final[str] = "bucket_counts"
    SHIFTING: Final[str] = "shifting"
    BUCKETS_STATUS: Final[str] = "buckets_status"


class EV_STATUS:
    GOOD: Final[str] = "🟢"
    BAD: Final[str] = "🔴"
    WARN: Final[str] = "🟡"
    INFO: Final[str] = "🔵"
