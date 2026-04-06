# ai_hydra/utils/HydraSnapshot.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from datetime import datetime
import traceback

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DHydraTui import DField, DLabel
from ai_hydra.constants.DNNet import DNetDef, DNetField, DLinear, DRNN, DGRU
from ai_hydra.constants.DGame import DGameDef
from ai_hydra.constants.DReplayMemory import DMemDef
from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.utils.HydraMetrics import HydraMetrics


class HydraSnapshot:
    def __init__(self, metrics: HydraMetrics):
        self.metrics = metrics

    def create_snapshot(self, snap_file: str, cfg: SimCfg) -> None:
        report = self._build_report(cfg)
        with open(snap_file, "w", encoding="utf-8") as f:
            f.write(report)

    def _build_report(self, cfg: SimCfg) -> str:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = []

        lines.extend(
            self._build_kv_section(
                "📸 AI Hydra - Snapshot Report",
                [
                    ("Timestamp", timestamp),
                    ("Simulation Run Time", self.metrics.get_elapsed_time()),
                    (
                        "Current Episode Number",
                        str(self.metrics.get_cur_epoch()),
                    ),
                    ("AI Hydra Version", f"v{DHydra.VERSION}"),
                    ("Random Seed", str(cfg.get(DNetField.RANDOM_SEED))),
                ],
            )
        )

        lines.extend(
            self._build_kv_section(
                "🎯 Epsilon Greedy",
                [
                    (
                        "Initial Epsilon",
                        str(cfg.get(DNetField.INITIAL_EPSILON)),
                    ),
                    ("Minimum Epsilon", str(cfg.get(DNetField.MIN_EPSILON))),
                    (
                        "Epsilon Decay Rate",
                        str(cfg.get(DNetField.EPSILON_DECAY)),
                    ),
                ],
            )
        )

        model_type = cfg.get(DNetField.MODEL_TYPE)

        lines.extend(self._build_model_section(cfg))
        lines.extend(self._build_rewards_section(cfg))
        lines.extend(self._build_memory_section(cfg))
        lines.extend(self._build_mcts_section(cfg))
        lines.extend(self._build_event_log_section())
        lines.extend(self._build_highscore_section())
        lines.extend(self._build_nice_section(cfg))
        lines.extend(self._build_nice_table())
        lines.extend(self._build_shift_mean_median_section())
        lines.extend(self._build_bucket_table())
        lines.extend(self._build_mcts_bucket_table())

        return "\n".join(lines).rstrip() + "\n"

    def _build_model_section(self, cfg: SimCfg) -> list[str]:
        model_type = cfg.get(DNetField.MODEL_TYPE)
        model_hidden_size = cfg.get(DNetField.HIDDEN_SIZE)
        model_input_size = DNetDef.INPUT_SIZE
        dropout_p = cfg.get(DNetField.DROPOUT_P)
        learning_rate = cfg.get(DNetField.LEARNING_RATE)
        gamma = cfg.get(DNetField.GAMMA)
        layers = cfg.get(DNetField.LAYERS)
        tau = cfg.get(DNetField.TAU)

        if model_type == DField.LINEAR:
            model_name = DLabel.LINEAR
        elif model_type == DField.RNN:
            model_name = DLabel.RNN
        elif model_type == DField.GRU:
            model_name = DLabel.GRU
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return self._build_kv_section(
            f"🧠 {model_name} Model",
            [
                ("Input Size", str(model_input_size)),
                ("Hidden Size", str(model_hidden_size)),
                ("Dropout Layer P-Value", str(dropout_p)),
                ("Layers", str(layers)),
                ("Learning Rate", str(learning_rate)),
                ("Discount/Gamma", str(gamma)),
                ("Tau", str(tau)),
            ],
        )

    def _build_rewards_section(self, cfg: SimCfg) -> list[str]:
        return self._build_kv_section(
            "💰 Rewards",
            [
                (
                    "Food Reward",
                    cfg.get(DNetField.FOOD_REWARD),
                ),
                (
                    "Collision Penalty",
                    cfg.get(DNetField.COLLISION_PENALTY),
                ),
                (
                    "Max Moves Penalty",
                    cfg.get(DNetField.MAX_MOVES_PENALTY),
                ),
                (
                    "Empty Move Reward",
                    cfg.get(DNetField.EMPTY_MOVE_REWARD),
                ),
                (
                    "Closer to Food",
                    cfg.get(DNetField.CLOSER_TO_FOOD),
                ),
                (
                    "Further from Food",
                    cfg.get(DNetField.FURTHER_FROM_FOOD),
                ),
                (
                    "Max Moves Multiplier",
                    cfg.get(DNetField.MAX_MOVES_MULTIPLIER),
                ),
            ],
        )

    def _build_memory_section(self, cfg: SimCfg) -> list[str]:
        return self._build_kv_section(
            "💾 Replay Memory",
            [
                ("Max Frames", cfg.get(DNetField.MAX_FRAMES)),
                ("Memory Buckets", cfg.get(DNetField.MAX_BUCKETS)),
                (
                    "Max Training Frames",
                    cfg.get(DNetField.MAX_TRAINING_FRAMES),
                ),
                ("Highest Gear", cfg.get(DNetField.MAX_GEAR)),
                (
                    "Cooldown Threshold",
                    cfg.get(DNetField.NUM_COOLDOWN_EPISODES),
                ),
                (
                    "Upshift Threshold",
                    cfg.get(DNetField.UPSHIFT_COUNT_THRESHOLD),
                ),
                (
                    "Downshift Threshold",
                    cfg.get(DNetField.DOWNSHIFT_COUNT_THRESHOLD),
                ),
                (
                    "Stagnant Threshold",
                    cfg.get(DNetField.MAX_STAGNANT_EPISODES),
                ),
                (
                    "Critical Stagnant Threshold",
                    cfg.get(DNetField.MAX_HARD_RESET_EPISODES),
                ),
            ],
        )

    def _build_mcts_section(self, cfg: SimCfg) -> list[str]:
        return self._build_kv_section(
            "🎲 Monte Carlo Tree Search",
            [
                (DLabel.MCTS_FREQUENCY, cfg.get(DNetField.MCTS_FREQUENCY)),
                (DLabel.SEARCH_DEPTH, cfg.get(DNetField.MCTS_DEPTH)),
                (DLabel.MCTS_ITER, cfg.get(DNetField.MCTS_ITER)),
                (
                    DLabel.MCTS_EXPLORE_P_VALUE,
                    cfg.get(DNetField.MCTS_EXPLORE_P_VALUE),
                ),
            ],
        )

    def _build_nice_section(self, cfg: SimCfg) -> list[str]:
        p_value = cfg.get(DNetField.NICE_P_VALUE)
        nice_steps = cfg.get(DNetField.NICE_STEPS)
        return self._build_kv_section(
            "🙂 Epsilon Nice",
            [("P-Value", p_value), ("Nice Steps", nice_steps)],
        )

    def _build_nice_table(self) -> list[str]:
        rows = self.metrics.get_epsilon_nice_events()
        headers = [
            "Window",
            "Epoch",
            "Calls",
            "Triggered",
            "Fatal Suggested",
            "Overrides",
            "No Safe Alt",
            "Trigger Rate",
            "Override Rate",
        ]
        table_rows: list[list[str]] = []

        for (
            window,
            epoch,
            calls,
            triggered,
            fatal_suggested,
            overrides,
            no_safe_alternatives,
            trigger_rate,
            override_rate,
        ) in rows:
            table_rows.append(
                [
                    window,
                    str(epoch),
                    str(calls),
                    str(triggered),
                    str(fatal_suggested),
                    str(overrides),
                    str(no_safe_alternatives),
                    f"{trigger_rate:.4f}",
                    f"{override_rate:.4f}",
                ]
            )

        return self._build_table_section(
            "🙂 Epsilon Nice Events", headers, table_rows
        )

    def _build_event_log_section(self) -> list[str]:
        rows = self.metrics.get_eventlog_msgs()
        lines = self._section_header("📚 Event Log Messages")

        if not rows:
            lines.append("(none)")
            lines.append("")
            return lines

        lines.extend(str(row) for row in rows)
        lines.append("")
        return lines

    def _build_highscore_section(self) -> list[str]:
        rows = self.metrics.get_highscore_snapshot_rows()
        headers = ["Epoch", "Highscore", "Time", "Epsilon"]
        table_rows: list[list[str]] = []

        for epoch, highscore, epsilon, ev_time in rows:
            try:
                print(f"Epsilon {epsilon}, type: {type(epsilon)}")
                epsilon_str = "" if epsilon is None else f"{epsilon:.4f}"
                time_str = "" if ev_time is None else str(ev_time)
                table_rows.append(
                    [
                        str(epoch),
                        str(highscore),
                        time_str,
                        epsilon_str,
                    ]
                )
            except Exception as e:
                print(f"ERROR: {e}")
                print(f"STACKTRACE: {traceback.format_exc()}")

        return self._build_table_section(
            "🏆 Highscore Events", headers, table_rows
        )

    def _build_shift_mean_median_section(self) -> list[str]:
        rows = self.metrics.get_shift_mean_median_snapshot_rows()
        headers = [
            "Epoch",
            "Gear",
            "Seq Length",
            "Batch Size",
            "Mean",
            "Median",
            "Recent Mean",
            "Recent Median",
        ]
        table_rows: list[list[str]] = []

        for (
            epoch,
            gear,
            seq_length,
            batch_size,
            mean_score,
            median_score,
            recent_mean_score,
            recent_median_score,
        ) in rows:
            table_rows.append(
                [
                    str(epoch),
                    str(gear),
                    str(seq_length),
                    str(batch_size),
                    f"{mean_score:.2f}",
                    f"{median_score:.2f}",
                    f"{recent_mean_score:.2f}",
                    f"{recent_median_score:.2f}",
                ]
            )

        return self._build_table_section(
            "⚙️  ATH Shift / Mean / Median",
            headers,
            table_rows,
        )

    def _build_bucket_table(self) -> list[str]:
        rows = self.metrics.get_bucket_snapshot_rows()
        table_rows: list[list[str]] = []

        for row in rows:
            table_rows.append([str(value) for value in row])

        if table_rows:
            bucket_count = len(table_rows[0]) - 1
            headers = ["Epoch", "Gear"] + [
                f"b{i}" for i in range(2, bucket_count + 1)
            ]
        else:
            headers = ["Epoch", "Gear"]

        return self._build_table_section(
            "🪣 ATH Memory Bucket Usage",
            headers,
            table_rows,
        )

    def _build_mcts_bucket_table(self) -> list[str]:
        rows = self.metrics.get_mcts_bucket_snapshot_rows()
        table_rows: list[list[str]] = []

        for row in rows:
            table_rows.append([str(value) for value in row])

        if table_rows:
            bucket_count = len(table_rows[0]) - 1
            headers = ["Epoch", "Gear"] + [
                f"b{i}" for i in range(2, bucket_count + 1)
            ]
        else:
            headers = ["Epoch", "Gear"]

        return self._build_table_section(
            "🪣 Monte-Carlo ATH Memory Bucket Usage",
            headers,
            table_rows,
        )

    def _build_kv_section(
        self,
        title: str,
        items: list[tuple[str, str]],
    ) -> list[str]:
        lines = self._section_header(title)
        max_key = max(len(key) for key, _ in items) if items else 0

        for key, value in items:
            lines.append(f"{key:<{max_key}} : {value}")

        lines.append("")
        return lines

    def _build_table_section(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> list[str]:
        lines = self._section_header(title)

        if not rows:
            lines.append("(none)")
            lines.append("")
            return lines

        lines.extend(self._format_table(headers, rows))
        lines.append("")
        return lines

    def _section_header(self, title: str) -> list[str]:
        return [title, "═" * len(title)]

    def _format_table(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> list[str]:

        for row in rows:
            if len(row) != len(headers):
                raise ValueError(
                    f"Table row/header mismatch: len(row)={len(row)} "
                    f"len(headers)={len(headers)} row={row}"
                )

        widths = [len(header) for header in headers]

        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        header_line = "  ".join(
            header.ljust(widths[i]) for i, header in enumerate(headers)
        )
        divider_line = "  ".join("═" * widths[i] for i in range(len(headers)))

        body_lines = [
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in rows
        ]

        return [header_line, divider_line, *body_lines]
