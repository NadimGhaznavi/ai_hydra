# ai_hydra/utils/HydraSnapshot.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from datetime import datetime

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DHydraTui import DLabel, DField
from ai_hydra.constants.DNNet import (
    DNetDef,
    DLinear,
    DRNN,
    DNetField,
    MODEL_TYPE_TABLE,
)

from ai_hydra.utils.SimCfg import SimCfg
from ai_hydra.utils.HydraMetrics import HydraMetrics


class HydraSnapshot:

    def __init__(self, metrics: HydraMetrics):
        self.metrics = metrics

    def create_snapshot(self, snap_file, cfg: SimCfg):

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Model info
        model_type = cfg.get(DNetField.MODEL_TYPE)
        model_hidden_size = cfg.get(DNetField.HIDDEN_SIZE)
        model_input_size = DNetDef.INPUT_SIZE
        dropout_p = cfg.get(DNetField.DROPOUT_P)
        rnn_layers = cfg.get(DNetField.RNN_LAYERS)
        # Epsilon info
        initial_epsilon = cfg.get(DNetField.INITIAL_EPSILON)
        min_epsilon = cfg.get(DNetField.MIN_EPSILON)
        epsilon_delay = cfg.get(DNetField.EPSILON_DECAY)
        # Training info
        learning_rate = cfg.get(DNetField.LEARNING_RATE)
        gamma = cfg.get(DNetField.GAMMA)
        batch_size = cfg.get(DNetField.BATCH_SIZE)
        seq_length = cfg.get(DNetField.SEQ_LENGTH)
        rnn_tau = cfg.get(DNetField.RNN_TAU)

        # Metrics data
        elapsed_time = self.metrics.get_elapsed_time()
        cur_epoch = self.metrics.get_cur_epoch()

        with open(snap_file, "w") as f:
            f.write(
                "📸 AI Hydra - Snapshot\n"
                "══════════════════════\n"
                f"Timestamp: {timestamp}\n"
                f"Simulation Run Time: {elapsed_time}\n"
                f"Current Episode Number: {cur_epoch}\n"
                f"AI Hydra Version: v{DHydra.VERSION}\n"
                f"Random Seed: {DHydra.RANDOM_SEED}\n\n"
            )

            f.write(
                "🎯 Epsilon Greedy\n"
                "═════════════════\n"
                f"Initial Epsilon: {initial_epsilon}\n"
                f"Minimum Epsilon: {min_epsilon}\n"
                f"Epsilon Decay Rate: {epsilon_delay}\n\n"
            )
            if model_type == DField.LINEAR:
                f.write(
                    "🧠 Linear Model\n"
                    "═══════════════\n"
                    f"Input Size: {model_input_size}\n"
                    f"Hidden Size: {model_hidden_size}\n"
                    f"Dropout Layer P-Value: {dropout_p}\n\n"
                    "🧙 Training\n"
                    "═══════════\n"
                    f"Learning Rate: {learning_rate}\n"
                    f"Discount/Gamma: {gamma}\n"
                    f"Batch Size: {batch_size}\n\n"
                )
            elif model_type == DField.RNN:
                f.write(
                    "🧠 RNN Model\n"
                    "════════════\n"
                    f"Input Size: {model_input_size}\n"
                    f"Hidden Size: {model_hidden_size}\n"
                    f"Dropout Layer P-Value: {dropout_p}\n"
                    f"RNN Layers: {rnn_layers}\n"
                    f"RNN Tau: {rnn_tau}\n\n"
                    "🧙 Training\n"
                    "═══════════\n"
                    f"Learning Rate: {learning_rate}\n"
                    f"Discount/Gamma: {gamma}\n"
                    f"Batch Size: {batch_size}\n"
                    f"Sequence Length: {seq_length}\n"
                    f"RNN Tau: {rnn_tau}\n\n"
                )

            f.write("📚 Event Log Messages\n" "═════════════════════\n")
            rows = self.metrics.get_eventlog_msgs()
            for row in rows:
                f.write(f"{row}\n")

            f.write(
                "\n🏆 Highscore Events\n"
                "═══════════════════\n"
                f"{'Episode':8s}{'Highscore':<10s}{'Time':>11s}{'Epsilon':>8s}\n"
                "═══════ ═════════ ═══════════ ═══════\n"
            )
            rows = self.metrics.get_highscore_snapshot_rows()
            for row in rows:
                epoch, highscore, epsilon, ev_time = row
                cur_ep = str(round(float(epsilon), 4))

                f.write(
                    f"{str(epoch):>7s}{str(highscore):>10s}{ev_time:>12s}{cur_ep:>8s}\n"
                )
