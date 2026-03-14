# ai_hydra/utils/HydraMetrics.py
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


class HydraMetrics:

    def __init__(self):
        self.epsilon = {}
        self.highscore_events = []
        self.linear_model = {}
        self.rnn_model = {}
        self.mean_median = []
        self.elapsed_time = None

    def add_elapsed_time(self, elapsed_time):
        self.elapsed_time = elapsed_time

    def add_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def add_highscore_event(self, episode, highscore, event_time, cur_ep):
        self.highscore_events.append((episode, highscore, event_time, cur_ep))

    def add_mean_median(self, episode, mean, median):
        self.mean_median.append((episode, mean, median))

    def add_trainer(self):
        pass

    def create_snapshot(self, snap_file, cfg: SimCfg):

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        model_type = cfg.get(DNetField.MODEL_TYPE)
        model_hidden_size = cfg.get(DNetField.HIDDEN_SIZE)
        model_input_size = DNetDef.INPUT_SIZE

        initial_epsilon = cfg.get(DNetField.INITIAL_EPSILON)
        min_epsilon = cfg.get(DNetField.MIN_EPSILON)
        epsilon_delay = cfg.get(DNetField.EPSILON_DECAY)

        learning_rate = cfg.get(DNetField.LEARNING_RATE)
        dropout_p = cfg.get(DNetField.DROPOUT_P)
        rnn_layers = cfg.get(DNetField.RNN_LAYERS)
        rnn_tau = cfg.get(DNetField.RNN_TAU)

        with open(snap_file, "w") as f:
            f.write(
                "📸 AI Hydra - Snapshot\n"
                "══════════════════════\n"
                f"Timestamp: {timestamp}\n"
                f"Simulation Run Time: {self.elapsed_time}\n"
                f"Current Episode Number: {self.cur_epoch}\n"
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
                    f"Dropout Layer P-Value: {dropout_p}\n"
                    f"Learning Rate: {learning_rate}\n\n"
                )
            elif model_type == DField.RNN:
                f.write(
                    "🧠 RNN Model\n"
                    "════════════\n"
                    f"Input Size: {model_input_size}\n"
                    f"Hidden Size: {model_hidden_size}\n"
                    f"RNN Layers: {rnn_layers}\n"
                    f"Dropout Layer P-Value: {dropout_p}\n"
                    f"RNN Tau: {rnn_tau}\n"
                    f"Sequence Length: {DRNN.SEQ_LENGTH}\n"
                    f"Batch Size: {DRNN.BATCH_SIZE}\n"
                    f"Learning Rate: {learning_rate}\n\n"
                )
            f.write(
                "🏆 Highscore Events\n"
                "═══════════════════\n"
                f"{'Episode':8s}{'Highscore':<10s}{'Time':>11s}{'Epsilon':>8s}\n"
                "═══════ ═════════ ═══════════ ═══════\n"
            )
            for event in self.highscore_events:
                episode, highscore, ev_time, cur_ep = event
                cur_ep = str(round(float(cur_ep), 4))
                f.write(
                    f"{str(episode):>7s}{str(highscore):>10s}{ev_time:>12s}{cur_ep:>8s}\n"
                )
