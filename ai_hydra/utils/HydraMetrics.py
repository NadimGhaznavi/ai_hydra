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

    def add_epsilon(self, initial, minimum, decay):
        self.epsilon[DField.INITIAL_EPSILON] = initial
        self.epsilon[DField.MIN_EPSILON] = minimum
        self.epsilon[DField.EPSILON_DECAY] = decay

    def add_epsilon_depleted(self, episode):
        self.epsilon[DField.EPSILON_DEPLETED] = episode

    def add_highscore_event(self, episode, highscore, event_time, cur_ep):
        self.highscore_events.append((episode, highscore, event_time, cur_ep))

    def add_mean_median(self, episode, mean, median):
        self.mean_median.append((episode, mean, median))

    def add_linear_model(self):
        self.linear_model[DField.INPUT_SIZE] = DNetDef.INPUT_SIZE
        self.linear_model[DField.DROPOUT_P] = DLinear.DROPOUT_P

    def add_rnn_model(self):
        self.rnn_model[DField.INPUT_SIZE] = DNetDef.INPUT_SIZE
        self.rnn_model[DField.RNN_LAYERS] = DRNN.RNN_LAYERS
        self.rnn_model[DField.RNN_DROPOUT] = DRNN.DROPOUT_P_VALUE

    def add_trainer(self):
        pass

    def create_snapshot(self, snap_file, model_type, model_hidden_size):

        if model_type == DField.RNN:
            self.rnn_model[DNetField.HIDDEN_SIZE] = model_hidden_size
        elif model_type == DField.LINEAR:
            self.linear_model[DNetField.HIDDEN_SIZE] = model_hidden_size

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        if model_type == DField.LINEAR:
            self.add_linear_model()
        else:
            self.add_rnn_model()

        with open(snap_file, "w") as f:
            f.write(
                "📸 AI Hydra - Snapshot\n"
                "══════════════════════\n"
                f"Timestamp: {timestamp}\n"
                f"Simulation Run Time: {self.elapsed_time}\n"
                f"Episode Number: {self.cur_epoch}\n"
                f"AI Hydra Version: v{DHydra.VERSION}\n"
                f"Random Seed: {DHydra.RANDOM_SEED}\n\n"
            )

            f.write(
                "🎯 Epsilon Greedy\n"
                "═════════════════\n"
                f"Initial Epsilon: {self.epsilon[DField.INITIAL_EPSILON]}\n"
                f"Minimum Epsilon: {self.epsilon[DField.MIN_EPSILON]}\n"
                f"Epsilon Decay Rate: {self.epsilon[DField.EPSILON_DECAY]}\n\n"
            )
            if model_type == DField.LINEAR:
                f.write(
                    "🧠 Linear Model\n"
                    "═══════════════\n"
                    f"Input Size: {self.linear_model[DField.INPUT_SIZE]}\n"
                    f"Hidden Size: {self.linear_model[DNetField.HIDDEN_SIZE]}\n"
                    f"Dropout Layer P-Value: {self.linear_model[DField.DROPOUT_P]}\n\n"
                )
            elif model_type == DField.RNN:
                f.write(
                    "🧠 RNN Model\n"
                    "════════════\n"
                    f"Input Size: {self.rnn_model[DField.INPUT_SIZE]}\n"
                    f"Hidden Size: {self.rnn_model[DNetField.HIDDEN_SIZE]}\n"
                    f"RNN Layers: {self.rnn_model[DField.RNN_LAYERS]}\n"
                    f"Dropout Layer P-Value: {self.rnn_model[DField.RNN_DROPOUT]}\n"
                    f"Sequence Length: {DRNN.SEQ_LENGTH}\n"
                    f"Batch Size: {DRNN.BATCH_SIZE}\n\n"
                )
            f.write(
                "🏆 Highscore Events\n"
                "═══════════════════\n"
                f"{'Episode':8s}{'Highscore':10s}{'Time':>11s}{'Epsilon':>8s}\n"
                "═══════ ═════════ ═══════════ ══════════ ═══════\n"
            )
            for event in self.highscore_events:
                episode, highscore, ev_time, cur_ep = event
                cur_ep = str(round(float(cur_ep), 4))
                f.write(
                    f"{str(episode):>8s}{str(highscore):>10s}{ev_time:>11s}{cur_ep:>8s}\n"
                )
