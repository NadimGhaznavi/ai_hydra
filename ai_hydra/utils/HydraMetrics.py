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
from ai_hydra.constants.DNNet import DNetDef, DLinear, DRNN, MODEL_TYPE_TABLE


class HydraMetrics:

    def __init__(self):
        self.epsilon = {}
        self.highscore_events = []
        self.linear_model = {}
        self.rnn_model = {}
        self.lookahead = None
        self.mean_median = []

    def add_epsilon(self, initial, minimum, decay):
        self.epsilon[DField.INITIAL_EPSILON] = initial
        self.epsilon[DField.MIN_EPSILON] = minimum
        self.epsilon[DField.EPSILON_DECAY] = decay

    def add_epsilon_depleted(self, episode):
        self.epsilon[DField.EPSILON_DEPLETED] = episode

    def add_highscore_event(
        self, episode, highscore, event_time, lookahead, cur_ep
    ):
        self.highscore_events.append(
            (episode, highscore, event_time, lookahead, cur_ep)
        )

    def add_lookahead(self, pvalue):
        self.lookahead = pvalue

    def add_mean_median(self, episode, mean, median):
        self.mean_median.append((episode, mean, median))

    def add_linear_model(self):
        self.linear_model[DField.INPUT_SIZE] = DNetDef.INPUT_SIZE
        self.linear_model[DField.HIDDEN_SIZE] = DLinear.HIDDEN_SIZE
        self.linear_model[DField.DROPOUT_P] = DLinear.DROPOUT_P

    def add_rnn_model(self):
        self.rnn_model[DField.INPUT_SIZE] = DNetDef.INPUT_SIZE
        self.rnn_model[DField.HIDDEN_SIZE] = DRNN.HIDDEN_SIZE
        self.rnn_model[DField.RNN_LAYERS] = DRNN.RNN_LAYERS
        self.rnn_model[DField.RNN_DROPOUT] = DRNN.P_VALUE

    def add_trainer(self):
        pass

    def create_snapshot(self, snap_file, model_type):
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
                f"AI Hydra Version: v{DHydra.VERSION}\n"
                f"Random Seed: {DHydra.RANDOM_SEED}\n"
                f"Look Ahead P-Value: {self.lookahead}\n\n"
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
                    f"Hidden Size: {self.linear_model[DField.HIDDEN_SIZE]}\n"
                    f"Dropout Layer P-Value: {self.linear_model[DField.DROPOUT_P]}\n\n"
                )
            else:
                f.write(
                    "🧠 RNN Model\n"
                    "════════════\n"
                    f"Input Size: {self.rnn_model[DField.INPUT_SIZE]}\n"
                    f"Hidden Size: {self.rnn_model[DField.HIDDEN_SIZE]}\n"
                    f"RNN Layers: {self.rnn_model[DField.RNN_LAYERS]}\n"
                    f"Dropout Layer P-Value: {self.rnn_model[DField.RNN_DROPOUT]}\n"
                    f"Batch Size: {DRNN.BATCH_SIZE}\n"
                    f"Sequence Length: {DRNN.SEQ_LENGTH}\n\n"
                )
            f.write(
                "🏆 Highscore Events\n"
                "═══════════════════\n"
                f"{'Episode':8s}{'Highscore':10s}{'Time':>11s}{'Look Ahead':>11}{'Epsilon':>8s}\n"
                "═══════ ═════════ ═══════════ ══════════ ═══════\n"
            )
            for event in self.highscore_events:
                episode, highscore, ev_time, lookahead, cur_ep = event
                cur_ep = str(round(float(cur_ep), 4))
                f.write(
                    f"{str(episode):>8s}{str(highscore):>10s}{ev_time:>11s}{str(lookahead):>11s}{cur_ep:>8s}\n"
                )
