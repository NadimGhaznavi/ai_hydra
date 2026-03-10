# ai_hydra/utils/HydraMetrics.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DHydraTui import DLabel, DField
from ai_hydra.constants.DNNet import DNetDef, DLinear, DRNN2


class HydraMetrics:

    def __init__(self):
        self._epsilon = {}
        self._highscore_events = []
        self._mean_median = []
        self._random_seed = DHydra.RANDOM_SEED
        self._linear_model = {}
        self._rnn_model = {}

    def add_epsilon(self, initial, min, decay):
        self._epsilon[DField.INITIAL_EPSILON] = initial
        self._epsilon[DField.MIN_EPSILON] = min
        self._epsilon[DField.EPSILON_DECAY] = decay

    def add_epsilon_depleted(self, episode):
        self._epsilon[DField.EPSILON_DEPLETED] = episode

    def add_highscore_event(self, episode, highscore, event_time, lookahead):
        self._highscore_events.append(
            (episode, highscore, event_time, lookahead)
        )

    def add_mean_median(self, episode, mean, median):
        self._mean_median.append((episode, mean, median))

    def add_linear_model(self):
        self._linear_model[DField.INPUT_SIZE] = DNetDef.INPUT_SIZE
        self._linear_model[DField.HIDDEN_SIZE] = DLinear.HIDDEN_SIZE
        self._linear_model[DField.DROPOUT_P] = DLinear.DROPOUT_P

    def add_rnn_model(self):
        self._rnn_model[DField.INPUT_SIZE] = DNetDef.INPUT_SIZE
        self._rnn_model[DField.HIDDEN_SIZE] = DRNN2.HIDDEN_SIZE
        self._rnn_model[DField.RNN_LAYERS] = DRNN2.RNN_LAYERS
        self._rnn_model[DField.RNN_DROPOUT] = DRNN2.P_VALUE

    def _init_headers(self):
        self.add_highscore_event(
            episode=DLabel.EPISODE,
            highscore=DLabel.HIGHSCORE,
            event_time=DLabel.EVENT_TIME,
            lookahead=DLabel.LOOKAHEAD,
        )
        self.add_mean_median(
            episode=DLabel.EPISODE, mean=DLabel.MEAN, median=DLabel.MEDIAN
        )
