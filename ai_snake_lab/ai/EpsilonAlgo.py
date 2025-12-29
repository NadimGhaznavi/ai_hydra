"""
ai_snake_lab/ai/EpsilonAlgo.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0

A class to encapsulate the functionality of the epsilon algorithm. The algorithm
injects random moves at the beginning of the simulation. The amount of moves
is controlled by the epsilon_value parameter which is in the AISnakeGame.ini and
can also be passed in when invoking the main asg.py front end.
"""

import random
from random import randint

from ai_snake_lab.constants.DEpsilon import DEpsilon


class EpsilonAlgo:

    def __init__(self, seed):
        # Set this random seed so things are repeatable
        random.seed(seed)
        self._initial_epsilon = DEpsilon.EPSILON_INITIAL
        self._epsilon_min = DEpsilon.EPSILON_MIN
        self._epsilon_decay = DEpsilon.EPSILON_DECAY
        self._epsilon = self._initial_epsilon
        self._injected = 0
        self._depleted = False

    def add_epsilon(self, score):
        pass

    def get_move(self, score=None):
        if random.random() < self._epsilon:
            rand_move = [0, 0, 0]
            rand_idx = randint(0, 2)
            rand_move[rand_idx] = 1
            self._injected += 1
            return rand_move
        return False

    def epsilon(self, score=None):
        return self._epsilon

    def epsilon_decay(self, epsilon_decay=None):
        if epsilon_decay is not None:
            self._epsilon_decay = epsilon_decay
        return self._epsilon_decay

    def epsilon_min(self, epsilon_min=None):
        if epsilon_min is not None:
            self._epsilon_min = epsilon_min
        return self._epsilon_min

    def initial_epsilon(self, initial_epsilon=None):
        if initial_epsilon is not None:
            self._initial_epsilon = initial_epsilon
        return self._initial_epsilon

    def injected(self):
        return self._injected

    def new_highscore(self, score=None):
        pass

    def played_game(self, score=None):
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        self.reset_injected()

    def reset_injected(self):
        self._injected = 0
