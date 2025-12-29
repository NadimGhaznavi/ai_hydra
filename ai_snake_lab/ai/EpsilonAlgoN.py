"""
ai_snake_lab/ai/EpsilonAlgoN.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0

This class provides a different approach to exploration in the context of the the
SnakeGame. Observation of the game score plot shows that the AI does not fully
master the games at low scores, even as it achieves higher scores. For example, even
after thousands of epochs it still regularly only gets a score of 1, 2 or 3.

This class provides an alternative mechanism to help the AI master the game at lower
scores.

Essentially, it maintains a dictionary of EpsilonAlgo instances, keyed by game score.
When the simulation (AISim) achieves a new highscore, it calls new_highscore() on this
class and this class then creates a new instance of EpsilonAlgo and adds it to the
dictionaary of EpsilonAlgo instances.

The get_move() method for this class requires a "score" parameter i.e. get_move(score).
This class then calls the get_move() method on the EpsilonAlgo instance associated
with the score.
"""

from ai_snake_lab.ai.EpsilonAlgo import EpsilonAlgo


class EpsilonAlgoN:

    def __init__(self, seed: int):
        self._seed = seed
        self._initial_epsilon = None
        self._epsilon_decay = None
        self._epsilon_min = None
        self._epsilons = {}
        self.add_epsilon(0)

    def add_epsilon(self, score):
        epsilon_algo = EpsilonAlgo(seed=self._seed)
        epsilon_algo.epsilon_min(self.epsilon_min())
        epsilon_algo.epsilon_decay(self.epsilon_decay())
        epsilon_algo.initial_epsilon(self.initial_epsilon())
        self._epsilons[score] = epsilon_algo

    def epsilon(self, score):
        return self._epsilons[score].epsilon()

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

    def get_move(self, score):
        return self._epsilons[score].get_move()

    def played_game(self, score):
        self._epsilons[score].played_game()

    def new_highscore(self, score):
        self.add_epsilon(score)
