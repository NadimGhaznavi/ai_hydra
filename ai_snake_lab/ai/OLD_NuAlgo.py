"""
NuAlgo.py

This class implements the nu algorithm. The algorithm injects random moves
whenever the AI reaches the current high score. This encourages the AI to
continue exploring as the snake grows in length. It is an alternative to
the epsilon algorithm. Unlike the epsilon algorithm, the nu algorithm is
dynamic and continues to inject random moves based on the AI agent's
performance..

Defaults

  nu_enabled: True
  nu_bad_games: 5
  nu_high_grace: 4
  nu_max_epochs: 100
  nu_pool: 6
  nu_score: 0
  nu_verbose: False

"""

import random
from random import randint


class NuAlgo:
    def __init__(self, ini, log, stats):
        # Constructor
        self.ini = ini
        self.log = log
        self.stats = stats
        # Set this random seed so things are repeatable
        random.seed(ini.get("random_seed"))
        self.pool = ini.get(
            "nu_pool"
        )  # Size of the pool (like epsilon, but dynamic)
        self.bad_games = ini.get(
            "nu_bad_games"
        )  # How many games in a row where no high score is reached
        self.new_high_grace = ini.get(
            "nu_high_grace"
        )  # Number of games after a high score has been found where no random moves are injected
        self.enabled = ini.get(
            "nu_enabled"
        )  # Whether this algorithm is enabled
        self.score = ini.get(
            "nu_score"
        )  # The game score that triggers the nu algorithm
        self.verbose = ini.get(
            "nu_verbose"
        )  # Whether to print additional messages

        # Initialization
        self.reset_count = 0  # How many times the pool has been refilled without finding a high score
        self.injected = 0  # Number of random moves injected in a game
        self.injected_flag = (
            False  # Whether random move(s) have been injected in a game
        )
        self.cur_pool = self.pool  # Size of the random move pool
        self.bad_game_count = 0  # A 'number of bad games' counter
        self.new_high = False  # Whether a new high score has been found
        self.cur_high = (
            False  # Whether the current score game matched the high score
        )
        self.new_high_grace_count = 0  # Counter for self.new_high_grace

        if self.enabled == False:
            self.log.log("NuAlgo: NuAlgo is disabled")
            self.ini.set("print_stats", "False")
            self.verbose = False
        else:
            self.stats.set("nu", "status", "enabled")
            if self.verbose:
                self.log.log(
                    f"NuAlgo: New instance with pool size ({self.pool}), score ({self.score}) and bad games ({self.bad_games+1})"
                )

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled

    def get_move(self, cur_score):
        """
        Return False or a random move.
        """
        if not self.ini.get("nu_enabled"):
            # NuAlgo is disabled
            return False

        if cur_score < self.score:
            # Current game score too low to inject random moves
            return False

        # if self.new_high or self.cur_high:
        # A new high score has been found, give the AI new_high_grace games to find a new high score
        # before starting to inject random moves
        #  self.new_high_grace_count += 1
        #  if self.new_high_grace_count == self.new_high_grace:
        #    self.new_high = False
        #    self.cur_high = False
        #    self.new_high_grace_count = 0
        #  return False

        if self.cur_pool == 0:
            # Pool is empty
            return False

        return self.get_random_move()

    def get_pool(self):
        return self.cur_pool

    def get_random_move(self):
        self.injected_flag = True
        self.injected += 1  # Increment the number of injected move counter
        rand_move = [0, 0, 0]
        rand_idx = randint(0, 2)
        rand_move[rand_idx] = 1
        return rand_move

    def new_highscore(self, score):
        self.score = score  # Set a new NuAlgo score
        self.cur_pool = self.pool  # Refill the pool
        self.reset_count = 0  # Reset the 'number of times the pool was refilled without finding a high score' counter
        self.bad_game_count = 1  # Reset the 'number of bad games' counter
        if self.verbose:
            self.log.log(
                f"NuAlgo: New high score, increasing score to ({self.score}) and refilling pool to ({self.cur_pool})"
            )

    def played_game(self, cur_score):
        self.bad_game_count += 1

        if cur_score > self.score:
            self.score = cur_score
            self.new_high = True

        if self.injected_flag:
            self.injected_flag = False
            self.cur_pool -= 1  # Decrement the pool
            if self.cur_pool < 0:  # Make sure the pool is not negative
                self.cur_pool = 0

        if cur_score == self.score and self.reset_count > 1:
            self.reset_count = 1
            self.cur_high = True
            if self.verbose:
                self.log.log(
                    f"NuAlgo: Played a game matching the high score, resetting the reset count to (1)"
                )

        ## Bad game logic
        if self.bad_game_count == self.bad_games:
            self.bad_game_count = 0

            if self.reset_count == 1:
                self.score -= 1
                self.cur_pool = self.pool  # Refill the pool
            elif self.reset_count > 1:
                self.score = self.score - self.reset_count
                self.cur_pool = self.pool  # Refill the pool

            if self.score < 0:
                self.score = 0

            self.reset_count += 1
            if self.verbose:
                self.log.log(
                    f"NuAlgo: Played ({self.reset_count * self.bad_games}) games without a new high score, incrementing reset count to ({self.reset_count + 1}), decreasing score to ({self.score})"
                )

        # Reset injected count
        self.reset_injected()

    def reset_injected(self):
        self.injected = 0

    def update_status(self):
        # Update status
        status = "score {:>2}, injected {:>3}, pool {:>2}, bad games {:>2}, reset {:>2}".format(
            self.score,
            self.injected,
            self.cur_pool,
            self.bad_game_count,
            self.reset_count,
        )
        self.stats.set("nu", "status", status)
