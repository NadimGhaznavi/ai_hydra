"""
ai_snake_lab/hydra/SnakeGameEnv.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import gym
import numpy as np
from gym import spaces


from ai_snake_lab.ai.ReplayMemory import ReplayMemory

from ai_snake_lab.game.ServerGameBoard import ServerGameBoard
from ai_snake_lab.game.SnakeGame import SnakeGame

from ai_snake_lab.constants.DSim import DSim


import numpy as np
import gym
from gym import spaces
from ai_snake_lab.game import SnakeGame, ServerGameBoard


class SnakeGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "zmq_msg", "none"]}

    def __init__(self, board_size=15, render_mode="none"):
        super().__init__()
        self.render_mode = render_mode
        self.game_board = ServerGameBoard(board_size)
        self.snake_game = SnakeGame(game_board=self.game_board)

        init_state = np.array(self.game_board.get_state(), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=init_state.shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.state = init_state
        self.terminated = False
        self.truncated = False
        self.np_random = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.snake_game.reset()
        self.terminated = False
        self.truncated = False
        self.state = np.array(self.game_board.get_state(), dtype=np.float32)
        return self.state, {}

    def step(self, action):
        reward, done, score, game_reward = self.snake_game.play_step(action)
        self.state = np.array(self.game_board.get_state(), dtype=np.float32)

        # mbrl-lib expects separate flags
        self.terminated = done
        self.truncated = False  # can add a max-steps cutoff later

        info = {
            "score": score,
            "game_reward": game_reward,
        }
        return self.state, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "human":
            print(self.snake_game)  # or some textual printout
        return None

    def close(self):
        pass
