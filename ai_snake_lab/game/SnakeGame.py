"""
ai_snake_lab/game/SnakeGame.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import time
import random
import numpy as np

from textual.geometry import Offset

from ai_snake_lab.game.GameBoard import GameBoard
from ai_snake_lab.game.GameElements import Direction

# Maximum number of moves. This is multiplied by the length of the snake. The game
# ends if game moves > MAX_MOVES * length-of-snake. This avoids enless AI looping behavior.
MAX_MOVES = 100


class SnakeGame:

    def __init__(self, game_board: GameBoard, id=None):
        # Make multiple runs predictable
        random.seed(1970)

        # Get an instance of the game board and it's dimensions
        self.game_board = game_board
        board_size = self.game_board.board_size()

        # Track the number of moves
        self.moves = 0

        # Track the reward within a given game
        self.game_reward = 0

        # Set the initial snake direction and position
        self.direction = Direction.RIGHT
        self.head = Offset(board_size // 2, board_size // 2)
        self.snake = [
            self.head,
            Offset(self.head.x - 1, self.head.y),
            Offset(self.head.x - 2, self.head.y),
        ]

        # Place a food in a random location (not occupied by the snake)
        self.food = self.place_food()

        # Update the game board
        self.game_board.update_snake(snake=self.snake, direction=self.direction)
        self.game_board.update_food(food=self.food)

        # Track the distance from the snake head to the food to feed the reward system
        self.distance_to_food = self.game_board.board_size() // 2

        # The current game score
        self.game_score = 0

    def get_direction(self):
        return self.direction

    def move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Mod 4 to avoid out of index error
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1] ... there are only 3 actions
            next_idx = (idx - 1) % 4  # Again, MOD 4 to avoid out of index error
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
        self.direction = new_dir

        old_head = self.head

        if self.direction == Direction.RIGHT:
            self.head = Offset(self.head.x + 1, self.head.y)
        elif self.direction == Direction.LEFT:
            self.head = Offset(self.head.x - 1, self.head.y)
        elif self.direction == Direction.DOWN:
            self.head = Offset(self.head.x, self.head.y + 1)
        elif self.direction == Direction.UP:
            self.head = Offset(self.head.x, self.head.y - 1)

        # self.snake.insert(0, self.head)

    def place_food(self):
        board_size = self.game_board.board_size()
        x = random.randint(0, board_size - 1)
        y = random.randint(0, board_size - 1)
        self.food = Offset(x, y)
        if self.food in self.snake:
            self.place_food()
        return self.food

    def play_step(self, action):
        ## 1. Move
        self.move(action)
        self.snake.insert(0, self.head)
        self.game_board.update_snake(snake=self.snake, direction=self.direction)
        snake_length = len(self.snake)
        max_moves = 100

        ## 2. Check if the game is over
        reward = 0
        game_over = False
        board_size = self.game_board.board_size()

        ## 3. Check for "game over" states

        # Wall collision
        if self.game_board.is_wall_collision(self.head):
            # Wall collision
            game_over = True
            reward = -10

        # Snake collision
        elif self.game_board.is_snake_collision(self.head):
            game_over = True
            reward = -10

        # Exceeded max moves
        if self.moves > max_moves * snake_length:
            game_over = True
            reward = -10

        if game_over == True:
            # Game is over: Snake or wall collision or exceeded max moves
            self.game_reward += reward
            return reward, game_over, self.game_score

        ## 4. Game is not over, lets see what else is going on

        if self.head == self.food:
            # We found food!!
            self.game_reward += 10
            self.place_food()
            self.game_score += 1

        else:
            self.snake.pop()

        ## 5. See if we're closer to the food than the last move, or further away
        cur_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        if cur_distance < self.distance_to_food:
            reward += 2
        elif cur_distance > self.distance_to_food:
            reward -= 2
        self.distance_to_food = cur_distance

        ## 6. Set a negative reward if the snake head is adjacent to the snake body.
        # This is to discourage snake collisions.
        # for segment in self.snake[1:]:
        #    if abs(self.head.x - segment.x) < 2 and abs(self.head.y - segment.y) < 2:
        #        reward -= -2

        self.game_reward += reward
        self.game_board.update_snake(snake=self.snake, direction=self.direction)
        self.game_board.update_food(food=self.food)
        self.moves += 1
        return reward, game_over, self.game_score, self.game_reward

    def reset(self):
        # Reset the game reward
        self.game_reward = 0

        # Get the board size
        board_size = self.game_board.board_size()

        # Reset the number of moves
        self.moves = 0

        # Set the initial snake direction and position
        self.direction = Direction.RIGHT
        self.head = Offset(board_size // 2, board_size // 2)
        self.snake = [
            self.head,
            Offset(self.head.x - 1, self.head.y),
            Offset(self.head.x - 2, self.head.y),
        ]

        # Place a food in a random location (not occupied by the snake)
        self.food = self.place_food()

        # Update the game board
        self.game_board.update_snake(snake=self.snake, direction=self.direction)
        self.game_board.update_food(food=self.food)

        # The current game score
        self.game_score = 0

        # Compute and return the initial state
        return self.get_state()

    def get_state(self):
        return self.game_board.get_state()
