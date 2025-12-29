from dataclasses import dataclass


from ai_snake_lab.game.GameElements import Direction
from ai_snake_lab.game.ServerGameBoard import ServerGameBoard


from ai_snake_lab.game.SnakeGame import SnakeGame

from ai_snake_lab.constants.DSim import DSim


class ReactivePolicies:

    def __init__(self, game: SnakeGame, board: ServerGameBoard):
        import random

        self.game = game
        self.board = board
        random.seed(DSim.RANDOM_SEED)
        self._move = None
        self._game_reward = 0
        self._policies = []

    def __repr__(self):
        return f"Policies:"

    def process_move(self, move, reward, game_over, score):
        """
        Returns a move vector [straight, right, left] that is guaranteed safe if possible.
        Prefers the network’s suggested move, but falls back to any safe move if necessary.
        """
        self._move = move

        if game_over:
            print(
                f"RP: {str(move):<10s}: reward:{str(reward):<4} score:{str(score):<2}: GAME OVER"
            )  # Reward:{self._game_reward:<4} Score:{score:<2}")
            self._game_reward = 0
        else:
            self._game_reward += reward
        return reward, game_over, score

    def next_position(self, move, direction, snake_head):
        from textual.geometry import Offset

        DIR_MAP = {
            Direction.UP: Offset(0, -1),
            Direction.DOWN: Offset(0, 1),
            Direction.LEFT: Offset(-1, 0),
            Direction.RIGHT: Offset(1, 0),
        }

        if direction not in DIR_MAP:
            raise ValueError(f"Invalid direction: {direction}")

        delta = DIR_MAP[direction]
        dx, dy = delta.x, delta.y

        if move == 0:  # straight
            return Offset(snake_head.x + dx, snake_head.y + dy)
        elif move == 1:  # right turn
            return Offset(snake_head.x + dy, snake_head.y - dx)
        elif move == 2:  # left turn
            return Offset(snake_head.x - dy, snake_head.y + dx)
        else:
            raise ValueError(f"Invalid move index: {move}. Must be 0, 1, or 2.")
