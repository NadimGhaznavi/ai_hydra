"""
ai_snake_lab/game/ServerGameBoard.py

    AI Snake Game Simulator
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import numpy as np

from textual.geometry import Offset, Size
from ai_snake_lab.game.GameElements import Direction

from ai_snake_lab.constants.DMQ import DMQ


class ServerGameBoard:

    food = Offset(0, 0)
    snake_head = Offset(0, 0)
    snake_body = []
    direction = Direction.RIGHT
    last_dirs = [0, 0, 1, 0]

    def __init__(self, board_size: int) -> None:
        self._board_size = board_size
        self.virtual_size = Size(board_size * 2, board_size)

    def board_size(self) -> int:
        return self._board_size

    def get_binary(self, bits_needed, some_int):
        # This is used in the state map, the get_state() function.
        some_int = int(some_int)
        bin_str = format(some_int, "b")
        out_list = []
        for bit in range(len(bin_str)):
            out_list.append(bin_str[bit])
        for zero in range(bits_needed - len(out_list)):
            out_list.insert(0, "0")
        for x in range(bits_needed):
            out_list[x] = int(out_list[x])
        return out_list

    def get_state(self):
        head = self.snake_head
        direction = self.direction

        # Adjacent points
        point_l = Offset(head.x - 1, head.y)
        point_r = Offset(head.x + 1, head.y)
        point_u = Offset(head.x, head.y - 1)
        point_d = Offset(head.x, head.y + 1)

        # Direction flags
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        # Length encoded in 7-bit binary
        slb = self.get_binary(7, len(self.snake_body))

        # Normalized distances to walls (0=touching, 1=center)
        width = height = self.board_size()
        dist_left = head.x / width
        dist_right = (width - head.x - 1) / width
        dist_up = head.y / height
        dist_down = (height - head.y - 1) / height

        # Relative food direction (normalized)
        dx = self.food.x - head.x
        dy = self.food.y - head.y
        food_dx = dx / max(1, width)
        food_dy = dy / max(1, height)

        # Free space straight ahead
        free_ahead = 0
        probe = Offset(head.x, head.y)
        while (
            0 <= probe.x < width
            and 0 <= probe.y < height
            and not self.is_snake_collision(probe)
        ):
            free_ahead += 1
            if dir_r:
                probe = Offset(probe.x + 1, probe.y)
            elif dir_l:
                probe = Offset(probe.x - 1, probe.y)
            elif dir_u:
                probe = Offset(probe.x, probe.y - 1)
            elif dir_d:
                probe = Offset(probe.x, probe.y + 1)
        free_ahead = free_ahead / max(width, height)  # normalize

        # Local free cell count (0–4)
        adjacent_points = [point_l, point_r, point_u, point_d]
        local_free = (
            sum(
                1
                for p in adjacent_points
                if not self.is_wall_collision(p) and not self.is_snake_collision(p)
            )
            / 4.0
        )

        state = [
            # 1-3. Snake collision directions
            (dir_r and self.is_snake_collision(point_r))
            or (dir_l and self.is_snake_collision(point_l))
            or (dir_u and self.is_snake_collision(point_u))
            or (dir_d and self.is_snake_collision(point_d)),
            (dir_u and self.is_snake_collision(point_r))
            or (dir_d and self.is_snake_collision(point_l))
            or (dir_l and self.is_snake_collision(point_u))
            or (dir_r and self.is_snake_collision(point_d)),
            (dir_d and self.is_snake_collision(point_r))
            or (dir_u and self.is_snake_collision(point_l))
            or (dir_r and self.is_snake_collision(point_u))
            or (dir_l and self.is_snake_collision(point_d)),
            # 4-6. Wall collision directions
            (dir_r and self.is_wall_collision(point_r))
            or (dir_l and self.is_wall_collision(point_l))
            or (dir_u and self.is_wall_collision(point_u))
            or (dir_d and self.is_wall_collision(point_d)),
            (dir_u and self.is_wall_collision(point_r))
            or (dir_d and self.is_wall_collision(point_l))
            or (dir_l and self.is_wall_collision(point_u))
            or (dir_r and self.is_wall_collision(point_d)),
            (dir_d and self.is_wall_collision(point_r))
            or (dir_u and self.is_wall_collision(point_l))
            or (dir_r and self.is_wall_collision(point_u))
            or (dir_l and self.is_wall_collision(point_d)),
            # 7-10. Direction flags
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # 11-14. Food relative direction
            food_dx,
            food_dy,
            # 15-21. Snake length bits
            *slb,
            # 22-26. Distances
            # dist_left,
            # dist_right,
            # dist_up,
            # dist_down,
            # free_ahead,
            # local_free,
            # recent_growth,
            # time_since_food,
        ]

        return [float(x) for x in state]

    def is_snake_collision(self, pt: Offset) -> bool:
        if pt in self.snake_body:
            return True
        return False

    def is_wall_collision(self, pt: Offset) -> bool:
        if pt.x >= self._board_size or pt.x < 0 or pt.y >= self._board_size or pt.y < 0:
            return True
        return False

    def location_data(self):
        return {
            DMQ.SNAKE_HEAD: (DMQ.SNAKE_HEAD, self.snake_head),
            DMQ.SNAKE_BODY: (DMQ.SNAKE_BODY, self.snake_body),
            DMQ.FOOD: (DMQ.FOOD, self.food),
        }

    def update_food(self, food: Offset) -> None:
        self.food = food

    def update_snake(self, snake: list[Offset], direction: Direction) -> None:
        self.direction = direction
        self.snake_head = snake[0]
        self.snake_body = snake[1:]
