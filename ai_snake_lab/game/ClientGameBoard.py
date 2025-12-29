"""
ai_snake_lab/game/ClientGameBoard.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import numpy as np

from textual.scroll_view import ScrollView
from textual.geometry import Offset, Region, Size
from textual.strip import Strip
from textual.reactive import var

from rich.segment import Segment
from rich.style import Style

from ai_snake_lab.game.GameElements import Direction

emptyA = "#111111"
emptyB = "#000000"
food = "#940101"
snake = "#025b02"
snake_head = "#16e116"


class ClientGameBoard(ScrollView):
    COMPONENT_CLASSES = {
        "clientgameboard--emptyA-square",
        "clientgameboard--emptyB-square",
        "clientgameboard--food-square",
        "clientgameboard--snake-square",
        "clientgameboard--snake-head-square",
    }

    DEFAULT_CSS = """
    ClientGameBoard > .clientgameboard--emptyA-square {
        background: #111111;
    }
    ClientGameBoard > .clientgameboard--emptyB-square {
        background: #000000;
    }
    ClientGameBoard > .clientgameboard--food-square {
        background: #940101;
    }
    ClientGameBoard > .clientgameboard--snake-square {
        background: #025b02;
    }
    ClientGameBoard > .clientgameboard--snake-head-square {
        background: #0ca30c;
    }
    """

    food = var(Offset(9, 9))
    snake_head = var(Offset())
    snake_body = var([])
    direction = Direction.RIGHT
    last_dirs = [0, 0, 1, 0]

    def __init__(self, board_size: int, id=None) -> None:
        super().__init__(id=id)
        self._board_size = board_size
        self.virtual_size = Size(board_size * 2, board_size)

    def board_size(self) -> int:
        return self._board_size

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        y += scroll_y
        row_index = y

        emptyA = self.get_component_rich_style("clientgameboard--emptyA-square")
        emptyB = self.get_component_rich_style("clientgameboard--emptyB-square")
        food = self.get_component_rich_style("clientgameboard--food-square")
        snake = self.get_component_rich_style("clientgameboard--snake-square")
        snake_head = self.get_component_rich_style("clientgameboard--snake-head-square")

        if row_index >= self._board_size:
            return Strip.blank(self.size.width)

        is_odd = row_index % 2

        def get_square_style(column: int, row: int) -> Style:
            if self.food == Offset(column, row):
                square_style = food
            elif self.snake_head == Offset(column, row):
                square_style = snake_head
            elif Offset(column, row) in self.snake_body:
                square_style = snake
            else:
                square_style = emptyA if (column + is_odd) % 2 else emptyB
            return square_style

        segments = [
            Segment(" " * 2, get_square_style(column, row_index))
            for column in range(self._board_size)
        ]
        strip = Strip(segments, self._board_size * 2)
        # Crop the strip so that is covers the visible area
        strip = strip.crop(scroll_x, scroll_x + self.size.width)
        return strip

    def watch_food(self, previous_food, food) -> None:
        """Called when the food square changes."""

        # Refresh the previous food square
        self.refresh(self.get_square_region(previous_food))

        # Refresh the new food square
        self.refresh(self.get_square_region(food))

    def watch_snake_head(self, previous_snake_head: Offset, snake_head: Offset) -> None:
        """Called when the snake head square changes."""
        self.refresh(self.get_square_region(previous_snake_head))
        self.refresh(self.get_square_region(snake_head))

    def watch_snake_body(self, previous_snake_body: list, snake_body: list) -> None:
        """Called when the snake body changes."""
        for segment in previous_snake_body:
            self.refresh(self.get_square_region(segment))

        for segment in snake_body:
            self.refresh(self.get_square_region(segment))

    def update_food(self, food_msg) -> None:
        self.food = Offset(food_msg[1][0], food_msg[1][1])
        self.refresh()

    def update_snake(self, snake_msg) -> None:
        snake_body = []
        for segment in snake_msg[1]:
            snake_body.append(Offset(segment[0], segment[1]))
        self.snake_body = snake_body
        self.refresh()

    def update_snake_head(self, snake_head_msg) -> None:
        self.snake_head = Offset(snake_head_msg[1][0], snake_head_msg[1][1])
        self.refresh()

    def get_square_region(self, square_offset: Offset) -> Region:
        """Get region relative to widget from square coordinate."""
        x, y = square_offset
        region = Region(x * 2, y, 2, 1)
        # Move the region into the widgets frame of reference
        region = region.translate(-self.scroll_offset)
        return region
