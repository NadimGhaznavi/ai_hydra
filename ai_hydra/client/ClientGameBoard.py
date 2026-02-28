# ai_hydra/client/ClientGameBoard.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from textual.scroll_view import ScrollView
from textual.geometry import Offset, Region, Size
from textual.strip import Strip
from textual.reactive import var

from rich.segment import Segment
from rich.style import Style

from ai_hydra.constants.DGame import DGameField

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

    # Reactive state (screen coords: Offset(x,y))
    food = var(Offset(9, 9))
    snake_head = var(Offset())
    snake_body = var([])
    direction = var(Offset(1, 0))

    def __init__(self, board_size: int, id=None) -> None:
        super().__init__(id=id)
        self._board_size = int(board_size)
        self.virtual_size = Size(board_size * 2, board_size)

    def board_size(self) -> int:
        return self._board_size

    # Updates
    # --------
    def apply_snapshot(self, snapshot: dict) -> None:
        """
        Apply a Hydra SnakeMgr snapshot payload

        Expects a snapshot like:
            {
                "board": {
                    "snake_head": {"x":..,"y":..},
                    "snake_body": [{"x":..,"y":..}, ...]
                    "food_position": {"x":..,"y":..},
                    "direction": {"dx":..,"dy":..},
                    "grid_size": {"w",.., "h":..},
                    ...
                }
            }
        """
        board = snapshot.get(DGameField.BOARD, {})
        if not isinstance(board, dict):
            raise TypeError(f"ERROR: Unrecognized board: {board}")

        grid = board.get(DGameField.GRID_SIZE, {})
        if isinstance(grid, dict):
            w = grid.get(DGameField.W)
            h = grid.get(DGameField.H)

        head = board.get(DGameField.SNAKE_HEAD, {})
        if isinstance(head, dict):
            self.snake_head = Offset(
                int(head[DGameField.X]), int(head[DGameField.Y])
            )

        body = board.get(DGameField.SNAKE_BODY, [])
        if isinstance(body, list):
            self.snake_body = [
                Offset(int(seg[DGameField.X]), int(seg[DGameField.Y]))
                for seg in body
                if isinstance(seg, dict)
            ]

        food = board.get(DGameField.FOOD_POSITION, {})
        if isinstance(food, dict):
            self.food = Offset(
                int(food[DGameField.X]), int(food[DGameField.Y])
            )

        d = board.get(DGameField.DIRECTION, {})
        if isinstance(d, dict):
            self.direction = Offset(
                int(d[DGameField.DX]), int(d[DGameField.DY])
            )

    # Rendering
    # ----------
    def get_square_region(self, square_offset: Offset) -> Region:
        """Get region relative to widget from square coordinate."""
        x, y = square_offset
        region = Region(x * 2, y, 2, 1)
        # Move the region into the widgets frame of reference
        return region.translate(-self.scroll_offset)

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        y += scroll_y
        row_index = y

        emptyA = self.get_component_rich_style(
            "clientgameboard--emptyA-square"
        )
        emptyB = self.get_component_rich_style(
            "clientgameboard--emptyB-square"
        )
        food = self.get_component_rich_style("clientgameboard--food-square")
        snake = self.get_component_rich_style("clientgameboard--snake-square")
        snake_head = self.get_component_rich_style(
            "clientgameboard--snake-head-square"
        )

        if row_index >= self._board_size:
            return Strip.blank(self.size.width)

        is_odd = row_index % 2

        def get_square_style(column: int, row: int) -> Style:
            pos = Offset(column, row)
            if self.food == pos:
                return food
            if self.snake_head == pos:
                return snake_head
            if pos in self.snake_body:
                return snake
            return emptyA if (column + is_odd) % 2 else emptyB

        segments = [
            Segment(" " * 2, get_square_style(column, row_index))
            for column in range(self._board_size)
        ]
        strip = Strip(segments, self._board_size * 2)
        # Crop the strip so that is covers the visible area
        return strip.crop(scroll_x, scroll_x + self.size.width)

    # Reactive watchers
    # ------------------
    def watch_food(self, previous_food, food) -> None:
        """Called when the food square changes."""
        # Refresh the previous food square
        self.refresh(self.get_square_region(previous_food))
        # Refresh the new food square
        self.refresh(self.get_square_region(food))

    def watch_snake_head(
        self, previous_snake_head: Offset, snake_head: Offset
    ) -> None:
        """Called when the snake head square changes."""
        self.refresh(self.get_square_region(previous_snake_head))
        self.refresh(self.get_square_region(snake_head))

    def watch_snake_body(
        self, previous_snake_body: list, snake_body: list
    ) -> None:
        """Called when the snake body changes."""
        for segment in previous_snake_body:
            self.refresh(self.get_square_region(segment))

        for segment in snake_body:
            self.refresh(self.get_square_region(segment))
