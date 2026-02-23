"""
AI Hydra Game Board Widget

Visual representation of the Snake game state using Textual.
Adapted from ai_snake_lab ClientGameBoard with AI Hydra integration.
"""

from typing import Dict, Any, List, Tuple, Optional

from textual.geometry import Offset, Region, Size
from textual.reactive import var
from textual.scroll_view import ScrollView
from textual.strip import Strip
from rich.segment import Segment
from rich.style import Style


class HydraGameBoard(ScrollView):
    """Real-time game board visualization for AI Hydra."""
    
    COMPONENT_CLASSES = {
        "hydragameboard--empty-a-square",
        "hydragameboard--empty-b-square", 
        "hydragameboard--food-square",
        "hydragameboard--snake-square",
        "hydragameboard--snake-head-square",
    }
    
    DEFAULT_CSS = """
    HydraGameBoard > .hydragameboard--empty-a-square {
        background: #111111;
    }
    HydraGameBoard > .hydragameboard--empty-b-square {
        background: #000000;
    }
    HydraGameBoard > .hydragameboard--food-square {
        background: #BF616A;
    }
    HydraGameBoard > .hydragameboard--snake-square {
        background: #A3BE8C;
    }
    HydraGameBoard > .hydragameboard--snake-head-square {
        background: #88C0D0;
    }
    """
    
    # Reactive properties for game state
    snake_head = var(Offset(10, 10))
    snake_body = var([])
    food_position = var(Offset(5, 5))
    grid_size = var((20, 20))
    
    def __init__(self, board_size: Tuple[int, int] = (20, 20), **kwargs):
        super().__init__(**kwargs)
        self.board_size = board_size
        self.grid_size = board_size
        
        # Set virtual size (width * 2 for double-width characters)
        self.virtual_size = Size(board_size[0] * 2, board_size[1])
        
        # Initialize default positions
        center_x, center_y = board_size[0] // 2, board_size[1] // 2
        self.snake_head = Offset(center_x, center_y)
        self.snake_body = [
            Offset(center_x - 1, center_y),
            Offset(center_x - 2, center_y)
        ]
        self.food_position = Offset(center_x + 5, center_y + 3)
    
    def render_line(self, y: int) -> Strip:
        """Render a single line of the game board."""
        scroll_x, scroll_y = self.scroll_offset
        y += scroll_y
        row_index = y
        
        # Get component styles
        empty_a = self.get_component_rich_style("hydragameboard--empty-a-square")
        empty_b = self.get_component_rich_style("hydragameboard--empty-b-square")
        food = self.get_component_rich_style("hydragameboard--food-square")
        snake = self.get_component_rich_style("hydragameboard--snake-square")
        snake_head = self.get_component_rich_style("hydragameboard--snake-head-square")
        
        # Return blank strip if outside board bounds
        if row_index >= self.board_size[1]:
            return Strip.blank(self.size.width)
        
        # Checkerboard pattern
        is_odd = row_index % 2
        
        def get_square_style(column: int, row: int) -> Style:
            """Get the style for a specific square."""
            pos = Offset(column, row)
            
            if pos == self.food_position:
                return food
            elif pos == self.snake_head:
                return snake_head
            elif pos in self.snake_body:
                return snake
            else:
                # Checkerboard pattern for empty squares
                return empty_a if (column + is_odd) % 2 else empty_b
        
        # Create segments for the row (double-width characters)
        segments = [
            Segment(" " * 2, get_square_style(column, row_index))
            for column in range(self.board_size[0])
        ]
        
        strip = Strip(segments, self.board_size[0] * 2)
        
        # Crop to visible area
        strip = strip.crop(scroll_x, scroll_x + self.size.width)
        return strip
    
    async def update_game_state(self, game_state: Dict[str, Any]) -> None:
        """Update the game board with new state from server."""
        try:
            # Update snake head
            if "snake_head" in game_state:
                head_pos = game_state["snake_head"]
                if isinstance(head_pos, (list, tuple)) and len(head_pos) >= 2:
                    self.snake_head = Offset(head_pos[0], head_pos[1])
            
            # Update snake body
            if "snake_body" in game_state:
                body_positions = []
                for pos in game_state["snake_body"]:
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        body_positions.append(Offset(pos[0], pos[1]))
                self.snake_body = body_positions
            
            # Update food position
            if "food_position" in game_state:
                food_pos = game_state["food_position"]
                if isinstance(food_pos, (list, tuple)) and len(food_pos) >= 2:
                    self.food_position = Offset(food_pos[0], food_pos[1])
            
            # Update grid size if changed
            if "grid_size" in game_state:
                grid_size = game_state["grid_size"]
                if isinstance(grid_size, (list, tuple)) and len(grid_size) >= 2:
                    new_size = (grid_size[0], grid_size[1])
                    if new_size != self.board_size:
                        self.board_size = new_size
                        self.grid_size = new_size
                        self.virtual_size = Size(new_size[0] * 2, new_size[1])
            
            # Refresh the display
            self.refresh()
            
        except Exception as e:
            # Log error but don't crash
            pass
    
    async def reset(self) -> None:
        """Reset the game board to initial state."""
        center_x, center_y = self.board_size[0] // 2, self.board_size[1] // 2
        
        self.snake_head = Offset(center_x, center_y)
        self.snake_body = [
            Offset(center_x - 1, center_y),
            Offset(center_x - 2, center_y)
        ]
        self.food_position = Offset(center_x + 5, center_y + 3)
        
        self.refresh()
    
    # Reactive watchers for smooth updates
    def watch_snake_head(self, old_head: Offset, new_head: Offset) -> None:
        """React to snake head position changes."""
        self.refresh(self.get_square_region(old_head))
        self.refresh(self.get_square_region(new_head))
    
    def watch_snake_body(self, old_body: List[Offset], new_body: List[Offset]) -> None:
        """React to snake body changes."""
        # Refresh old body positions
        for segment in old_body:
            self.refresh(self.get_square_region(segment))
        
        # Refresh new body positions
        for segment in new_body:
            self.refresh(self.get_square_region(segment))
    
    def watch_food_position(self, old_food: Offset, new_food: Offset) -> None:
        """React to food position changes."""
        self.refresh(self.get_square_region(old_food))
        self.refresh(self.get_square_region(new_food))
    
    def watch_grid_size(self, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> None:
        """React to grid size changes."""
        self.board_size = new_size
        self.virtual_size = Size(new_size[0] * 2, new_size[1])
        self.refresh()
    
    def get_square_region(self, square_offset: Offset) -> Region:
        """Get region relative to widget from square coordinate."""
        x, y = square_offset
        region = Region(x * 2, y, 2, 1)
        # Move the region into the widget's frame of reference
        region = region.translate(-self.scroll_offset)
        return region