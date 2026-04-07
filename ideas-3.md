def get_state(self) -> list[float]:
    head = self.snake_head
    radius = 3
    forward = self.direction
    right = Direction(-forward.dy, forward.dx)  # 90° clockwise

    grid: list[float] = []
    tail_pos = self.snake_body[-1] if self.snake_body else None  # last element is tail

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                grid.append(0.0)  # head
                continue

            # Egocentric world position
            wx = head.x + (dx * right.dx) - (dy * forward.dx)
            wy = head.y + (dx * right.dy) - (dy * forward.dy)
            pos = Position(wx, wy)

            if not self.is_position_within_bounds(pos):
                grid.append(-1.0)                    # wall
            elif pos == self.food_position:
                grid.append(1.0)                     # food (highest priority)
            elif self.snake_body and pos == tail_pos:
                grid.append(0.2)                     # tail = low danger
            elif pos in self.snake_body:
                grid.append(0.8)                     # body segment = high danger
            else:
                grid.append(0.0)                     # empty

    # Continuous relative food position (very useful for planning)
    rel_x = self.food_position.x - head.x
    rel_y = self.food_position.y - head.y

    food_local_x = (rel_x * right.dx) + (rel_y * right.dy)
    food_local_y = -((rel_x * forward.dx) + (rel_y * forward.dy))

    # Normalize by half board size (keeps values reasonable even on larger boards)
    half_size = max(self.grid_size[0], self.grid_size[1]) / 2.0

    state = [
        *grid,                          # 49 values
        food_local_x / half_size,       # 50
        food_local_y / half_size,       # 51
    ]

    return [float(x) for x in state]