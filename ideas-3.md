# === 1. Direction one-hot ===
dir_l = int(dir_vec.dx == -1 and dir_vec.dy == 0)
dir_r = int(dir_vec.dx ==  1 and dir_vec.dy == 0)
dir_u = int(dir_vec.dx ==  0 and dir_vec.dy == -1)
dir_d = int(dir_vec.dx ==  0 and dir_vec.dy ==  1)

# === 2. Relative dangers (snake body only + wall) ===
def get_relative_dangers():
    if dir_r:
        straight = Position(head.x + 1, head.y)
        right    = Position(head.x, head.y + 1)
        left     = Position(head.x, head.y - 1)
    elif dir_l:
        straight = Position(head.x - 1, head.y)
        right    = Position(head.x, head.y - 1)
        left     = Position(head.x, head.y + 1)
    elif dir_u:
        straight = Position(head.x, head.y - 1)
        right    = Position(head.x + 1, head.y)
        left     = Position(head.x - 1, head.y)
    else:  # down
        straight = Position(head.x, head.y + 1)
        right    = Position(head.x - 1, head.y)
        left     = Position(head.x + 1, head.y)

    snake_d_straight = self._is_snake_collision(straight)
    snake_d_right    = self._is_snake_collision(right)
    snake_d_left     = self._is_snake_collision(left)
    wall_d_straight  = self._is_wall_collision(straight)
    wall_d_right     = self._is_wall_collision(right)
    wall_d_left      = self._is_wall_collision(left)

    return [snake_d_left, snake_d_straight, snake_d_right,
            wall_d_left,  wall_d_straight,  wall_d_right]

# === 3. Food & Tail cardinal flags (keeping your style) ===
food_left  = int(self.food_position.x < head.x)
food_right = int(self.food_position.x > head.x)
food_up    = int(self.food_position.y < head.y)
food_down  = int(self.food_position.y > head.y)
food_on_x  = int(head.x == self.food_position.x)
food_on_y  = int(head.y == self.food_position.y)

tail = self.snake_body[-1] if self.snake_body else head
tail_left  = int(tail.x < head.x)
tail_right = int(tail.x > head.x)
tail_up    = int(tail.y < head.y)
tail_down  = int(tail.y > head.y)

# === 4. Length bits ===
length_bits = self._int_to_bits(self.STATE_LENGTH_BITS, self.get_snake_length())

# === 5. Local 7x7 directional occupancy grid (the key addition) ===
grid = []
# We rotate so current direction is always "up" (negative y)
for dy in range(-3, 4):      # 7 rows
    for dx in range(-3, 4):  # 7 columns
        if dx == 0 and dy == 0:
            grid.append(0.0)          # head = special value
            continue

        # Transform to world coordinates based on current heading
        if dir_r:   # right = current "up" becomes world +x? Wait, let's define properly
            wx = head.x + dy   # adjust mapping so snake "looks" up in the grid
            wy = head.y - dx
        elif dir_l:
            wx = head.x - dy
            wy = head.y + dx
        elif dir_u:
            wx = head.x + dx
            wy = head.y + dy
        else:  # down
            wx = head.x - dx
            wy = head.y - dy

        pos = Position(wx, wy)
        if not self.is_position_within_bounds(pos):
            grid.append(-1.0)                    # wall
        elif self._is_snake_collision(pos):
            grid.append(1.0)                     # body
        elif pos == self.food_position:
            grid.append(0.5)                     # food
        else:
            grid.append(0.0)                     # empty

# Build final state
state = [
    *get_relative_dangers(),           # 6 features
    dir_l, dir_r, dir_u, dir_d,        # 4
    food_left, food_right, food_up, food_down, food_on_x, food_on_y,  # 6
    tail_left, tail_right, tail_up, tail_down,                        # 4
    *length_bits,                      # 7
    *grid,                             # 49 (7×7)
]

assert len(state) == DNetDef.INPUT_SIZE