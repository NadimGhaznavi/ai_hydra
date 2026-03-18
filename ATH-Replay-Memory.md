


## Adding Memories

- Initialization: `cur_game: list = []` and `games: list[list[t]] = []`
- Memories, or transitions will be added just like before `replay.append(t)`.
- `append(t)` will simply `cur_game.append(t)`
- When the game is over (`t.done == True`), the complete game will be saved:
  - `games.append(cur_game) ; cur_game = []`

## Managing Memory Size

- The Replay Memory will manage a **MAX_MEM_SIZE** setting that represents the total number of stored transitions. When this is reached, games will be removed from the beginning of the `games` list to keep the size relatively stable.

## The Chunk

- The smallest *memory unit* consists of 4 transitions; a **chunk**.
- Valid chunk sizes are 4, 8, 16, 32 or 64.
- The transitions in a chunk **must be ordered**.

## Sampling

Sampling is a bit more complex.



## The Clean Position

Avoid the Bitter Lesson:

- keep real experience
- keep bad experience
- keep embarrassing experience
- discard only what is too short to form a meaningful temporal unit

# Proposed Gears

Sequence Length | Batch Size | Score Threshold 
----------------|------------|-----------------
4               | 128        | 10
8               | 64         | 20
16              | 32         | 30
32              | 16         | 40
64              | 8          | 50
