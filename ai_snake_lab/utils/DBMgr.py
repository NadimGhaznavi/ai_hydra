"""
ai_snake_lab/utils/DBMgr.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0

This file contains the ReplayMemory class.
"""

import os
import random
import sqlite3, pickle
from pathlib import Path

from ai_snake_lab.constants.DDir import DDir
from ai_snake_lab.constants.DFile import DFile
from ai_snake_lab.constants.DReplayMemory import MEM, MEM_TYPE
from ai_snake_lab.constants.DPlot import Plot


class DBMgr:

    def __init__(self, seed):
        # Set the seed value to ensure deterministic simulation runs
        random.seed(seed)

        # The minimum number of games that need to be stored before any random
        # frames or random games are returned
        self.min_games = MEM.MIN_GAMES

        # The snake lab stores temporary and persistent files in this directory
        snake_dir = os.path.join(Path.home(), DDir.DOT + DDir.AI_SNAKE_LAB)
        if not os.path.exists(snake_dir):
            os.mkdir(snake_dir)
        self.db_file = os.path.join(snake_dir, DFile.RUNTIME_DB)
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

        # Connect to SQLite
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)

        # Get a cursor
        self._cursor = self.conn.cursor()

        # Intialize the schema
        self.init_db()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # avoid errors on interpreter shutdown

    def add_avg_loss(self, epoch, avg_loss):
        # Record the average loss after each epoch
        self._cursor.execute(
            "INSERT INTO avg_loss (epoch, avg_loss) VALUES (?, ?)",
            (epoch, avg_loss),
        )
        self.conn.commit()

    def add_game(self, final_score, total_frames):
        # Record the game
        self._cursor.execute(
            "INSERT INTO games (score, total_frames) VALUES (?, ?)",
            (final_score, total_frames),
        )
        self.conn.commit()
        return self._cursor.lastrowid  # game_id

    def add_game_score(self, epoch, score):
        # Record the game score
        self._cursor.execute(
            "INSERT INTO game_score (epoch, score) VALUES (?, ?)",
            (epoch, score),
        )

        # Limit the amount of data in this table. We only need it for the
        # game score plot which has a MAX_GAMESCORE_DATA_POINTS limit for the
        # amount of data being plotted.
        self._cursor.execute(
            f"""
            DELETE FROM game_score
            WHERE id NOT IN (
                SELECT id FROM game_score
                ORDER BY epoch DESC
                LIMIT {Plot.MAX_GAMESCORE_DATA_POINTS}
            )
            """
        )
        self.conn.commit()

    def add_highscore_event(self, epoch, score, runtime):
        # Record the highscore event
        self._cursor.execute(
            "INSERT INTO highscore_events (epoch, score, runtime) VALUES (?, ?, ?)",
            (epoch, score, runtime),
        )
        self.conn.commit()

    def clear_runtime_data(self):
        """Clear all data out of the runtime DB"""
        self._cursor.execute("DELETE FROM games")
        self._cursor.execute("DELETE FROM frames")
        self._cursor.execute("DELETE from highscore_events")
        self._cursor.execute("DELETE from game_score")
        self._cursor.execute("DELETE from avg_loss")
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if getattr(self, "conn", None):
            self.conn.close()
            self.conn = None
        if getattr(self, "db_file", None) and os.path.exists(self.db_file):
            os.remove(self.db_file)
            self.db_file = None

    def cursor(self):
        return self._cursor

    def get_avg_loss_data(self):
        self._cursor.execute("SELECT epoch, avg_loss FROM avg_loss ORDER BY epoch ASC")
        return self._cursor.fetchall()

    def get_average_game_length(self):
        self._cursor.execute("SELECT AVG(total_frames) FROM games")
        avg = self._cursor.fetchone()[0]
        return int(avg) if avg else 0

    def get_game_score_data(self):
        self._cursor.execute("SELECT epoch, score from game_score ORDER by epoch ASC")
        return self._cursor.fetchall()

    def get_highscore_events(self):
        self._cursor.execute(
            "SELECT epoch, score, runtime FROM highscore_events ORDER BY epoch ASC"
        )
        return self._cursor.fetchall()

    def get_num_games(self):
        """Return number of games stored in the database."""
        self._cursor.execute("SELECT COUNT(*) FROM games")
        return self._cursor.fetchone()[0]

    def get_num_frames(self):
        """Return number of frames stored in the database."""
        self._cursor.execute("SELECT COUNT(*) FROM frames")
        return self._cursor.fetchone()[0]

    def get_random_frames(self):
        """Return a random set of frames from the database. Do not use the SQLite3
        RANDOM() function, because we can't set the seed."""
        num_frames = self.get_average_game_length() or 32  # fallback if no data

        # Retrieve the id values from the frames table
        self._cursor.execute("SELECT id FROM frames")
        all_ids = [row[0] for row in self._cursor.fetchall()]

        # Get n random ids
        sample_ids = random.sample(all_ids, min(num_frames, len(all_ids)))
        if not sample_ids:
            return []

        # Build placeholders for the SQL IN clause
        placeholders = ",".join("?" for _ in sample_ids)

        # Execute the query with the unpacked tuple
        self._cursor.execute(
            f"SELECT state, action, reward, next_state, done FROM frames WHERE id IN ({placeholders})",
            sample_ids,
        )
        rows = self._cursor.fetchall()
        frames = [
            (
                pickle.loads(state_blob),
                pickle.loads(action),
                float(reward),
                pickle.loads(next_state_blob),
                bool(done),
            )
            for state_blob, action, reward, next_state_blob, done in rows
        ]
        return frames, len(frames)

    def get_random_game(self):
        self._cursor.execute("SELECT id FROM games")
        all_ids = [row[0] for row in self._cursor.fetchall()]
        if not all_ids or len(all_ids) < self.min_games:
            return [], MEM.NO_DATA  # no games available

        rand_id = random.choice(all_ids)
        self._cursor.execute(
            "SELECT state, action, reward, next_state, done "
            "FROM frames WHERE game_id = ? ORDER BY frame_index ASC",
            (rand_id,),
        )
        rows = self._cursor.fetchall()
        if not rows:
            return [], rand_id  # game exists but no frames

        game = [
            (
                pickle.loads(state_blob),
                pickle.loads(action),
                float(reward),
                pickle.loads(next_state_blob),
                bool(done),
            )
            for state_blob, action, reward, next_state_blob, done in rows
        ]
        return game, rand_id

    def get_num_games(self):
        """Return number of games stored in the database."""
        self._cursor.execute("SELECT COUNT(*) FROM games")
        return self._cursor.fetchone()[0]

    def get_num_frames(self):
        """Return number of frames stored in the database."""
        self._cursor.execute("SELECT COUNT(*) FROM frames")
        return self._cursor.fetchone()[0]

    def get_training_data(self, mem_type):
        if mem_type == MEM_TYPE.NONE:
            return None, MEM.NO_DATA  # No data available

        # RANDOM_GAME mode: return full ordered frames from one random game
        elif mem_type == MEM_TYPE.RANDOM_GAME:
            frames, game_id = self.get_random_game()
            if not frames:  # no frames available
                return None, MEM.NO_DATA
            training_data = frames
            metadata = game_id

        # SHUFFLE mode: return a random set of frames
        elif mem_type == MEM_TYPE.SHUFFLE:
            frames, num_frames = self.get_random_frames()
            if not frames:  # no frames available
                return None, MEM.NO_DATA
            training_data = frames
            metadata = num_frames

        else:
            raise ValueError(f"Unknown memory type: {mem_type}")

        # Split into arrays for vectorized training
        states = [d[0] for d in training_data]
        actions = [d[1] for d in training_data]
        rewards = [d[2] for d in training_data]
        next_states = [d[3] for d in training_data]
        dones = [d[4] for d in training_data]

        return (states, actions, rewards, next_states, dones), metadata

    def init_db(self):
        # Create the games table
        self._cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score INTEGER NOT NULL,
                total_frames INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                state BLOB NOT NULL,
                action BLOB NOT NULL,      
                reward INTEGER NOT NULL,
                next_state BLOB NOT NULL,
                done INTEGER NOT NULL,        -- 0 or 1
                FOREIGN KEY (game_id) REFERENCES games(id)
            );

            CREATE TABLE IF NOT EXISTS highscore_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                score INTEGER NOT NULL,
                runtime TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS avg_loss (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                avg_loss REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS game_score (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                score INTEGER NOT NULL            
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_game_frame ON frames (game_id, frame_index);

            CREATE INDEX IF NOT EXISTS idx_frames_game_id ON frames (game_id);

            CREATE INDEX IF NOT EXISTS idx_highscore_events ON highscore_events (epoch);

            CREATE INDEX IF NOT EXISTS idx_avg_loss ON avg_loss (epoch);

            CREATE INDEX IF NOT EXISTS idx_game_score ON game_score (epoch);
            """
        )
        self.conn.commit()

        # If the games or frames tables do exist, clear the data
        self.clear_runtime_data()
