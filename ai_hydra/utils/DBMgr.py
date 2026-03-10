# ai_hydra/utils/DBMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import os
import random
import sqlite3, pickle
from pathlib import Path

from ai_hydra.constants.DHydra import DHydra, DHydraLog
from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydraTui import DFile

from ai_hydra.utils.HydraLog import HydraLog


class DBMgr:

    def __init__(self, log_level: DHydraLog):

        self.log = HydraLog(
            client_id="DBMgr", log_level=log_level, to_console=True
        )

        # We need determinism for RL experiments
        random.seed(DHydra.RANDOM_SEED)

        # The minimum number of episodes, before anyting is returned from
        # the replay memory.
        self._min_games = DMemory.MIN_GAMES
        # How large batches of frames should be
        self._batch_size = DMemory.BATCH_SIZE  # 256

        # AI Hydra stores temporary and persistent files in this directory
        ai_hydra_dir = os.path.join(Path.home(), DHydra.HYDRA_DIR)
        if not os.path.exists(ai_hydra_dir):
            os.mkdir(ai_hydra_dir)
        self._db_file = os.path.join(ai_hydra_dir, DFile.HYDRA_SERVER_DB)

        # Start with a new DB file (no persistence)
        if os.path.exists(self._db_file):
            os.remove(self._db_file)

        # Connect to SQLite
        self._conn = sqlite3.connect(self._db_file, check_same_thread=False)

        # Get a cursor
        self._cursor = self._conn.cursor()

        # Initialize the DB
        self._init_db()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Try to ensure the DB shuts down cleanly.
        """
        self.close()

    def __del__(self):
        """
        Try to ensure the DB shuts down cleanly.
        """
        try:
            self.close()
        except Exception:
            pass

    def add_game(self, final_score, total_frames):
        """
        Add a game into the games table.
        """
        self._cursor.execute(
            "INSERT INTO games (score, total_frames) VALUES (?, ?)",
            (final_score, total_frames),
        )
        self._conn.commit()
        return self._cursor.lastrowid  # game_id

    def clear_runtime_data(self) -> None:
        self._cursor.executescript(
            """
            DELETE from games;

            DELETE from frames;
            """
        )
        self._conn.commit()

    def close(self):
        """
        Close the database.
        """
        if getattr(self, "_conn", None):
            self._conn.close()
            self._conn = None
        if getattr(self, "_db_file", None) and os.path.exists(self._db_file):
            os.remove(self._db_file)
            self._db_file = None

    def cursor(self) -> None:
        """
        Return a cursor (used by ReplayMemory).
        """
        return self._cursor

    def get_avg_game_length(self):
        self._cursor.execute("SELECT AVG(total_frames) FROM games")
        avg = self._cursor.fetchone()[0]
        return int(avg) if avg else 0

    def get_random_frames(self, batch_size=None):
        """Return a random set of frames from the database. Do not use the SQLite3
        RANDOM() function, because we can't set the seed."""

        if batch_size:
            num_frames = batch_size
        else:
            num_frames = (
                self.get_avg_game_length() or 32
            )  # fallback if no data

        # Retrieve the id values from the frames table
        self._cursor.execute("SELECT id FROM frames")
        all_ids = [row[0] for row in self._cursor.fetchall()]

        # if len(all_ids) < self._batch_size * 10:
        if len(all_ids) < 1000:
            return [], 0

        # Get n random ids
        sample_ids = random.sample(all_ids, min(num_frames, len(all_ids)))
        if not sample_ids:
            return [], 0

        # Build placeholders for the SQL IN clause
        placeholders = ",".join("?" for _ in sample_ids)

        # Execute the query with the unpacked tuple
        self._cursor.execute(
            f"SELECT state, action, reward, next_state, done FROM frames "
            f"WHERE id IN ({placeholders}) ORDER BY id ASC",
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
        if not all_ids or len(all_ids) < DMemory.MIN_GAMES:
            return [], DMemory.NO_DATA  # no games available

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

    def _init_db(self) -> None:
        # Create the tables
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

            CREATE UNIQUE INDEX IF NOT EXISTS idx_game_frame ON frames (game_id, frame_index);

            CREATE INDEX IF NOT EXISTS idx_frames_game_id ON frames (game_id);
            """
        )
        self._conn.commit()

        # If the games or frames tables do exit, clear the data
        self.clear_runtime_data()
