"""
ai_snake_lab/ai/ReplayMemory.py

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

from ai_snake_lab.utils.DBMgr import DBMgr

from ai_snake_lab.constants.DReplayMemory import MEM_TYPE, MEM
from ai_snake_lab.constants.DDir import DDir
from ai_snake_lab.constants.DFile import DFile


class ReplayMemory:

    def __init__(self, db_mgr: DBMgr):
        # SQLite3 database manager
        self.db_mgr = db_mgr

        # How large the batches of frames should be
        self.batch_size = 250

        # Valid options: shuffle, random_game or none
        self._mem_type = MEM_TYPE.RANDOM_GAME

        # All of the states for a game are stored, in order.
        self.cur_memory = []

    def append(self, transition, final_score=None):
        """Add a transition to the current game."""
        if self.mem_type() == MEM_TYPE.NONE:
            return

        (old_state, move, reward, new_state, done) = transition

        self.cur_memory.append((old_state, move, reward, new_state, done))

        if done:
            if final_score is None:
                raise ValueError("final_score must be provided when the game ends")

            total_frames = len(self.cur_memory)

            game_id = self.db_mgr.add_game(
                final_score=final_score, total_frames=total_frames
            )

            ## NOTE: This is low-level SQLite DB code that *should* be in DBMgr, but
            ## that caused a huge performance hit. Moving it here is an 10x+ performance
            ## optimization.
            ##
            db_cursor = self.db_mgr.cursor()
            for index, (state, action, reward, next_state, done) in enumerate(
                self.cur_memory
            ):
                db_cursor.execute(
                    """
                    INSERT INTO frames (game_id, frame_index, state, action, reward, next_state, done)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        game_id,
                        index,
                        pickle.dumps(state),
                        pickle.dumps(action),
                        reward,
                        pickle.dumps(next_state),
                        done,
                    ),
                )
            self.db_mgr.conn.commit()
            self.cur_memory = []

    def get_training_data(self):
        mem_type = self.mem_type()

        if mem_type == MEM_TYPE.NONE:
            return None, MEM.NO_DATA  # No data available

        # RANDOM_GAME mode: return full ordered frames from one random game
        elif mem_type == MEM_TYPE.RANDOM_GAME:
            frames, game_id = self.db_mgr.get_random_game()
            if not frames:  # no frames available
                return None, MEM.NO_DATA
            training_data = frames
            metadata = game_id

        # SHUFFLE mode: return a random set of frames
        elif mem_type == MEM_TYPE.SHUFFLE:
            frames, num_frames = self.db_mgr.get_random_frames()
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

    def mem_type(self, mem_type=None):
        if mem_type is not None:
            self._mem_type = mem_type
        return self._mem_type

    def set_memory(self, memory):
        self.memory = memory
