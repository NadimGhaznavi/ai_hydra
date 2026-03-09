# ai_hydra/nnet/ReplayMemory.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from random import Random
from typing import Deque, List
import pickle

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.DBMgr import DBMgr
from ai_hydra.constants.DReplayMemory import DMemory


class ReplayMemory:

    def __init__(self, db_mgr: DBMgr):
        # Our SQLite DB manager
        self._db_mgr = db_mgr

        # How large batches of frames should be
        self._batch_size = DMemory.FRAME_BATCH_SIZE  # 256

        # Memory type, valid choices are ran_frames, ran_game or none.
        self._mem_type = DMemory.RAN_GAMES

        # Frames for the current game are stored here.
        self._cur_mem = []

    def append(self, transition, final_score=None):
        """
        Add a transition to the current game frames list.
        """
        if self.mem_type() == DMemory.NO_MEMORY:
            return

        (old_state, action, reward, new_state, done) = transition

        self._cur_mem.append((old_state, action, reward, new_state, done))

        if done:
            if final_score is None:
                raise ValueError(
                    "final_score must be provided when the game ends"
                )

            total_frames = len(self._cur_mem)

            game_id = self._db_mgr.add_game(
                final_score=final_score, total_frames=total_frames
            )
            ## NOTE: This is low-level SQLite DB code that *should* be in
            ## DBMgr, but that resulted in a huge performance hit. Moving it
            ## here means we don't pass in one frame at a time.
            ##
            db_cursor = self._db_mgr.cursor()
            for index, (state, action, reward, next_state, done) in enumerate(
                self._cur_mem
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
            self._db_mgr._conn.commit()
            self.cur_memory = []

    def get_training_data(self):
        mem_type = self.mem_type()

        if mem_type == DMemory.NO_MEMORY:
            return None, DMemory.NO_DATA  # No data available

        # RANDOM_GAME mode: return full ordered frames from one random game
        elif mem_type == DMemory.RAN_GAMES:
            frames, game_id = self._db_mgr.get_random_game()
            if not frames:  # no frames available
                return None, DMemory.NO_DATA
            training_data = frames
            metadata = game_id

        # SHUFFLE mode: return a random set of frames
        elif mem_type == DMemory.RAN_FRAMES:
            frames, num_frames = self._db_mgr.get_random_frames()
            if not frames:  # no frames available
                return None, DMemory.NO_DATA
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

    def mem_type(self, mem_type: str = None) -> None:
        if mem_type is not None:
            self._mem_type = mem_type
        return self._mem_type
