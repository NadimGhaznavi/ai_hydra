"""
ai_snake_lab/ai/Agent.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import torch
from ai_snake_lab.ai.EpsilonAlgo import EpsilonAlgo
from ai_snake_lab.ai.EpsilonAlgoN import EpsilonAlgoN
from ai_snake_lab.ai.ReplayMemory import ReplayMemory
from ai_snake_lab.ai.AITrainer import AITrainer

from ai_snake_lab.ai.models.ModelL import ModelL
from ai_snake_lab.ai.models.ModelRNN import ModelRNN

from ai_snake_lab.utils.DBMgr import DBMgr

from ai_snake_lab.constants.DLabels import DLabel
from ai_snake_lab.constants.DReplayMemory import MEM, MEM_TYPE
from ai_snake_lab.constants.DModelL import DModelL
from ai_snake_lab.constants.DModelLRNN import DModelRNN
from ai_snake_lab.constants.DAIAgent import DAIAgent
from ai_snake_lab.constants.DEpsilon import DEpsilon
from ai_snake_lab.constants.DSim import DSim


class AIAgent:

    def __init__(self, seed: int, db_mgr: DBMgr):
        self.explore = None
        self.memory = ReplayMemory(db_mgr=db_mgr)
        self.trainer = None
        self._training_data = []
        self._game_id = None
        self._model_type = None
        self._num_frames = None
        self._seed = seed
        self._dynamic_training = None
        self._epoch = None
        self._model = None

    def dynamic_training(self, enable_flag=None):
        if enable_flag is not None:
            self._dynamic_training = enable_flag
        return self._dynamic_training

    def epoch(self, epoch=None):
        if epoch is not None:
            self._epoch = epoch
        return self._epoch

    def game_id(self, game_id=None):
        if game_id is not None:
            self._game_id = game_id
        return self._game_id

    def get_move(self, state, score):
        random_move = self.explore.get_move(score=score)  # Explore with epsilon
        if random_move != False:
            return random_move  # Random move was returned

        # Exploit with an AI agent based action
        final_move = [0, 0, 0]
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float)  # Convert to a tensor
        prediction = self._model(state)  # Get the prediction
        move = torch.argmax(prediction).item()  # Select the move with the highest value
        final_move[move] = 1  # Set the move
        return final_move  # Return

    def get_optimizer(self):
        return self.trainer.get_optimizer()

    def load_training_data(self):
        # Ask ReplayMemory for training data and check that it was provided
        batch, metadata = self.memory.get_training_data()
        if batch is None:
            self.game_id(MEM.NO_DATA)
            self.num_frames(MEM.NO_DATA)
            return

        states, actions, rewards, next_states, dones = batch

        # Update current game id for TUI
        mem_type = self.memory.mem_type()
        if mem_type == MEM_TYPE.RANDOM_GAME:
            self.game_id(metadata)
        # Update the number of frames for the TUI
        elif mem_type == MEM_TYPE.SHUFFLE:
            self.num_frames(metadata)

        # Store the training data in the agent (without frame index)
        self._training_data = list(zip(states, actions, rewards, next_states, dones))

    def model_type(self, model_type=None):
        if model_type is not None:
            if model_type == DModelL.MODEL:
                self._model = ModelL(seed=self._seed)
            elif model_type == DModelRNN.MODEL:
                self._model = ModelRNN(seed=self._seed)
            self.trainer = AITrainer(seed=self._seed)
            self.trainer.set_model(self._model)
        return self._model_type

    def model_type_name(self):
        if type(self._model) == ModelL:
            return DLabel.LINEAR_MODEL
        elif type(self._model) == ModelRNN:
            return DLabel.RNN_MODEL

    def model(self):
        return self._model

    def num_frames(self, frames=None):
        if frames is not None:
            self._num_frames = frames
        return self._num_frames

    def set_explore(self, explore_type):
        if explore_type == DEpsilon.EPSILON:
            self.explore = EpsilonAlgo(seed=self._seed)
        elif explore_type == DEpsilon.EPSILON_N:
            self.explore = EpsilonAlgoN(seed=self._seed)
        else:
            raise ValueError(f"Unknown exploration type: {explore_type}")

    def set_optimizer(self, optimizer):
        self.trainer.set_optimizer(optimizer)

    def train_long_memory(self, batch_size=DSim.BATCH_SIZE):

        if self.epoch() < MEM.MIN_GAMES:
            self.game_id(MEM.NO_DATA)
            self.num_frames(MEM.NO_DATA)
            return

        # Adaptive training

        if self.dynamic_training():
            loops = max(
                1, min(self.epoch() // 250, DAIAgent.MAX_DYNAMIC_TRAINING_LOOPS)
            )
        else:
            loops = 1

        while loops > 0:
            loops -= 1

            # No training data is available
            if self.game_id() == MEM.NO_DATA:
                return

            training_batch = self.training_data()
            if not training_batch:
                return

            states, actions, rewards, next_states, dones = zip(*training_batch)
            n_samples = len(states)
            total_loss = 0.0

            # Slice into mini-batches
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_rewards = rewards[start:end]
                batch_next_states = next_states[start:end]
                batch_dones = dones[start:end]

                # Vectorized training step
                loss = self.trainer.train_step(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                )
                total_loss += loss

            avg_loss = total_loss / (n_samples / batch_size)
            return avg_loss

    def train_short_memory(self, state, action, reward, next_state, done):
        # Always train on the current frame
        self.trainer.train_step(state, action, reward, next_state, [done])

    def training_data(self):
        return self._training_data
