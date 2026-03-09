# ai_hydra/nnet/Trainer.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

from ai_hydra.constants.DReplayMemory import DMemory

from ai_hydra.nnet.ReplayMemory import ReplayMemory
from ai_hydra.constants.DNNet import DNetDef, DLinear
from ai_hydra.constants.DHydra import DHydra
from ai_hydra.nnet.models.LinearModel import LinearModel
from ai_hydra.nnet.models.RNNModel import RNNModel
from ai_hydra.nnet.Transition import Transition


class Trainer:
    def __init__(
        self,
        model,
        replay: ReplayMemory,
        lr: float,
        *,
        device: torch.device | None = None,
        gamma: float = DNetDef.GAMMA,
    ):
        torch.manual_seed(DHydra.RANDOM_SEED)
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        ### TODO explore this
        self.tau = 0.01
        self.target_model = deepcopy(model)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.replay = replay
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # nn.SmoothL1Loss()

        ### TODO explore this
        self._target_update_freq = 100
        self._update_counter = 0

        self._losses = []

        # Track epochs
        self._epoch = 0

        # The game that we're training on, if we're doing DMemory.RAN_GAMES
        self._game_id = None

        # The number of frames we're training on, if we're doing DMemory.RAN_FRAMES.
        self._num_frames = None

    def game_id(self, value: int = None) -> int:
        if value is not None:
            self._game_id = value

    def get_loss(self):
        if not self._losses:
            return None
        avg_loss = sum(self._losses) / len(self._losses)
        self._losses = []
        return avg_loss

    def load_training_data(self):
        # Ask ReplayMemory for training data and check that it was provided
        batch, metadata = self.replay.get_training_data()
        if batch is None:
            self.game_id(DMemory.NO_DATA)
            self.num_frames(DMemory.NO_DATA)
            return

        states, actions, rewards, next_states, dones = batch

        ### TODO: Update current game id/num_frames in the TUI
        mem_type = self.replay.mem_type()
        if mem_type == DMemory.RAN_GAMES:
            self.game_id(metadata)
        # Update the number of frames for the TUI
        elif mem_type == DMemory.RAN_FRAMES:
            self.num_frames(metadata)

        # Store the training data in the agent (without frame index)
        self._training_data = list(
            zip(states, actions, rewards, next_states, dones)
        )

    def num_frames(self, value: int = None) -> int:
        if value is not None:
            self._num_frames = value
        return self._num_frames

    def soft_update_target(self):
        """Soft update: θ_target ← τ*θ_main + (1-τ)*θ_target"""
        for target_param, main_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * main_param.data
                + (1.0 - self.tau) * target_param.data
            )

    def train_long_memory(self, batch_size=DMemory.BATCH_SIZE) -> float:
        self._epoch += 1

        # Let the memory "warm up"
        if self._epoch < DMemory.MIN_GAMES:
            self.game_id(DMemory.NO_DATA)
            self.num_frames(DMemory.NO_DATA)

        # Get the training data from the ReplayMemory
        self.load_training_data()

        # No training data is available
        if self.game_id() == DMemory.NO_DATA:
            return

        training_batch = self.training_data()

        # No training data is available
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

    def train_step(self, t: Transition) -> float:
        (old_state, action, reward, new_state, done) = t

        # Convert t values into tensors
        old_state = torch.tensor(np.array(old_state), dtype=torch.float)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.bool)

        # Handle single transition case → batch size 1
        if len(old_state.shape) == 1:
            old_state = old_state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
            reward = reward.squeeze(-1)
            done = done.squeeze(-1)

        # Predicted Q-values (main network)
        pred_Q_all = self.model(old_state)  # shape: [batch, n_actions]
        action_indices = torch.argmax(action, dim=1)
        pred_Q = pred_Q_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Select best actions using online (policy) network
            next_actions = torch.argmax(
                self.model(next_state), dim=1, keepdim=True
            )
            # Evaluate them using target network
            next_Q_target = (
                self.target_model(next_state)
                .gather(1, next_actions)
                .squeeze(1)
            )
            # Compute final target values
            target_Q = reward + self.gamma * next_Q_target * (~done)

        # Compute loss
        loss = self.criterion(pred_Q, target_Q)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodic soft update of target model
        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self.soft_update_target()

        return loss.item()

    def training_data(self):
        return self._training_data
