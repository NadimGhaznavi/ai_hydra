"""
ai_snake_lab/ai/AITrainer.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import copy

from ai_snake_lab.utils.LabLogger import LabLogger

from ai_snake_lab.ai.models.ModelL import ModelL
from ai_snake_lab.ai.models.ModelRNN import ModelRNN
from ai_snake_lab.constants.DModelL import DModelL
from ai_snake_lab.constants.DModelLRNN import DModelRNN
from ai_snake_lab.constants.DSim import DSim


class AITrainer:

    def __init__(self, seed):
        torch.manual_seed(seed)

        self.model = None
        self.target_model = None
        self.optimizer = None

        self.log = LabLogger(client_id="AITrainer", to_console=True)

        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.gamma = DSim.DISCOUNT_RATE  # Huber loss
        self.tau = 0.01  # For soft target updates
        self.update_counter = 0
        self.target_update_freq = 100  # Every N training steps
        self._epoch_losses = []

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_learning_rate(self, learning_rate):
        # NOTE: You need to set the model before you set the learning rate!
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def set_model(self, model):
        self.model = model

        # Initialize target network as a frozen copy
        self.target_model = copy.deepcopy(self.model)
        for param in self.target_model.parameters():
            param.requires_grad = False

    def get_epoch_loss(self):
        if not self._epoch_losses:
            return 0.0
        avg_loss = sum(self._epoch_losses) / len(self._epoch_losses)
        self._epoch_losses = []
        return avg_loss

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def soft_update_target(self):
        """Soft update: θ_target ← τ*θ_main + (1-τ)*θ_target"""
        for target_param, main_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1.0 - self.tau) * target_param.data
            )

    def train_step(self, state, action, reward, next_state, game_over):
        # Convert inputs to tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(game_over), dtype=torch.bool)

        # Handle single transition case → batch size 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
            reward = reward.squeeze(-1)
            done = done.squeeze(-1)

        # Predicted Q-values (main network)
        pred_Q_all = self.model(state)  # shape: [batch, n_actions]
        action_indices = torch.argmax(action, dim=1)
        pred_Q = pred_Q_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        # Compute target Q-values

        # Orig
        # with torch.no_grad():
        #    next_Q_values = self.target_model(next_state)
        #    max_next_Q = next_Q_values.max(dim=1)[0]  # shape: [batch_size]
        #    target_Q = reward + self.gamma * max_next_Q * (~done)  # shape: [batch_size]
        # Ver 1
        # with torch.no_grad():
        #    next_actions = self.model(next_state).argmax(dim=1)
        #    next_Q_target = self.target_model(next_state)
        #    max_next_Q = next_Q_target.gather(1, next_actions.unsqueeze(1)).squeeze()
        # Ver 2
        # with torch.no_grad():
        #    # Actions chosen by main network
        #    next_actions = self.model(next_state).argmax(dim=1)
        #    # Q-values evaluated by target network
        #    next_Q = self.target_model(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        #    target_Q = reward + self.gamma * next_Q * (~done)
        # Ver 3
        with torch.no_grad():
            # Select best actions using online (policy) network
            next_actions = torch.argmax(self.model(next_state), dim=1, keepdim=True)
            # Evaluate them using target network
            next_Q_target = (
                self.target_model(next_state).gather(1, next_actions).squeeze(1)
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
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.soft_update_target()

        # Keep the step loss value
        self._epoch_losses.append(loss.item())
        return loss.item()
