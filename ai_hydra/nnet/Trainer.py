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
from ai_hydra.constants.DNNet import DNetDef
from ai_hydra.constants.DHydra import DHydra, DHydraLog

from ai_hydra.nnet.ReplayMemory import ReplayMemory
from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.nnet.models.LinearModel import LinearModel
from ai_hydra.nnet.models.RNNModel import RNNModel


class Trainer:
    def __init__(
        self,
        model,
        replay: ReplayMemory,
        lr: float,
        log_level: DHydraLog,
        *,
        device: torch.device | None = None,
        gamma: float = DNetDef.GAMMA,
    ):
        torch.manual_seed(DHydra.RANDOM_SEED)
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.log = HydraLog(
            client_id="Trainer", log_level=log_level, to_console=True
        )

        ### TODO explore this
        self.tau = 0.01
        self.target_model = deepcopy(self.model)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.replay = replay
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # nn.SmoothL1Loss()

        ### TODO explore this
        self._target_update_freq = 100
        self._update_counter = 0

        self._per_step_losses = []
        self._per_ep_loss = None

        # Track epochs
        self._epoch = 0
        # The game that we're training on, if we're doing DMemory.RAN_GAMES
        self._game_id = None
        # The number of frames we're training on, if we're doing DMemory.RAN_FRAMES.
        self._num_frames = None
        # A buffer to hold the training data, this allows the code to handle
        # both Replay memory modes: Random game or random frames.
        self._training_data = None

    def game_id(self, value: int = None) -> int:
        if value is not None:
            self._game_id = value
        return self._game_id

    def get_per_ep_loss(self):
        return self._per_ep_loss

    def get_avg_per_step_loss(self):
        if not self._per_step_losses:
            return None
        avg_loss = sum(self._per_step_losses) / len(self._per_step_losses)
        self._per_step_losses = []
        return avg_loss

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
        if isinstance(self.model, LinearModel):
            return self._train_long_memory_linear(batch_size=batch_size)
        elif isinstance(self.model, RNNModel):
            return self._train_long_memory_rnn()
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _train_long_memory_linear(
        self, batch_size=DMemory.BATCH_SIZE
    ) -> float | None:

        batch = self.replay.sample(batch_size)
        if batch is None:
            return

        states = torch.tensor(
            [t.old_state for t in batch],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            [t.new_state for t in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.model(next_states)
            max_next_q = next_q.max(dim=1).values
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.criterion(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._per_step_losses.append(loss.item())
        self._per_ep_loss = loss.item()

        return float(loss.item())

    def _train_long_memory_rnn(self) -> float | None:
        self._epoch += 1

        # Get the training data from ReplayMemory
        self.load_training_data()

        training_batch = self.replay.sample()

        # No training data is available
        if not training_batch:
            return

        states, actions, rewards, new_states, dones = zip(*training_batch)

        # Convert full game sequence to tensors
        states = torch.tensor(
            np.array(states), dtype=torch.float, device=self.device
        )
        new_states = torch.tensor(
            np.array(new_states), dtype=torch.float, device=self.device
        )
        actions = torch.tensor(
            np.array(actions), dtype=torch.long, device=self.device
        ).view(-1)
        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float, device=self.device
        ).view(-1)
        dones = torch.tensor(
            np.array(dones), dtype=torch.bool, device=self.device
        ).view(-1)

        # Get Q-values for every timestep in the sequence
        pred_Q_all = self.model.forward_sequence(
            states
        )  # [seq_len, n_actions]
        pred_Q = pred_Q_all.gather(1, actions.unsqueeze(1)).squeeze(
            1
        )  # [seq_len]

        with torch.no_grad():
            next_Q_online = self.model.forward_sequence(
                new_states
            )  # [seq_len, n_actions]
            next_actions = torch.argmax(
                next_Q_online, dim=1, keepdim=True
            )  # [seq_len, 1]

            next_Q_target_all = self.target_model.forward_sequence(
                new_states
            )  # [seq_len, n_actions]
            next_Q_target = next_Q_target_all.gather(1, next_actions).squeeze(
                1
            )  # [seq_len]

            target_Q = rewards + self.gamma * next_Q_target * (~dones)

        loss = self.criterion(pred_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self.soft_update_target()

        self._per_ep_loss = loss.item()
        return loss.item()

    def train_step(self, t: Transition) -> float:
        if isinstance(self.model, LinearModel):
            return self._train_step_linear(t)
        elif isinstance(self.model, RNNModel):
            return self._train_step_rnn(t)
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _train_step_linear(self, t: Transition) -> float:
        old_state = torch.tensor(
            np.array(t.old_state), dtype=torch.float32, device=self.device
        )
        new_state = torch.tensor(
            np.array(t.new_state), dtype=torch.float32, device=self.device
        )
        action = torch.tensor(
            np.array(t.action), dtype=torch.long, device=self.device
        )
        reward = torch.tensor(
            np.array(t.reward), dtype=torch.float32, device=self.device
        )
        done = torch.tensor(
            np.array(t.done), dtype=torch.float32, device=self.device
        )

        if old_state.ndim == 1:
            old_state = old_state.unsqueeze(0)
            new_state = new_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        action = action.view(-1)
        reward = reward.view(-1)
        done = done.view(-1)

        q_values = self.model(old_state)
        q_selected = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.model(new_state)
            max_next_q = next_q.max(dim=1).values
            target = reward + self.gamma * max_next_q * (1.0 - done)

        loss = self.criterion(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._per_step_losses.append(loss.item())
        return float(loss.item())

    def _train_step_rnn(self, t: Transition) -> float:
        old_state = t.old_state
        action = t.action
        reward = t.reward
        new_state = t.new_state
        done = t.done

        # Convert to tensors on the correct device
        old_state = torch.tensor(
            np.array(old_state), dtype=torch.float, device=self.device
        )
        new_state = torch.tensor(
            np.array(new_state), dtype=torch.float, device=self.device
        )
        action = torch.tensor(
            np.array(action), dtype=torch.long, device=self.device
        )
        reward = torch.tensor(
            np.array(reward), dtype=torch.float, device=self.device
        )
        done = torch.tensor(
            np.array(done), dtype=torch.bool, device=self.device
        )

        # Single-step online training:
        # ensure one state becomes a sequence of length 1
        if old_state.dim() == 1:
            old_state = old_state.unsqueeze(0)  # [1, input_size]
        if new_state.dim() == 1:
            new_state = new_state.unsqueeze(0)  # [1, input_size]

        # Scalars for the single transition
        action = action.view(-1)[0]
        reward = reward.view(-1)[0]
        done = done.view(-1)[0]

        # Model returns Q-values for the last timestep of the sequence: [n_actions]
        pred_Q_all = self.model(old_state)  # [n_actions]
        pred_Q = pred_Q_all[action]  # scalar

        with torch.no_grad():
            next_Q_all = self.target_model(new_state)  # [n_actions]
            next_Q = torch.max(next_Q_all)  # scalar
            target_Q = reward + self.gamma * next_Q * (~done)

        loss = self.criterion(pred_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self.soft_update_target()

        self._per_step_losses.append(loss.item())
        return loss.item()

    def training_data(self):
        return self._training_data
