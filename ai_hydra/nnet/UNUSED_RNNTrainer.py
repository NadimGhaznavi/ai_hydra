# ai_hydra/nnet/RNNTrainer.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DNNet import DNetDef
from ai_hydra.constants.DHydra import DHydraLog

from ai_hydra.nnet.ReplayMemory import ReplayMemory
from ai_hydra.utils.HydraLog import HydraLog


class RNNTrainer:
    def __init__(
        self,
        model,
        replay: ReplayMemory,
        lr: float,
        log_level: DHydraLog,
        device: torch.device | None = None,
        gamma: float = DNetDef.GAMMA,
        target_update_freq: int = 100,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.target_model = deepcopy(model).to(self.device)
        self.target_model.eval()

        self.replay = replay
        self.gamma = gamma
        self._update_counter = 0
        self._target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.log = HydraLog(
            client_id="RNNTrainer",
            log_level=log_level,
            to_console=True,
        )

        self._per_step_losses: list[float] = []
        self._per_ep_loss: float | None = None

    def get_per_ep_loss(self) -> float | None:
        return self._per_ep_loss

    def get_avg_per_step_loss(self) -> float | None:
        if not self._per_step_losses:
            return None
        avg_loss = sum(self._per_step_losses) / len(self._per_step_losses)
        self._per_step_losses = []
        return avg_loss

    def soft_update_target(self, tau: float = 1.0) -> None:
        for target_param, param in zip(
            self.target_model.parameters(),
            self.model.parameters(),
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def train_long_memory(
        self,
        batch_size: int = DMemory.BATCH_SIZE,
    ) -> float | None:
        self.model.train()
        self.target_model.eval()

        training_batch = self.replay.sample_sequences(batch_size=batch_size)
        if not training_batch:
            return None

        seq_losses: list[torch.Tensor] = []

        for seq in training_batch:
            states = torch.tensor(
                np.array([t.old_state for t in seq]),
                dtype=torch.float32,
                device=self.device,
            )
            actions = torch.tensor(
                np.array([t.action for t in seq]),
                dtype=torch.long,
                device=self.device,
            )
            rewards = torch.tensor(
                np.array([t.reward for t in seq]),
                dtype=torch.float32,
                device=self.device,
            )
            new_states = torch.tensor(
                np.array([t.new_state for t in seq]),
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.tensor(
                np.array([t.done for t in seq]),
                dtype=torch.float32,
                device=self.device,
            )

            pred_q_all = self.model.forward_sequence(states)
            pred_q = pred_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_online = self.model.forward_sequence(new_states)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

                next_q_target_all = self.target_model.forward_sequence(
                    new_states
                )
                next_q_target = next_q_target_all.gather(
                    1, next_actions
                ).squeeze(1)

                target_q = rewards + self.gamma * next_q_target * (1.0 - dones)

            seq_loss = self.criterion(pred_q, target_q)
            seq_losses.append(seq_loss)

        loss = torch.stack(seq_losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self.soft_update_target()

        loss_value = float(loss.item())
        self._per_step_losses.append(loss_value)
        self._per_ep_loss = loss_value
        return loss_value
