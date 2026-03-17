# ai_hydra/nnet/LinearTrainer.py
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

from ai_hydra.constants.DNNet import DLinear
from ai_hydra.constants.DHydra import DHydra, DHydraLog

from ai_hydra.nnet.ReplayMemory import ReplayMemory
from ai_hydra.utils.HydraLog import HydraLog


class LinearTrainer:
    def __init__(
        self,
        model,
        replay: ReplayMemory,
        lr: float,
        log_level: DHydraLog,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.replay = replay
        self._gamma = None
        self._batch_size = None

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # nn.MSELoss()
        torch.manual_seed(DHydra.RANDOM_SEED)
        self.log = HydraLog(
            client_id="LinearTrainer", log_level=log_level, to_console=True
        )
        self._losses = []
        self._cold_memory = True
        self.log.info(f"Setting the learning rate to {lr}")
        self.log.info("Initialized")

    def get_avg_loss(self) -> float | None:
        if not self._losses:
            return None
        avg_loss = sum(self._losses) / len(self._losses)
        self._losses = []
        return avg_loss

    def train_long_memory(self) -> float | None:

        batch = self.replay.sample_transitions(self._batch_size)
        if batch is None:
            return None

        if self._cold_memory:
            self._cold_memory = False
            self.log.debug(f"Training with {self._batch_size} transitions")

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

        self.model.train()

        q_values = self.model(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        self.model.eval()
        with torch.no_grad():
            next_q = self.model(next_states)
            max_next_q = next_q.max(dim=1).values
            q_target = rewards + self._gamma * max_next_q * (1.0 - dones)

        loss = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._losses.append(loss.item())

        return float(loss.item())

    def set_params(self, gamma: float, batch_size: int):
        self._gamma = gamma
        self.log.info(f"Setting the Discount/Gamma to {gamma}")
        self._batch_size = batch_size
        self.log.info(f"Setting the Batch Size to {batch_size}")
