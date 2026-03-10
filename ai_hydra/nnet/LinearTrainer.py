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

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DNNet import DNetDef
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
        gamma: float = DNetDef.GAMMA,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.replay = replay
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # nn.MSELoss()
        torch.manual_seed(DHydra.RANDOM_SEED)
        self.log = HydraLog(
            client_id="LinearTrainer", log_level=log_level, to_console=True
        )
        self._per_step_losses = []
        self._per_ep_loss = None

    def get_per_ep_loss(self) -> float | None:
        return self._per_ep_loss

    def get_avg_per_step_loss(self) -> float | None:
        if not self._per_step_losses:
            return None
        avg_loss = sum(self._per_step_losses) / len(self._per_step_losses)
        self._per_step_losses = []
        return avg_loss

    def train_long_memory(
        self, batch_size: int = DMemory.BATCH_SIZE
    ) -> float | None:

        batch = self.replay.sample_transitions(batch_size)
        if batch is None:
            return None

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
