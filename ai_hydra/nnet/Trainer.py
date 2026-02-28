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

from ai_hydra.nnet.ReplayMemory import ReplayMemory


class Trainer:
    def __init__(
        self,
        model,
        replay_buffer: ReplayMemory,
        *,
        device: torch.device | None = None,
        gamma: float = 0.9,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.replay_buffer = replay_buffer
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train_step(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)

        states = torch.tensor(
            [t.state for t in batch], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            [t.next_state for t in batch],
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

        return float(loss.item())
