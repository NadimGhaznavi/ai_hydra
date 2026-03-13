# ai_hydra/nnet/RNNTrainer2.py
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

from ai_hydra.constants.DNNet import DNetDef, DRNN
from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DNNet import DRNNTrainer

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
        tau: float = 0.005,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.target_model = deepcopy(model).to(self.device)
        self.target_model.eval()

        self.replay = replay
        self.gamma = gamma
        self._target_update_freq = target_update_freq
        self._update_counter = 0
        self._tau = tau

        self.optimizer = DRNNTrainer.OPTIM(self.model.parameters(), lr=lr)
        self.criterion = DRNNTrainer.CRITERION()

        self.log = HydraLog(
            client_id="RNNTrainer",
            log_level=log_level,
            to_console=True,
        )

        self._per_ep_loss: float | None = None
        self._per_step_losses: list[float] = []
        self.log.debug("Initialized")
        self._cold_memory = True

    def get_per_ep_loss(self) -> float | None:
        return self._per_ep_loss

    def get_avg_per_step_loss(self) -> float | None:
        if not self._per_step_losses:
            return None
        avg_loss = sum(self._per_step_losses) / len(self._per_step_losses)
        self._per_step_losses = []
        return avg_loss

    def train_long_memory(self, batch_size=DRNN.BATCH_SIZE) -> None:
        chunks = self.replay.sample_chunks(batch_size=batch_size)
        if chunks is None:
            self._per_ep_loss = None
            return

        if self._cold_memory:
            self._cold_memory = False
            self.log.debug("Training has warmed up...")

        # Shapes (B = batch_size, T = seq_length, F = feature/input size)
        # states      -> [B, T, F]
        # actions     -> [B, T]
        # rewards     -> [B, T]
        # next_states -> [B, T, F]
        # dones       -> [B, T]
        states = torch.tensor(
            np.array(
                [[t.old_state for t in chunk] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.array(
                [[t.action for t in chunk] for chunk in chunks], dtype=np.int64
            ),
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            np.array(
                [[t.reward for t in chunk] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.array(
                [[t.new_state for t in chunk] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            np.array(
                [[t.done for t in chunk] for chunk in chunks], dtype=np.float32
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.model.train()

        # Online network Q(s_t, ·)
        q_pred_all = self.model.forward_sequence(states)  # [B, T, A]
        q_pred = q_pred_all.gather(2, actions.unsqueeze(-1)).squeeze(
            -1
        )  # [B, T]

        with torch.no_grad():
            # Target network max_a Q_target(s_{t+1}, a)
            q_next_all = self.target_model.forward_sequence(
                next_states
            )  # [B, T, A]
            q_next_max = q_next_all.max(dim=2).values  # [B, T]

            q_target = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._per_ep_loss = float(loss.item())
        self._per_step_losses.append(self._per_ep_loss)
        self.model.eval()

        # self._update_counter += 1
        # if self._update_counter % self._target_update_freq == 0:
        #    self._soft_update_target()
        self._soft_update_target()

    def _soft_update_target(self) -> None:
        for target_param, param in zip(
            self.target_model.parameters(),
            self.model.parameters(),
        ):
            target_param.data.copy_(
                self._tau * param.data + (1.0 - self._tau) * target_param.data
            )
