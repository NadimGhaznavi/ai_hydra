# ai_hydra/nnet/RecurrentTrainer.py
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

from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DNNet import DRecurrentTrainer

from ai_hydra.nnet.ATH.ATHMemory import ATHMemory
from ai_hydra.utils.HydraLog import HydraLog

GRAD_CLIPPING = True
DQN_WITH_TARGET = False
DOUBLE_DQN = True


class RecurrentTrainer:
    def __init__(
        self,
        model,
        replay: ATHMemory,
        lr: float,
        log_level: DHydraLog,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(DField.CPU)
        self.model = model.to(self.device)

        if DQN_WITH_TARGET or DOUBLE_DQN:
            self.target_model = deepcopy(model).to(self.device)
            self.target_model.eval()

        self.replay = replay
        self._tau = None
        self._gamma = None

        self.optimizer = DRecurrentTrainer.OPTIM(
            self.model.parameters(), lr=lr
        )
        self.criterion = DRecurrentTrainer.CRITERION()

        self.log = HydraLog(
            client_id=DModule.RECURRENT_TRAINER,
            log_level=log_level,
            to_console=True,
        )

        self._losses: list[float] = []
        self._cold_memory = True
        if DQN_WITH_TARGET and DOUBLE_DQN:
            raise ValueError(
                "You cannot enable both DQN_WITH_TARGET and DOUBLE_DQN"
            )
        self.log.info(f"Setting the learning rate to {lr}")
        self.log.debug("Initialized")

    def get_avg_loss(self) -> float | None:
        if not self._losses:
            return None
        avg_loss = sum(self._losses) / len(self._losses)
        self._losses = []
        return avg_loss

    def reset(self):
        self.optimizer.state = {}
        if DQN_WITH_TARGET or DOUBLE_DQN:
            self.target_model = deepcopy(self.model).to(self.device)
            self.target_model.eval()

    async def train_long_memory(self) -> float | None:

        chunks = await self.replay.data_mgr.sample_chunks()
        if chunks is None:
            return

        if self._cold_memory:
            self._cold_memory = False
            self.log.debug(
                f"Training with {len(chunks)} batches with sequence length {len(chunks[0])}"
            )

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

        q_pred_all = self.model.forward_sequence(states)  # [B, T, A]
        q_pred = q_pred_all.gather(2, actions.unsqueeze(-1)).squeeze(
            -1
        )  # [B, T]

        with torch.no_grad():
            if DOUBLE_DQN:
                self.model.eval()  # Don't include dropout noise
                # [B, T, A]
                online_next_q = self.model.forward_sequence(next_states)
                self.model.train()
                next_actions = online_next_q.argmax(
                    dim=2, keepdim=True
                )  # [B, T, 1]

                # target net evaluates those actions
                target_next_q = self.target_model.forward_sequence(
                    next_states
                )  # [B, T, A]
                max_next_q = target_next_q.gather(2, next_actions).squeeze(
                    -1
                )  # [B, T]

            elif DQN_WITH_TARGET:
                # [B, T, A]
                next_q = self.target_model.forward_sequence(next_states)
                max_next_q = next_q.max(dim=2).values  # [B, T]

            else:
                # Don't include dropout noise (eval()/train() toggle)
                self.model.eval()
                next_q = self.model.forward_sequence(next_states)
                self.model.train()
                max_next_q = next_q.max(dim=2).values  # [B, T]

            q_target = rewards + self._gamma * max_next_q * (1.0 - dones)

        loss = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        if GRAD_CLIPPING:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

        self.optimizer.step()

        if DQN_WITH_TARGET or DOUBLE_DQN:
            self._soft_update_target()

        self._losses.append(loss.item())
        return float(loss.item())

    def _soft_update_target(self) -> None:
        for target_param, param in zip(
            self.target_model.parameters(),
            self.model.parameters(),
        ):
            target_param.data.copy_(
                self._tau * param.data + (1.0 - self._tau) * target_param.data
            )

    def set_params(self, tau: float, gamma: float):
        self._tau = tau
        self.log.info(f"Setting Tau to {tau}")
        self._gamma = gamma
        self.log.info(f"Setting the Discount/Gamma to {gamma}")
