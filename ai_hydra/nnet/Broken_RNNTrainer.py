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
import torch.nn.functional as F
from copy import deepcopy

from ai_hydra.constants.DNNet import DRNNTrainer
from ai_hydra.constants.DHydra import DHydraLog

from ai_hydra.nnet.ReplayMemory import ReplayMemory, RNNChunk
from ai_hydra.utils.HydraLog import HydraLog

GRAD_CLIPPING = True
DQN_WITH_TARGET = False
DOUBLE_DQN = True


class RNNTrainer:
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

        if DQN_WITH_TARGET or DOUBLE_DQN:
            self.target_model = deepcopy(model).to(self.device)
            self.target_model.eval()

        self.replay = replay
        self._tau = None
        self._gamma = None
        self._batch_size = None

        self.optimizer = DRNNTrainer.OPTIM(self.model.parameters(), lr=lr)
        self.criterion = DRNNTrainer.CRITERION()

        self.log = HydraLog(
            client_id="RNNTrainer",
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

    def train_long_memory(self) -> float | None:
        chunks = self.replay.sample_chunks(batch_size=self._batch_size)
        if chunks is None:
            return None

        seq_length = len(chunks[0].transitions)

        if self._cold_memory:
            self._cold_memory = False
            self.log.debug(
                f"Training with {self._batch_size} batches with sequence length {seq_length}"
            )

        # Fast path: all chunks are full, no masking needed
        if all(chunk.valid_len == seq_length for chunk in chunks):
            return self._train_full_chunks_fast(chunks)

        # Cold-start path: one or more chunks are padded
        return self._train_padded_chunks_masked(chunks)

    def _build_batch_tensors(
        self,
        chunks: list[RNNChunk],
    ) -> tuple[torch.Tensor, ...]:
        """
        Build batched tensors from replay chunks.

        Returns:
            states:      [B, T, F]
            actions:     [B, T]
            rewards:     [B, T]
            next_states: [B, T, F]
            dones:       [B, T]
            valid_lens:  [B]
        """
        states = torch.tensor(
            np.array(
                [[t.old_state for t in chunk.transitions] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            np.array(
                [[t.action for t in chunk.transitions] for chunk in chunks],
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            np.array(
                [[t.reward for t in chunk.transitions] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            np.array(
                [[t.new_state for t in chunk.transitions] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            np.array(
                [[t.done for t in chunk.transitions] for chunk in chunks],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        valid_lens = torch.tensor(
            [chunk.valid_len for chunk in chunks],
            dtype=torch.long,
            device=self.device,
        )

        return states, actions, rewards, next_states, dones, valid_lens

    def _compute_q_pred_and_target(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute:
            q_pred   [B, T]
            q_target [B, T]
        """
        self.model.train()

        q_pred_all = self.model.forward_sequence(states)  # [B, T, A]
        q_pred = q_pred_all.gather(2, actions.unsqueeze(-1)).squeeze(
            -1
        )  # [B, T]

        with torch.no_grad():
            if DOUBLE_DQN:
                self.model.eval()  # avoid dropout noise in action selection
                online_next_q = self.model.forward_sequence(
                    next_states
                )  # [B, T, A]
                self.model.train()

                next_actions = online_next_q.argmax(
                    dim=2, keepdim=True
                )  # [B, T, 1]

                target_next_q = self.target_model.forward_sequence(
                    next_states
                )  # [B, T, A]
                max_next_q = target_next_q.gather(2, next_actions).squeeze(
                    -1
                )  # [B, T]

            elif DQN_WITH_TARGET:
                next_q = self.target_model.forward_sequence(
                    next_states
                )  # [B, T, A]
                max_next_q = next_q.max(dim=2).values  # [B, T]

            else:
                self.model.eval()
                next_q = self.model.forward_sequence(next_states)
                self.model.train()
                max_next_q = next_q.max(dim=2).values  # [B, T]

            q_target = rewards + self._gamma * max_next_q * (1.0 - dones)

        return q_pred, q_target

    def _train_full_chunks_fast(
        self,
        chunks: list[RNNChunk],
    ) -> float:
        """
        Hot path:
        all chunks are full-length, so we keep the simple fully-vectorized loss.
        """
        states, actions, rewards, next_states, dones, _valid_lens = (
            self._build_batch_tensors(chunks)
        )

        q_pred, q_target = self._compute_q_pred_and_target(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        loss = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        if GRAD_CLIPPING:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

        self.optimizer.step()

        if DQN_WITH_TARGET or DOUBLE_DQN:
            self._soft_update_target()

        loss_value = float(loss.item())
        self._losses.append(loss_value)
        return loss_value

    def _train_padded_chunks_masked(
        self,
        chunks: list[RNNChunk],
    ) -> float:
        """
        Cold-start path:
        one or more chunks are padded on the LEFT. Only the real suffix
        (last valid_len timesteps) contributes to loss.
        """
        states, actions, rewards, next_states, dones, valid_lens = (
            self._build_batch_tensors(chunks)
        )

        q_pred, q_target = self._compute_q_pred_and_target(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        # Build mask: padded prefix = False, real suffix = True
        # mask shape: [B, T]
        seq_length = states.shape[1]
        time_idx = torch.arange(seq_length, device=self.device).unsqueeze(
            0
        )  # [1, T]
        start_idx = (seq_length - valid_lens).unsqueeze(1)  # [B, 1]
        mask = time_idx >= start_idx  # [B, T]

        # Elementwise MSE, then reduce only over valid timesteps
        loss_per_timestep = F.mse_loss(
            q_pred, q_target, reduction="none"
        )  # [B, T]
        masked_loss = loss_per_timestep * mask.float()

        valid_count = mask.sum().clamp_min(1).float()
        loss = masked_loss.sum() / valid_count

        self.optimizer.zero_grad()
        loss.backward()

        if GRAD_CLIPPING:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

        self.optimizer.step()

        if DQN_WITH_TARGET or DOUBLE_DQN:
            self._soft_update_target()

        loss_value = float(loss.item())
        self._losses.append(loss_value)
        return loss_value

    def _soft_update_target(self) -> None:
        for target_param, param in zip(
            self.target_model.parameters(),
            self.model.parameters(),
        ):
            target_param.data.copy_(
                self._tau * param.data + (1.0 - self._tau) * target_param.data
            )

    def set_params(self, tau: float, gamma: float, batch_size: int) -> None:
        self._tau = tau
        self.log.info(f"Setting Tau to {tau}")

        self._gamma = gamma
        self.log.info(f"Setting the Discount/Gamma to {gamma}")

        self._batch_size = batch_size
        self.log.info(f"Setting Batch Size to {batch_size}")
