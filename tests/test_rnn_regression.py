import torch

from ai_hydra.constants.DHydra import DHydraLogDef
from ai_hydra.constants.DNNet import DNetDef, DRNN
from ai_hydra.nnet.Policy.RNNPolicy import RNNPolicy
from ai_hydra.nnet.RNNTrainer import RNNTrainer
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.models.RNNModel import RNNModel


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().cpu().clone())
        return torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)


class StubReplayMemory:
    def __init__(self, chunks: list[list[Transition]]) -> None:
        self._chunks = chunks

    def sample_chunks(self, batch_size: int) -> list[list[Transition]] | None:
        if len(self._chunks) < batch_size:
            return None
        return self._chunks[:batch_size]


def make_state(value: float) -> tuple[float, ...]:
    return (value,) * DNetDef.INPUT_SIZE


def make_chunk() -> list[Transition]:
    chunk: list[Transition] = []
    for idx in range(DRNN.SEQ_LENGTH):
        chunk.append(
            Transition(
                old_state=make_state(float(idx)),
                action=idx % 3,
                reward=float(idx % 2),
                new_state=make_state(float(idx + 1)),
                done=idx == (DRNN.SEQ_LENGTH - 1),
            )
        )
    return chunk


def test_rnn_policy_uses_rolling_sequence_window() -> None:
    model = RecordingModel()
    policy = RNNPolicy(model=model)

    action = policy.select_action(make_state(1.0))
    assert action == 1
    assert model.inputs[-1].shape == (1, DNetDef.INPUT_SIZE)

    policy.select_action(make_state(2.0))
    assert model.inputs[-1].shape == (2, DNetDef.INPUT_SIZE)

    for idx in range(DRNN.SEQ_LENGTH + 5):
        policy.select_action(make_state(float(idx)))

    assert model.inputs[-1].shape == (DRNN.SEQ_LENGTH, DNetDef.INPUT_SIZE)


def test_rnn_policy_reset_episode_clears_history() -> None:
    model = RecordingModel()
    policy = RNNPolicy(model=model)

    policy.select_action(make_state(1.0))
    policy.select_action(make_state(2.0))
    assert model.inputs[-1].shape == (2, DNetDef.INPUT_SIZE)

    policy.reset_episode()
    policy.select_action(make_state(3.0))
    assert model.inputs[-1].shape == (1, DNetDef.INPUT_SIZE)


def test_rnn_trainer_returns_model_to_eval_mode() -> None:
    model = RNNModel()
    replay = StubReplayMemory([make_chunk()])
    trainer = RNNTrainer(
        model=model,
        replay=replay,
        lr=0.001,
        log_level=DHydraLogDef.DEFAULT_LOG_LEVEL,
    )

    trainer.train_long_memory(batch_size=1)

    assert trainer.get_per_ep_loss() is not None
    assert trainer.model.training is False
