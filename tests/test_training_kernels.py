import copy
import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mock_d1 import MockD1Config, MockD1ForCausalLM
from mock_d1.focus import ChunkedFocusAttentionFunction
from mock_d1.retention import ChunkedRetentionFunction, MockD1RetentionMechanism
from engine.optimiser import Adafactor
from scripts.training_common import PackedTokens, RankSampler


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def focus_reference(q, k, v, gamma, scale):
    state = torch.zeros_like(q[:, :, 0, :, None] * k[:, :, 0, None, :])
    outputs = []
    for t in range(q.shape[2]):
        state = gamma * state + scale * q[:, :, t, :, None] * k[:, :, t, None, :]
        outputs.append((v[:, :, t, None, :] @ state.softmax(-1)).squeeze(-2))
    return torch.stack(outputs, dim=2)


def retention_reference(q, k, v, scale, activation):
    score = (q @ k.transpose(-1, -2)) * scale
    mask = torch.ones_like(score, dtype=torch.bool).tril()
    phi = {"silu": F.silu, "relu": F.relu, "gelu": F.gelu}[activation](score)
    cumulative = phi.masked_fill(~mask, 0).cumsum(-2)
    return cumulative.masked_fill(~mask, -torch.inf).softmax(-1) @ v


def check_gradients(actual, expected, inputs):
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    probe = torch.randn_like(actual)
    observed = torch.autograd.grad(actual, inputs, probe, retain_graph=True)
    reference = torch.autograd.grad(expected, inputs, probe)
    for left, right in zip(observed, reference):
        torch.testing.assert_close(left, right, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("length,chunk", [(1, 1), (3, 8), (8, 4), (9, 4)])
def test_focus_chunked_matches_independent_recurrence(length, chunk):
    torch.manual_seed(123)
    q, k, v = [torch.randn(2, 2, length, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    gamma = torch.tensor([0.3, 0.91], dtype=torch.float64).view(1, 2, 1, 1).requires_grad_()
    actual = ChunkedFocusAttentionFunction.apply(q, k, v, gamma, 0.5, chunk)
    expected = focus_reference(q, k, v, gamma, 0.5)
    check_gradients(actual, expected, (q, k, v, gamma))


def test_focus_future_tokens_cannot_change_prefix():
    torch.manual_seed(12)
    q, k, v = [torch.randn(1, 2, 9, 3) for _ in range(3)]
    gamma = torch.full((1, 2, 1, 1), 0.8)
    original = ChunkedFocusAttentionFunction.apply(q, k, v, gamma, 0.5, 4)
    for tensor in (q, k, v):
        tensor[:, :, 2:] += 7
    changed = ChunkedFocusAttentionFunction.apply(q, k, v, gamma, 0.5, 4)
    torch.testing.assert_close(original[:, :, :2], changed[:, :, :2])


@pytest.mark.parametrize("activation", ["silu", "relu", "gelu"])
@pytest.mark.parametrize("length,chunk", [(3, 8), (8, 4), (9, 4)])
def test_retention_boundary_reconstruction_gradients(length, chunk, activation):
    torch.manual_seed(456)
    q, k, v = [torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    actual = ChunkedRetentionFunction.apply(q, k, v, 0.5, activation, chunk)
    expected = retention_reference(q, k, v, 0.5, activation)
    check_gradients(actual, expected, (q, k, v))


def tiny_config():
    return MockD1Config(vocab_size=32, hidden_dim=8, kqv_dim=8, intermediate_dim=16,
                        num_blocks=1, num_layers=4, focus_heads=2, retention_latent_dim=3,
                        lora_deembed_rank=2, chunk_size=3)


def test_retention_latent_reassociation_preserves_all_gradients():
    torch.manual_seed(789)
    module = MockD1RetentionMechanism(tiny_config()).double()
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    latent = module.w_kv_latent(x)
    expected = module.o_proj(retention_reference(module.q_proj(x), module.w_k_expand(latent),
                                                 module.w_v_expand(latent), module.scale, "silu"))
    actual, _ = module(x)
    check_gradients(actual, expected, (x, *module.parameters()))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_loss_only_matches_full_projection_and_parameter_gradients(checkpointing):
    torch.manual_seed(33)
    model = MockD1ForCausalLM(tiny_config())
    if checkpointing:
        model.gradient_checkpointing_enable()
    other = copy.deepcopy(model)
    ids = torch.randint(0, 32, (2, 7))
    targets = ids.clone()
    targets[0, 3] = -100
    logits, _, _ = other(ids)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 32), targets[:, 1:].reshape(-1))
    no_logits, actual, _ = model(ids, labels=targets, return_logits=False, loss_chunk_size=4)
    assert no_logits is None
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for p, reference in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, reference.grad, atol=2e-6, rtol=2e-4)


def test_adafactor_updates_factored_moments_and_matches_numpy():
    initial = np.array([[1., 2., 3.], [2., -1., 0.5]], dtype=np.float32)
    parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    optimizer = Adafactor([parameter], lr=0.01, scale_parameter=False, weight_decay=0.1)
    expected = initial.copy()
    row, col = np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)
    for step, gradient in enumerate([np.array([[1., 2., -3.], [4., 5., 6.]], dtype=np.float32),
                                      np.array([[3., -2., 1.], [-1., 4., 2.]], dtype=np.float32)], 1):
        parameter.grad = torch.from_numpy(gradient.copy())
        optimizer.step()
        beta = 1 - step ** -0.8
        row = beta * row + (1 - beta) * (gradient ** 2).mean(-1)
        col = beta * col + (1 - beta) * (gradient ** 2).mean(-2)
        update = gradient / np.sqrt((row / row.mean())[:, None] * col[None, :])
        update /= max(1, float(np.sqrt((update ** 2).mean())))
        expected = expected * (1 - 0.01 * 0.1) - 0.01 * update
        np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(optimizer.state[parameter]["exp_avg_sq_row"]) == 2


def test_rank_partitions_and_resume_offsets():
    streams = [list(itertools.islice(iter(RankSampler(40, rank, 2, 2)), 20)) for rank in range(2)]
    assert set(streams[0]).isdisjoint(streams[1])
    resumed = list(itertools.islice(iter(RankSampler(40, 1, 2, 2, consumed=6)), 10))
    assert resumed == streams[1][6:16]


def test_packed_tokens(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype="<u4").tofile(path)
    dataset = PackedTokens(path, 8, 32)
    torch.testing.assert_close(dataset[2], torch.arange(16, 24))
    assert len(dataset) == 4


def test_focus_operation_count_does_not_grow_per_token():
    # This guards against reintroducing Python token loops in either direction.
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.operations += 1
            return func(*args, **(kwargs or {}))

    counts = []
    for length in (8, 64):
        q, k, v = [torch.randn(1, 2, length, 3, requires_grad=True) for _ in range(3)]
        gamma = torch.full((1, 2, 1, 1), 0.8, requires_grad=True)
        with Count() as counter:
            ChunkedFocusAttentionFunction.apply(q, k, v, gamma, 0.5, length).sum().backward()
        counts.append(counter.operations)
    assert counts[1] <= counts[0] * 1.2


def test_xla_execution_boundary_preserves_gradient(monkeypatch):
    import sys
    import types
    from mock_d1.model_mock import _XLAGraphBoundary
    calls = []
    fake = types.ModuleType("torch_xla.core.xla_model")
    fake.mark_step = lambda: calls.append("execute")
    monkeypatch.setitem(sys.modules, "torch_xla", types.ModuleType("torch_xla"))
    monkeypatch.setitem(sys.modules, "torch_xla.core", types.ModuleType("torch_xla.core"))
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", fake)
    x = torch.tensor([2., 3.], requires_grad=True)
    (_XLAGraphBoundary.apply(x.square()) * 3).sum().backward()
    torch.testing.assert_close(x.grad, 6 * x.detach())
    assert calls == ["execute", "execute"]
