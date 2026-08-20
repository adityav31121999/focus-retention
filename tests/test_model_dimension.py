"""
End-to-End 36-Layer Tensor Dimension Verification Suite
Focus-Retention Architecture (Mock-D1:7B)

Tests all 36 layers (9 blocks x (3 Focus + 1 Retention)) in:
1. Parallel Sequence Mode (Training & Loss Backprop)
2. Recurrent Autoregressive Mode (Inference Step-by-Step with MockD1StateCache)
3. Layer-by-layer intermediate tensor shapes
"""

import pytest
import torch
import torch.nn.functional as F

from mock_d1.configure_mockd1 import MockD1Config
from mock_d1.model_mock import MockD1ForCausalLM, MockD1Model
from mock_d1.block import MockD1Block
from mock_d1.focus import MockD1FocusAttention
from mock_d1.retention import MockD1RetentionMechanism
from mock_d1.feedforward import MockD1MLP, RMSNorm
from inference.cache import MockD1StateCache


def test_36_layer_model_dimensions():
    """Verify full 36-layer (9-block) model shapes during parallel forward and backward pass."""
    torch.manual_seed(42)
    B, C = 2, 16
    
    # Test with scaled-down dimensions for fast unit testing
    config = MockD1Config(
        vocab_size=1024,
        max_seq_len=2048,
        hidden_dim=256,
        kqv_dim=256,
        intermediate_dim=512,
        num_layers=36,
        num_blocks=9,
        focus_layers_per_block=3,
        retention_layers_per_block=1,
        focus_heads=4,
        retention_heads=1,
        retention_latent_dim=64,
        lora_deembed_rank=32,
    )

    model = MockD1ForCausalLM(config)
    assert len(model.model.blocks) == 9, f"Expected 9 blocks, got {len(model.model.blocks)}"

    input_ids = torch.randint(0, config.vocab_size, (B, C))
    labels = torch.randint(0, config.vocab_size, (B, C))

    # Forward pass
    logits, loss, next_states = model(input_ids=input_ids, labels=labels)

    # Assert Logits & Loss shapes
    assert logits.shape == (B, C, config.vocab_size), f"Expected logits {(B, C, config.vocab_size)}, got {logits.shape}"
    assert loss is not None and loss.numel() == 1, "Loss must be a scalar"
    assert next_states is None, "Parallel mode should return None for next_states"

    # Backward pass verification
    loss.backward()
    assert model.model.embed_tokens.weight.grad is not None, "Embedding gradient must not be None"
    print("[PASS] Full 36-Layer Parallel Forward & Backward Pass Passed!")


def test_layer_by_layer_block_dimensions():
    """Verify exact tensor transformations across a single MockD1Block (3 Focus + 1 Retention)."""
    torch.manual_seed(42)
    B, C = 2, 8
    config = MockD1Config(
        vocab_size=512,
        hidden_dim=128,
        kqv_dim=128,
        intermediate_dim=256,
        num_blocks=1,
        focus_heads=4,
        retention_latent_dim=32,
    )

    block = MockD1Block(config)
    x = torch.randn(B, C, config.hidden_dim)
    x0 = torch.randn(B, C, config.hidden_dim)

    # Parallel mode
    out, states = block(x, x0=x0, states=None)
    assert out.shape == (B, C, config.hidden_dim), f"Block output mismatch: {out.shape}"
    assert states is None

    # Recurrent mode step
    step_x = torch.randn(B, 1, config.hidden_dim)
    step_x0 = torch.randn(B, 1, config.hidden_dim)
    
    # Initialize cache states: 3 Focus tensors + 1 Retention (empty dict)
    head_dim = config.focus_head_dim
    initial_states = [
        torch.zeros(B, config.focus_heads, head_dim, head_dim),
        torch.zeros(B, config.focus_heads, head_dim, head_dim),
        torch.zeros(B, config.focus_heads, head_dim, head_dim),
        {}  # Retention state dictionary
    ]

    out_step1, new_states1 = block(step_x, x0=step_x0, states=initial_states)
    assert out_step1.shape == (B, 1, config.hidden_dim)
    assert len(new_states1) == 4
    assert new_states1[0].shape == (B, config.focus_heads, head_dim, head_dim)
    assert isinstance(new_states1[3], dict)
    assert "l_kv_cache" in new_states1[3] and "running_S" in new_states1[3]
    assert new_states1[3]["l_kv_cache"].shape == (B, 1, config.retention_latent_dim)

    # Step 2 with accumulated cache
    out_step2, new_states2 = block(step_x, x0=step_x0, states=new_states1)
    assert out_step2.shape == (B, 1, config.hidden_dim)
    assert new_states2[3]["l_kv_cache"].shape == (B, 2, config.retention_latent_dim)
    assert new_states2[3]["running_S"].shape == (B, 1, 2)
    print("[PASS] Block Layer-by-Layer Dimension Assertions Passed!")


def test_autoregressive_state_cache_generation():
    """Verify MockD1StateCache autoregressive inference step decoding across all 36 layers."""
    torch.manual_seed(42)
    B = 1
    config = MockD1Config(
        vocab_size=256,
        max_seq_len=512,
        hidden_dim=64,
        kqv_dim=64,
        intermediate_dim=128,
        num_layers=36,
        num_blocks=9,
        focus_heads=2,
        retention_latent_dim=16,
        lora_deembed_rank=8,
    )

    model = MockD1ForCausalLM(config).eval()
    cache = MockD1StateCache(config, batch_size=B)
    assert len(cache.states) == 9

    # Generate 5 tokens autoregressively
    curr_token = torch.tensor([[42]], dtype=torch.long)
    for step in range(5):
        logits, _, new_states = model(curr_token, past_states=cache.states)
        assert logits.shape == (B, 1, config.vocab_size)
        cache.update(new_states)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_token = next_token

    print("[PASS] 36-Layer Autoregressive State Caching Verification Passed!")


if __name__ == "__main__":
    test_36_layer_model_dimensions()
    test_layer_by_layer_block_dimensions()
    test_autoregressive_state_cache_generation()
