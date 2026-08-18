import pytest
import torch
from mock_d1.focus import FocusAttentionFunction
from mock_d1.retention import RetentionFunction

def test_focus_attention_gradcheck():
    torch.manual_seed(42)
    # Small double-precision tensors for gradcheck
    B, H, C, d_h = 2, 4, 8, 16
    scale = 1.0 / (d_h ** 0.5)

    q = torch.randn(B, H, C, d_h, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H, C, d_h, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H, C, d_h, dtype=torch.float64, requires_grad=True)

    # PyTorch gradcheck tests analytical vs numerical gradients
    passed = torch.autograd.gradcheck(FocusAttentionFunction.apply, (q, k, v, scale), eps=1e-6, atol=1e-4)
    assert passed, "Focus Attention analytical gradients failed gradcheck!"
    print("✅ Focus Attention Gradcheck Passed!")

def test_retention_gradcheck():
    torch.manual_seed(42)
    B, C, D = 2, 8, 16
    scale = 1.0 / (D ** 0.5)

    q = torch.randn(B, C, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, C, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, C, D, dtype=torch.float64, requires_grad=True)

    passed = torch.autograd.gradcheck(RetentionFunction.apply, (q, k, v, scale, "silu"), eps=1e-6, atol=1e-4)
    assert passed, "Retention analytical gradients failed gradcheck!"
    print("✅ Retention Gradcheck Passed!")

if __name__ == "__main__":
    test_focus_attention_gradcheck()
    test_retention_gradcheck()