import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1 import MockD1Config


class FocusAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        # q, k, v: [B, H, C, d_h]
        # 1. Compute M = (Q^T @ K) * scale -> [B, H, d_h, d_h]
        M = torch.matmul(q.transpose(-1, -2), k) * scale
        
        # 2. Row-wise Softmax: S = softmax(M, dim=-1)
        S = F.softmax(M, dim=-1)
        
        # 3. Output: A = V @ S -> [B, H, C, d_h]
        A = torch.matmul(v, S)
        
        ctx.save_for_backward(q, k, v, S)
        ctx.scale = scale
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, S = ctx.saved_tensors
        scale = ctx.scale

        # 1. Grad w.r.t V: [B, H, C, d_h] @ [B, H, d_h, d_h]^T -> [B, H, C, d_h]
        grad_V = torch.matmul(grad_A, S.transpose(-1, -2))

        # 2. Grad w.r.t S: [B, H, d_h, C] @ [B, H, C, d_h] -> [B, H, d_h, d_h]
        grad_S = torch.matmul(v.transpose(-1, -2), grad_A)

        # 3. Softmax VJP: grad_M = S * (grad_S - sum(grad_S * S, dim=-1, keepdim=True))
        sum_grad_S_S = torch.sum(grad_S * S, dim=-1, keepdim=True)
        grad_M = S * (grad_S - sum_grad_S_S)

        # 4. Grad w.r.t Q and K
        # grad_Q = scale * (K @ grad_M^T)
        grad_Q = torch.matmul(k, grad_M.transpose(-1, -2)) * scale
        # grad_K = scale * (Q @ grad_M)
        grad_K = torch.matmul(q, grad_M) * scale

        return grad_Q, grad_K, grad_V, None


class MockD1FocusAttention(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.num_heads = config.focus_heads
        self.head_dim = config.focus_head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.o_proj = nn.Linear(self.kqv_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, _ = x.shape

        q = self.q_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        if state is None:
            # Analytical manual backward via FocusAttentionFunction
            A = FocusAttentionFunction.apply(q, k, v, self.scale)
            next_state = None
        else:
            # Recurrent step inference mode
            delta_M = torch.matmul(q.transpose(-1, -2), k) * self.scale
            next_state = state + delta_M
            S = F.softmax(next_state, dim=-1)
            A = torch.matmul(v, S)

        A = A.transpose(1, 2).contiguous().view(B, C, self.kqv_dim)
        return self.o_proj(A), next_state