import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1 import MockD1Config


class FocusRoPE(nn.Module):
    """Rotary Positional Embeddings for Focus Attention Heads"""
    def __init__(self, dim: int, max_seq_len: int = 262144, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # inv_freq: [dim // 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)  # [1, 1, seq_len, dim]
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        # x: [B, H, C, d_h]
        if offset + seq_len > self.cos_cached.shape[2]:
            self._build_cache(offset + seq_len)
        cos = self.cos_cached[:, :, offset : offset + seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :, offset : offset + seq_len, :].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class FocusAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, scale: float):
        # q, k, v: [B, H, C, d_h], gamma: [1, H, 1, 1]
        B, H, C, d_h = q.shape

        # Compute position-wise outer products P_t = scale * (q_t^T @ k_t): [B, H, C, d_h, d_h]
        # q: [B, H, C, d_h, 1], k: [B, H, C, 1, d_h] -> [B, H, C, d_h, d_h]
        P = scale * torch.matmul(q.unsqueeze(-1), k.unsqueeze(-2))  # [B, H, C, d_h, d_h]
        
        # Accumulate decayed M_t across time: M_t = gamma * M_{t-1} + P_t
        M = torch.zeros(B, H, C, d_h, d_h, device=q.device, dtype=q.dtype)
        curr_M = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)
        for t in range(C):
            curr_M = gamma * curr_M + P[:, :, t]
            M[:, :, t] = curr_M

        # S_t = softmax(M_t, dim=-1)
        S = F.softmax(M, dim=-1)  # [B, H, C, d_h, d_h]

        # A_t = v_t @ S_t: [B, H, C, 1, d_h] @ [B, H, C, d_h, d_h] -> [B, H, C, d_h]
        A = torch.matmul(v.unsqueeze(-2), S).squeeze(-2)

        ctx.save_for_backward(q, k, v, S, gamma, M)
        ctx.scale = scale
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, S, gamma, M = ctx.saved_tensors
        scale = ctx.scale
        B, H, C, d_h = q.shape

        # 1. Grad w.r.t V
        grad_V = torch.matmul(grad_A.unsqueeze(-2), S.transpose(-1, -2)).squeeze(-2)  # [B, H, C, d_h]

        # 2. Grad w.r.t S
        grad_S = torch.matmul(v.unsqueeze(-1), grad_A.unsqueeze(-2))  # [B, H, C, d_h, d_h]

        # 3. Softmax VJP for M_t
        sum_grad_S = torch.sum(grad_S * S, dim=-1, keepdim=True)
        grad_M = S * (grad_S - sum_grad_S)  # [B, H, C, d_h, d_h]

        # 4. Reverse discounted suffix scan: grad_P_t = grad_M_t + gamma * grad_P_{t+1}
        grad_P = torch.zeros_like(grad_M)
        curr_grad_P = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)
        grad_gamma = torch.zeros_like(gamma)

        for t in reversed(range(C)):
            curr_grad_P = grad_M[:, :, t] + gamma * curr_grad_P
            grad_P[:, :, t] = curr_grad_P
            if t > 0:
                grad_gamma += torch.sum(curr_grad_P * M[:, :, t - 1], dim=(0, 2, 3), keepdim=True)

        # 5. Grad w.r.t Q and K
        grad_Q = scale * torch.matmul(k.unsqueeze(-2), grad_P.transpose(-1, -2)).squeeze(-2)
        grad_K = scale * torch.matmul(q.unsqueeze(-2), grad_P).squeeze(-2)

        return grad_Q, grad_K, grad_V, grad_gamma, None


class MockD1FocusAttention(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.num_heads = config.focus_heads
        self.head_dim = config.focus_head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = config.use_rope
        self.use_decay = config.use_focus_decay

        self.q_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.o_proj = nn.Linear(self.kqv_dim, self.hidden_dim, bias=False)

        # Rotary Embeddings
        if self.use_rope:
            self.rope = FocusRoPE(self.head_dim, max_seq_len=config.max_seq_len, base=config.rope_base)

        # Learnable log-space decay per head: gamma = exp(-exp(w)) in (0, 1)
        if self.use_decay:
            self.decay_param = nn.Parameter(torch.linspace(-5.0, -1.0, self.num_heads).view(1, self.num_heads, 1, 1))
        else:
            self.register_buffer("decay_param", torch.full((1, self.num_heads, 1, 1), -100.0))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        seq_offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, _ = x.shape

        q = self.q_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, seq_len=C, offset=seq_offset)
            k = self.rope(k, seq_len=C, offset=seq_offset)

        gamma = torch.exp(-torch.exp(self.decay_param))

        if state is None:
            # Parallel Training Mode via Decayed Autograd Function
            A = FocusAttentionFunction.apply(q, k, v, gamma, self.scale)
            next_state = None
        else:
            # Recurrent Autoregressive Inference: M_t = gamma * M_{t-1} + (q_t^T @ k_t) / sqrt(d)
            delta_M = self.scale * torch.matmul(q.transpose(-1, -2), k)  # [B, H, d_h, d_h]
            next_state = gamma * state + delta_M
            S = F.softmax(next_state, dim=-1)
            A = torch.matmul(v, S)

        A = A.transpose(1, 2).contiguous().view(B, C, self.kqv_dim)
        return self.o_proj(A), next_state