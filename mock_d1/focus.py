# focus.py
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1_mini import MockD1Config


class FocusRoPE(nn.Module):
    """
    Dynamic on-demand Rotary Positional Embeddings.
    Allocates an initial lightweight cache (2,048 tokens) and dynamically grows,
    preventing multi-gigabyte static VRAM allocation during model initialization.
    """
    def __init__(self, dim: int, max_seq_len: int = 262144, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(2048)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        if self.cos_cached.numel() == 0 or (offset + seq_len > self.cos_cached.shape[2]):
            self._build_cache(max(offset + seq_len, 2048))
        cos = self.cos_cached[:, :, offset : offset + seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :, offset : offset + seq_len, :].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class ChunkedFocusAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, scale: float, chunk_size: int):
        dtype = q.dtype
        gamma = gamma.to(dtype)
        B, H, C, d_h = q.shape
        num_chunks = (C + chunk_size - 1) // chunk_size

        arange_b = torch.arange(chunk_size, device=q.device)
        dist = arange_b.unsqueeze(0) - arange_b.unsqueeze(1)
        mask = dist >= 0
        
        # 1. Shape gamma for intra-chunk decay: [H, 1, 1] vs [1, chunk_size, chunk_size]
        gamma_s = gamma.view(H, 1, 1)
        decay_weights = (gamma_s ** dist.unsqueeze(0).clamp(min=0)) * mask.unsqueeze(0).to(dtype)

        # 2. Shape gamma for inter-chunk carry: [1, H, 1, 1, 1] (5D)
        gamma_5d = gamma.view(1, H, 1, 1, 1)

        A = torch.empty_like(v)
        boundary_states = torch.zeros(num_chunks + 1, B, H, d_h, d_h, device=q.device, dtype=dtype)
        curr_state = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=dtype)
        boundary_states[0] = curr_state

        for c_idx in range(num_chunks):
            start = c_idx * chunk_size
            end = min(start + chunk_size, C)
            curr_len = end - start

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]

            P_c = scale * torch.matmul(q_c.unsqueeze(-1), k_c.unsqueeze(-2))
            
            decay_c = decay_weights[:, :curr_len, :curr_len].unsqueeze(0)
            P_flat = P_c.view(B, H, curr_len, d_h * d_h)
            
            # 1. Allocate intra-chunk accumulation directly in M_c
            M_c = torch.matmul(decay_c, P_flat).view(B, H, curr_len, d_h, d_h)
            del P_flat, P_c

            # 2. Add carry decay in-place (avoids allocating temporary 5D tensors)
            power = torch.arange(1, curr_len + 1, device=q.device).view(1, 1, curr_len, 1, 1)
            M_c.add_((gamma_5d ** power) * curr_state.unsqueeze(2))

            S_c = F.softmax(M_c, dim=-1).to(dtype)
            A[:, :, start:end] = torch.matmul(v_c.unsqueeze(-2), S_c).squeeze(-2)

            curr_state = M_c[:, :, -1]
            boundary_states[c_idx + 1] = curr_state

        ctx.save_for_backward(q, k, v, gamma, boundary_states)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, gamma, boundary_states = ctx.saved_tensors
        dtype = q.dtype
        grad_A = grad_A.to(dtype)
        gamma = gamma.to(dtype)
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        B, H, C, d_h = q.shape
        num_chunks = (C + chunk_size - 1) // chunk_size

        grad_Q = torch.empty_like(q)
        grad_K = torch.empty_like(k)
        grad_V = torch.empty_like(v)
        grad_gamma = torch.zeros_like(gamma)

        arange_b = torch.arange(chunk_size, device=q.device)
        dist = arange_b.unsqueeze(0) - arange_b.unsqueeze(1)
        mask = dist >= 0
        
        gamma_s = gamma.view(H, 1, 1)
        decay_weights = (gamma_s ** dist.unsqueeze(0).clamp(min=0)) * mask.unsqueeze(0).to(dtype)
        gamma_5d = gamma.view(1, H, 1, 1, 1)

        curr_grad_carry = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=dtype)

        for c_idx in reversed(range(num_chunks)):
            start = c_idx * chunk_size
            end = min(start + chunk_size, C)
            curr_len = end - start

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            gA_c = grad_A[:, :, start:end]
            start_state = boundary_states[c_idx]

            P_c = scale * torch.matmul(q_c.unsqueeze(-1), k_c.unsqueeze(-2))
            decay_c = decay_weights[:, :curr_len, :curr_len].unsqueeze(0)
            P_flat = P_c.view(B, H, curr_len, d_h * d_h)
            
            # Reconstruct M_c in-place
            M_c = torch.matmul(decay_c, P_flat).view(B, H, curr_len, d_h, d_h)
            del P_flat, P_c

            power = torch.arange(1, curr_len + 1, device=q.device).view(1, 1, curr_len, 1, 1)
            M_c.add_((gamma_5d ** power) * start_state.unsqueeze(2))
            
            S_c = F.softmax(M_c, dim=-1).to(dtype)

            # Gradient V
            grad_V[:, :, start:end] = torch.matmul(gA_c.unsqueeze(-2), S_c.transpose(-1, -2)).squeeze(-2)

            # Softmax VJP
            grad_S_c = torch.matmul(v_c.unsqueeze(-1), gA_c.unsqueeze(-2))
            sum_grad_S = torch.sum(grad_S_c * S_c, dim=-1, keepdim=True)
            grad_M_c = S_c * (grad_S_c - sum_grad_S)

            del grad_S_c, sum_grad_S

            grad_M_c[:, :, -1] += curr_grad_carry

            grad_P_c = torch.zeros_like(grad_M_c)
            curr_grad_P = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=dtype)

            for t in reversed(range(curr_len)):
                curr_grad_P = grad_M_c[:, :, t] + gamma * curr_grad_P
                grad_P_c[:, :, t] = curr_grad_P
                prev_M = start_state if t == 0 else M_c[:, :, t - 1]
                grad_gamma += torch.sum(curr_grad_P * prev_M, dim=(0, 2, 3), keepdim=True)

            del grad_M_c, M_c

            curr_grad_carry = curr_grad_P * gamma

            grad_Q[:, :, start:end] = scale * torch.matmul(k_c.unsqueeze(-2), grad_P_c.transpose(-1, -2)).squeeze(-2)
            grad_K[:, :, start:end] = scale * torch.matmul(q_c.unsqueeze(-2), grad_P_c).squeeze(-2)

        return grad_Q, grad_K, grad_V, grad_gamma, None, None


class FocusAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, scale: float):
        dtype = q.dtype
        gamma = gamma.to(dtype)
        B, H, C, d_h = q.shape

        M = torch.zeros(B, H, C, d_h, d_h, device=q.device, dtype=dtype)
        S = torch.zeros(B, H, C, d_h, d_h, device=q.device, dtype=dtype)
        A = torch.zeros(B, H, C, d_h, device=q.device, dtype=dtype)

        curr_M = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=dtype)

        for t in range(C):
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            v_t = v[:, :, t]

            delta_M = scale * torch.matmul(q_t.unsqueeze(-1), k_t.unsqueeze(-2))
            curr_M = gamma * curr_M + delta_M
            curr_S = F.softmax(curr_M, dim=-1).to(dtype)
            curr_A = torch.matmul(v_t.unsqueeze(-2), curr_S).squeeze(-2)

            M[:, :, t] = curr_M
            S[:, :, t] = curr_S
            A[:, :, t] = curr_A

        ctx.save_for_backward(q, k, v, S, gamma, M)
        ctx.scale = scale
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, S, gamma, M = ctx.saved_tensors
        dtype = q.dtype
        grad_A = grad_A.to(dtype)
        gamma = gamma.to(dtype)
        S = S.to(dtype)
        scale = ctx.scale
        B, H, C, d_h = q.shape

        grad_V = torch.matmul(grad_A.unsqueeze(-2), S.transpose(-1, -2)).squeeze(-2)
        grad_S = torch.matmul(v.unsqueeze(-1), grad_A.unsqueeze(-2))
        sum_grad_S = torch.sum(grad_S * S, dim=-1, keepdim=True)
        grad_M = S * (grad_S - sum_grad_S)

        del grad_S, sum_grad_S

        grad_P = torch.zeros_like(grad_M)
        curr_grad_P = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=dtype)
        grad_gamma = torch.zeros_like(gamma)

        for t in reversed(range(C)):
            curr_grad_P = grad_M[:, :, t] + gamma * curr_grad_P
            grad_P[:, :, t] = curr_grad_P
            if t > 0:
                grad_gamma += torch.sum(curr_grad_P * M[:, :, t - 1], dim=(0, 2, 3), keepdim=True)

        del grad_M

        grad_Q = scale * torch.matmul(k.unsqueeze(-2), grad_P.transpose(-1, -2)).squeeze(-2)
        grad_K = scale * torch.matmul(q.unsqueeze(-2), grad_P).squeeze(-2)
        return grad_Q, grad_K, grad_V, grad_gamma, None


class MockD1FocusAttention(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.num_heads = config.focus_heads
        self.head_dim = config.focus_head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = config.use_rope
        self.use_decay = config.use_focus_decay
        self.chunk_size = getattr(config, "chunk_size", 128)

        self.q_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.o_proj = nn.Linear(self.kqv_dim, self.hidden_dim, bias=False)

        if self.use_rope:
            self.rope = FocusRoPE(self.head_dim, max_seq_len=config.max_seq_len, base=config.rope_base)

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
            if C >= self.chunk_size:
                A = ChunkedFocusAttentionFunction.apply(q, k, v, gamma, self.scale, self.chunk_size)
            else:
                A = FocusAttentionFunction.apply(q, k, v, gamma, self.scale)
            next_state = None
        else:
            delta_M = self.scale * torch.matmul(q.transpose(-1, -2), k)
            next_state = gamma * state + delta_M
            S = F.softmax(next_state, dim=-1).to(q.dtype)
            A = torch.matmul(v, S)

        A = A.transpose(1, 2).contiguous().view(B, C, self.kqv_dim)
        return self.o_proj(A), next_state