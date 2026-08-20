# focus.py
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd17B import MockD1Config


class FocusRoPE(nn.Module):
    """Rotary Positional Embeddings for Focus Attention Heads"""
    def __init__(self, dim: int, max_seq_len: int = 262144, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

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
        if offset + seq_len > self.cos_cached.shape[2]:
            self._build_cache(offset + seq_len)
        cos = self.cos_cached[:, :, offset : offset + seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :, offset : offset + seq_len, :].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class ChunkedFocusAttentionFunction(torch.autograd.Function):
    """
    Stage 5 Policy: O(C / Chunk_Size) Blockwise Parallel Scan.
    Memory Footprint: O(B_chunk * d_h^2) instead of O(C * d_h^2).
    """
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, scale: float, chunk_size: int):
        # q, k, v: [B, H, C, d_h], gamma: [1, H, 1, 1]
        B, H, C, d_h = q.shape
        num_chunks = (C + chunk_size - 1) // chunk_size

        # Precompute intra-chunk causal decay matrix D: [H, chunk_size, chunk_size]
        # D[h, i, j] = gamma[h]^(i - j) for i >= j else 0
        arange_b = torch.arange(chunk_size, device=q.device)
        dist = arange_b.unsqueeze(0) - arange_b.unsqueeze(1)  # [B, B]
        mask = dist >= 0
        gamma_s = gamma.squeeze(0).squeeze(-1)  # [H, 1]
        decay_weights = (gamma_s ** dist.unsqueeze(0).clamp(min=0)) * mask.unsqueeze(0).to(q.dtype)  # [H, B, B]

        A = torch.empty_like(v)
        # Checkpoints of carry state M at block boundaries: [num_chunks + 1, B, H, d_h, d_h]
        boundary_states = torch.zeros(num_chunks + 1, B, H, d_h, d_h, device=q.device, dtype=q.dtype)

        curr_state = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)
        boundary_states[0] = curr_state

        for c_idx in range(num_chunks):
            start = c_idx * chunk_size
            end = min(start + chunk_size, C)
            curr_len = end - start

            q_c = q[:, :, start:end]  # [B, H, L, d_h]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]

            # Intra-chunk raw outer products: [B, H, L, d_h, d_h]
            P_c = scale * torch.matmul(q_c.unsqueeze(-1), k_c.unsqueeze(-2))

            # 1. Past carry decay inside chunk: [B, H, L, 1, 1] * [B, H, 1, d_h, d_h]
            power = torch.arange(1, curr_len + 1, device=q.device).view(1, 1, curr_len, 1, 1)
            carry_decay = (gamma ** power) * curr_state.unsqueeze(2)  # [B, H, L, d_h, d_h]

            # 2. Intra-chunk accumulated contribution via batched matrix multiply
            # decay_c: [1, H, L, L] @ [B, H, L, d_h * d_h] -> [B, H, L, d_h, d_h]
            decay_c = decay_weights[:, :curr_len, :curr_len].unsqueeze(0)
            P_flat = P_c.view(B, H, curr_len, d_h * d_h)
            intra_accum = torch.matmul(decay_c, P_flat).view(B, H, curr_len, d_h, d_h)

            M_c = carry_decay + intra_accum  # [B, H, L, d_h, d_h]
            S_c = F.softmax(M_c, dim=-1)

            # Output A_c: [B, H, L, d_h]
            A[:, :, start:end] = torch.matmul(v_c.unsqueeze(-2), S_c).squeeze(-2)

            # Update boundary state for next chunk
            curr_state = M_c[:, :, -1]
            boundary_states[c_idx + 1] = curr_state

        ctx.save_for_backward(q, k, v, gamma, boundary_states)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, gamma, boundary_states = ctx.saved_tensors
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        B, H, C, d_h = q.shape
        num_chunks = (C + chunk_size - 1) // chunk_size

        grad_Q = torch.empty_like(q)
        grad_K = torch.empty_like(k)
        grad_V = torch.empty_like(v)
        grad_gamma = torch.zeros_like(gamma)

        # Precompute intra-chunk decay weights
        arange_b = torch.arange(chunk_size, device=q.device)
        dist = arange_b.unsqueeze(0) - arange_b.unsqueeze(1)
        mask = dist >= 0
        gamma_s = gamma.squeeze(0).squeeze(-1)
        decay_weights = (gamma_s ** dist.unsqueeze(0).clamp(min=0)) * mask.unsqueeze(0).to(q.dtype)

        curr_grad_carry = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)

        # Reverse block-by-block execution
        for c_idx in reversed(range(num_chunks)):
            start = c_idx * chunk_size
            end = min(start + chunk_size, C)
            curr_len = end - start

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            gA_c = grad_A[:, :, start:end]
            start_state = boundary_states[c_idx]

            # Recompute M_c on-the-fly for this chunk
            P_c = scale * torch.matmul(q_c.unsqueeze(-1), k_c.unsqueeze(-2))
            power = torch.arange(1, curr_len + 1, device=q.device).view(1, 1, curr_len, 1, 1)
            carry_decay = (gamma ** power) * start_state.unsqueeze(2)
            decay_c = decay_weights[:, :curr_len, :curr_len].unsqueeze(0)
            P_flat = P_c.view(B, H, curr_len, d_h * d_h)
            intra_accum = torch.matmul(decay_c, P_flat).view(B, H, curr_len, d_h, d_h)
            M_c = carry_decay + intra_accum
            S_c = F.softmax(M_c, dim=-1)

            # Grad V
            grad_V[:, :, start:end] = torch.matmul(gA_c.unsqueeze(-2), S_c.transpose(-1, -2)).squeeze(-2)

            # Softmax VJP for M_c
            grad_S_c = torch.matmul(v_c.unsqueeze(-1), gA_c.unsqueeze(-2))
            sum_grad_S = torch.sum(grad_S_c * S_c, dim=-1, keepdim=True)
            grad_M_c = S_c * (grad_S_c - sum_grad_S)

            # Add downstream carry gradient to the last position of the chunk
            grad_M_c[:, :, -1] += curr_grad_carry

            # Reverse suffix scan within chunk
            grad_P_c = torch.zeros_like(grad_M_c)
            curr_grad_P = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)

            for t in reversed(range(curr_len)):
                curr_grad_P = grad_M_c[:, :, t] + gamma * curr_grad_P
                grad_P_c[:, :, t] = curr_grad_P
                prev_M = start_state if t == 0 else M_c[:, :, t - 1]
                grad_gamma += torch.sum(curr_grad_P * prev_M, dim=(0, 2, 3), keepdim=True)

            # Carry gradient propagating to previous chunk
            curr_grad_carry = curr_grad_P * gamma

            # Grad Q and K for this chunk
            grad_Q[:, :, start:end] = scale * torch.matmul(k_c.unsqueeze(-2), grad_P_c.transpose(-1, -2)).squeeze(-2)
            grad_K[:, :, start:end] = scale * torch.matmul(q_c.unsqueeze(-2), grad_P_c).squeeze(-2)

        return grad_Q, grad_K, grad_V, grad_gamma, None, None


class FocusAttentionFunction(torch.autograd.Function):
    """Reference Small-Context Autograd (Stages 1-4)."""
    @staticmethod
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

        # Apply RoPE using the true sequence offset
        if self.use_rope:
            q = self.rope(q, seq_len=C, offset=seq_offset)
            k = self.rope(k, seq_len=C, offset=seq_offset)

        gamma = torch.exp(-torch.exp(self.decay_param))

        if state is None:
            # Parallel training path
            if (self.config.curriculum_stage == 5 or self.config.use_chunked_scan) and C > self.chunk_size:
                A = ChunkedFocusAttentionFunction.apply(q, k, v, gamma, self.scale, self.chunk_size)
            else:
                A = FocusAttentionFunction.apply(q, k, v, gamma, self.scale)
            next_state = None
        else:
            # Recurrent inference path
            delta_M = self.scale * torch.matmul(q.transpose(-1, -2), k)  # [B, H, d_h, d_h]
            next_state = gamma * state + delta_M
            S = F.softmax(next_state, dim=-1)
            A = torch.matmul(v, S)

        A = A.transpose(1, 2).contiguous().view(B, C, self.kqv_dim)
        return self.o_proj(A), next_state

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, S, gamma, M = ctx.saved_tensors
        scale = ctx.scale
        B, H, C, d_h = q.shape

        grad_V = torch.matmul(grad_A.unsqueeze(-2), S.transpose(-1, -2)).squeeze(-2)
        grad_S = torch.matmul(v.unsqueeze(-1), grad_A.unsqueeze(-2))
        sum_grad_S = torch.sum(grad_S * S, dim=-1, keepdim=True)
        grad_M = S * (grad_S - sum_grad_S)

        grad_P = torch.zeros_like(grad_M)
        curr_grad_P = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)
        grad_gamma = torch.zeros_like(gamma)

        for t in reversed(range(C)):
            curr_grad_P = grad_M[:, :, t] + gamma * curr_grad_P
            grad_P[:, :, t] = curr_grad_P
            if t > 0:
                grad_gamma += torch.sum(curr_grad_P * M[:, :, t - 1], dim=(0, 2, 3), keepdim=True)

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
        self.chunk_size = config.chunk_size

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
            # Policy Dispatcher: Stage 5 uses chunked blockwise scan
            if (self.config.curriculum_stage == 5 or self.config.use_chunked_scan) and C > self.chunk_size:
                A = ChunkedFocusAttentionFunction.apply(q, k, v, gamma, self.scale, self.chunk_size)
            else:
                A = FocusAttentionFunction.apply(q, k, v, gamma, self.scale)
            next_state = None
        else:
            # Step-by-step Autoregressive inference
            delta_M = self.scale * torch.matmul(q.transpose(-1, -2), k)
            next_state = gamma * state + delta_M
            S = F.softmax(next_state, dim=-1)
            A = torch.matmul(v, S)

        A = A.transpose(1, 2).contiguous().view(B, C, self.kqv_dim)
        return self.o_proj(A), next_state