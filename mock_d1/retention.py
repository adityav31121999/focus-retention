# retention.py
import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1_mini import MockD1Config


def _phi(score: torch.Tensor, phi_act: str) -> torch.Tensor:
    if phi_act == "silu":
        return F.silu(score)
    elif phi_act == "relu":
        return F.relu(score)
    elif phi_act == "gelu":
        return F.gelu(score)
    return score


def _dphi(score: torch.Tensor, phi_act: str) -> torch.Tensor:
    if phi_act == "silu":
        sig = torch.sigmoid(score)
        return sig * (1.0 + score * (1.0 - sig))
    elif phi_act == "relu":
        return (score > 0).to(score.dtype)
    elif phi_act == "gelu":
        return 0.5 * (1.0 + torch.erf(score / math.sqrt(2.0))) + \
               (score / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * score.pow(2))
    return torch.ones_like(score)


class ChunkedRetentionFunction(torch.autograd.Function):
    """
    Stage 5 Policy: Chunked Retention without materializing O(C^2) matrices in VRAM.
    """
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, phi_act: str, chunk_size: int):
        dtype = q.dtype
        B, C, D = q.shape
        num_chunks = (C + chunk_size - 1) // chunk_size
        A = torch.empty_like(v)

        running_prefix_S = torch.empty(0, device=q.device, dtype=dtype)

        for c_idx in range(num_chunks):
            start = c_idx * chunk_size
            end = min(start + chunk_size, C)
            curr_len = end - start

            q_c = q[:, start:end]
            k_c = k[:, :end]
            v_c = v[:, :end]

            causal_mask = torch.tril(torch.ones(curr_len, end, device=q.device, dtype=torch.bool), diagonal=start)
            score_c = torch.matmul(q_c, k_c.transpose(-1, -2)) * scale
            phi_c = _phi(score_c, phi_act).masked_fill(~causal_mask, 0.0)

            S_c = torch.cumsum(phi_c, dim=-2)

            if c_idx > 0:
                S_c[:, :, :start] += running_prefix_S

            running_prefix_S = S_c[:, -1:, :end]

            S_c_masked = S_c.masked_fill(~causal_mask, -1e4 if dtype == torch.float16 else float("-inf"))
            alpha_c = F.softmax(S_c_masked, dim=-1).to(dtype)

            A[:, start:end] = torch.matmul(alpha_c, v_c)

        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.phi_act = phi_act
        ctx.chunk_size = chunk_size
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v = ctx.saved_tensors
        dtype = q.dtype
        grad_A = grad_A.to(dtype)
        scale = ctx.scale
        phi_act = ctx.phi_act
        chunk_size = ctx.chunk_size
        B, C, D = q.shape
        num_chunks = (C + chunk_size - 1) // chunk_size

        grad_Q = torch.zeros_like(q)
        grad_K = torch.zeros_like(k)
        grad_V = torch.zeros_like(v)

        running_grad_prefix = torch.zeros(B, 1, C, device=q.device, dtype=dtype)

        for c_idx in reversed(range(num_chunks)):
            start = c_idx * chunk_size
            end = min(start + chunk_size, C)
            curr_len = end - start

            q_c = q[:, start:end]
            k_c = k[:, :end]
            v_c = v[:, :end]
            gA_c = grad_A[:, start:end]

            causal_mask = torch.tril(torch.ones(curr_len, end, device=q.device, dtype=torch.bool), diagonal=start)
            score_c = torch.matmul(q_c, k_c.transpose(-1, -2)) * scale
            phi_c = _phi(score_c, phi_act).masked_fill(~causal_mask, 0.0)

            if start > 0:
                full_q_prefix = q[:, :start]
                full_k_prefix = k[:, :start]
                full_score_prefix = torch.matmul(full_q_prefix, full_k_prefix.transpose(-1, -2)) * scale
                full_mask_prefix = torch.tril(torch.ones(start, start, device=q.device, dtype=torch.bool))
                full_phi_prefix = _phi(full_score_prefix, phi_act).masked_fill(~full_mask_prefix, 0.0)
                prefix_sum = torch.sum(full_phi_prefix, dim=-2, keepdim=True)
                S_c = torch.cumsum(phi_c, dim=-2)
                S_c[:, :, :start] += prefix_sum
            else:
                S_c = torch.cumsum(phi_c, dim=-2)

            alpha_c = F.softmax(S_c.masked_fill(~causal_mask, -1e4 if dtype == torch.float16 else float("-inf")), dim=-1).to(dtype)

            # 1. Grad V
            grad_V[:, :end] += torch.matmul(alpha_c.transpose(-1, -2), gA_c)

            # 2. Grad alpha & Softmax VJP
            grad_alpha_c = torch.matmul(gA_c, v_c.transpose(-1, -2))
            sum_grad_alpha = torch.sum(grad_alpha_c * alpha_c, dim=-1, keepdim=True)
            grad_S_c = torch.where(causal_mask, alpha_c * (grad_alpha_c - sum_grad_alpha), torch.zeros_like(alpha_c))

            # 3. Reverse suffix scan
            grad_phi_c = torch.flip(torch.cumsum(torch.flip(grad_S_c, dims=[-2]), dim=-2), dims=[-2])
            if c_idx < num_chunks - 1:
                grad_phi_c += running_grad_prefix[:, :, :end]

            grad_phi_c = torch.where(causal_mask, grad_phi_c, torch.zeros_like(grad_phi_c))
            running_grad_prefix[:, :, :end] += torch.sum(grad_S_c, dim=-2, keepdim=True)

            # 4. Score gradient
            grad_score_c = grad_phi_c * _dphi(score_c, phi_act).to(dtype)

            # 5. Grad Q and K
            grad_Q[:, start:end] += torch.matmul(grad_score_c, k_c) * scale
            grad_K[:, :end] += torch.matmul(grad_score_c.transpose(-1, -2), q_c) * scale

        return grad_Q, grad_K, grad_V, None, None, None


class RetentionFunction(torch.autograd.Function):
    """Reference Small-Context Retention (Stages 1-4)."""
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, phi_act: str):
        dtype = q.dtype
        B, C, _ = q.shape
        causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
        score = torch.matmul(q, k.transpose(-1, -2)) * scale
        phi_score = _phi(score, phi_act)
        phi_masked = phi_score.masked_fill(~causal_mask, 0.0)
        S = torch.cumsum(phi_masked, dim=-2)
        S_masked = S.masked_fill(~causal_mask, -1e4 if dtype == torch.float16 else float("-inf"))
        alpha = F.softmax(S_masked, dim=-1).to(dtype)
        A = torch.matmul(alpha, v)

        ctx.save_for_backward(q, k, v, score, alpha, causal_mask)
        ctx.scale = scale
        ctx.phi_act = phi_act
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, score, alpha, causal_mask = ctx.saved_tensors
        dtype = q.dtype
        grad_A = grad_A.to(dtype)
        alpha = alpha.to(dtype)
        score = score.to(dtype)
        scale = ctx.scale
        phi_act = ctx.phi_act

        grad_V = torch.matmul(alpha.transpose(-1, -2), grad_A)
        grad_alpha = torch.matmul(grad_A, v.transpose(-1, -2))
        sum_grad_alpha = torch.sum(grad_alpha * alpha, dim=-1, keepdim=True)
        grad_S = alpha * (grad_alpha - sum_grad_alpha)
        grad_S = torch.where(causal_mask, grad_S, torch.zeros_like(grad_S))

        grad_phi_masked = torch.flip(torch.cumsum(torch.flip(grad_S, dims=[-2]), dim=-2), dims=[-2])
        grad_phi_score = torch.where(causal_mask, grad_phi_masked, torch.zeros_like(grad_phi_masked))
        grad_score = grad_phi_score * _dphi(score, phi_act).to(dtype)

        grad_Q = torch.matmul(grad_score, k) * scale
        grad_K = torch.matmul(grad_score.transpose(-1, -2), q) * scale
        return grad_Q, grad_K, grad_V, None, None


class MockD1RetentionMechanism(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.latent_dim = config.retention_latent_dim
        self.scale = 1.0 / math.sqrt(self.kqv_dim)
        self.phi_act = config.phi_act
        self.chunk_size = config.chunk_size

        self.q_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.w_kv_latent = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.w_k_expand = nn.Linear(self.latent_dim, self.kqv_dim, bias=False)
        self.w_v_expand = nn.Linear(self.latent_dim, self.kqv_dim, bias=False)
        self.o_proj = nn.Linear(self.kqv_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, C, _ = x.shape
        q = self.q_proj(x)
        l_kv_curr = self.w_kv_latent(x)

        if state is None:
            k = self.w_k_expand(l_kv_curr)
            v = self.w_v_expand(l_kv_curr)
            
            if (self.config.curriculum_stage == 5 or self.config.use_chunked_scan) and C > self.chunk_size:
                A = ChunkedRetentionFunction.apply(q, k, v, self.scale, self.phi_act, self.chunk_size)
            else:
                A = RetentionFunction.apply(q, k, v, self.scale, self.phi_act)
            next_state = None
        else:
            prev_l_kv = state.get("l_kv_cache", None)
            prev_S = state.get("running_S", None)
            l_kv_all = torch.cat([prev_l_kv, l_kv_curr], dim=1) if prev_l_kv is not None else l_kv_curr

            k_all = self.w_k_expand(l_kv_all)
            v_all = self.w_v_expand(l_kv_all)

            curr_score = torch.matmul(q, k_all.transpose(-1, -2)) * self.scale
            phi_curr = _phi(curr_score, self.phi_act)

            if prev_S is not None:
                prev_S_padded = F.pad(prev_S, (0, 1), value=0.0)
                next_S = prev_S_padded + phi_curr
            else:
                next_S = phi_curr

            alpha = F.softmax(next_S, dim=-1).to(q.dtype)
            A = torch.matmul(alpha, v_all)

            next_state = {
                "l_kv_cache": l_kv_all,
                "running_S": next_S
            }

        return self.o_proj(A), next_state