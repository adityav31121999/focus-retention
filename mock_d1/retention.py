import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1 import MockD1Config


class RetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, phi_act: str):
        # q, k, v: [B, C, D]
        B, C, _ = q.shape

        # 1. Pre-phi score: [B, C, C]
        score = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        # 2. Apply phi activation
        if phi_act == "silu":
            phi_score = F.silu(score)
        elif phi_act == "relu":
            phi_score = F.relu(score)
        elif phi_act == "gelu":
            phi_score = F.gelu
        else:
            phi_score = score

        # 3. Causal mask & Softmax retention: alpha = softmax(tril(phi_score))
        causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
        masked_phi = phi_score.masked_fill(~causal_mask, float("-inf"))
        alpha = F.softmax(masked_phi, dim=-1)

        # 4. Output: A = alpha @ V
        A = torch.matmul(alpha, v)

        ctx.save_for_backward(q, k, v, score, alpha, causal_mask)
        ctx.scale = scale
        ctx.phi_act = phi_act
        return A

    @staticmethod
    def backward(ctx, grad_A: torch.Tensor):
        q, k, v, score, alpha, causal_mask = ctx.saved_tensors
        scale = ctx.scale
        phi_act = ctx.phi_act

        # 1. Grad w.r.t V
        grad_V = torch.matmul(alpha.transpose(-1, -2), grad_A)

        # 2. Grad w.r.t alpha
        grad_alpha = torch.matmul(grad_A, v.transpose(-1, -2))

        # 3. Softmax VJP: grad_S = alpha * (grad_alpha - sum(grad_alpha * alpha))
        sum_grad_a = torch.sum(grad_alpha * alpha, dim=-1, keepdim=True)
        grad_S = alpha * (grad_alpha - sum_grad_a)
        grad_S = torch.where(causal_mask, grad_S, torch.zeros_like(grad_S))

        # 4. Reverse Suffix Scan (Backpropagating through recursive S_i = S_{i-1} + P_i)
        # Suffix sum along sequence dimension C
        grad_P = torch.flip(torch.cumsum(torch.flip(grad_S, dims=[-2]), dim=-2), dims=[-2])

        # 5. Derivative of phi(score)
        if phi_act == "silu":
            sig = torch.sigmoid(score)
            d_phi = sig * (1.0 + score * (1.0 - sig))
        elif phi_act == "relu":
            d_phi = (score > 0).float()
        elif phi_act == "gelu":
            d_phi = 0.5 * (1.0 + torch.erf(score / math.sqrt(2.0))) + \
                    (score / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * score.pow(2))
        else:
            d_phi = torch.ones_like(score)

        grad_score = grad_P * d_phi

        # 6. Grad w.r.t Q and K
        grad_Q = torch.matmul(grad_score, k) * scale
        grad_K = torch.matmul(grad_score.transpose(-1, -2), q) * scale

        return grad_Q, grad_K, grad_V, None, None


class MockD1RetentionAttention(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.latent_dim = config.retention_latent_dim
        self.scale = 1.0 / math.sqrt(self.kqv_dim)
        self.phi_act = config.phi_act

        self.q_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)
        self.w_kv_latent = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.w_k_expand = nn.Linear(self.latent_dim, self.kqv_dim, bias=False)
        self.w_v_expand = nn.Linear(self.latent_dim, self.kqv_dim, bias=False)
        self.o_proj = nn.Linear(self.kqv_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, _ = x.shape

        q = self.q_proj(x)
        l_kv = self.w_kv_latent(x)
        k = self.w_k_expand(l_kv)
        v = self.w_v_expand(l_kv)

        if state is None:
            # Analytical manual backward via RetentionFunction
            A = RetentionFunction.apply(q, k, v, self.scale, self.phi_act)
            next_state = None
        else:
            # Step-by-step inference
            score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            phi_score = F.silu(score) if self.phi_act == "silu" else score
            next_state = state + phi_score
            alpha = F.softmax(next_state, dim=-1)
            A = torch.matmul(alpha, v)

        return self.o_proj(A), next_state