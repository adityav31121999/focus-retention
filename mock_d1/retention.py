import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def _phi(score: torch.Tensor, phi_act: str) -> torch.Tensor:
    """Feature activation function phi for non-linear history weighting."""
    if phi_act == "silu":
        return F.silu(score)
    elif phi_act == "relu":
        return F.relu(score)
    elif phi_act == "gelu":
        return F.gelu(score)
    else:
        return score


def _dphi(score: torch.Tensor, phi_act: str) -> torch.Tensor:
    """Exact analytical derivative phi'(score)."""
    if phi_act == "silu":
        sig = torch.sigmoid(score)
        return sig * (1.0 + score * (1.0 - sig))
    elif phi_act == "relu":
        return (score > 0).to(score.dtype)
    elif phi_act == "gelu":
        return 0.5 * (1.0 + torch.erf(score / math.sqrt(2.0))) + \
               (score / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * score.pow(2))
    else:
        return torch.ones_like(score)


class RetentionFunction(torch.autograd.Function):
    """
    Custom Autograd Function for Global Retention:
    
    Forward:
        1. Score: Z = (Q @ K^T) / sqrt(D_KQV)
        2. Feature map: P = phi(Z)
        3. Recursive accumulation: S_i = S_{i-1} + P_i  =>  S = cumsum(tril(P), dim=-2)
        4. Causal retention weights: alpha = softmax(S_masked, dim=-1)
        5. Output: A = alpha @ V
    
    Backward:
        - Closed-form Softmax VJP
        - Vectorized reverse suffix scan: grad_P = flip(cumsum(flip(grad_S)))
    """
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, phi_act: str):
        # q, k, v: [B, C, D_KQV]
        B, C, _ = q.shape
        causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))

        # 1. Scaled dot-product interaction
        score = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, C, C]

        # 2. phi feature activation
        phi_score = _phi(score, phi_act)

        # 3. Mask future tokens before recursive accumulation
        phi_masked = phi_score.masked_fill(~causal_mask, 0.0)

        # 4. Causal recursive state accumulation: S_i = S_{i-1} + phi(Q_i @ K^T)
        S = torch.cumsum(phi_masked, dim=-2)

        # 5. Re-mask before softmax so future positions do not receive probability mass
        S_masked = S.masked_fill(~causal_mask, float("-inf"))

        # 6. Retention probability distribution: alpha = softmax(S)
        alpha = F.softmax(S_masked, dim=-1)

        # 7. Output aggregation: A = alpha @ V
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

        # 1. Gradient w.r.t V: [B, C, C]^T @ [B, C, D] -> [B, C, D]
        grad_V = torch.matmul(alpha.transpose(-1, -2), grad_A)

        # 2. Gradient w.r.t alpha: [B, C, D] @ [B, C, D]^T -> [B, C, C]
        grad_alpha = torch.matmul(grad_A, v.transpose(-1, -2))

        # 3. Softmax VJP on alpha -> grad_S
        sum_grad_alpha = torch.sum(grad_alpha * alpha, dim=-1, keepdim=True)
        grad_S = alpha * (grad_alpha - sum_grad_alpha)
        grad_S = torch.where(causal_mask, grad_S, torch.zeros_like(grad_S))

        # 4. Backward through cumsum along query axis (dim=-2):
        # Suffix-sum accumulation over downstream tokens i..C
        grad_phi_masked = torch.flip(torch.cumsum(torch.flip(grad_S, dims=[-2]), dim=-2), dims=[-2])
        grad_phi_score = torch.where(causal_mask, grad_phi_masked, torch.zeros_like(grad_phi_masked))

        # 5. Gradient through activation phi
        grad_score = grad_phi_score * _dphi(score, phi_act)

        # 6. Gradients w.r.t Q and K
        grad_Q = torch.matmul(grad_score, k) * scale
        grad_K = torch.matmul(grad_score.transpose(-1, -2), q) * scale

        return grad_Q, grad_K, grad_V, None, None


class MockD1RetentionAttention(nn.Module):
    """
    Global Retention Layer with Latent KV Cache:
    - L_KV = X @ W_KV^L                  [B, C, d_L]      (512-dim Latent KV Cache)
    - K = L_KV @ W_K^L                   [B, C, D_KQV]    (Key Expansion)
    - V = L_KV @ W_V^L                   [B, C, D_KQV]    (Value Expansion)
    - Q = X @ W_Q                        [B, C, D_KQV]    (Query Projection)
    - S_i = S_{i-1} + phi(Q_i @ K^T)
    - alpha_i = softmax(S_i)
    - A_i = (alpha_i @ L_KV) @ W_V^L = alpha_i @ V
    - Y = A @ W_O                        [B, C, D_E]
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.latent_dim = config.retention_latent_dim
        self.scale = 1.0 / math.sqrt(self.kqv_dim)
        self.phi_act = config.phi_act

        # Query projection
        self.q_proj = nn.Linear(self.hidden_dim, self.kqv_dim, bias=False)

        # Latent KV compression & expansions (DeepSeek-style)
        self.w_kv_latent = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)  # W_KV^L (3072 -> 512)
        self.w_k_expand = nn.Linear(self.latent_dim, self.kqv_dim, bias=False)       # W_K^L  (512 -> 4096)
        self.w_v_expand = nn.Linear(self.latent_dim, self.kqv_dim, bias=False)       # W_V^L  (512 -> 4096)

        # Output projection
        self.o_proj = nn.Linear(self.kqv_dim, self.hidden_dim, bias=False)          # W_O    (4096 -> 3072)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [B, C, D_E]
            state: Optional dictionary for step-by-step generation:
                   - "l_kv_cache": Past compressed latent KV [B, t_prev, 512]
                   - "running_S": Running cumulative state S [B, 1, t_prev]
        Returns:
            Output tensor [B, C, D_E] and next state dict.
        """
        B, C, _ = x.shape

        # Compute Query and Latent KV
        q = self.q_proj(x)                  # [B, C, D_KQV]
        l_kv_curr = self.w_kv_latent(x)     # [B, C, d_L] (512-dim)

        if state is None:
            # -------------------------------------------------------------
            # Parallel Training Path: Fast full-sequence execution via Autograd Function
            # -------------------------------------------------------------
            k = self.w_k_expand(l_kv_curr)  # [B, C, D_KQV]
            v = self.w_v_expand(l_kv_curr)  # [B, C, D_KQV]
            A = RetentionFunction.apply(q, k, v, self.scale, self.phi_act)
            next_state = None

        else:
            # -------------------------------------------------------------
            # Recurrent Inference Path: Step-by-step decoding with Latent KV Cache
            # -------------------------------------------------------------
            prev_l_kv = state.get("l_kv_cache", None)
            prev_S = state.get("running_S", None)

            # Append current 512-dim latent token to compressed cache
            if prev_l_kv is not None:
                l_kv_all = torch.cat([prev_l_kv, l_kv_curr], dim=1)  # [B, t_total, 512]
            else:
                l_kv_all = l_kv_curr

            t_total = l_kv_all.shape[1]

            # Expand Keys and Values from compressed cache
            k_all = self.w_k_expand(l_kv_all)  # [B, t_total, 4096]
            v_all = self.w_v_expand(l_kv_all)  # [B, t_total, 4096]

            # Current step interaction: score = (Q_i @ K^T) / sqrt(D)
            curr_score = torch.matmul(q, k_all.transpose(-1, -2)) * self.scale  # [B, 1, t_total]
            phi_curr = _phi(curr_score, self.phi_act)                            # [B, 1, t_total]

            # Update recursive state: S_i = S_{i-1} + phi(Q_i @ K^T)
            if prev_S is not None:
                # Pad prev_S by 1 for the newly arrived token position
                prev_S_padded = F.pad(prev_S, (0, 1), value=0.0)  # [B, 1, t_total]
                next_S = prev_S_padded + phi_curr
            else:
                next_S = phi_curr

            # Probability distribution: alpha = softmax(S_i)
            alpha = F.softmax(next_S, dim=-1)  # [B, 1, t_total]

            # Output: A = (alpha @ L_KV) @ W_V^L = alpha @ V
            A = torch.matmul(alpha, v_all)     # [B, 1, 4096]

            next_state = {
                "l_kv_cache": l_kv_all,        # Compressed 512-dim cache
                "running_S": next_S            # Cumulative state
            }

        return self.o_proj(A), next_state