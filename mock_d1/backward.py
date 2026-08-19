"""
Analytical Mathematical Reference & Backward Pass Derivations
Focus-Retention Architecture (Mock-D1:7B)

This module provides mathematical definitions and reference formulations
for the custom Autograd functions implemented in `focus.py` and `retention.py`.
"""

import math
import torch
import torch.nn.functional as F


def focus_attention_forward_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, scale: float):
    """
    Reference Focus Attention Forward Pass:
    - q, k, v: [B, H, C, d_h]
    - gamma: [1, H, 1, 1] (learnable head-specific temporal decay)
    - scale: 1 / sqrt(d_h)
    
    Formula:
    P_t = scale * (q_t^T @ k_t)  in R^{d_h x d_h}
    M_t = gamma * M_{t-1} + P_t  in R^{d_h x d_h}
    S_t = softmax(M_t, dim=-1)   in R^{d_h x d_h}
    A_t = v_t @ S_t              in R^{d_h}
    """
    B, H, C, d_h = q.shape
    P = scale * torch.matmul(q.unsqueeze(-1), k.unsqueeze(-2))  # [B, H, C, d_h, d_h]
    
    M = torch.zeros(B, H, C, d_h, d_h, device=q.device, dtype=q.dtype)
    curr_M = torch.zeros(B, H, d_h, d_h, device=q.device, dtype=q.dtype)
    for t in range(C):
        curr_M = gamma * curr_M + P[:, :, t]
        M[:, :, t] = curr_M
        
    S = F.softmax(M, dim=-1)
    A = torch.matmul(v.unsqueeze(-2), S).squeeze(-2)
    return A, S, M


def retention_forward_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, phi_act: str = "silu"):
    """
    Reference Global Retention Forward Pass:
    - q, k, v: [B, C, D_KQV]
    - scale: 1 / sqrt(D_KQV)
    - phi_act: Feature map ('silu', 'relu', 'gelu', 'identity')
    
    Formula:
    score = (q @ k^T) * scale    in R^{C x C}
    P = phi(score)               in R^{C x C} (masked lower triangular)
    S = cumsum(P, dim=-2)        in R^{C x C} (recursive history accumulation)
    alpha = softmax(S, dim=-1)   in R^{C x C} (causal attention probabilities)
    A = alpha @ v                in R^{C x D_KQV}
    """
    B, C, D = q.shape
    causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
    score = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    if phi_act == "silu":
        phi_score = F.silu(score)
    elif phi_act == "relu":
        phi_score = F.relu(score)
    elif phi_act == "gelu":
        phi_score = F.gelu(score)
    else:
        phi_score = score
        
    phi_masked = phi_score.masked_fill(~causal_mask, 0.0)
    S = torch.cumsum(phi_masked, dim=-2)
    S_masked = S.masked_fill(~causal_mask, float("-inf"))
    alpha = F.softmax(S_masked, dim=-1)
    A = torch.matmul(alpha, v)
    return A, alpha, score
