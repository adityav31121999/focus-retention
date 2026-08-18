import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1 import MockD1Config

class MockD1RetentionAttention(nn.Module):
    """
    Retention Layer with Latent KV Cache:
    - W_Q: 3072 x 4096
    - W_KV^L: 3072 x 512 (Latent compression)
    - W_K^L, W_V^L: 512 x 4096 (Expansions)
    - W_O: 4096 x 3072
    """
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.kqv_dim = config.kqv_dim
        self.latent_dim = config.retention_latent_dim
        self.scale = 1.0 / math.sqrt(self.kqv_dim)

        if config.phi_act == "silu":
            self.phi = F.silu
        elif config.phi_act == "relu":
            self.phi = F.relu
        elif config.phi_act == "gelu":
            self.phi = F.gelu
        else:
            self.phi = nn.Identity()

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

        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        phi_score = self.phi(score)

        if state is None:
            causal_mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
            alpha = F.softmax(phi_score.masked_fill(~causal_mask, float("-inf")), dim=-1)
            A = torch.matmul(alpha, v)
            next_state = None
        else:
            next_state = state + phi_score
            alpha = F.softmax(next_state, dim=-1)
            A = torch.matmul(alpha, v)

        return self.o_proj(A), next_state