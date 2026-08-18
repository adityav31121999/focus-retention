import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1 import MockD1Config

class MockD1FocusAttention(nn.Module):
    """
    Focus Attention Layer:
    - W_Q, W_K, W_V: 3072 x 4096
    - Head Dim d_h: 128 (32 heads)
    - S = softmax(Q^T @ K / sqrt(d_h)) -> Matrix size 128 x 128
    - A = V @ S
    - W_O: 4096 x 3072
    """
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
            # Parallel sequence mode: [B, H, 128, 128]
            M = torch.matmul(q.transpose(-1, -2), k) * self.scale
            S = F.softmax(M, dim=-1)
            A = torch.matmul(v, S)
            next_state = None
        else:
            # Recurrent step mode: M_i = M_{i-1} + Q_i^T @ K_i / sqrt(d)
            delta_M = torch.matmul(q.transpose(-1, -2), k) * self.scale
            next_state = state + delta_M
            S = F.softmax(next_state, dim=-1)
            A = torch.matmul(v, S)

        A = A.transpose(1, 2).contiguous().view(B, C, self.kqv_dim)
        return self.o_proj(A), next_state