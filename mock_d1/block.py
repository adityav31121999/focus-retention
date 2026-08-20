# block.py
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from .configure_mockd17B import MockD1Config
from .feedforward import RMSNorm, MockD1MLP
from .focus import MockD1FocusAttention
from .retention import MockD1RetentionMechanism

class MockD1Block(nn.Module):
    """
    3 Focus Layers + 1 Retention Layer Block
    - Residual connections across F1, F2
    - F3 skip bridge connecting directly to Retention output
    - Token embedding injection (X0) into Retention layer
    """
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.focus_norms = nn.ModuleList([RMSNorm(config.hidden_dim, eps=config.rms_norm_eps) for _ in range(3)])
        self.focus_layers = nn.ModuleList([MockD1FocusAttention(config) for _ in range(3)])
        self.focus_mlp_norms = nn.ModuleList([RMSNorm(config.hidden_dim, eps=config.rms_norm_eps) for _ in range(3)])
        self.focus_mlps = nn.ModuleList([MockD1MLP(config) for _ in range(3)])

        self.ret_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ret_layer = MockD1RetentionMechanism(config)
        self.ret_mlp_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ret_mlp = MockD1MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        states: Optional[List[Optional[torch.Tensor]]] = None,
        seq_offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]]]:
        new_states = [] if states is not None else None

        # F1
        s0 = states[0] if states is not None else None
        f1_out, next_s0 = self.focus_layers[0](self.focus_norms[0](x), state=s0, seq_offset=seq_offset)
        h = x + f1_out
        h = h + self.focus_mlps[0](self.focus_mlp_norms[0](h))
        if new_states is not None:
            new_states.append(next_s0)

        # F2
        s1 = states[1] if states is not None else None
        f2_out, next_s1 = self.focus_layers[1](self.focus_norms[1](h), state=s1, seq_offset=seq_offset)
        h = h + f2_out
        h = h + self.focus_mlps[1](self.focus_mlp_norms[1](h))
        if new_states is not None:
            new_states.append(next_s1)

        # Bridge input saved before F3
        f3_bridge_in = h

        # F3
        s2 = states[2] if states is not None else None
        f3_out, next_s2 = self.focus_layers[2](self.focus_norms[2](h), state=s2, seq_offset=seq_offset)
        f3_out = f3_out + self.focus_mlps[2](self.focus_mlp_norms[2](f3_out))
        if new_states is not None:
            new_states.append(next_s2)

        # Retention with X0 injection
        ret_in = f3_out + x0
        s3 = states[3] if states is not None else None
        ret_out, next_s3 = self.ret_layer(self.ret_norm(ret_in), state=s3)
        ret_out = ret_out + self.ret_mlp(self.ret_mlp_norm(ret_out))
        if new_states is not None:
            new_states.append(next_s3)

        # Skip bridge output
        out = f3_bridge_in + ret_out
        return out, new_states