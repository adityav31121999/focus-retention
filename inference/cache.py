from typing import List, Optional, Union
import torch
from mock_d1.configure_mockd1_mini import MockD1Config


class MockD1StateCache:
    """Fixed-dimension recursive state caching for O(1) memory step inference."""
    def __init__(self, config: MockD1Config, batch_size: int, device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.float32):
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.states = self._init_cache()

    def _init_cache(self) -> List[List[Optional[torch.Tensor]]]:
        cache = []
        head_dim = self.config.focus_head_dim
        num_heads = self.config.focus_heads

        for _ in range(self.config.num_blocks):
            block_states = []
            # 3 Focus states: [B, num_heads, head_dim, head_dim]
            for _ in range(3):
                m_state = torch.zeros(
                    (self.batch_size, num_heads, head_dim, head_dim),
                    device=self.device,
                    dtype=self.dtype
                )
                block_states.append(m_state)
            # 1 Retention state (dictionary containing 'l_kv_cache' and 'running_S')
            block_states.append({"l_kv_cache": None, "running_S": None})
            cache.append(block_states)
        return cache

    def update(self, new_states: List[List[Optional[torch.Tensor]]]):
        self.states = new_states