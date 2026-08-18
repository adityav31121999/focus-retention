from typing import List, Optional
import torch
from mock_d1.configure_mockd1 import MockD1Config

class MockD1StateCache:
    """Fixed-dimension recursive state caching for O(1) memory step inference"""
    def __init__(self, config: MockD1Config, batch_size: int, device: str = "cpu"):
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.states = self._init_cache()

    def _init_cache(self) -> List[List[Optional[torch.Tensor]]]:
        cache = []
        head_dim = self.config.focus_head_dim  # 128
        
        for _ in range(self.config.num_blocks):
            block_states = []
            # 3 Focus states: [B, 32, 128, 128]
            for _ in range(3):
                m_state = torch.zeros(
                    (self.batch_size, self.config.focus_heads, head_dim, head_dim),
                    device=self.device,
                    dtype=torch.float32
                )
                block_states.append(m_state)
            # 1 Retention state
            s_state = torch.zeros((self.batch_size, 1, 1), device=self.device, dtype=torch.float32)
            block_states.append(s_state)
            cache.append(block_states)
        return cache

    def update(self, new_states: List[List[Optional[torch.Tensor]]]):
        self.states = new_states