from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configure_mockd1 import MockD1Config
from .feedforward import RMSNorm
from .block import MockD1Block

class MockD1LoRADeEmbeddingHead(nn.Module):
    """LoRA De-Embedding Head: D = W_emb^T + (A x B)"""
    def __init__(self, config: MockD1Config, shared_embed: nn.Embedding):
        super().__init__()
        self.shared_embed = shared_embed
        self.use_lora = config.use_lora_deembed
        self.r = config.lora_deembed_rank

        if self.use_lora:
            self.lora_A = nn.Parameter(torch.empty(config.hidden_dim, self.r))
            self.lora_B = nn.Parameter(torch.empty(self.r, config.vocab_size))
            nn.init.normal_(self.lora_A, mean=0.0, std=config.initializer_range)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.shared_embed.weight)
        if self.use_lora:
            lora_out = torch.matmul(x, self.lora_A)
            lora_out = torch.matmul(lora_out, self.lora_B)
            logits = logits + lora_out
        return logits


class MockD1Model(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([MockD1Block(config) for _ in range(config.num_blocks)])
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_states: Optional[List[List[Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[List[Optional[torch.Tensor]]]]]:
        x0 = self.embed_tokens(input_ids)
        h = x0
        new_states = [] if past_states is not None else None

        for i, block in enumerate(self.blocks):
            block_states = past_states[i] if past_states is not None else None
            h, next_b_states = block(h, x0=x0, states=block_states)
            if new_states is not None:
                new_states.append(next_b_states)

        h = self.norm(h)
        return h, new_states


class MockD1ForCausalLM(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.config = config
        self.model = MockD1Model(config)
        self.lm_head = MockD1LoRADeEmbeddingHead(config, self.model.embed_tokens)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        past_states: Optional[List[List[Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[List[Optional[torch.Tensor]]]]]:
        
        hidden_states, next_states = self.model(input_ids, past_states=past_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return logits, loss, next_states