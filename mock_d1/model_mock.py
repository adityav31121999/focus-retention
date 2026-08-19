from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .configure_mockd1 import MockD1Config
from .feedforward import RMSNorm
from .block import MockD1Block


def init_laplace_weights_(tensor: torch.Tensor, scale: float = 0.02, min_val: float = -1.0, max_val: float = 1.0):
    """
    Initializes tensor in-place with Laplace distribution centered at 0.0
    and clamped to the range (min_val, max_val).
    """
    with torch.no_grad():
        laplace = torch.distributions.Laplace(
            loc=torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device),
            scale=torch.tensor(scale, dtype=tensor.dtype, device=tensor.device)
        )
        sample = laplace.sample(tensor.shape)
        tensor.copy_(sample.clamp_(min_val, max_val))
    return tensor


class MockD1LoRADeEmbeddingHead(nn.Module):
    """
    De-Embedding Head with Shared Base Embeddings and LoRA Residual:
    Output_logits = X @ (W_emb^T + A @ B) = (X @ W_emb^T) + ((X @ A) @ B)
    """
    def __init__(self, config: MockD1Config, shared_embed: nn.Embedding):
        super().__init__()
        self.shared_embed = shared_embed
        self.use_lora = config.use_lora_deembed
        self.r = config.lora_deembed_rank

        if self.use_lora:
            # Low-rank factor A: [D_E, r] = [3072, 256]
            self.lora_A = nn.Parameter(torch.empty(config.hidden_dim, self.r))
            # Low-rank factor B: [r, VocabSize] = [256, 262144]
            self.lora_B = nn.Parameter(torch.empty(self.r, config.vocab_size))
            
            init_laplace_weights_(self.lora_A, scale=config.initializer_range, min_val=-1.0, max_val=1.0)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base shared de-embedding: [B, C, D_E] @ [D_E, Vocab] -> [B, C, Vocab]
        logits = F.linear(x, self.shared_embed.weight)

        if self.use_lora:
            # Low-rank residual path: [B, C, D_E] @ [D_E, r] @ [r, Vocab]
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
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_states: Optional[List[List[Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[List[Optional[torch.Tensor]]]]]:
        
        # Initial token embeddings X0 (injected into each Retention layer)
        x0 = self.embed_tokens(input_ids)
        h = x0
        new_states = [] if past_states is not None else None

        for i, block in enumerate(self.blocks):
            block_states = past_states[i] if past_states is not None else None

            # Activation Checkpointing for training long sequences
            if self.gradient_checkpointing and self.training and block_states is None:
                def create_custom_forward(module):
                    def custom_forward(hidden_states, token_x0):
                        out, _ = module(hidden_states, token_x0, states=None)
                        return out
                    return custom_forward

                h = checkpoint.checkpoint(
                    create_custom_forward(block),
                    h,
                    x0,
                    use_reentrant=False
                )
            else:
                h, next_block_states = block(h, x0=x0, states=block_states)
                if new_states is not None:
                    new_states.append(next_block_states)

        h = self.norm(h)
        return h, new_states


class MockD1ForCausalLM(nn.Module):
    def __init__(self, config: MockD1Config):
        super().__init__()
        self.config = config
        self.model = MockD1Model(config)
        self.lm_head = MockD1LoRADeEmbeddingHead(config, self.model.embed_tokens)

        # Initialize parameter weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            init_laplace_weights_(module.weight, scale=self.config.initializer_range, min_val=-1.0, max_val=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init_laplace_weights_(module.weight, scale=self.config.initializer_range, min_val=-1.0, max_val=1.0)

    def gradient_checkpointing_enable(self):
        """Enables activation checkpointing across all transformer blocks."""
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disables activation checkpointing."""
        self.model.gradient_checkpointing = False

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