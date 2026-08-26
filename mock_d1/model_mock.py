# model_mock.py
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .configure_mockd1_mini import MockD1Config
from .feedforward import RMSNorm
from .block import MockD1Block


def init_laplace_weights_(tensor: torch.Tensor, scale: float = 0.02, min_val: float = -1.0, max_val: float = 1.0):
    """
    Initializes tensor in-place with Laplace distribution centered at 0.0
    and clamped to the range (min_val, max_val).
    
    Uses an in-place Inverse-CDF transform:
        X = -sgn(u) * scale * ln(1 - 2|u|),  where u in (-0.5, 0.5)
    
    Zero heap/CPU memory allocation overhead to avoid crashing on RAM-limited environments (e.g., Colab/Kaggle).
    """
    with torch.no_grad():
        # 1. Generate uniform samples in (-0.5, 0.5) in-place
        tensor.uniform_(-0.499999, 0.499999)
        # 2. Extract sign
        sgn = torch.sign(tensor)
        # 3. Transform uniform distribution into Laplace distribution in-place
        tensor.abs_().mul_(2.0).neg_().add_(1.0).log_().mul_(-scale).mul_(sgn)
        # 4. Clamp in-place
        tensor.clamp_(min_val, max_val)
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
            # Low-rank factor A: [D_E, r]
            self.lora_A = nn.Parameter(torch.empty(config.hidden_dim, self.r))
            # Low-rank factor B: [r, VocabSize]
            self.lora_B = nn.Parameter(torch.empty(self.r, config.vocab_size))
            
            init_laplace_weights_(self.lora_A, scale=config.initializer_range, min_val=-1.0, max_val=1.0)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base shared de-embedding: [B, C, D_E] @ [D_E, Vocab] -> [B, C, Vocab]
        logits = F.linear(x, self.shared_embed.weight)

        if self.use_lora and self.lora_A is not None and self.lora_B is not None:
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

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Returns total parameter count of the base model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
            self,
            input_ids: torch.LongTensor,
            past_states: Optional[List[List[Optional[torch.Tensor]]]] = None,
            seq_offset: int = 0
        ) -> Tuple[torch.Tensor, Optional[List[List[Optional[torch.Tensor]]]]]:
        
        x0 = self.embed_tokens(input_ids)
        h = x0
        new_states = [] if past_states is not None else None

        # Check once if running on TPU / XLA device
        is_xla = input_ids.device.type == "xla"

        for i, block in enumerate(self.blocks):
            block_states = past_states[i] if past_states is not None else None

            if self.gradient_checkpointing and self.training and block_states is None:
                def create_custom_forward(module):
                    def custom_forward(hidden_states, token_x0):
                        out, _ = module(hidden_states, token_x0, states=None, seq_offset=seq_offset)
                        return out
                    return custom_forward

                # --- Device-Specific Safe Activation Checkpointing ---
                if is_xla:
                    # TPU / XLA: Must use reentrant mode and disable RNG state preservation
                    h = checkpoint.checkpoint(
                        create_custom_forward(block),
                        h,
                        x0,
                        use_reentrant=True,
                        preserve_rng_state=False
                    )
                else:
                    # CUDA GPU / ROCm / CPU: Modern non-reentrant checkpointing
                    h = checkpoint.checkpoint(
                        create_custom_forward(block),
                        h,
                        x0,
                        use_reentrant=False
                    )
            else:
                h, next_block_states = block(h, x0=x0, states=block_states, seq_offset=seq_offset)
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

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Returns total parameter count of the causal LM (accounting for tied weights)."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        # model.embed_tokens and lm_head.shared_embed share parameters
        unique_params = set(self.parameters())
        return sum(p.numel() for p in unique_params)

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
        past_states: Optional[List[List[Optional[torch.Tensor]]]] = None,
        seq_offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[List[Optional[torch.Tensor]]]]]:
        
        hidden_states, next_states = self.model(input_ids, past_states=past_states, seq_offset=seq_offset)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Calculate cross-entropy in float32 for mixed-precision numerical stability
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size).float(),
                shift_labels.view(-1)
            )

        return logits, loss, next_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fast token-by-token autoregressive generation using recurrent state carry.
        O(1) memory per step.
        """
        self.eval()
        B, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        head_dim = self.config.focus_head_dim
        num_heads = self.config.focus_heads

        # 1. Initialize empty recurrent states across all 9 blocks
        past_states = []
        for _ in range(self.config.num_blocks):
            block_states = [
                torch.zeros(B, num_heads, head_dim, head_dim, device=device, dtype=dtype),  # F1
                torch.zeros(B, num_heads, head_dim, head_dim, device=device, dtype=dtype),  # F2
                torch.zeros(B, num_heads, head_dim, head_dim, device=device, dtype=dtype),  # F3
                {"l_kv_cache": None, "running_S": None}                                     # Retention
            ]
            past_states.append(block_states)

        # 2. Prompt Prefill Phase (build initial states token-by-token)
        logits = None
        for t in range(seq_len):
            token = input_ids[:, t : t + 1]
            logits, _, past_states = self.forward(token, past_states=past_states, seq_offset=t)

        generated = input_ids

        # 3. Autoregressive Decoding Loop
        for step in range(max_new_tokens):
            curr_pos = seq_len + step
            last_logits = logits[:, -1, :]

            if temperature > 0:
                last_logits = last_logits / max(temperature, 1e-5)
                if top_k > 0:
                    v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                    last_logits[last_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Single token forward step with carried past_states
            logits, _, past_states = self.forward(next_token, past_states=past_states, seq_offset=curr_pos)

        return generated