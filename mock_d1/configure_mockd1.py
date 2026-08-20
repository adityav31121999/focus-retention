# configure_mockd1.py
from dataclasses import dataclass

# 7B Model Configuration
@dataclass
class MockD1Config7B:
    # General Dimensions
    vocab_size: int = 262_144               # Gemma Vocab (2^18)
    max_seq_len: int = 262_144              # Context Window C_W (2^18)
    hidden_dim: int = 3_072                 # D_E: Embedding Dimension
    kqv_dim: int = 4_096                    # D_KQV: Key, Query, Value Dimension
    intermediate_dim: int = 12_288          # D_FFN: SwiGLU Up-projection

    # Topology
    num_layers: int = 36                    # N_L
    num_blocks: int = 9                     # N_B = 36 / (3 + 1)
    focus_layers_per_block: int = 3         # N_F (27 Focus layers total)
    retention_layers_per_block: int = 1     # N_R (9 Retention layers total)

    # Focus Attention + RoPE & Decay
    focus_heads: int = 32                   # N_H (d_h = 4096 / 32 = 128)
    use_rope: bool = True                   # Enable Rotary Position Embeddings
    rope_base: float = 500_000.0            # High base frequency for 262k context
    use_focus_decay: bool = True            # Enable learnable head-specific temporal decay

    # Retention Attention
    retention_heads: int = 1
    retention_latent_dim: int = 512         # d_L (Latent KV cache dimension)
    retention_decay: float = 0.9            # Causal discount factor lambda
    phi_act: str = "silu"

    # Curriculum Stage 5 Policy Settings
    curriculum_stage: int = 5               # Current training stage (1 to 5)
    chunk_size: int = 512                   # Intra-block chunk size B for 262k context
    use_chunked_scan: bool = True           # Enable O(C/B) blockwise parallel scan

    # Embeddings & LoRA De-Embedding Head
    tie_word_embeddings: bool = True
    use_lora_deembed: bool = True
    lora_deembed_rank: int = 256            # LoRA rank r = 256

    # Normalization & Regularization
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    dropout: float = 0.0

    @property
    def focus_head_dim(self) -> int:
        return self.kqv_dim // self.focus_heads  # 128

# Mini Model Configuration for 1.7B
@dataclass
class MockD1Config:
    # General Dimensions (Mock-D1:1.7B)
    vocab_size: int = 128_256               # SmolLM3 Tokenizer Vocab
    max_seq_len: int = 65_536               # Context Window C_W (64k)
    hidden_dim: int = 1_536                 # D_E: Embedding Dimension
    kqv_dim: int = 2_304                    # D_KQV: 18 heads * 128 head_dim
    intermediate_dim: int = 6_144           # D_FFN: SwiGLU Up-projection (4 * D_E)

    # Block & Layer Topology (27 Focus + 9 Retention across 9 Blocks)
    num_blocks: int = 9                     # N_B = 9 Blocks
    num_layers: int = 36                    # N_L = 36 Total Layers
    focus_layers_per_block: int = 3         # N_F = 3 (27 Focus layers total)
    retention_layers_per_block: int = 1     # N_R = 1 (9 Retention layers total)

    # Focus Attention (18 Heads)
    focus_heads: int = 18                   # N_H = 18 (d_h = 2304 / 18 = 128)
    use_rope: bool = True
    rope_base: float = 500_000.0            # Base frequency for 64k window
    use_focus_decay: bool = True

    # Retention Mechanism
    retention_heads: int = 1
    retention_latent_dim: int = 256         # d_L (Latent KV Cache Dimension)
    phi_act: str = "silu"

    # Curriculum & Chunked Scan Policy
    curriculum_stage: int = 1
    chunk_size: int = 512                   # Intra-block chunk size
    use_chunked_scan: bool = True

    # Embeddings & LoRA De-Embedding Head
    tie_word_embeddings: bool = True
    use_lora_deembed: bool = True
    lora_deembed_rank: int = 128            # r = 128

    # Normalization & Regularization
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    dropout: float = 0.0

    @property
    def focus_head_dim(self) -> int:
        return self.kqv_dim // self.focus_heads  # 128