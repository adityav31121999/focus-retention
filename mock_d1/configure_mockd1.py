from dataclasses import dataclass


@dataclass
class MockD1Config:
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

    # Retention Attention (Exact Associative Memory)
    retention_heads: int = 1
    retention_latent_dim: int = 512         # d_L (Latent KV cache dimension)
    retention_decay: float = 0.9            # Causal discount factor lambda
    phi_act: str = "silu"

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