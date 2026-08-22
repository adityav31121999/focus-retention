# configure_mockd1_mini.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class MockD1Config:
    """
    Configuration dataclass for the Focus-Retention Architecture (Mock-D1).
    Default parameters represent the 1.7B Mini Model optimized for Kaggle/single-GPU training.
    """
    # General Dimensions (Default: 1.7B Mini Model)
    model_name: str = "Mock-D1-1.7B"
    vocab_size: int = 128_256               # SmolLM3 Tokenizer Vocab (128k)
    max_seq_len: int = 65_536               # Context Window C_W (64k)
    hidden_dim: int = 1_536                 # D_E: Embedding Dimension
    kqv_dim: int = 2_304                    # D_KQV: 18 heads * 128 head_dim
    intermediate_dim: int = 6_144           # D_FFN: SwiGLU Up-projection (4 * D_E)

    # Block & Layer Topology (9 Blocks * (3 Focus + 1 Retention) = 36 Layers)
    num_blocks: int = 9                     # N_B = 9 Blocks
    num_layers: int = 36                    # N_L = 36 Total Layers
    focus_layers_per_block: int = 3         # N_F = 3 (27 Focus layers total)
    retention_layers_per_block: int = 1     # N_R = 1 (9 Retention layers total)

    # Focus Attention (18 Heads @ 128 dim)
    focus_heads: int = 18                   # N_H = 18 (d_h = 2304 / 18 = 128)
    use_rope: bool = True                   # Enable Rotary Position Embeddings
    rope_base: float = 500_000.0            # Base frequency for long context
    use_focus_decay: bool = True            # Learnable head-specific temporal decay

    # Retention Mechanism (1 Head Global Attention + Latent KV)
    retention_heads: int = 1
    retention_latent_dim: int = 256         # d_L (Latent KV Cache Dimension)
    retention_decay: float = 0.9            # Causal discount factor
    phi_act: str = "silu"                   # Feature activation function: 'silu', 'relu', 'gelu'

    # Curriculum & Chunked Scan Policy
    curriculum_stage: int = 1               # Current training stage (1 to 5)
    chunk_size: int = 512                   # Intra-block chunk size B for parallel scan
    use_chunked_scan: bool = True           # Enable O(C/B) blockwise parallel scan

    # Embeddings & LoRA De-Embedding Head
    tie_word_embeddings: bool = True
    use_lora_deembed: bool = True
    lora_deembed_rank: int = 128            # LoRA rank r (128 for 1.7B, 256 for 7B)

    # Normalization & Regularization
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    dropout: float = 0.0

    @property
    def focus_head_dim(self) -> int:
        return self.kqv_dim // self.focus_heads

    @classmethod
    def get_1_7b_config(cls, **overrides) -> "MockD1Config":
        """Returns standard configuration for the 1.7B Mini Model."""
        config_kwargs: Dict[str, Any] = {
            "model_name": "Mock-D1-1.7B",
            "vocab_size": 128_256,
            "max_seq_len": 65_536,
            "hidden_dim": 1_536,
            "kqv_dim": 2_304,
            "intermediate_dim": 6_144,
            "num_blocks": 9,
            "num_layers": 36,
            "focus_layers_per_block": 3,
            "retention_layers_per_block": 1,
            "focus_heads": 18,
            "use_rope": True,
            "rope_base": 500_000.0,
            "use_focus_decay": True,
            "retention_heads": 1,
            "retention_latent_dim": 256,
            "retention_decay": 0.9,
            "phi_act": "silu",
            "curriculum_stage": 1,
            "chunk_size": 512,
            "use_chunked_scan": True,
            "tie_word_embeddings": True,
            "use_lora_deembed": True,
            "lora_deembed_rank": 128,
            "rms_norm_eps": 1e-6,
            "initializer_range": 0.02,
            "dropout": 0.0,
        }
        config_kwargs.update(overrides)
        return cls(**config_kwargs)

    @classmethod
    def get_7b_config(cls, **overrides) -> "MockD1Config":
        """Returns standard configuration for the 7B Full Model."""
        config_kwargs: Dict[str, Any] = {
            "model_name": "Mock-D1-7B",
            "vocab_size": 262_144,
            "max_seq_len": 262_144,
            "hidden_dim": 3_072,
            "kqv_dim": 4_096,
            "intermediate_dim": 12_288,
            "num_blocks": 9,
            "num_layers": 36,
            "focus_layers_per_block": 3,
            "retention_layers_per_block": 1,
            "focus_heads": 32,
            "use_rope": True,
            "rope_base": 500_000.0,
            "use_focus_decay": True,
            "retention_heads": 1,
            "retention_latent_dim": 512,
            "retention_decay": 0.9,
            "phi_act": "silu",
            "curriculum_stage": 1,
            "chunk_size": 512,
            "use_chunked_scan": True,
            "tie_word_embeddings": True,
            "use_lora_deembed": True,
            "lora_deembed_rank": 256,
            "rms_norm_eps": 1e-6,
            "initializer_range": 0.02,
            "dropout": 0.0,
        }
        config_kwargs.update(overrides)
        return cls(**config_kwargs)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "MockD1Config":
        """Factory method to instantiate a config by preset string ('1.7b', '1.7B', '7b', '7B')."""
        normalized = str(preset_name).lower().replace("_", "").replace("-", "").replace(".", "")
        if "17b" in normalized or "mini" in normalized:
            return cls.get_1_7b_config(**overrides)
        elif "7b" in normalized or "full" in normalized:
            return cls.get_7b_config(**overrides)
        else:
            raise ValueError(f"Unknown preset '{preset_name}'. Supported presets: '1.7B', '7B'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockD1Config":
        """Instantiate config from dictionary, ignoring unrecognized extra keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MockD1Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
        if "model" in raw_dict:
            return cls.from_dict(raw_dict["model"])
        return cls.from_dict(raw_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def count_parameters(self) -> Dict[str, int]:
        """Calculates theoretical parameter count breakdown for the configured architecture."""
        d_e = self.hidden_dim
        d_kqv = self.kqv_dim
        d_ffn = self.intermediate_dim
        v = self.vocab_size
        d_l = self.retention_latent_dim

        # 1. Embeddings: V x D_E
        emb_params = v * d_e

        # 2. Focus Attention Layer (Q, K, V, O projections + Decay parameter)
        focus_attn_per_layer = (3 * d_e * d_kqv) + (d_kqv * d_e) + (self.focus_heads if self.use_focus_decay else 0)
        # Focus MLP (Gate: D_E x D_FFN, Up: D_E x D_FFN, Down: D_FFN x D_E)
        mlp_per_layer = (2 * d_e * d_ffn) + (d_ffn * d_e)
        # RMSNorms (2 per layer: attn norm + mlp norm)
        norms_per_focus_layer = 2 * d_e
        total_focus_layer = focus_attn_per_layer + mlp_per_layer + norms_per_focus_layer

        # 3. Retention Layer
        ret_attn_per_layer = (d_e * d_kqv) + (d_e * d_l) + (2 * d_l * d_kqv) + (d_kqv * d_e)
        norms_per_ret_layer = 2 * d_e
        total_ret_layer = ret_attn_per_layer + mlp_per_layer + norms_per_ret_layer

        # 4. Total Blocks: 9 blocks * (3 Focus + 1 Retention)
        total_blocks_params = self.num_blocks * (
            self.focus_layers_per_block * total_focus_layer +
            self.retention_layers_per_block * total_ret_layer
        )

        # 5. Final RMSNorm
        final_norm_params = d_e

        # 6. LoRA De-Embedding Head: A [D_E x r] + B [r x V]
        lora_params = (d_e * self.lora_deembed_rank) + (self.lora_deembed_rank * v) if self.use_lora_deembed else 0

        total_params = emb_params + total_blocks_params + final_norm_params + lora_params

        return {
            "embeddings": emb_params,
            "focus_layers_total": self.num_blocks * self.focus_layers_per_block * total_focus_layer,
            "retention_layers_total": self.num_blocks * self.retention_layers_per_block * total_ret_layer,
            "final_norm": final_norm_params,
            "lora_deembed_head": lora_params,
            "total_parameters": total_params,
        }


# Backward-compatibility helpers and aliases
def MockD1Config1_7B(**kwargs) -> MockD1Config:
    return MockD1Config.get_1_7b_config(**kwargs)

def MockD1Config7B(**kwargs) -> MockD1Config:
    return MockD1Config.get_7b_config(**kwargs)