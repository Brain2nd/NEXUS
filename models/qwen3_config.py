"""
SpikeQwen3Config - Configuration for SNN Qwen3 Model
=====================================================

Matches the Qwen3 architecture from HuggingFace Transformers.
Simplified default values for testing purposes.

Author: MofNeuroSim Project
"""


class SpikeQwen3Config:
    """Configuration class for SpikeQwen3 models.

    This configuration matches the Qwen3 architecture with sensible defaults
    for testing and smaller-scale experiments.

    Args:
        vocab_size: Vocabulary size (default: 1000 for testing)
        hidden_size: Hidden dimension (default: 64)
        intermediate_size: MLP intermediate size (default: ~2.6875 * hidden_size)
        num_hidden_layers: Number of decoder layers (default: 2)
        num_attention_heads: Number of attention heads (default: 4)
        num_key_value_heads: Number of KV heads for GQA (default: same as attention heads)
        head_dim: Dimension per attention head (default: hidden_size // num_attention_heads)
        rms_norm_eps: Epsilon for RMSNorm (default: 1e-6)
        rope_theta: Base for RoPE (default: 10000.0)
        max_position_embeddings: Maximum sequence length (default: 128)
        hidden_act: Activation function (default: 'silu')
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        intermediate_size: int = None,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = None,
        head_dim: int = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 128,
        hidden_act: str = 'silu',
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # intermediate_size: default to ~2.6875 * hidden_size (Qwen3 style)
        if intermediate_size is None:
            self.intermediate_size = int(hidden_size * 2.6875)
        else:
            self.intermediate_size = intermediate_size

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # GQA: num_key_value_heads <= num_attention_heads
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads

        # head_dim: default to hidden_size // num_attention_heads
        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act

    def __repr__(self):
        return (
            f"SpikeQwen3Config(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  rms_norm_eps={self.rms_norm_eps},\n"
            f"  rope_theta={self.rope_theta},\n"
            f"  max_position_embeddings={self.max_position_embeddings}\n"
            f")"
        )

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'head_dim': self.head_dim,
            'rms_norm_eps': self.rms_norm_eps,
            'rope_theta': self.rope_theta,
            'max_position_embeddings': self.max_position_embeddings,
            'hidden_act': self.hidden_act,
        }
