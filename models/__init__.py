"""
SNN Transformer Models - 100% Pure SNN Gate Circuit Implementation
===================================================================

Provides pre-built Transformer models based on Qwen3 architecture.

All computations are implemented using SNN gates (AND, OR, NOT, XOR, MUX).
No Python arithmetic operators in computation paths.

Usage:
    from models import SpikeQwen3ForCausalLM, SpikeQwen3Config

    config = SpikeQwen3Config(vocab_size=1000, hidden_size=64)
    model = SpikeQwen3ForCausalLM(config)

    input_ids = torch.randint(0, 1000, (1, 8))
    model.reset()
    logits = model(input_ids)  # [1, 8, 1000, 32] FP32 pulse
"""

from .qwen3_config import SpikeQwen3Config
from .qwen3_mlp import SpikeQwen3MLP
from .qwen3_attention import SpikeQwen3Attention
from .qwen3_decoder_layer import SpikeQwen3DecoderLayer
from .qwen3_model import SpikeQwen3Model, SpikeQwen3ForCausalLM

__all__ = [
    'SpikeQwen3Config',
    'SpikeQwen3MLP',
    'SpikeQwen3Attention',
    'SpikeQwen3DecoderLayer',
    'SpikeQwen3Model',
    'SpikeQwen3ForCausalLM',
]
