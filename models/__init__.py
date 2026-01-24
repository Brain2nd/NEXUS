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

Note:
    All SNN model implementations are in models/reference/modeling_qwen3.py.
    This is the authoritative source - SNN-ized from the HuggingFace Qwen3 source.
"""

# Import from the authoritative SNN implementation
from .reference.modeling_qwen3 import (
    SpikeQwen3Config,
    SpikeQwen3RMSNorm,
    SpikeQwen3MLP,
    SpikeQwen3Attention,
    SpikeQwen3DecoderLayer,
    SpikeQwen3Model,
    SpikeQwen3ForCausalLM,
)

__all__ = [
    'SpikeQwen3Config',
    'SpikeQwen3RMSNorm',
    'SpikeQwen3MLP',
    'SpikeQwen3Attention',
    'SpikeQwen3DecoderLayer',
    'SpikeQwen3Model',
    'SpikeQwen3ForCausalLM',
]
