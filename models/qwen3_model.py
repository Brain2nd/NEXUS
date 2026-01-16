"""
SpikeQwen3Model & SpikeQwen3ForCausalLM - Full Qwen3 Model
===========================================================

100% Pure SNN Gate Circuit Implementation

Architecture:
    SpikeQwen3ForCausalLM
    ├── model (SpikeQwen3Model)
    │   ├── embed_tokens (SpikeFP32Embedding)
    │   ├── layers × N (SpikeQwen3DecoderLayer)
    │   └── norm (SpikeFP32RMSNormFullFP64)
    └── lm_head (SpikeFP32Linear)

Author: MofNeuroSim Project
"""
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    SpikeFP32Embedding,
    SpikeFP32RMSNormFullFP64,
    SpikeFP32Linear,
)

from .qwen3_config import SpikeQwen3Config
from .qwen3_decoder_layer import SpikeQwen3DecoderLayer


class SpikeQwen3Model(nn.Module):
    """Qwen3 Base Model - 100% Pure SNN Implementation

    Components:
    - embed_tokens: Token embedding
    - layers: N decoder layers
    - norm: Final RMSNorm

    Args:
        config: SpikeQwen3Config instance
        neuron_template: Neuron template for physical simulation

    Input:
        input_ids: [batch, seq_len] integer tensor
        positions: [seq_len] position indices (optional, auto-generated if None)
        attention_mask: [seq_len, seq_len] bool, True = masked

    Output:
        [batch, seq_len, hidden_size, 32] FP32 pulse
    """

    def __init__(self, config: SpikeQwen3Config, neuron_template=None):
        super().__init__()
        self.config = config
        nt = neuron_template

        # Token embedding
        self.embed_tokens = SpikeFP32Embedding(
            config.vocab_size,
            config.hidden_size,
            neuron_template=nt
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            SpikeQwen3DecoderLayer(config, neuron_template=nt)
            for _ in range(config.num_hidden_layers)
        ])

        # Final RMSNorm
        self.norm = SpikeFP32RMSNormFullFP64(
            config.hidden_size,
            config.rms_norm_eps,
            neuron_template=nt
        )

    def forward(self, input_ids, positions=None, attention_mask=None):
        """
        Args:
            input_ids: [batch, seq_len] integer tensor
            positions: [seq_len] position indices (optional)
            attention_mask: [seq_len, seq_len] bool, True = masked

        Returns:
            [batch, seq_len, hidden_size, 32] FP32 pulse
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)  # [batch, seq, hidden, 32]

        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, attention_mask)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def set_embedding_weight(self, weight):
        """Set embedding weights from float tensor.

        Args:
            weight: [vocab_size, hidden_size]
        """
        self.embed_tokens.set_weight_from_float(weight)

    def reset(self):
        """Reset all submodules."""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()


class SpikeQwen3ForCausalLM(nn.Module):
    """Qwen3 Causal Language Model - 100% Pure SNN Implementation

    Components:
    - model (SpikeQwen3Model)
    - lm_head (Linear projection to vocabulary)

    Args:
        config: SpikeQwen3Config instance
        neuron_template: Neuron template for physical simulation

    Input:
        input_ids: [batch, seq_len] integer tensor
        positions: [seq_len] position indices (optional)
        attention_mask: [seq_len, seq_len] bool, True = masked

    Output:
        [batch, seq_len, vocab_size, 32] FP32 pulse (logits)
    """

    def __init__(self, config: SpikeQwen3Config, neuron_template=None):
        super().__init__()
        self.config = config
        nt = neuron_template

        # Base model
        self.model = SpikeQwen3Model(config, neuron_template=nt)

        # LM head: hidden_size -> vocab_size
        self.lm_head = SpikeFP32Linear(
            config.hidden_size,
            config.vocab_size,
            neuron_template=nt
        )

    def forward(self, input_ids, positions=None, attention_mask=None):
        """
        Args:
            input_ids: [batch, seq_len] integer tensor
            positions: [seq_len] position indices (optional)
            attention_mask: [seq_len, seq_len] bool, True = masked

        Returns:
            [batch, seq_len, vocab_size, 32] FP32 pulse (logits)
        """
        # Get hidden states from base model
        hidden_states = self.model(input_ids, positions, attention_mask)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits

    def generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask.

        Args:
            seq_len: Sequence length
            device: Target device

        Returns:
            [seq_len, seq_len] bool tensor, True = masked
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def set_weights_from_hf_model(self, hf_model):
        """Import weights from HuggingFace Qwen3 model.

        Args:
            hf_model: HuggingFace Qwen3ForCausalLM instance
        """
        # Embedding
        self.model.set_embedding_weight(hf_model.model.embed_tokens.weight.data)

        # LM head
        self.lm_head.set_weight_from_float(hf_model.lm_head.weight.data)

        # Final norm
        self.model.norm.weight.data = hf_model.model.norm.weight.data

        # Layers
        for i, (snn_layer, hf_layer) in enumerate(zip(self.model.layers, hf_model.model.layers)):
            # Input layernorm
            snn_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data

            # Post-attention layernorm
            snn_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data

            # Attention
            snn_layer.self_attn.set_weights_from_float(
                hf_layer.self_attn.q_proj.weight.data,
                hf_layer.self_attn.k_proj.weight.data,
                hf_layer.self_attn.v_proj.weight.data,
                hf_layer.self_attn.o_proj.weight.data,
                hf_layer.self_attn.q_norm.weight.data if hasattr(hf_layer.self_attn, 'q_norm') else None,
                hf_layer.self_attn.k_norm.weight.data if hasattr(hf_layer.self_attn, 'k_norm') else None,
            )

            # MLP
            snn_layer.mlp.set_weights_from_float(
                hf_layer.mlp.gate_proj.weight.data,
                hf_layer.mlp.up_proj.weight.data,
                hf_layer.mlp.down_proj.weight.data,
            )

    def reset(self):
        """Reset all submodules."""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()
