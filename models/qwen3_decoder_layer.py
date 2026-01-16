"""
SpikeQwen3DecoderLayer - Qwen3 Decoder Layer
=============================================

100% Pure SNN Gate Circuit Implementation

Pre-LN structure:
    residual = x
    x = LayerNorm(x)
    x = Attention(x)
    x = residual + x

    residual = x
    x = LayerNorm(x)
    x = MLP(x)
    x = residual + x

Author: MofNeuroSim Project
"""
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import SpikeFP32RMSNormFullFP64, SpikeFP32Adder

from .qwen3_attention import SpikeQwen3Attention
from .qwen3_mlp import SpikeQwen3MLP


class SpikeQwen3DecoderLayer(nn.Module):
    """Qwen3 Decoder Layer - Pre-LN Structure

    100% Pure SNN Implementation

    Components:
    - input_layernorm (RMSNorm before attention)
    - self_attn (SpikeQwen3Attention)
    - post_attention_layernorm (RMSNorm before MLP)
    - mlp (SpikeQwen3MLP with SwiGLU)
    - Residual connections using SpikeFP32Adder

    Args:
        config: SpikeQwen3Config instance
        neuron_template: Neuron template for physical simulation

    Input:
        hidden_states: [batch, seq_len, hidden_size, 32] FP32 pulse
        positions: [seq_len] position indices
        attention_mask: [seq_len, seq_len] bool, True = masked

    Output:
        [batch, seq_len, hidden_size, 32] FP32 pulse
    """

    def __init__(self, config, neuron_template=None):
        super().__init__()
        self.config = config
        nt = neuron_template

        # Pre-attention RMSNorm
        self.input_layernorm = SpikeFP32RMSNormFullFP64(
            config.hidden_size,
            config.rms_norm_eps,
            neuron_template=nt
        )

        # Self Attention
        self.self_attn = SpikeQwen3Attention(config, neuron_template=nt)

        # Post-attention RMSNorm
        self.post_attention_layernorm = SpikeFP32RMSNormFullFP64(
            config.hidden_size,
            config.rms_norm_eps,
            neuron_template=nt
        )

        # MLP
        self.mlp = SpikeQwen3MLP(
            config.hidden_size,
            config.intermediate_size,
            neuron_template=nt
        )

        # Residual additions (pure SNN)
        self.residual_add_attn = SpikeFP32Adder(neuron_template=nt)
        self.residual_add_mlp = SpikeFP32Adder(neuron_template=nt)

    def forward(self, hidden_states, positions, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size, 32] FP32 pulse
            positions: [seq_len] position indices
            attention_mask: [seq_len, seq_len] bool, True = masked

        Returns:
            [batch, seq_len, hidden_size, 32] FP32 pulse
        """
        # ===== Attention Block =====
        # Pre-LN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states = self.self_attn(hidden_states, positions, attention_mask)

        # Residual connection
        hidden_states = self.residual_add_attn(residual, hidden_states)

        # ===== MLP Block =====
        # Pre-LN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = self.residual_add_mlp(residual, hidden_states)

        return hidden_states

    def reset(self):
        """Reset all submodules."""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()
