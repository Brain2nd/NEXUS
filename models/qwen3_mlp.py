"""
SpikeQwen3MLP - SwiGLU MLP Module
==================================

100% Pure SNN Gate Circuit Implementation

SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))

Author: MofNeuroSim Project
"""
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import SpikeFP32Linear, SpikeFP32SiLU, SpikeFP32Multiplier


class SpikeQwen3MLP(nn.Module):
    """Qwen3 SwiGLU MLP - 100% Pure SNN Implementation

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden MLP dimension
        neuron_template: Neuron template for physical simulation (optional)

    Input:
        x: [..., hidden_size, 32] FP32 pulse

    Output:
        [..., hidden_size, 32] FP32 pulse
    """

    def __init__(self, hidden_size: int, intermediate_size: int, neuron_template=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        nt = neuron_template

        # Three Linear projections
        self.gate_proj = SpikeFP32Linear(hidden_size, intermediate_size, neuron_template=nt)
        self.up_proj = SpikeFP32Linear(hidden_size, intermediate_size, neuron_template=nt)
        self.down_proj = SpikeFP32Linear(intermediate_size, hidden_size, neuron_template=nt)

        # SiLU activation
        self.act_fn = SpikeFP32SiLU(neuron_template=nt)

        # Element-wise multiplication (gate * up)
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

    def forward(self, x):
        """
        Args:
            x: [..., hidden_size, 32] FP32 pulse

        Returns:
            [..., hidden_size, 32] FP32 pulse
        """
        # Gate path: silu(gate_proj(x))
        gate = self.gate_proj(x)      # [..., intermediate_size, 32]
        gate = self.act_fn(gate)       # SiLU activation

        # Up path: up_proj(x)
        up = self.up_proj(x)           # [..., intermediate_size, 32]

        # Element-wise multiplication
        hidden = self.mul(gate, up)    # [..., intermediate_size, 32]

        # Down projection
        output = self.down_proj(hidden)  # [..., hidden_size, 32]

        return output

    def set_weights_from_float(self, gate_weight, up_weight, down_weight):
        """Set weights from float tensors.

        Args:
            gate_weight: [intermediate_size, hidden_size]
            up_weight: [intermediate_size, hidden_size]
            down_weight: [hidden_size, intermediate_size]
        """
        self.gate_proj.set_weight_from_float(gate_weight)
        self.up_proj.set_weight_from_float(up_weight)
        self.down_proj.set_weight_from_float(down_weight)

    def reset(self):
        """Reset all submodules."""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()
