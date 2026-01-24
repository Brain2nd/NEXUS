# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# SNN-ized by MofNeuroSim Project - 100% Pure SNN Gate Circuit Implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SpikeQwen3 - 100% Pure SNN Gate Circuit Implementation of Qwen3
================================================================

This module provides a complete SNN implementation of the Qwen3 architecture,
where all computations are performed using pulse neurons and logic gates.

Architecture:
    SpikeQwen3ForCausalLM
    ├── model (SpikeQwen3Model)
    │   ├── embed_tokens (SpikeFP32Embedding)
    │   ├── layers × N (SpikeQwen3DecoderLayer)
    │   │   ├── input_layernorm (SpikeQwen3RMSNorm)
    │   │   ├── self_attn (SpikeQwen3Attention)
    │   │   ├── post_attention_layernorm (SpikeQwen3RMSNorm)
    │   │   └── mlp (SpikeQwen3MLP)
    │   └── norm (SpikeQwen3RMSNorm)
    └── lm_head (SpikeFP32Linear)

Data Flow:
    input_ids → Embedding → [batch, seq, hidden, 32] pulse
              ↓
    RMSNorm → Attention → residual_add → RMSNorm → MLP → residual_add
              ↓ (repeat N layers)
    Final Norm → LM Head → [batch, seq, vocab, 32] pulse
              ↓ (boundary decode)
    pulse_to_float32 → logits

Author: MofNeuroSim Project
"""
from typing import Optional

import torch
import torch.nn as nn
import math

from atomic_ops import (
    SpikeFP32Linear,
    SpikeFP32Embedding,
    SpikeFP32RMSNormFullFP64,
    SpikeFP32Multiplier,
    SpikeFP32Adder,
    SpikeFP32SiLU,
    SpikeFP32Softmax,
    SpikeFP32RoPE,
    float32_to_pulse,
    pulse_to_float32,
)
from atomic_ops.core.vec_logic_gates import VecMUX
from atomic_ops.core.reset_utils import reset_children


# =============================================================================
# Configuration
# =============================================================================

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
        attention_bias: Whether to use bias in attention projections (default: False)
        attention_dropout: Dropout rate for attention (default: 0.0, unused in SNN)
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
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
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
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

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

    @classmethod
    def from_hf_config(cls, hf_config):
        """Create config from HuggingFace Qwen3Config.

        Args:
            hf_config: HuggingFace Qwen3Config instance

        Returns:
            SpikeQwen3Config instance
        """
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hf_config.hidden_act,
            attention_bias=getattr(hf_config, 'attention_bias', False),
            attention_dropout=getattr(hf_config, 'attention_dropout', 0.0),
        )

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
            'attention_bias': self.attention_bias,
            'attention_dropout': self.attention_dropout,
        }


# =============================================================================
# SpikeQwen3RMSNorm - RMS Normalization using SNN gates
# =============================================================================

class SpikeQwen3RMSNorm(nn.Module):
    """SpikeQwen3RMSNorm - 100% Pure SNN Implementation

    Wrapper around SpikeFP32RMSNormFullFP64 for the Qwen3 architecture.

    Args:
        hidden_size: Dimension to normalize over
        eps: Epsilon for numerical stability (default: 1e-6)
        neuron_template: Neuron template for physical simulation

    Input:
        hidden_states: [..., hidden_size, 32] FP32 pulse

    Output:
        [..., hidden_size, 32] FP32 pulse
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, neuron_template=None):
        super().__init__()
        self.norm = SpikeFP32RMSNormFullFP64(hidden_size, eps, neuron_template=neuron_template)
        # Expose weight for direct access (for weight loading)
        self.weight = self.norm.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [..., hidden_size, 32] FP32 pulse

        Returns:
            [..., hidden_size, 32] FP32 pulse
        """
        return self.norm(hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.norm.eps}"

    def reset(self):
        """Reset all child modules."""
        reset_children(self)


# =============================================================================
# SpikeQwen3MLP - SwiGLU MLP using SNN gates
# =============================================================================

class SpikeQwen3MLP(nn.Module):
    """SpikeQwen3MLP - SwiGLU MLP, 100% Pure SNN Implementation

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        config: SpikeQwen3Config instance
        neuron_template: Neuron template for physical simulation

    Input:
        x: [..., hidden_size, 32] FP32 pulse

    Output:
        [..., hidden_size, 32] FP32 pulse
    """

    def __init__(self, config: SpikeQwen3Config, neuron_template=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        nt = neuron_template

        # Three Linear projections
        self.gate_proj = SpikeFP32Linear(self.hidden_size, self.intermediate_size, neuron_template=nt)
        self.up_proj = SpikeFP32Linear(self.hidden_size, self.intermediate_size, neuron_template=nt)
        self.down_proj = SpikeFP32Linear(self.intermediate_size, self.hidden_size, neuron_template=nt)

        # SiLU activation
        self.act_fn = SpikeFP32SiLU(neuron_template=nt)

        # Element-wise multiplication (gate * up)
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size, 32] FP32 pulse

        Returns:
            [..., hidden_size, 32] FP32 pulse
        """
        # Gate path: silu(gate_proj(x))
        gate = self.gate_proj(x)       # [..., intermediate_size, 32]
        gate = self.act_fn(gate)        # SiLU activation

        # Up path: up_proj(x)
        up = self.up_proj(x)            # [..., intermediate_size, 32]

        # Element-wise multiplication
        hidden = self.mul(gate, up)     # [..., intermediate_size, 32]

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
        """Reset all child modules."""
        reset_children(self)


# =============================================================================
# SpikeQwen3Attention - Multi-Head Attention with QK Norm using SNN gates
# =============================================================================

class SpikeQwen3Attention(nn.Module):
    """SpikeQwen3Attention - 100% Pure SNN Implementation

    Multi-headed attention with QK Normalization and RoPE.

    Features:
    - QK Norm: RMSNorm applied to Q and K projections
    - GQA (Grouped Query Attention): num_key_value_heads <= num_attention_heads
    - RoPE (Rotary Position Embedding)
    - Causal masking support

    Args:
        config: SpikeQwen3Config instance
        layer_idx: Layer index (for future cache support)
        neuron_template: Neuron template for physical simulation

    Input:
        hidden_states: [batch, seq_len, hidden_size, 32] FP32 pulse
        positions: [seq_len] position indices
        attention_mask: [seq_len, seq_len] bool, True = masked

    Output:
        [batch, seq_len, hidden_size, 32] FP32 pulse
    """

    def __init__(self, config: SpikeQwen3Config, layer_idx: int = 0, neuron_template=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        nt = neuron_template

        # Q/K/V/O Linear projections
        self.q_proj = SpikeFP32Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            neuron_template=nt
        )
        self.k_proj = SpikeFP32Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            neuron_template=nt
        )
        self.v_proj = SpikeFP32Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            neuron_template=nt
        )
        self.o_proj = SpikeFP32Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            neuron_template=nt
        )

        # QK Norm (Qwen3 specific: RMSNorm on head_dim only)
        self.q_norm = SpikeQwen3RMSNorm(config.head_dim, config.rms_norm_eps, neuron_template=nt)
        self.k_norm = SpikeQwen3RMSNorm(config.head_dim, config.rms_norm_eps, neuron_template=nt)

        # RoPE
        self.rope = SpikeFP32RoPE(config.head_dim, config.rope_theta, neuron_template=nt)

        # Attention computation components
        self.qk_mul = SpikeFP32Multiplier(neuron_template=nt)
        self.qk_adder = SpikeFP32Adder(neuron_template=nt)

        # Scaling: 1 / sqrt(head_dim)
        self.scale_mul = SpikeFP32Multiplier(neuron_template=nt)
        scale_val = 1.0 / math.sqrt(config.head_dim)
        self.register_buffer('scale_pulse', float32_to_pulse(
            torch.tensor(scale_val), device='cpu'
        ).squeeze(0))  # [32]

        # Softmax
        self.softmax = SpikeFP32Softmax(neuron_template=nt)

        # AV matmul
        self.av_mul = SpikeFP32Multiplier(neuron_template=nt)
        self.av_adder = SpikeFP32Adder(neuron_template=nt)

        # Mask MUX (处理单比特 mask)
        self.mask_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # -inf constant for masking
        self.register_buffer('neg_inf_pulse', float32_to_pulse(
            torch.tensor(float('-inf')), device='cpu'
        ).squeeze(0))  # [32]

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size, 32] FP32 pulse
            positions: [seq_len] position indices
            attention_mask: [seq_len, seq_len] bool, True = masked

        Returns:
            [batch, seq_len, hidden_size, 32] FP32 pulse
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # 1. Q/K/V projections
        Q = self.q_proj(hidden_states)  # [batch, seq, num_heads * head_dim, 32]
        K = self.k_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim, 32]
        V = self.v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim, 32]

        # 2. Reshape to multi-head format
        Q = self._reshape_to_heads(Q, batch_size, seq_len, self.num_attention_heads)
        K = self._reshape_to_heads(K, batch_size, seq_len, self.num_key_value_heads)
        V = self._reshape_to_heads(V, batch_size, seq_len, self.num_key_value_heads)
        # Now: [batch, heads, seq, head_dim, 32]

        # 3. QK Norm
        Q = self._apply_head_norm(Q, self.q_norm)
        K = self._apply_head_norm(K, self.k_norm)

        # 4. RoPE
        Q = self._apply_rope(Q, positions)
        K = self._apply_rope(K, positions)

        # 5. GQA: repeat K/V heads if needed
        if self.num_key_value_groups > 1:
            K = self._repeat_kv(K)
            V = self._repeat_kv(V)

        # 6. Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn_scores = self._batched_matmul_qk(Q, K)  # [batch, heads, seq_q, seq_k, 32]

        # 7. Scale
        attn_scores = self.scale_mul(attn_scores, self.scale_pulse)

        # 8. Apply mask
        if attention_mask is not None:
            attn_scores = self._apply_mask(attn_scores, attention_mask)

        # 9. Softmax along seq_k dimension
        original_shape = attn_scores.shape
        attn_scores_flat = attn_scores.reshape(-1, seq_len, 32)
        attn_weights_flat = self.softmax(attn_scores_flat)
        attn_weights = attn_weights_flat.reshape(original_shape)

        # 10. Weighted sum: Attn @ V
        output = self._batched_matmul_av(attn_weights, V)  # [batch, heads, seq_q, head_dim, 32]

        # 11. Merge heads
        output = output.transpose(1, 2)  # [batch, seq_q, heads, head_dim, 32]
        output = output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim, 32)

        # 12. Output projection
        output = self.o_proj(output)

        return output

    def _reshape_to_heads(self, x, batch_size, seq_len, num_heads):
        """Reshape [batch, seq, num_heads * head_dim, 32] to [batch, heads, seq, head_dim, 32]"""
        x = x.view(batch_size, seq_len, num_heads, self.head_dim, 32)
        return x.transpose(1, 2)

    def _apply_head_norm(self, x, norm):
        """Apply RMSNorm to each head's vectors.

        Args:
            x: [batch, heads, seq, head_dim, 32]
            norm: SpikeQwen3RMSNorm module

        Returns:
            [batch, heads, seq, head_dim, 32]
        """
        batch, heads, seq, head_dim, bits = x.shape
        x_flat = x.reshape(-1, head_dim, bits)
        x_norm = norm(x_flat)
        return x_norm.reshape(batch, heads, seq, head_dim, bits)

    def _apply_rope(self, x, positions):
        """Apply RoPE to multi-head tensor (vectorized).

        Args:
            x: [batch, heads, seq, head_dim, 32]
            positions: [seq_len]

        Returns:
            [batch, heads, seq, head_dim, 32]
        """
        batch_size, num_heads, seq_len, head_dim, _ = x.shape

        # Flatten: [batch * heads * seq, head_dim, 32]
        x_flat = x.contiguous().reshape(-1, head_dim, 32)

        # Expand positions: [batch * heads * seq]
        pos_expanded = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        pos_flat = pos_expanded.reshape(-1).float()

        # Vectorized RoPE: process all positions in parallel
        result_flat = self.rope(x_flat, pos_flat)

        return result_flat.reshape(batch_size, num_heads, seq_len, head_dim, 32)

    def _repeat_kv(self, x):
        """Repeat KV heads for GQA.

        Args:
            x: [batch, kv_heads, seq, head_dim, 32]

        Returns:
            [batch, num_heads, seq, head_dim, 32]
        """
        batch, kv_heads, seq, head_dim, bits = x.shape
        x = x.unsqueeze(2).expand(batch, kv_heads, self.num_key_value_groups, seq, head_dim, bits)
        return x.reshape(batch, kv_heads * self.num_key_value_groups, seq, head_dim, bits)

    def _parallel_reduce(self, x, dim, adder):
        """Parallel tree-based reduction.

        Args:
            x: Input tensor with shape [..., n, ..., 32] where n is reduction dimension
            dim: Dimension to reduce (negative indexing supported)
            adder: SpikeFP32Adder instance for addition

        Returns:
            Tensor with reduction dimension squeezed out
        """
        ndim = x.ndim
        if dim < 0:
            dim = ndim + dim

        # Move reduction dimension to -2 if needed
        if dim != ndim - 2:
            x = x.movedim(dim, -2)

        n = x.shape[-2]

        # Slice into list - each element has shape [..., 32] (consistent!)
        elements = [x[..., i, :] for i in range(n)]

        # Pairwise tree reduction
        while len(elements) > 1:
            new_elements = []
            for i in range(0, len(elements), 2):
                if i + 1 < len(elements):
                    result = adder(elements[i], elements[i + 1])
                    new_elements.append(result)
                else:
                    new_elements.append(elements[i])
            elements = new_elements

        return elements[0]

    def _batched_matmul_qk(self, Q, K):
        """Q @ K^T matrix multiplication.

        Args:
            Q: [batch, heads, seq_q, head_dim, 32]
            K: [batch, heads, seq_k, head_dim, 32]

        Returns:
            [batch, heads, seq_q, seq_k, 32]
        """
        # Broadcast expansion
        Q_expanded = Q.unsqueeze(-3)  # [batch, heads, seq_q, 1, head_dim, 32]
        K_expanded = K.unsqueeze(-4)  # [batch, heads, 1, seq_k, head_dim, 32]

        # Element-wise multiplication
        products = self.qk_mul(Q_expanded, K_expanded)  # [batch, heads, seq_q, seq_k, head_dim, 32]

        # Parallel tree reduction along head_dim (dim=-2)
        return self._parallel_reduce(products, dim=-2, adder=self.qk_adder)

    def _batched_matmul_av(self, attn, V):
        """Attn @ V matrix multiplication.

        Args:
            attn: [batch, heads, seq_q, seq_k, 32]
            V: [batch, heads, seq_k, head_dim, 32]

        Returns:
            [batch, heads, seq_q, head_dim, 32]
        """
        # Broadcast expansion
        attn_expanded = attn.unsqueeze(-2)  # [batch, heads, seq_q, seq_k, 1, 32]
        V_expanded = V.unsqueeze(-4)        # [batch, heads, 1, seq_k, head_dim, 32]

        # Element-wise multiplication
        products = self.av_mul(attn_expanded, V_expanded)  # [batch, heads, seq_q, seq_k, head_dim, 32]

        # Parallel tree reduction along seq_k (dim=-3)
        return self._parallel_reduce(products, dim=-3, adder=self.av_adder)

    def _apply_mask(self, scores, mask):
        """Apply attention mask.

        Args:
            scores: [batch, heads, seq_q, seq_k, 32]
            mask: [seq_q, seq_k] bool, True = masked

        Returns:
            [batch, heads, seq_q, seq_k, 32]
        """
        # Expand mask
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        mask_expanded = mask_expanded.expand_as(scores).float()

        # -inf expansion
        neg_inf_expanded = self.neg_inf_pulse.expand_as(scores)

        # MUX: mask ? neg_inf : scores
        result = self.mask_mux(mask_expanded, neg_inf_expanded, scores)

        return result

    def set_weights_from_float(self, q_weight, k_weight, v_weight, o_weight,
                               q_norm_weight=None, k_norm_weight=None):
        """Set weights from float tensors.

        Args:
            q_weight: [num_heads * head_dim, hidden_size]
            k_weight: [num_kv_heads * head_dim, hidden_size]
            v_weight: [num_kv_heads * head_dim, hidden_size]
            o_weight: [hidden_size, num_heads * head_dim]
            q_norm_weight: [head_dim] (optional)
            k_norm_weight: [head_dim] (optional)
        """
        self.q_proj.set_weight_from_float(q_weight)
        self.k_proj.set_weight_from_float(k_weight)
        self.v_proj.set_weight_from_float(v_weight)
        self.o_proj.set_weight_from_float(o_weight)

        if q_norm_weight is not None:
            self.q_norm.weight.data = q_norm_weight
        if k_norm_weight is not None:
            self.k_norm.weight.data = k_norm_weight

    def reset(self):
        """Reset all child modules."""
        reset_children(self)


# =============================================================================
# SpikeQwen3DecoderLayer - Decoder Layer using SNN gates
# =============================================================================

class SpikeQwen3DecoderLayer(nn.Module):
    """SpikeQwen3DecoderLayer - 100% Pure SNN Implementation

    Pre-LN structure:
        residual = x
        x = LayerNorm(x)
        x = Attention(x)
        x = residual + x

        residual = x
        x = LayerNorm(x)
        x = MLP(x)
        x = residual + x

    Args:
        config: SpikeQwen3Config instance
        layer_idx: Layer index
        neuron_template: Neuron template for physical simulation

    Input:
        hidden_states: [batch, seq_len, hidden_size, 32] FP32 pulse
        positions: [seq_len] position indices
        attention_mask: [seq_len, seq_len] bool, True = masked

    Output:
        [batch, seq_len, hidden_size, 32] FP32 pulse
    """

    def __init__(self, config: SpikeQwen3Config, layer_idx: int = 0, neuron_template=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        nt = neuron_template

        # Pre-attention RMSNorm
        self.input_layernorm = SpikeQwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            neuron_template=nt
        )

        # Self Attention
        self.self_attn = SpikeQwen3Attention(config, layer_idx, neuron_template=nt)

        # Post-attention RMSNorm
        self.post_attention_layernorm = SpikeQwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            neuron_template=nt
        )

        # MLP
        self.mlp = SpikeQwen3MLP(config, neuron_template=nt)

        # Residual additions (pure SNN)
        self.residual_add_attn = SpikeFP32Adder(neuron_template=nt)
        self.residual_add_mlp = SpikeFP32Adder(neuron_template=nt)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        """Reset all child modules."""
        reset_children(self)


# =============================================================================
# SpikeQwen3Model - Base Model using SNN gates
# =============================================================================

class SpikeQwen3Model(nn.Module):
    """SpikeQwen3Model - 100% Pure SNN Implementation

    Base model containing:
    - Token embedding
    - N decoder layers
    - Final RMSNorm

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
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        nt = neuron_template

        # Token embedding
        self.embed_tokens = SpikeFP32Embedding(
            config.vocab_size,
            config.hidden_size,
            neuron_template=nt
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            SpikeQwen3DecoderLayer(config, layer_idx, neuron_template=nt)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final RMSNorm
        self.norm = SpikeQwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            neuron_template=nt
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        """Reset all child modules."""
        reset_children(self)


# =============================================================================
# SpikeQwen3ForCausalLM - Causal Language Model using SNN gates
# =============================================================================

class SpikeQwen3ForCausalLM(nn.Module):
    """SpikeQwen3ForCausalLM - 100% Pure SNN Implementation

    Causal language model containing:
    - SpikeQwen3Model base
    - Linear LM head projection

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
        self.vocab_size = config.vocab_size
        nt = neuron_template

        # Base model
        self.model = SpikeQwen3Model(config, neuron_template=nt)

        # LM head: hidden_size -> vocab_size
        self.lm_head = SpikeFP32Linear(
            config.hidden_size,
            config.vocab_size,
            neuron_template=nt
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

    def generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
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
        """Reset all child modules."""
        reset_children(self)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SpikeQwen3Config",
    "SpikeQwen3RMSNorm",
    "SpikeQwen3MLP",
    "SpikeQwen3Attention",
    "SpikeQwen3DecoderLayer",
    "SpikeQwen3Model",
    "SpikeQwen3ForCausalLM",
]
