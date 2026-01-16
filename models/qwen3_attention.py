"""
SpikeQwen3Attention - Qwen3 Attention with QK Norm
===================================================

100% Pure SNN Gate Circuit Implementation

Features:
- QK Norm: RMSNorm applied to Q and K projections
- GQA (Grouped Query Attention): num_key_value_heads <= num_attention_heads
- RoPE (Rotary Position Embedding)
- Causal masking support

Author: MofNeuroSim Project
"""
import torch
import torch.nn as nn
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    SpikeFP32Linear,
    SpikeFP32RMSNormFullFP64,
    SpikeFP32RoPE,
    SpikeFP32Multiplier,
    SpikeFP32Adder,
    SpikeFP32Softmax,
    float32_to_pulse,
)
from atomic_ops.vec_logic_gates import VecMUX


class SpikeQwen3Attention(nn.Module):
    """Qwen3 Attention - 100% Pure SNN Implementation

    Features:
    - QK Norm: RMSNorm on Q and K after projection
    - GQA: Supports grouped query attention
    - RoPE: Rotary position embedding

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
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

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

        # QK Norm
        self.q_norm = SpikeFP32RMSNormFullFP64(
            config.head_dim, config.rms_norm_eps, neuron_template=nt
        )
        self.k_norm = SpikeFP32RMSNormFullFP64(
            config.head_dim, config.rms_norm_eps, neuron_template=nt
        )

        # RoPE
        self.rope = SpikeFP32RoPE(config.head_dim, config.rope_theta, neuron_template=nt)

        # Attention computation components
        # QK matmul: need head_dim - 1 adders for dot product
        self.qk_mul = SpikeFP32Multiplier(neuron_template=nt)
        self.qk_adders = nn.ModuleList([
            SpikeFP32Adder(neuron_template=nt) for _ in range(max(1, config.head_dim - 1))
        ])

        # Scaling: 1 / sqrt(head_dim)
        self.scale_mul = SpikeFP32Multiplier(neuron_template=nt)
        scale_val = 1.0 / math.sqrt(config.head_dim)
        self.register_buffer('scale_pulse', float32_to_pulse(
            torch.tensor(scale_val), device='cpu'
        ).squeeze(0))  # [32]

        # Softmax
        self.softmax = SpikeFP32Softmax(neuron_template=nt)

        # AV matmul: reuse single adder
        self.av_mul = SpikeFP32Multiplier(neuron_template=nt)
        self.av_adder = SpikeFP32Adder(neuron_template=nt)

        # Mask MUX
        self.mask_mux = VecMUX(neuron_template=nt)

        # -inf constant for masking
        self.register_buffer('neg_inf_pulse', float32_to_pulse(
            torch.tensor(float('-inf')), device='cpu'
        ).squeeze(0))  # [32]

    def forward(self, hidden_states, positions, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size, 32] FP32 pulse
            positions: [seq_len] position indices
            attention_mask: [seq_len, seq_len] bool, True = masked

        Returns:
            [batch, seq_len, hidden_size, 32] FP32 pulse
        """
        device = hidden_states.device
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
        # Q: [batch, heads, seq_q, head_dim, 32]
        # K: [batch, heads, seq_k, head_dim, 32]
        attn_scores = self._batched_matmul_qk(Q, K)  # [batch, heads, seq_q, seq_k, 32]

        # 7. Scale
        attn_scores = self.scale_mul(attn_scores, self.scale_pulse.to(device))

        # 8. Apply mask
        if attention_mask is not None:
            attn_scores = self._apply_mask(attn_scores, attention_mask, device)

        # 9. Softmax along seq_k dimension
        original_shape = attn_scores.shape
        attn_scores_flat = attn_scores.reshape(-1, seq_len, 32)
        attn_weights_flat = self.softmax(attn_scores_flat)
        attn_weights = attn_weights_flat.reshape(original_shape)

        # 10. Weighted sum: Attn @ V
        # V: [batch, heads, seq_k, head_dim, 32]
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
            norm: RMSNorm module

        Returns:
            [batch, heads, seq, head_dim, 32]
        """
        batch, heads, seq, head_dim, bits = x.shape
        x_flat = x.reshape(-1, head_dim, bits)
        x_norm = norm(x_flat)
        return x_norm.reshape(batch, heads, seq, head_dim, bits)

    def _apply_rope(self, x, positions):
        """Apply RoPE to multi-head tensor.

        Args:
            x: [batch, heads, seq, head_dim, 32]
            positions: [seq_len]

        Returns:
            [batch, heads, seq, head_dim, 32]
        """
        batch_size, num_heads, seq_len, head_dim, _ = x.shape

        # Flatten: [batch * heads * seq, head_dim, 32]
        x_contig = x.contiguous()
        x_flat = x_contig.reshape(-1, head_dim, 32)

        # Expand positions: [batch * heads * seq]
        pos_expanded = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        pos_flat = pos_expanded.reshape(-1)

        # Apply RoPE per position
        results = []
        for i in range(x_flat.shape[0]):
            pos_i = pos_flat[i].unsqueeze(0)
            x_i = x_flat[i:i+1]  # [1, head_dim, 32]
            result_i = self.rope(x_i, pos_i)
            results.append(result_i)

        result = torch.cat(results, dim=0)
        return result.reshape(batch_size, num_heads, seq_len, head_dim, 32)

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

        # Sum along head_dim (dot product)
        acc = products[..., 0, :]  # [batch, heads, seq_q, seq_k, 32]

        for i in range(1, self.head_dim):
            acc = self.qk_adders[i - 1](acc, products[..., i, :])

        return acc

    def _batched_matmul_av(self, attn, V):
        """Attn @ V matrix multiplication.

        Args:
            attn: [batch, heads, seq_q, seq_k, 32]
            V: [batch, heads, seq_k, head_dim, 32]

        Returns:
            [batch, heads, seq_q, head_dim, 32]
        """
        seq_k = V.shape[2]

        # Broadcast expansion
        attn_expanded = attn.unsqueeze(-2)  # [batch, heads, seq_q, seq_k, 1, 32]
        V_expanded = V.unsqueeze(-4)        # [batch, heads, 1, seq_k, head_dim, 32]

        # Element-wise multiplication
        products = self.av_mul(attn_expanded, V_expanded)  # [batch, heads, seq_q, seq_k, head_dim, 32]

        # Sum along seq_k
        acc = products[..., 0, :, :]  # [batch, heads, seq_q, head_dim, 32]

        for i in range(1, seq_k):
            acc = self.av_adder(acc, products[..., i, :, :])

        return acc

    def _apply_mask(self, scores, mask, device):
        """Apply attention mask.

        Args:
            scores: [batch, heads, seq_q, seq_k, 32]
            mask: [seq_q, seq_k] bool, True = masked
            device: Target device

        Returns:
            [batch, heads, seq_q, seq_k, 32]
        """
        # Expand mask
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        mask_expanded = mask_expanded.expand_as(scores).float().to(device)

        # -inf expansion
        neg_inf = self.neg_inf_pulse.to(device)
        neg_inf_expanded = neg_inf.expand_as(scores)

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
        """Reset all submodules."""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()
