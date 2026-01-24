"""
多头注意力机制 (MultiHeadAttention) - 100%纯SNN门电路实现
============================================================

支持三种输入精度 (FP8/FP16/FP32)，中间使用 FP32 计算。

数学原理:
    Attention(Q, K, V) = Softmax(Q × K^T / √d_k) × V

    其中:
    - Q, K, V: [batch, num_heads, seq_len, head_dim]
    - Q × K^T: [batch, num_heads, seq_len, seq_len] (注意力分数)
    - √d_k: 缩放因子 (head_dim 的平方根)

设计原则:
- 输入输出精度一致
- 中间计算使用 FP32 (最高精度)
- 100% 复用现有 SNN 组件
- BatchMatMul 通过 Multiplier + Adder 组合实现

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
import math

from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.core.accumulator import ParallelAccumulator
from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision as SpikeFP32Linear
from atomic_ops.activation.fp32.fp32_softmax import SpikeFP32Softmax
from .rope import SpikeFP32RoPE
from atomic_ops.core.vec_logic_gates import VecMUX
from atomic_ops.encoding.converters import float32_to_pulse
from atomic_ops.arithmetic.fp32.fp32_components import FP8ToFP32Converter, FP32ToFP8Converter, FP32ToFP16Converter
from atomic_ops.arithmetic.fp16.fp16_mul_to_fp32 import FP16ToFP32Converter


# ==============================================================================
# FP32 多头注意力核心实现
# ==============================================================================

class SpikeFP32MultiHeadAttention(nn.Module):
    """FP32 多头注意力 - 100%纯SNN门电路实现

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数量
        head_dim: 每个头的维度 (默认 embed_dim // num_heads)
        use_rope: 是否使用 RoPE 位置编码
        rope_base: RoPE 基数 (默认 10000)
        neuron_template: 神经元模板

    输入:
        query: [batch, seq_len, embed_dim, 32] FP32 脉冲
        key: [batch, seq_len, embed_dim, 32] FP32 脉冲
        value: [batch, seq_len, embed_dim, 32] FP32 脉冲
        positions: [seq_len] 位置索引 (RoPE 使用，可选)
        attn_mask: [seq_len, seq_len] 注意力掩码 (可选)

    输出:
        [batch, seq_len, embed_dim, 32] FP32 脉冲
    """

    def __init__(self, embed_dim, num_heads, head_dim=None,
                 use_rope=True, rope_base=10000.0, neuron_template=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        assert embed_dim % num_heads == 0 or head_dim is not None, \
            "embed_dim must be divisible by num_heads"

        nt = neuron_template
        total_head_dim = num_heads * self.head_dim

        # Q/K/V 投影
        self.q_proj = SpikeFP32Linear(embed_dim, total_head_dim, neuron_template=nt)
        self.k_proj = SpikeFP32Linear(embed_dim, total_head_dim, neuron_template=nt)
        self.v_proj = SpikeFP32Linear(embed_dim, total_head_dim, neuron_template=nt)
        self.out_proj = SpikeFP32Linear(total_head_dim, embed_dim, neuron_template=nt)

        # RoPE (可选)
        self.use_rope = use_rope
        if use_rope:
            self.rope = SpikeFP32RoPE(self.head_dim, rope_base, neuron_template=nt)
        else:
            self.rope = None

        # BatchMatMul 组件: Q×K^T (使用 ParallelAccumulator 树形归约)
        self.qk_mul = SpikeFP32Multiplier(neuron_template=nt)
        self.qk_adder = SpikeFP32Adder(neuron_template=nt)
        self.qk_acc = ParallelAccumulator(self.qk_adder)

        # BatchMatMul 组件: Attn×V (使用 ParallelAccumulator 树形归约)
        self.av_mul = SpikeFP32Multiplier(neuron_template=nt)
        self.av_adder = SpikeFP32Adder(neuron_template=nt)
        self.av_acc = ParallelAccumulator(self.av_adder)

        # Softmax
        self.softmax = SpikeFP32Softmax(neuron_template=nt)

        # 缩放乘法器
        self.scale_mul = SpikeFP32Multiplier(neuron_template=nt)

        # 预计算缩放因子 1/√d_k
        scale_val = 1.0 / math.sqrt(self.head_dim)
        self.register_buffer('scale_pulse', float32_to_pulse(
            torch.tensor(scale_val), device='cpu'
        ).squeeze(0))  # [32]

        # 掩码 MUX (FP32 = 32位)
        self.mask_mux = VecMUX(neuron_template=nt, max_param_shape=(32,))

        # -inf 常量 (用于掩码)
        self.register_buffer('neg_inf_pulse', float32_to_pulse(
            torch.tensor(float('-inf')), device='cpu'
        ).squeeze(0))  # [32]

    def forward(self, query, key, value, positions=None, attn_mask=None):
        """前向传播

        Args:
            query: [batch, seq_len, embed_dim, 32]
            key: [batch, seq_len, embed_dim, 32]
            value: [batch, seq_len, embed_dim, 32]
            positions: [seq_len] 或 None
            attn_mask: [seq_len, seq_len] bool 掩码，True 表示屏蔽

        Returns:
            [batch, seq_len, embed_dim, 32]
        """
        device = query.device
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 1. Q/K/V 投影
        Q = self.q_proj(query)  # [batch, seq_len, total_head_dim, 32]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2. 重塑为多头格式
        # [batch, seq_len, num_heads, head_dim, 32]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim, 32)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim, 32)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim, 32)

        # [batch, num_heads, seq_len, head_dim, 32]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. 应用 RoPE (如果启用)
        if self.rope is not None and positions is not None:
            Q = self._apply_rope(Q, positions)
            K = self._apply_rope(K, positions)

        # 4. 计算注意力分数: Q × K^T
        # Q: [batch, heads, seq_q, head_dim, 32]
        # K: [batch, heads, seq_k, head_dim, 32]
        # result: [batch, heads, seq_q, seq_k, 32]
        attn_scores = self._batched_matmul_qk(Q, K)

        # 5. 缩放
        attn_scores = self.scale_mul(attn_scores, self.scale_pulse)

        # 6. 应用注意力掩码 (如果有)
        if attn_mask is not None:
            attn_scores = self._apply_mask(attn_scores, attn_mask, device)

        # 7. Softmax
        # attn_scores: [batch, heads, seq_q, seq_k, 32]
        # 需要沿 seq_k 维度做 softmax
        # 重塑为 [..., seq_k, 32] 的格式
        original_shape = attn_scores.shape
        attn_scores_flat = attn_scores.reshape(-1, seq_len, 32)
        attn_weights_flat = self.softmax(attn_scores_flat)
        attn_weights = attn_weights_flat.reshape(original_shape)

        # 8. 加权求和: Attn × V
        # attn_weights: [batch, heads, seq_q, seq_k, 32]
        # V: [batch, heads, seq_k, head_dim, 32]
        # result: [batch, heads, seq_q, head_dim, 32]
        output = self._batched_matmul_av(attn_weights, V)

        # 9. 合并多头
        output = output.transpose(1, 2)  # [batch, seq_q, heads, head_dim, 32]
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim, 32)

        # 10. 输出投影
        output = self.out_proj(output)

        return output

    def _apply_rope(self, x, positions):
        """应用 RoPE 到多头张量

        Args:
            x: [batch, num_heads, seq_len, head_dim, 32]
            positions: [seq_len]

        Returns:
            [batch, num_heads, seq_len, head_dim, 32]
        """
        batch_size, num_heads, seq_len, head_dim, _ = x.shape

        # 展平为 [batch * num_heads * seq_len, head_dim, 32]
        # 使用 contiguous() 确保内存连续
        x_contig = x.contiguous()
        x_flat = x_contig.reshape(-1, head_dim, 32)

        # 为每个位置创建 position tensor
        # positions: [seq_len] -> 扩展为 [batch * num_heads * seq_len]
        pos_expanded = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        pos_flat = pos_expanded.reshape(-1)  # [batch * num_heads * seq_len]

        # 应用 RoPE (逐位置处理)
        results = []
        for i in range(x_flat.shape[0]):
            pos_i = pos_flat[i].unsqueeze(0)
            x_i = x_flat[i:i+1]  # [1, head_dim, 32]
            result_i = self.rope(x_i, pos_i)  # [1, head_dim, 32]
            results.append(result_i)

        result = torch.cat(results, dim=0)  # [batch * num_heads * seq_len, head_dim, 32]
        return result.reshape(batch_size, num_heads, seq_len, head_dim, 32)

    def _batched_matmul_qk(self, Q, K):
        """Q × K^T 矩阵乘法

        Args:
            Q: [batch, heads, seq_q, head_dim, 32]
            K: [batch, heads, seq_k, head_dim, 32]

        Returns:
            [batch, heads, seq_q, seq_k, 32]
        """
        # 广播扩展
        # Q: [batch, heads, seq_q, 1, head_dim, 32]
        # K: [batch, heads, 1, seq_k, head_dim, 32]
        Q_expanded = Q.unsqueeze(-3)
        K_expanded = K.unsqueeze(-4)

        # 元素级乘法 (广播)
        # products: [batch, heads, seq_q, seq_k, head_dim, 32]
        products = self.qk_mul(Q_expanded, K_expanded)

        # 沿 head_dim 维度累加 (点积) - 使用 ParallelAccumulator 树形归约
        # products: [batch, heads, seq_q, seq_k, head_dim, 32]
        # reduce dim=-2 (head_dim) -> [batch, heads, seq_q, seq_k, 32]
        return self.qk_acc.reduce(products, dim=-2)

    def _batched_matmul_av(self, attn, V):
        """Attn × V 矩阵乘法

        Args:
            attn: [batch, heads, seq_q, seq_k, 32]
            V: [batch, heads, seq_k, head_dim, 32]

        Returns:
            [batch, heads, seq_q, head_dim, 32]
        """
        # 广播扩展
        # attn: [batch, heads, seq_q, seq_k, 1, 32]
        # V: [batch, heads, 1, seq_k, head_dim, 32]
        attn_expanded = attn.unsqueeze(-2)
        V_expanded = V.unsqueeze(-4)

        # 元素级乘法 (广播)
        # products: [batch, heads, seq_q, seq_k, head_dim, 32]
        products = self.av_mul(attn_expanded, V_expanded)

        # 沿 seq_k 维度累加 - 使用 ParallelAccumulator 树形归约
        # products: [batch, heads, seq_q, seq_k, head_dim, 32]
        # 需要先转置让 seq_k 在 -2 位置，归约后再转回
        # 原: [batch, heads, seq_q, seq_k, head_dim, 32]
        # 转: [batch, heads, seq_q, head_dim, seq_k, 32]
        products_t = products.transpose(-3, -2)
        # reduce dim=-2 (seq_k) -> [batch, heads, seq_q, head_dim, 32]
        return self.av_acc.reduce(products_t, dim=-2)

    def _apply_mask(self, scores, mask, device):
        """应用注意力掩码

        Args:
            scores: [batch, heads, seq_q, seq_k, 32]
            mask: [seq_q, seq_k] bool, True = 屏蔽
            device: 设备

        Returns:
            [batch, heads, seq_q, seq_k, 32]
        """
        # 扩展掩码到 scores 形状
        # mask: [seq_q, seq_k] -> [1, 1, seq_q, seq_k, 1]
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        mask_expanded = mask_expanded.expand_as(scores).float().to(device)

        # neg_inf_pulse 扩展
        neg_inf = self.neg_inf_pulse.to(device)
        neg_inf_expanded = neg_inf.expand_as(scores)

        # MUX: mask ? neg_inf : scores
        result = self.mask_mux(mask_expanded, neg_inf_expanded, scores)

        return result

    def set_weights_from_float(self, q_weight, k_weight, v_weight, out_weight):
        """设置所有投影权重

        Args:
            q_weight: [total_head_dim, embed_dim]
            k_weight: [total_head_dim, embed_dim]
            v_weight: [total_head_dim, embed_dim]
            out_weight: [embed_dim, total_head_dim]
        """
        self.q_proj.set_weight_from_float(q_weight)
        self.k_proj.set_weight_from_float(k_weight)
        self.v_proj.set_weight_from_float(v_weight)
        self.out_proj.set_weight_from_float(out_weight)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 多精度多头注意力包装器
# ==============================================================================

class SpikeMultiHeadAttention(nn.Module):
    """多精度多头注意力 - 支持 FP8/FP16/FP32

    输入输出精度一致，中间使用 FP32 计算。

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数量
        head_dim: 每个头的维度 (默认 embed_dim // num_heads)
        input_precision: 输入精度 'fp8' / 'fp16' / 'fp32'
        use_rope: 是否使用 RoPE 位置编码
        rope_base: RoPE 基数 (默认 10000)
        neuron_template: 神经元模板

    输入:
        query: [batch, seq_len, embed_dim, bits]
        key: [batch, seq_len, embed_dim, bits]
        value: [batch, seq_len, embed_dim, bits]
        positions: [seq_len] 位置索引 (RoPE 使用，可选)
        attn_mask: [seq_len, seq_len] 注意力掩码 (可选)

    输出:
        [batch, seq_len, embed_dim, bits]
    """

    def __init__(self, embed_dim, num_heads, head_dim=None,
                 input_precision='fp32', use_rope=True,
                 rope_base=10000.0, neuron_template=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.input_precision = input_precision.lower()

        assert self.input_precision in ('fp8', 'fp16', 'fp32'), \
            f"input_precision must be 'fp8', 'fp16', or 'fp32', got {input_precision}"

        nt = neuron_template

        # 输入/输出转换器
        if self.input_precision == 'fp8':
            self.input_converter = FP8ToFP32Converter(neuron_template=nt)
            self.output_converter = FP32ToFP8Converter(neuron_template=nt)
            self.input_bits = 8
        elif self.input_precision == 'fp16':
            self.input_converter = FP16ToFP32Converter(neuron_template=nt)
            self.output_converter = FP32ToFP16Converter(neuron_template=nt)
            self.input_bits = 16
        else:  # fp32
            self.input_converter = None
            self.output_converter = None
            self.input_bits = 32

        # 核心注意力 (FP32)
        self.attention = SpikeFP32MultiHeadAttention(
            embed_dim, num_heads, head_dim, use_rope, rope_base, nt
        )

    def forward(self, query, key, value, positions=None, attn_mask=None):
        """前向传播

        Args:
            query: [batch, seq_len, embed_dim, bits]
            key: [batch, seq_len, embed_dim, bits]
            value: [batch, seq_len, embed_dim, bits]
            positions: [seq_len] 或 None
            attn_mask: [seq_len, seq_len] bool 掩码

        Returns:
            [batch, seq_len, embed_dim, bits]
        """
        # 输入转换
        if self.input_converter is not None:
            query = self._convert_input(query)
            key = self._convert_input(key)
            value = self._convert_input(value)

        # FP32 计算
        output = self.attention(query, key, value, positions, attn_mask)

        # 输出转换
        if self.output_converter is not None:
            output = self._convert_output(output)

        return output

    def _convert_input(self, x):
        """输入精度转换 (向量化)

        Args:
            x: [batch, seq_len, embed_dim, input_bits]

        Returns:
            [batch, seq_len, embed_dim, 32]
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_bits)
        x_fp32_flat = self.input_converter(x_flat)
        return x_fp32_flat.reshape(original_shape + (32,))

    def _convert_output(self, x):
        """输出精度转换 (向量化)

        Args:
            x: [batch, seq_len, embed_dim, 32]

        Returns:
            [batch, seq_len, embed_dim, input_bits]
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, 32)
        x_out_flat = self.output_converter(x_flat)
        return x_out_flat.reshape(original_shape + (self.input_bits,))

    def set_weights_from_float(self, q_weight, k_weight, v_weight, out_weight):
        """设置所有投影权重"""
        self.attention.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 便捷别名
# ==============================================================================

class SpikeFP8MultiHeadAttention(SpikeMultiHeadAttention):
    """FP8 多头注意力"""

    def __init__(self, embed_dim, num_heads, head_dim=None,
                 use_rope=True, rope_base=10000.0, neuron_template=None):
        super().__init__(
            embed_dim, num_heads, head_dim,
            input_precision='fp8',
            use_rope=use_rope, rope_base=rope_base,
            neuron_template=neuron_template
        )


class SpikeFP16MultiHeadAttention(SpikeMultiHeadAttention):
    """FP16 多头注意力"""

    def __init__(self, embed_dim, num_heads, head_dim=None,
                 use_rope=True, rope_base=10000.0, neuron_template=None):
        super().__init__(
            embed_dim, num_heads, head_dim,
            input_precision='fp16',
            use_rope=use_rope, rope_base=rope_base,
            neuron_template=neuron_template
        )
