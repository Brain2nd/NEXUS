"""
旋转位置编码 (RoPE) - 100%纯SNN门电路实现 (向量化版本)
======================================================

支持三种输入精度 (FP8/FP16/FP32)，中间精度可配置。

RoPE 数学原理:
对于位置 p 和维度对 (2i, 2i+1):
    θ_i = base^(-2i/d)，通常 base=10000

    x'[2i]   = x[2i] * cos(θ_i * p) - x[2i+1] * sin(θ_i * p)
    x'[2i+1] = x[2i] * sin(θ_i * p) + x[2i+1] * cos(θ_i * p)

设计原则:
- 输入输出精度一致
- 中间计算精度可配置（更高精度 = 更高准确度）
- 100% 复用现有 SNN 组件
- 完全向量化，禁止位级循环

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
import math
import struct

from atomic_ops.trigonometry.fp32.fp32_sincos import SpikeFP32SinCos
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp32.fp32_components import FP8ToFP32Converter, FP32ToFP8Converter, FP32ToFP16Converter
from atomic_ops.arithmetic.fp16.fp16_components import FP16ToFP8Converter
from atomic_ops.arithmetic.fp16.fp16_mul_to_fp32 import FP16ToFP32Converter
from atomic_ops.core.vec_logic_gates import VecXOR, VecNOT


# ==============================================================================
# 辅助函数
# ==============================================================================

def float32_to_bits(f):
    """将Python float转换为32位整数表示"""
    return struct.unpack('>I', struct.pack('>f', f))[0]


def float32_tensor_to_bits(vals):
    """将 float32 张量转换为整数位表示 (向量化)

    Args:
        vals: 任意形状的 float32 张量

    Returns:
        同形状的 int32 张量，包含位表示
    """
    # 使用 view 将 float32 重新解释为 int32 (无数据拷贝)
    return vals.view(torch.int32)


def make_fp32_constant(val, batch_shape, device):
    """创建FP32常量脉冲 (向量化)"""
    bits = float32_to_bits(float(val))
    # 向量化位提取
    bit_positions = torch.arange(31, -1, -1, device=device, dtype=torch.int32)
    pulse_1d = ((bits >> bit_positions) & 1).float()  # [32]
    # 广播到 batch_shape
    pulse = pulse_1d.expand(batch_shape + (32,)).clone()
    return pulse


def make_fp32_constant_batch(vals, device):
    """创建批量FP32常量脉冲 (向量化)

    Args:
        vals: [N] 浮点数张量
        device: 设备

    Returns:
        [N, 32] FP32 脉冲
    """
    N = vals.numel()
    vals_flat = vals.flatten().float().to(device)

    # 向量化: 将所有 float32 转换为整数位表示
    bits_tensor = float32_tensor_to_bits(vals_flat)  # [N] int32

    # 向量化位提取: bits_tensor[:, None] >> bit_positions[None, :]
    bit_positions = torch.arange(31, -1, -1, device=device, dtype=torch.int32)  # [32]
    # 扩展维度并进行位运算
    pulse = ((bits_tensor.unsqueeze(1) >> bit_positions.unsqueeze(0)) & 1).float()  # [N, 32]

    return pulse


def precompute_theta(head_dim, base=10000.0):
    """预计算 θ_i = base^(-2i/d)

    Args:
        head_dim: 注意力头维度
        base: RoPE 基数

    Returns:
        theta: [head_dim // 2] 的 theta 值
    """
    i = torch.arange(0, head_dim // 2, dtype=torch.float32)
    theta = base ** (-2.0 * i / head_dim)
    return theta


# ==============================================================================
# FP32 RoPE (向量化实现)
# ==============================================================================

class SpikeFP32RoPE(nn.Module):
    """FP32 旋转位置编码 - 纯SNN门电路实现 (向量化)

    Args:
        head_dim: 注意力头维度 (必须是偶数)
        base: RoPE 基数，默认 10000
        neuron_template: 神经元模板

    输入:
        x: [batch, head_dim, 32] FP32 脉冲
        position: 位置索引 (标量或 [batch] 张量)

    输出:
        [batch, head_dim, 32] 应用 RoPE 后的 FP32 脉冲
    """
    def __init__(self, head_dim, base=10000.0, neuron_template=None):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"

        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.base = base
        nt = neuron_template

        # 预计算 theta (浮点值，懒加载编码为脉冲)
        self.register_buffer('theta', precompute_theta(head_dim, base))
        self.register_buffer('theta_pulse', None)  # 懒加载脉冲编码

        # SNN 组件 - 需要为每个维度对创建独立实例
        # 因为它们可能被不同 batch 的数据调用

        # theta × position 乘法器 (纯 SNN)
        self.mul_theta_pos = SpikeFP32Multiplier(neuron_template=nt)

        # sin/cos 计算器 (向量化处理所有维度)
        self.sincos = SpikeFP32SinCos(neuron_template=nt)

        # 乘法器 (向量化处理所有维度)
        self.mul_cos_even = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_sin_odd = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_sin_even = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_cos_odd = SpikeFP32Multiplier(neuron_template=nt)

        # 加法器 (向量化处理所有维度)
        self.add_even = SpikeFP32Adder(neuron_template=nt)
        self.add_odd = SpikeFP32Adder(neuron_template=nt)

        # 符号翻转 (使用 VecXOR)
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=None)

    def forward(self, x, position):
        """应用 RoPE (向量化) - 与 HuggingFace 实现一致

        使用前后半分割 (HF 标准):
            x_first = x[:, :half_dim]
            x_second = x[:, half_dim:]
            rotate_half(x) = [-x_second, x_first]
            result = x * cos + rotate_half(x) * sin

        Args:
            x: [batch, head_dim, 32] FP32 脉冲
            position: 标量或 [batch] 位置索引

        Returns:
            [batch, head_dim, 32] 旋转后的 FP32 脉冲
        """
        device = x.device
        batch_size = x.shape[0]

        # 确保 position 是 tensor
        if not isinstance(position, torch.Tensor):
            position = torch.tensor([position], device=device, dtype=torch.float32)
        position = position.float().to(device)

        # 如果 position 是标量，扩展为 batch
        if position.numel() == 1:
            position = position.expand(batch_size)

        # HF 风格: 前后半分割 (不是奇偶分割)
        # x: [batch, head_dim, 32]
        x_first = x[:, :self.half_dim, :]   # [batch, half_dim, 32] 前半
        x_second = x[:, self.half_dim:, :]  # [batch, half_dim, 32] 后半

        # ========== 纯 SNN 计算 theta × position ==========
        # 1. 懒加载编码 theta 到脉冲 (边界编码，仅执行一次)
        if self.theta_pulse is None or self.theta_pulse.device != device:
            self.theta_pulse = make_fp32_constant_batch(self.theta, device)
            # theta_pulse: [half_dim, 32]

        # 2. 编码 position 到脉冲 (边界编码)
        position_pulse = make_fp32_constant_batch(position, device)  # [batch, 32]

        # 3. 广播并使用 SNN 乘法器计算 theta × position
        # theta_pulse: [half_dim, 32] → [batch, half_dim, 32]
        # position_pulse: [batch, 32] → [batch, half_dim, 32]
        theta_expanded = self.theta_pulse.unsqueeze(0).expand(batch_size, -1, -1)
        position_expanded = position_pulse.unsqueeze(1).expand(-1, self.half_dim, -1)

        # 展平并用 SNN 乘法器计算
        theta_flat = theta_expanded.reshape(-1, 32)  # [batch*half_dim, 32]
        position_flat = position_expanded.reshape(-1, 32)  # [batch*half_dim, 32]
        angle_pulses_flat = self.mul_theta_pos(theta_flat, position_flat)  # [batch*half_dim, 32]
        angle_pulses = angle_pulses_flat.view(batch_size, self.half_dim, 32)

        # 计算 sin/cos (向量化处理所有角度)
        # 展平为 [batch * half_dim, 32] 以便向量化计算
        angle_flat = angle_pulses.view(-1, 32)
        sin_flat, cos_flat = self.sincos(angle_flat)

        # 恢复形状: [batch, half_dim, 32]
        sin_angle = sin_flat.view(batch_size, self.half_dim, 32)
        cos_angle = cos_flat.view(batch_size, self.half_dim, 32)

        # HF 风格: cos/sin 需要扩展到完整 head_dim (通过 cat 重复)
        # cos/sin 在 HF 中是 [batch, seq, head_dim]，其中前后半相同
        # emb = torch.cat((freqs, freqs), dim=-1) 所以 cos/sin 前后半相同
        cos_full = torch.cat([cos_angle, cos_angle], dim=1)  # [batch, head_dim, 32]
        sin_full = torch.cat([sin_angle, sin_angle], dim=1)  # [batch, head_dim, 32]

        # rotate_half: [-x_second, x_first]
        # 翻转 x_second 的符号位
        x_second_flat = x_second.reshape(-1, 32)
        ones = torch.ones_like(x_second_flat[..., 0:1])
        neg_sign = self.sign_xor(x_second_flat[..., 0:1], ones)
        neg_x_second_flat = torch.cat([neg_sign, x_second_flat[..., 1:]], dim=-1)
        neg_x_second = neg_x_second_flat.view(batch_size, self.half_dim, 32)

        # rotate_half(x) = [-x_second, x_first]
        x_rotated = torch.cat([neg_x_second, x_first], dim=1)  # [batch, head_dim, 32]

        # result = x * cos + rotate_half(x) * sin
        # 展平处理
        x_flat = x.reshape(-1, 32)  # [batch*head_dim, 32]
        x_rotated_flat = x_rotated.reshape(-1, 32)
        cos_flat = cos_full.reshape(-1, 32)
        sin_flat = sin_full.reshape(-1, 32)

        # 乘法
        x_cos = self.mul_cos_even(x_flat, cos_flat)  # [batch*head_dim, 32]
        xr_sin = self.mul_sin_odd(x_rotated_flat, sin_flat)

        # 加法: x * cos + rotate_half(x) * sin
        result_flat = self.add_even(x_cos, xr_sin)

        # 恢复形状
        result = result_flat.view(batch_size, self.head_dim, 32)

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 多精度 RoPE (向量化实现)
# ==============================================================================

class SpikeRoPE_MultiPrecision(nn.Module):
    """旋转位置编码 (RoPE) - 多精度支持 (向量化)

    输入输出精度一致，中间计算精度可配置。

    Args:
        head_dim: 注意力头维度 (必须是偶数)
        base: RoPE 基数，默认 10000
        input_precision: 输入精度 'fp8' / 'fp16' / 'fp32'
        intermediate_precision: 中间计算精度 'fp32' / 'fp64'
        neuron_template: 神经元模板

    输入:
        x: [batch, head_dim, bits] 脉冲 (bits = 8/16/32 根据 input_precision)
        position: 位置索引

    输出:
        [batch, head_dim, bits] 应用 RoPE 后的脉冲
    """
    def __init__(self, head_dim, base=10000.0,
                 input_precision='fp32', intermediate_precision='fp64',
                 neuron_template=None):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"

        self.head_dim = head_dim
        self.base = base
        self.input_precision = input_precision.lower()
        self.intermediate_precision = intermediate_precision.lower()

        assert self.input_precision in ('fp8', 'fp16', 'fp32'), \
            f"input_precision must be 'fp8', 'fp16', or 'fp32', got {input_precision}"
        assert self.intermediate_precision in ('fp32', 'fp64'), \
            f"intermediate_precision must be 'fp32' or 'fp64', got {intermediate_precision}"

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

        # 核心 RoPE (在 FP32/FP64 精度下计算)
        self.rope_core = SpikeFP32RoPE(head_dim, base, neuron_template=nt)

    def forward(self, x, position):
        """应用 RoPE (向量化)

        Args:
            x: [batch, head_dim, bits] 脉冲
            position: 位置索引

        Returns:
            [batch, head_dim, bits] 旋转后的脉冲
        """
        device = x.device
        batch_size = x.shape[0]

        # 输入转换 (向量化)
        if self.input_converter is not None:
            # 展平为 [batch * head_dim, input_bits]
            x_flat = x.reshape(-1, self.input_bits)

            # 向量化转换
            x_fp32_flat = self.input_converter(x_flat)

            # 恢复形状
            x_fp32 = x_fp32_flat.reshape(batch_size, self.head_dim, 32)
        else:
            x_fp32 = x

        # 应用 RoPE
        result_fp32 = self.rope_core(x_fp32, position)

        # 输出转换 (向量化)
        if self.output_converter is not None:
            # 展平
            result_flat = result_fp32.reshape(-1, 32)

            # 向量化转换
            result_out_flat = self.output_converter(result_flat)

            # 恢复形状
            result = result_out_flat.reshape(batch_size, self.head_dim, self.input_bits)
        else:
            result = result_fp32

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 便捷别名
# ==============================================================================

# FP8 RoPE
class SpikeFP8RoPE(SpikeRoPE_MultiPrecision):
    """FP8 旋转位置编码"""
    def __init__(self, head_dim, base=10000.0, intermediate_precision='fp32',
                 neuron_template=None):
        super().__init__(head_dim, base, 'fp8', intermediate_precision, neuron_template)


# FP16 RoPE
class SpikeFP16RoPE(SpikeRoPE_MultiPrecision):
    """FP16 旋转位置编码"""
    def __init__(self, head_dim, base=10000.0, intermediate_precision='fp32',
                 neuron_template=None):
        super().__init__(head_dim, base, 'fp16', intermediate_precision, neuron_template)


# FP32 RoPE (别名)
SpikeFP32RoPE_MultiPrecision = SpikeFP32RoPE
