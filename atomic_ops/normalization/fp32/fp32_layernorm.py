"""
FP32 LayerNorm - 100%纯SNN门电路实现 (向量化版本)
=================================================

LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)

其中:
- mean(x) = sum(x) / N
- var(x) = sum((x - mean)^2) / N

使用FP64内部精度以减少累积误差。

向量化优化:
- 累加操作使用解耦的 Accumulator
- 元素级操作 (中心化、平方、归一化) 使用向量化广播

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.core.accumulator import create_accumulator
import struct
from atomic_ops.core.logic_gates import NOTGate
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_mul import SpikeFP64Multiplier
from atomic_ops.arithmetic.fp64.fp64_div import SpikeFP64Divider
from atomic_ops.arithmetic.fp64.fp64_sqrt import SpikeFP64Sqrt


def make_fp64_constant(val, batch_shape, device):
    """从浮点数构造FP64脉冲常量 (向量化实现)"""
    bits = struct.unpack('>Q', struct.pack('>d', val))[0]
    # 向量化位提取
    bit_positions = torch.arange(63, -1, -1, device=device)
    bit_values = ((bits >> bit_positions) & 1).float()
    # 广播到 batch_shape
    pulse = bit_values.expand(batch_shape + (64,)).clone()
    return pulse


class SpikeFP32LayerNorm(nn.Module):
    """FP32 LayerNorm - 100%纯SNN门电路实现 (向量化版本)

    LayerNorm(x) = (x - mean) / sqrt(var + eps)

    输入: x [..., N, 32] FP32脉冲，其中N是归一化维度
    输出: LayerNorm(x) [..., N, 32] FP32脉冲

    Args:
        eps: epsilon (default: 1e-6)
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式 (None/TrainingMode.STE/TrainingMode.TEMPORAL)
        accumulator_mode: 累加模式 ('sequential' 或 'parallel')
    """
    def __init__(self, eps=1e-6, neuron_template=None, training_mode=None,
                 accumulator_mode='sequential'):
        super().__init__()
        self.eps = eps
        self.training_mode = TrainingMode.validate(training_mode)
        nt = neuron_template

        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)

        # 加法器 - 用于累加和元素级操作
        self.fp64_adder_sum = SpikeFP64Adder(neuron_template=nt)
        self.fp64_adder_center = SpikeFP64Adder(neuron_template=nt)
        self.fp64_adder_var_sum = SpikeFP64Adder(neuron_template=nt)
        self.fp64_adder_eps = SpikeFP64Adder(neuron_template=nt)

        # 乘法器
        self.fp64_mul_sq = SpikeFP64Multiplier(neuron_template=nt)
        self.fp64_mul_norm = SpikeFP64Multiplier(neuron_template=nt)

        # 除法器
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_divider_inv = SpikeFP64Divider(neuron_template=nt)

        # Sqrt (使用 SpikeFP64Sqrt 代替 Newton-Raphson)
        self.fp64_sqrt = SpikeFP64Sqrt(iterations=12, neuron_template=nt)

        # 解耦累加器
        self.accumulator_mean = create_accumulator(self.fp64_adder_sum, mode=accumulator_mode)
        self.accumulator_var = create_accumulator(self.fp64_adder_var_sum, mode=accumulator_mode)

        self.sign_not = NOTGate(neuron_template=nt, max_param_shape=(1,))

    def forward(self, x):
        """
        x: [..., N, 32] FP32脉冲
        Returns: LayerNorm(x) [..., N, 32] FP32脉冲
        """
        device = x.device
        batch_shape = x.shape[:-2]
        N = x.shape[-2]

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # FP32 -> FP64
            x_fp64 = self.fp32_to_fp64(x)  # [..., N, 64]

            # ============================================================
            # Step 1: 计算 mean(x) = sum(x) / N (使用累加器)
            # ============================================================
            sum_x = self.accumulator_mean.reduce(x_fp64, dim=-2)  # [..., 64]
            N_const = make_fp64_constant(float(N), batch_shape, device)
            mean_x = self.fp64_divider(sum_x, N_const)  # [..., 64]

            # ============================================================
            # Step 2: 计算 x - mean (向量化)
            # ============================================================
            # 广播 mean 到 [..., N, 64]
            neg_mean = mean_x.clone()
            neg_mean[..., 0:1] = self.sign_not(neg_mean[..., 0:1])
            neg_mean_expanded = neg_mean.unsqueeze(-2).expand_as(x_fp64)

            x_centered = self.fp64_adder_center(x_fp64, neg_mean_expanded)  # [..., N, 64]

            # ============================================================
            # Step 3: 计算 var = sum((x - mean)^2) / N
            # ============================================================
            # 向量化平方
            x_sq = self.fp64_mul_sq(x_centered, x_centered)  # [..., N, 64]

            # 使用累加器求和
            sum_sq = self.accumulator_var.reduce(x_sq, dim=-2)  # [..., 64]
            var = self.fp64_divider(sum_sq, N_const)

            # ============================================================
            # Step 4: var + eps
            # ============================================================
            eps_const = make_fp64_constant(self.eps, batch_shape, device)
            var_plus_eps = self.fp64_adder_eps(var, eps_const)

            # ============================================================
            # Step 5: sqrt(var + eps) 使用 SpikeFP64Sqrt
            # ============================================================
            std = self.fp64_sqrt(var_plus_eps)

            # ============================================================
            # Step 6: rsqrt = 1 / sqrt(var + eps)
            # ============================================================
            one_const = make_fp64_constant(1.0, batch_shape, device)
            rsqrt_var = self.fp64_divider_inv(one_const, std)

            # ============================================================
            # Step 7: (x - mean) * rsqrt(var + eps) (向量化)
            # ============================================================
            # 广播 rsqrt 到 [..., N, 64]
            rsqrt_expanded = rsqrt_var.unsqueeze(-2).expand_as(x_centered)
            result_fp64 = self.fp64_mul_norm(x_centered, rsqrt_expanded)  # [..., N, 64]

            # FP64 -> FP32
            out_pulse = self.fp64_to_fp32(result_fp64)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_layernorm
            return ste_layernorm(x, out_pulse, self.eps)

        return out_pulse

    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_to_fp32.reset()
        self.fp64_adder_sum.reset()
        self.fp64_adder_center.reset()
        self.fp64_adder_var_sum.reset()
        self.fp64_adder_eps.reset()
        self.fp64_mul_sq.reset()
        self.fp64_mul_norm.reset()
        self.fp64_divider.reset()
        self.fp64_divider_inv.reset()
        self.fp64_sqrt.reset()
        self.accumulator_mean.reset()
        self.accumulator_var.reset()
        self.sign_not.reset()
