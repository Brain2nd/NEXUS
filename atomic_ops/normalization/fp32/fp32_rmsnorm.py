"""
FP32 RMSNorm - 全链路FP64高精度实现
======================================

RMSNorm(x) = x * w / sqrt(mean(x^2) + epsilon)

流程:
1. x (FP32) -> FP64
2. x^2 (FP64 Mul)
3. sum(x^2) (FP64 Adder Tree/Accumulator)
4. mean = sum / N (FP64 Div)
5. rms = sqrt(mean + eps) (FP64 Sqrt)
6. scale = 1 / rms (FP64 Div)
7. y = x * scale * w (FP64 Mul)
8. y -> FP32

100% 纯SNN门电路实现。

使用示例：
```python
# 推理模式 (默认)
rmsnorm = SpikeFP32RMSNormFullFP64(dim=64)
y_pulse = rmsnorm(x_pulse)  # 纯 SNN

# 训练模式
rmsnorm = SpikeFP32RMSNormFullFP64(dim=64, training_mode=TrainingMode.STE)
rmsnorm.train()
optimizer = torch.optim.Adam([rmsnorm.weight], lr=1e-4)
```
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.core.accumulator import create_accumulator
from atomic_ops.core.logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate)
from atomic_ops.arithmetic.fp64.fp64_mul import SpikeFP64Multiplier
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_div import SpikeFP64Divider
from atomic_ops.arithmetic.fp64.fp64_sqrt import SpikeFP64Sqrt
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from atomic_ops.activation.fp64.fp64_exp import make_fp64_constant

class SpikeFP32RMSNormFullFP64(nn.Module):
    """FP32 RMSNorm 高精度版

    Args:
        normalized_shape: 输入特征维度 (int)
        eps: epsilon (default: 1e-6)
        neuron_template: 神经元模板
        training_mode: 训练模式 (None/TrainingMode.STE/TrainingMode.TEMPORAL)
            - False (默认): 纯推理模式
            - True: 训练模式，使用 STE 反向传播
        accumulator_mode: 累加模式 ('sequential' 或 'parallel')
    """
    def __init__(self, normalized_shape, eps=1e-6, neuron_template=None,
                 training_mode=None, accumulator_mode='sequential'):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.dim = normalized_shape
        else:
            self.dim = normalized_shape[0]

        self.eps = eps
        self.training_mode = TrainingMode.validate(training_mode)
        nt = neuron_template

        self.to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.to_fp64_w = FP32ToFP64Converter(neuron_template=nt)  # Separate converter for weights
        self.to_fp32 = FP64ToFP32Converter(neuron_template=nt)

        self.mul_sq = SpikeFP64Multiplier(neuron_template=nt)

        # 解耦累加器 - 使用可复用的加法器 + 累加策略
        self.adder_accum = SpikeFP64Adder(neuron_template=nt)
        self.accumulator = create_accumulator(self.adder_accum, mode=accumulator_mode)

        self.div_mean = SpikeFP64Divider(neuron_template=nt)
        self.adder_eps = SpikeFP64Adder(neuron_template=nt)
        self.sqrt = SpikeFP64Sqrt(iterations=12, neuron_template=nt)  # 12次迭代确保最高精度
        self.div_inv = SpikeFP64Divider(neuron_template=nt)
        self.mul_scale = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_w = SpikeFP64Multiplier(neuron_template=nt)

        # 权重 (always a Parameter for compatibility, but gradient flow depends on trainable)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        device = x.device
        batch_shape = x.shape[:-2]

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # 1. FP32 -> FP64
            x_fp64 = self.to_fp64(x)

            # 2. x^2
            x_sq = self.mul_sq(x_fp64, x_fp64)

            # 3. Sum(x^2) using decoupled accumulator
            sum_sq = self.accumulator.reduce(x_sq, dim=-2)

            # 4. Mean = Sum / N
            N_const = make_fp64_constant(float(self.dim), batch_shape, device)
            mean = self.div_mean(sum_sq, N_const)

            # 5. Sqrt(Mean + eps)
            eps_const = make_fp64_constant(self.eps, batch_shape, device)
            radicand = self.adder_eps(mean, eps_const)
            rms = self.sqrt(radicand)

            # 6. Scale = 1 / rms
            one_const = make_fp64_constant(1.0, batch_shape, device)
            inv_rms = self.div_inv(one_const, rms)

            # 7. y = x * inv_rms * w
            # Broadcast inv_rms: [..., 64] -> [..., dim, 64]
            inv_rms_expanded = inv_rms.unsqueeze(-2).expand_as(x_fp64)

            x_norm = self.mul_scale(x_fp64, inv_rms_expanded)

            # Weight multiplication
            w_pulse_32 = self._float_to_pulse_tensor(self.weight.data, device)
            w_pulse_64 = self.to_fp64_w(w_pulse_32)  # Use separate converter for weights
            w_expanded = w_pulse_64.expand_as(x_norm)

            y_fp64 = self.mul_w(x_norm, w_expanded)

            # 8. FP64 -> FP32
            out_pulse = self.to_fp32(y_fp64)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_rmsnorm
            return ste_rmsnorm(x, self.weight, out_pulse, self.eps)

        return out_pulse

    def _float_to_pulse_tensor(self, t, device):
        """Convert tensor of floats to tensor of bits [..., 32] (向量化实现)"""
        from atomic_ops.encoding.converters import float32_to_pulse
        return float32_to_pulse(t.to(device), device=device)

    def reset(self):
        self.to_fp64.reset()
        self.to_fp64_w.reset()
        self.to_fp32.reset()
        self.mul_sq.reset()
        self.adder_accum.reset()
        self.accumulator.reset()
        self.div_mean.reset()
        self.adder_eps.reset()
        self.sqrt.reset()
        self.div_inv.reset()
        self.mul_scale.reset()
        self.mul_w.reset()
