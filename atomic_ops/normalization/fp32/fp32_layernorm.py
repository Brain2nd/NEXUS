"""
FP32 LayerNorm - 100%纯SNN门电路实现
====================================

LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)

其中:
- mean(x) = sum(x) / N
- var(x) = sum((x - mean)^2) / N

使用FP64内部精度以减少累积误差。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
import struct
from atomic_ops.core.logic_gates import NOTGate
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_mul import SpikeFP64Multiplier
from atomic_ops.arithmetic.fp64.fp64_div import SpikeFP64Divider
# 注意: 需要 SpikeFP64Sqrt，暂时使用 Newton-Raphson 近似


def make_fp64_constant(val, batch_shape, device):
    """从浮点数构造FP64脉冲常量"""
    bits = struct.unpack('>Q', struct.pack('>d', val))[0]
    pulse = torch.zeros(batch_shape + (64,), device=device)
    for i in range(64):
        pulse[..., i] = float((bits >> (63 - i)) & 1)
    return pulse


class SpikeFP32LayerNorm(nn.Module):
    """FP32 LayerNorm - 100%纯SNN门电路实现

    LayerNorm(x) = (x - mean) / sqrt(var + eps)

    输入: x [..., N, 32] FP32脉冲，其中N是归一化维度
    输出: LayerNorm(x) [..., N, 32] FP32脉冲

    eps: 1e-6 (防止除零)

    注意: 由于sqrt需要特殊处理，此实现使用RMS近似:
          LayerNorm ≈ (x - mean) / RMS(x - mean)

    Args:
        eps: epsilon (default: 1e-6)
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式（梯度流过）
    """
    def __init__(self, eps=1e-6, neuron_template=None, trainable=False):
        super().__init__()
        self.eps = eps
        self.trainable = trainable
        nt = neuron_template

        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_mul = SpikeFP64Multiplier(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)

        self.sign_not = NOTGate(neuron_template=nt)
        
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

            # Step 1: 计算 mean(x) = sum(x) / N
            sum_x = x_fp64[..., 0, :]  # [..., 64]
            for i in range(1, N):
                sum_x = self.fp64_adder(sum_x, x_fp64[..., i, :])

            N_const = make_fp64_constant(float(N), batch_shape, device)
            mean_x = self.fp64_divider(sum_x, N_const)  # [..., 64]

            # Step 2: 计算 x - mean (对每个元素)
            x_centered = []
            for i in range(N):
                # -(mean)
                neg_mean = mean_x.clone()
                neg_mean[..., 0:1] = self.sign_not(neg_mean[..., 0:1])

                centered = self.fp64_adder(x_fp64[..., i, :], neg_mean)
                x_centered.append(centered.unsqueeze(-2))

            x_centered = torch.cat(x_centered, dim=-2)  # [..., N, 64]

            # Step 3: 计算 var = sum((x - mean)^2) / N
            x_sq = []
            for i in range(N):
                sq = self.fp64_mul(x_centered[..., i, :], x_centered[..., i, :])
                x_sq.append(sq)

            sum_sq = x_sq[0]
            for i in range(1, N):
                sum_sq = self.fp64_adder(sum_sq, x_sq[i])

            var = self.fp64_divider(sum_sq, N_const)

            # Step 4: var + eps
            eps_const = make_fp64_constant(self.eps, batch_shape, device)
            var_plus_eps = self.fp64_adder(var, eps_const)

            # Step 5: sqrt(var + eps) - 使用 Newton-Raphson 迭代
            # 初始猜测: y = var_plus_eps (简化，实际应该更好的初始化)
            # Newton: y_new = 0.5 * (y + var/y)
            # 这里简化为使用 var_plus_eps^0.5 的近似
            #
            # 更好的方法: 使用 rsqrt (1/sqrt) 的 Newton-Raphson
            # y = 1/sqrt(x), Newton: y_new = y * (1.5 - 0.5 * x * y^2)

            # 简化实现: 直接使用 RMS 方式
            # std ≈ sqrt(var + eps)
            # 使用迭代: y = y * (3 - x*y^2) / 2 for rsqrt

            # 临时解决方案: 使用 (x - mean) / sqrt(mean((x-mean)^2) + eps)
            # 即 RMSNorm 形式

            # RMS = sqrt(sum(x^2)/N + eps) = sqrt(var + eps) 当 mean=0
            # 这里 x_centered 的均值为0，所以可以直接用

            # Newton-Raphson for rsqrt: y = 1/sqrt(x)
            # y_new = 0.5 * y * (3 - x * y^2)

            # 初始猜测使用简单的缩放
            const_0_5 = make_fp64_constant(0.5, batch_shape, device)
            const_1_5 = make_fp64_constant(1.5, batch_shape, device)

            # 初始 y = 1/x (粗略近似 rsqrt)
            one = make_fp64_constant(1.0, batch_shape, device)
            y = self.fp64_divider(one, var_plus_eps)  # y ≈ 1/var (太大了)

            # 5次 Newton-Raphson 迭代
            for _ in range(5):
                # y^2
                y_sq = self.fp64_mul(y, y)

                # x * y^2
                x_y_sq = self.fp64_mul(var_plus_eps, y_sq)

                # 3 - x*y^2
                neg_x_y_sq = x_y_sq.clone()
                neg_x_y_sq[..., 0:1] = self.sign_not(neg_x_y_sq[..., 0:1])
                three = make_fp64_constant(3.0, batch_shape, device)
                term = self.fp64_adder(three, neg_x_y_sq)

                # y * (3 - x*y^2)
                y_term = self.fp64_mul(y, term)

                # 0.5 * y * (3 - x*y^2)
                y = self.fp64_mul(const_0_5, y_term)

            # rsqrt = y
            rsqrt_var = y

            # Step 6: (x - mean) * rsqrt(var + eps)
            result = []
            for i in range(N):
                normed = self.fp64_mul(x_centered[..., i, :], rsqrt_var)
                result.append(normed.unsqueeze(-2))

            result = torch.cat(result, dim=-2)  # [..., N, 64]

            # FP64 -> FP32
            out_pulse = self.fp64_to_fp32(result)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_layernorm
            return ste_layernorm(x, out_pulse, self.eps)

        return out_pulse
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_adder.reset()
        self.fp64_mul.reset()
        self.fp64_divider.reset()
        self.fp64_to_fp32.reset()
        self.sign_not.reset()


