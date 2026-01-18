"""
FP32 Tanh - 100%纯SNN门电路实现
=================================

tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        = (e^2x - 1) / (e^2x + 1)
        = 2 * sigmoid(2x) - 1

使用最后一个形式实现，因为它只需要一次exp计算。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
import struct
from atomic_ops.core.logic_gates import NOTGate
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from atomic_ops.activation.fp64.fp64_exp import SpikeFP64Exp
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_mul import SpikeFP64Multiplier
from atomic_ops.arithmetic.fp64.fp64_div import SpikeFP64Divider


def make_fp64_constant(val, batch_shape, device):
    """从浮点数构造FP64脉冲常量"""
    bits = struct.unpack('>Q', struct.pack('>d', val))[0]
    pulse = torch.zeros(batch_shape + (64,), device=device)
    for i in range(64):
        pulse[..., i] = float((bits >> (63 - i)) & 1)
    return pulse


class SpikeFP32Tanh(nn.Module):
    """FP32 Tanh - 100%纯SNN门电路实现

    tanh(x) = (e^2x - 1) / (e^2x + 1)

    流程: FP32输入 -> FP64 -> 全部FP64计算 -> FP64 -> FP32输出

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式（梯度流过）
    """
    def __init__(self, neuron_template=None, trainable=False):
        super().__init__()
        self.trainable = trainable
        nt = neuron_template

        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_mul = SpikeFP64Multiplier(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)

    def forward(self, x):
        """
        x: [..., 32] FP32脉冲
        Returns: tanh(x) [..., 32] FP32脉冲
        """
        device = x.device
        batch_shape = x.shape[:-1]

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            x_fp64 = self.fp32_to_fp64(x)
            const_2 = make_fp64_constant(2.0, batch_shape, device)
            const_1 = make_fp64_constant(1.0, batch_shape, device)
            const_neg1 = make_fp64_constant(-1.0, batch_shape, device)
            two_x = self.fp64_mul(const_2, x_fp64)
            exp_2x = self.fp64_exp(two_x)
            exp_minus_1 = self.fp64_adder(exp_2x, const_neg1)
            exp_plus_1 = self.fp64_adder(exp_2x, const_1)
            result_fp64 = self.fp64_divider(exp_minus_1, exp_plus_1)
            out_pulse = self.fp64_to_fp32(result_fp64)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_tanh
            return ste_tanh(x, out_pulse)

        return out_pulse
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_mul.reset()
        self.fp64_exp.reset()
        self.fp64_adder.reset()
        self.fp64_divider.reset()
        self.fp64_to_fp32.reset()


