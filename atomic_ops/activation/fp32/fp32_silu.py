"""
FP32 SiLU (Swish)函数 - 100%纯SNN门电路实现
=============================================

SiLU(x) = x * sigmoid(x)

使用已有的Multiplier和Sigmoid组合实现。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from .fp32_sigmoid import SpikeFP32Sigmoid


class SpikeFP32SiLU(nn.Module):
    """FP32 SiLU (Swish)函数 - 100%纯SNN门电路实现

    SiLU(x) = x * sigmoid(x)

    输入: x [..., 32] FP32脉冲
    输出: SiLU(x) [..., 32] FP32脉冲

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式 (None/TrainingMode.STE/TrainingMode.TEMPORAL)（梯度流过）
    """
    def __init__(self, neuron_template=None, training_mode=None):
        super().__init__()
        self.training_mode = TrainingMode.validate(training_mode)
        nt = neuron_template

        self.sigmoid = SpikeFP32Sigmoid(neuron_template=nt)
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

    def forward(self, x):
        """
        x: [..., 32] FP32脉冲
        Returns: SiLU(x) [..., 32] FP32脉冲
        """
        # SNN 前向 (纯门电路)
        with torch.no_grad():
            sig_x = self.sigmoid(x)
            out_pulse = self.mul(x, sig_x)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_silu
            return ste_silu(x, out_pulse, sig_x)

        return out_pulse

    def reset(self):
        self.sigmoid.reset()
        self.mul.reset()

