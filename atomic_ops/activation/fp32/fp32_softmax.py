"""
FP32 Softmax - 100%纯SNN门电路实现
====================================

Softmax(x_i) = exp(x_i) / sum(exp(x_j))

使用已有的Exp、Adder和Divider组合实现。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from .fp32_exp import SpikeFP32Exp
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp32.fp32_div import SpikeFP32Divider


class SpikeFP32Softmax(nn.Module):
    """FP32 Softmax - 100%纯SNN门电路实现

    Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    输入: x [..., N, 32] FP32脉冲，其中N是softmax维度
    输出: Softmax(x) [..., N, 32] FP32脉冲

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式（梯度流过）
    """
    def __init__(self, neuron_template=None, trainable=False):
        super().__init__()
        self.trainable = trainable
        nt = neuron_template
        
        self.exp = SpikeFP32Exp(neuron_template=nt)
        self.adder = SpikeFP32Adder(neuron_template=nt)
        self.divider = SpikeFP32Divider(neuron_template=nt)
        
    def forward(self, x):
        """
        x: [..., N, 32] FP32脉冲
        Returns: Softmax(x) [..., N, 32] FP32脉冲
        """
        N = x.shape[-2]

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            exp_x = self.exp(x)
            sum_exp = exp_x[..., 0, :]
            for i in range(1, N):
                sum_exp = self.adder(sum_exp, exp_x[..., i, :])
            sum_exp_expanded = sum_exp.unsqueeze(-2).expand_as(exp_x)
            out_pulse = self.divider(exp_x, sum_exp_expanded)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_softmax
            # pulse 格式的 softmax 维度是 -2 (N 维度)
            return ste_softmax(x, out_pulse, dim=-2)

        return out_pulse
    
    def reset(self):
        self.exp.reset()
        self.adder.reset()
        self.divider.reset()

