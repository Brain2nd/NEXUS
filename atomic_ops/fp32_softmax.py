"""
FP32 Softmax - 100%纯SNN门电路实现
====================================

Softmax(x_i) = exp(x_i) / sum(exp(x_j))

使用已有的Exp、Adder和Divider组合实现。

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from .fp32_exp import SpikeFP32Exp
from .fp32_adder import SpikeFP32Adder
from .fp32_div import SpikeFP32Divider


class SpikeFP32Softmax(nn.Module):
    """FP32 Softmax - 100%纯SNN门电路实现
    
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    输入: x [..., N, 32] FP32脉冲，其中N是softmax维度
    输出: Softmax(x) [..., N, 32] FP32脉冲
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        self.exp = SpikeFP32Exp(neuron_template=nt)
        self.adder = SpikeFP32Adder(neuron_template=nt)
        self.divider = SpikeFP32Divider(neuron_template=nt)
        
    def forward(self, x):
        """
        x: [..., N, 32] FP32脉冲
        Returns: Softmax(x) [..., N, 32] FP32脉冲
        """
        self.reset()
        device = x.device
        
        # x的形状: [..., N, 32]
        batch_shape = x.shape[:-2]
        N = x.shape[-2]
        
        # Step 1: 计算 exp(x_i) for all i
        exp_x = self.exp(x)  # [..., N, 32]
        
        # Step 2: 计算 sum(exp(x_j))
        sum_exp = exp_x[..., 0, :]  # [..., 32]
        for i in range(1, N):
            self.adder.reset()
            sum_exp = self.adder(sum_exp, exp_x[..., i, :])
        
        # Step 3: exp(x_i) / sum(exp(x_j)) for each i
        result = []
        for i in range(N):
            self.divider.reset()
            softmax_i = self.divider(exp_x[..., i, :], sum_exp)
            result.append(softmax_i.unsqueeze(-2))
        
        result = torch.cat(result, dim=-2)  # [..., N, 32]
        
        return result
    
    def reset(self):
        self.exp.reset()
        self.adder.reset()
        self.divider.reset()

