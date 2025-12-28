"""
FP32 Sigmoid函数 - 100%纯SNN门电路实现
=========================================

sigmoid(x) = 1 / (1 + exp(-x))

使用已有的Exp、Adder、Divider组合实现。

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from .logic_gates import NOTGate, MUXGate, ANDGate, ORGate
from .fp32_exp import SpikeFP32Exp
from .fp32_adder import SpikeFP32Adder
from .fp32_div import SpikeFP32Divider


class SpikeFP32Sigmoid(nn.Module):
    """FP32 Sigmoid函数 - 100%纯SNN门电路实现
    
    sigmoid(x) = 1 / (1 + exp(-x))
    
    输入: x [..., 32] FP32脉冲
    输出: sigmoid(x) [..., 32] FP32脉冲
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # 组件
        self.exp = SpikeFP32Exp(neuron_template=nt)
        self.adder = SpikeFP32Adder(neuron_template=nt)
        self.divider = SpikeFP32Divider(neuron_template=nt)
        
        # 符号翻转 (用于计算-x)
        self.sign_not = NOTGate(neuron_template=nt)
        
    def forward(self, x):
        """
        x: [..., 32] FP32脉冲
        Returns: sigmoid(x) [..., 32] FP32脉冲
        """
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # Step 1: 计算 -x (翻转符号位)
        neg_x_sign = self.sign_not(x[..., 0:1])
        neg_x = torch.cat([neg_x_sign, x[..., 1:]], dim=-1)
        
        # Step 2: exp(-x)
        exp_neg_x = self.exp(neg_x)
        
        # Step 3: 1 + exp(-x)
        # 构造常量1.0 = 0x3F800000
        c1 = self._make_constant(0x3F800000, batch_shape, device)
        one_plus_exp = self.adder(c1, exp_neg_x)
        
        # Step 4: 1 / (1 + exp(-x))
        result = self.divider(c1, one_plus_exp)
        
        return result
    
    def _make_constant(self, bits, batch_shape, device):
        """从32位整数构造FP32脉冲常量"""
        pulse = torch.zeros(batch_shape + (32,), device=device)
        for i in range(32):
            pulse[..., i] = float((bits >> (31 - i)) & 1)
        return pulse
    
    def reset(self):
        self.exp.reset()
        self.adder.reset()
        self.divider.reset()
        self.sign_not.reset()

