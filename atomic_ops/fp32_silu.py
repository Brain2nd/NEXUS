"""
FP32 SiLU (Swish)函数 - 100%纯SNN门电路实现
=============================================

SiLU(x) = x * sigmoid(x)

使用已有的Multiplier和Sigmoid组合实现。

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from .fp32_mul import SpikeFP32Multiplier
from .fp32_sigmoid import SpikeFP32Sigmoid


class SpikeFP32SiLU(nn.Module):
    """FP32 SiLU (Swish)函数 - 100%纯SNN门电路实现
    
    SiLU(x) = x * sigmoid(x)
    
    输入: x [..., 32] FP32脉冲
    输出: SiLU(x) [..., 32] FP32脉冲
    """
    def __init__(self):
        super().__init__()
        
        self.sigmoid = SpikeFP32Sigmoid()
        self.mul = SpikeFP32Multiplier()
        
    def forward(self, x):
        """
        x: [..., 32] FP32脉冲
        Returns: SiLU(x) [..., 32] FP32脉冲
        """
        self.reset()
        
        # Step 1: sigmoid(x)
        sig_x = self.sigmoid(x)
        
        # Step 2: x * sigmoid(x)
        result = self.mul(x, sig_x)
        
        return result
    
    def reset(self):
        self.sigmoid.reset()
        self.mul.reset()

