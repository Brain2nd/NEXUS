"""
FP32 倒数函数 - 100%纯SNN门电路实现
======================================

使用除法器实现: 1/x = 1.0 / x

完全避免迭代近似算法，保证100%位精确。

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from .fp32_div import SpikeFP32Divider


class SpikeFP32Reciprocal(nn.Module):
    """FP32 倒数 1/x - 100%纯SNN门电路实现
    
    使用除法器计算: 1/x = 1.0 / x
    
    保证100%位精确，与PyTorch完全一致。
    
    输入: x [..., 32] FP32脉冲
    输出: 1/x [..., 32] FP32脉冲
    
    特殊情况由除法器自动处理:
    - 1/0 = Inf
    - 1/Inf = 0
    - 1/NaN = NaN
    """
    def __init__(self):
        super().__init__()
        self.divider = SpikeFP32Divider()
        
    def forward(self, x):
        """
        x: [..., 32] FP32脉冲
        Returns: 1/x [..., 32] FP32脉冲
        """
        device = x.device
        batch_shape = x.shape[:-1]
        
        # 构建常量 1.0: [0, 01111111, 00...0]
        # 符号=0, 指数=127 (0b01111111), 尾数=0
        ones = torch.ones(batch_shape + (1,), device=device)
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        const_1 = torch.cat([
            zeros,  # 符号 = 0
            zeros, ones, ones, ones, ones, ones, ones, ones,  # 指数 = 127 = 0b01111111
        ] + [zeros]*23, dim=-1)  # 尾数 = 0
        
        # 1/x = 1.0 / x
        return self.divider(const_1, x)
    
    def reset(self):
        self.divider.reset()

