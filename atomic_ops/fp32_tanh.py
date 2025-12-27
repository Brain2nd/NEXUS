"""
FP32 Tanh - 100%纯SNN门电路实现
=================================

tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        = (e^2x - 1) / (e^2x + 1)
        = 2 * sigmoid(2x) - 1

使用最后一个形式实现，因为它只需要一次exp计算。

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
import struct
from .logic_gates import NOTGate
from .fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from .fp64_exp import SpikeFP64Exp
from .fp64_adder import SpikeFP64Adder
from .fp64_mul import SpikeFP64Multiplier
from .fp64_div import SpikeFP64Divider


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
    """
    def __init__(self):
        super().__init__()
        
        self.fp32_to_fp64 = FP32ToFP64Converter()
        self.fp64_mul = SpikeFP64Multiplier()
        self.fp64_exp = SpikeFP64Exp()
        self.fp64_adder = SpikeFP64Adder()
        self.fp64_divider = SpikeFP64Divider()
        self.fp64_to_fp32 = FP64ToFP32Converter()
        
    def forward(self, x):
        """
        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲 tanh(x)
        """
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)
        
        # 常量
        const_2 = make_fp64_constant(2.0, batch_shape, device)
        const_1 = make_fp64_constant(1.0, batch_shape, device)
        const_neg1 = make_fp64_constant(-1.0, batch_shape, device)
        
        # 2x
        two_x = self.fp64_mul(const_2, x_fp64)
        
        # e^2x
        exp_2x = self.fp64_exp(two_x)
        
        # e^2x - 1
        exp_minus_1 = self.fp64_adder(exp_2x, const_neg1)
        
        # e^2x + 1
        self.fp64_adder.reset()
        exp_plus_1 = self.fp64_adder(exp_2x, const_1)
        
        # (e^2x - 1) / (e^2x + 1)
        result_fp64 = self.fp64_divider(exp_minus_1, exp_plus_1)
        
        # FP64 -> FP32
        result_fp32 = self.fp64_to_fp32(result_fp64)
        
        return result_fp32
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_mul.reset()
        self.fp64_exp.reset()
        self.fp64_adder.reset()
        self.fp64_divider.reset()
        self.fp64_to_fp32.reset()


