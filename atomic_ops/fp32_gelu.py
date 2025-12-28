"""
FP32 GELU - 100%纯SNN门电路实现
================================

GELU(x) = x * Φ(x) ≈ x * σ(1.702 * x)

其中 Φ(x) 是标准正态分布的CDF，σ 是sigmoid函数。

使用快速近似: GELU(x) ≈ x * sigmoid(1.702 * x)
这个近似在实践中非常准确且计算效率高。

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


class SpikeFP32GELU(nn.Module):
    """FP32 GELU - 100%纯SNN门电路实现
    
    使用快速近似: GELU(x) ≈ x * sigmoid(1.702 * x)
    
    流程: FP32输入 -> FP64 -> 全部FP64计算 -> FP64 -> FP32输出
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_mul = SpikeFP64Multiplier(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        
        self.sign_not = NOTGate(neuron_template=nt)
        
    def forward(self, x):
        """
        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲 GELU(x)
        """
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)
        
        # 1.702 * x
        const_1702 = make_fp64_constant(1.702, batch_shape, device)
        x_scaled = self.fp64_mul(const_1702, x_fp64)
        
        # sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
        # -1.702 * x (翻转符号位)
        neg_x_scaled_sign = self.sign_not(x_scaled[..., 0:1])
        neg_x_scaled = torch.cat([neg_x_scaled_sign, x_scaled[..., 1:]], dim=-1)
        
        # exp(-1.702 * x)
        exp_neg = self.fp64_exp(neg_x_scaled)
        
        # 1 + exp(-1.702 * x)
        one_fp64 = make_fp64_constant(1.0, batch_shape, device)
        self.fp64_adder.reset()
        one_plus_exp = self.fp64_adder(one_fp64, exp_neg)
        
        # 1 / (1 + exp(-1.702 * x)) = sigmoid(1.702 * x)
        self.fp64_divider.reset()
        sigmoid_result = self.fp64_divider(one_fp64, one_plus_exp)
        
        # x * sigmoid(1.702 * x)
        self.fp64_mul.reset()
        result_fp64 = self.fp64_mul(x_fp64, sigmoid_result)
        
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
        self.sign_not.reset()


class SpikeFP32GELUExact(nn.Module):
    """FP32 GELU 精确版 - 使用完整公式
    
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    计算量更大，但更精确。
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_mul = SpikeFP64Multiplier(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        
        self.sign_not = NOTGate(neuron_template=nt)
        
    def forward(self, x):
        """
        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲 GELU(x)
        """
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)
        
        # 常量
        const_0_5 = make_fp64_constant(0.5, batch_shape, device)
        const_1 = make_fp64_constant(1.0, batch_shape, device)
        const_2 = make_fp64_constant(2.0, batch_shape, device)
        const_0044715 = make_fp64_constant(0.044715, batch_shape, device)
        sqrt_2_pi = make_fp64_constant(0.7978845608028654, batch_shape, device)  # sqrt(2/π)
        
        # x³ = x * x * x
        x_sq = self.fp64_mul(x_fp64, x_fp64)
        self.fp64_mul.reset()
        x_cubed = self.fp64_mul(x_sq, x_fp64)
        
        # 0.044715 * x³
        self.fp64_mul.reset()
        term = self.fp64_mul(const_0044715, x_cubed)
        
        # x + 0.044715 * x³
        inner = self.fp64_adder(x_fp64, term)
        
        # sqrt(2/π) * (x + 0.044715 * x³)
        self.fp64_mul.reset()
        inner_scaled = self.fp64_mul(sqrt_2_pi, inner)
        
        # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        self.fp64_mul.reset()
        two_z = self.fp64_mul(const_2, inner_scaled)
        
        exp_2z = self.fp64_exp(two_z)
        
        # exp(2z) - 1
        neg_1 = make_fp64_constant(-1.0, batch_shape, device)
        self.fp64_adder.reset()
        exp_minus_1 = self.fp64_adder(exp_2z, neg_1)
        
        # exp(2z) + 1
        self.fp64_adder.reset()
        exp_plus_1 = self.fp64_adder(exp_2z, const_1)
        
        # tanh = (exp(2z) - 1) / (exp(2z) + 1)
        tanh_result = self.fp64_divider(exp_minus_1, exp_plus_1)
        
        # 1 + tanh(...)
        self.fp64_adder.reset()
        one_plus_tanh = self.fp64_adder(const_1, tanh_result)
        
        # x * (1 + tanh(...))
        self.fp64_mul.reset()
        x_times = self.fp64_mul(x_fp64, one_plus_tanh)
        
        # 0.5 * x * (1 + tanh(...))
        self.fp64_mul.reset()
        result_fp64 = self.fp64_mul(const_0_5, x_times)
        
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
        self.sign_not.reset()


