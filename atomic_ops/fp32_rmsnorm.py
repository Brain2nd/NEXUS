"""
FP32 RMSNorm - 全链路FP64高精度实现
======================================

RMSNorm(x) = x * w / sqrt(mean(x^2) + epsilon)

流程:
1. x (FP32) -> FP64
2. x^2 (FP64 Mul)
3. sum(x^2) (FP64 Adder Tree/Accumulator)
4. mean = sum / N (FP64 Div)
5. rms = sqrt(mean + eps) (FP64 Sqrt)
6. scale = 1 / rms (FP64 Div)
7. y = x * scale * w (FP64 Mul)
8. y -> FP32

100% 纯SNN门电路实现。
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate)
from .fp64_mul import SpikeFP64Multiplier
from .fp64_adder import SpikeFP64Adder
from .fp64_div import SpikeFP64Divider
from .fp64_sqrt import SpikeFP64Sqrt
from .fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from .fp64_exp import make_fp64_constant

class SpikeFP32RMSNormFullFP64(nn.Module):
    """FP32 RMSNorm 高精度版
    
    Args:
        normalized_shape: 输入特征维度 (int)
        eps: epsilon (default: 1e-6)
    """
    def __init__(self, normalized_shape, eps=1e-6, neuron_template=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.dim = normalized_shape
        else:
            self.dim = normalized_shape[0]

        self.eps = eps
        nt = neuron_template

        self.to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.to_fp32 = FP64ToFP32Converter(neuron_template=nt)

        self.mul_sq = SpikeFP64Multiplier(neuron_template=nt)

        # Reduction Tree Adders: Need dim-1 adders
        self.adders_tree = nn.ModuleList([SpikeFP64Adder(neuron_template=nt) for _ in range(self.dim - 1)])

        self.div_mean = SpikeFP64Divider(neuron_template=nt)
        self.adder_eps = SpikeFP64Adder(neuron_template=nt)
        self.sqrt = SpikeFP64Sqrt(iterations=12, neuron_template=nt)  # 12次迭代确保最高精度
        self.div_inv = SpikeFP64Divider(neuron_template=nt)
        self.mul_scale = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_w = SpikeFP64Multiplier(neuron_template=nt)
        
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        self.reset()
        device = x.device
        batch_shape = x.shape[:-2]
        
        # 1. FP32 -> FP64
        x_fp64 = self.to_fp64(x)
        
        # 2. x^2
        x_sq = self.mul_sq(x_fp64, x_fp64)
        
        # 3. Sum(x^2) using tree
        # List of tensors to sum
        to_sum = [x_sq[..., i, :] for i in range(self.dim)]
        
        adder_idx = 0
        while len(to_sum) > 1:
            new_sum = []
            # Pairwise sum
            for i in range(0, len(to_sum), 2):
                if i + 1 < len(to_sum):
                    # Use unique adder
                    res = self.adders_tree[adder_idx](to_sum[i], to_sum[i+1])
                    adder_idx += 1
                    new_sum.append(res)
                else:
                    new_sum.append(to_sum[i])
            to_sum = new_sum
            
        sum_sq = to_sum[0]
        
        # 4. Mean = Sum / N
        N_const = make_fp64_constant(float(self.dim), batch_shape, device)
        mean = self.div_mean(sum_sq, N_const)
        
        # 5. Sqrt(Mean + eps)
        eps_const = make_fp64_constant(self.eps, batch_shape, device)
        radicand = self.adder_eps(mean, eps_const)
        rms = self.sqrt(radicand)
        
        # 6. Scale = 1 / rms
        one_const = make_fp64_constant(1.0, batch_shape, device)
        inv_rms = self.div_inv(one_const, rms)
        
        # 7. y = x * inv_rms * w
        # Broadcast inv_rms: [..., 64] -> [..., dim, 64]
        inv_rms_expanded = inv_rms.unsqueeze(-2).expand_as(x_fp64)
        
        x_norm = self.mul_scale(x_fp64, inv_rms_expanded)
        
        # Weight multiplication
        w_pulse_32 = self._float_to_pulse_tensor(self.weight, device)
        w_pulse_64 = self.to_fp64(w_pulse_32)
        w_expanded = w_pulse_64.expand_as(x_norm)
        
        y_fp64 = self.mul_w(x_norm, w_expanded)
        
        # 8. FP64 -> FP32
        y = self.to_fp32(y_fp64)
        
        return y

    def _float_to_pulse_tensor(self, t, device):
        # Convert tensor of floats to tensor of bits [..., 32]
        import struct
        flat = t.view(-1)
        pulses = []
        for val in flat:
            bits = struct.unpack('>I', struct.pack('>f', val.item()))[0]
            p = torch.zeros(32, device=device)
            for i in range(32):
                p[i] = float((bits >> (31-i)) & 1)
            pulses.append(p)
        return torch.stack(pulses).view(t.shape + (32,))

    def reset(self):
        self.to_fp64.reset()
        self.to_fp32.reset()
        self.mul_sq.reset()
        self.div_mean.reset()
        self.adder_eps.reset()
        self.sqrt.reset()
        self.div_inv.reset()
        self.mul_scale.reset()
        self.mul_w.reset()
        
        if hasattr(self, 'adders_tree'):
            for a in self.adders_tree: a.reset()
