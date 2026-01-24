"""
FP64 平方根函数 Sqrt(x) - 100%纯SNN门电路实现 (向量化版本)
==============================================

使用改进的 Newton-Raphson 迭代，带有精确的初始猜测。

y_{n+1} = 0.5 * (y_n + x / y_n)

向量化原则:
1. 使用 VecAND, VecOR, VecXOR, VecNOT, VecMUX 代替 ModuleList
2. 可并行操作一次处理所有位
3. 串行依赖仍保留循环
"""
import torch
import torch.nn as nn
import struct
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecORTree, VecANDTree, VecAdder
)
from .fp64_mul import SpikeFP64Multiplier
from .fp64_adder import SpikeFP64Adder
from .fp64_div import SpikeFP64Divider


# ==============================================================================
# FP64 辅助函数
# ==============================================================================
def float64_to_bits(f):
    return struct.unpack('>Q', struct.pack('>d', f))[0]

def make_fp64_constant(val, batch_shape, device):
    bits = float64_to_bits(val)
    pulse = torch.zeros(batch_shape + (64,), device=device)
    for i in range(64):
        pulse[..., i] = float((bits >> (63 - i)) & 1)
    return pulse


# ==============================================================================
# FP64 初始猜测生成器 - 向量化版本
# ==============================================================================
class SpikeFP64SqrtGuess(nn.Module):
    """生成 Sqrt(x) 的初始猜测值 y_0 - 向量化版本
    
    对于 x = 2^E * 1.M:
    - 如果 E 是偶数: y_0 ≈ 2^(E/2) * 1.0
    - 如果 E 是奇数: y_0 ≈ 2^((E-1)/2) * sqrt(2)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 预分配参数形状
        max_shape_52 = (52,)
        max_shape_11 = (11,)
        max_shape_1 = (1,)

        # 向量化门电路
        self.vec_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_52)

        # 指数加法器
        self.exp_adder = VecAdder(11, neuron_template=nt, max_param_shape=max_shape_11)
        
    def forward(self, x):
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        s_x = x[..., 0:1]
        e_x = x[..., 1:12]
        m_x = x[..., 12:64]
        
        # 1. E_real = E - 1023 (使用补码减法)
        e_x_le = e_x.flip(-1)  # LSB first
        
        # -1023 的补码 = NOT(1023) + 1 = NOT(0b01111111111) + 1
        # 1023 = 01111111111, NOT = 10000000000, +1 = 10000000001
        neg_1023_le = torch.cat([ones, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, ones], dim=-1)
        
        e_real_le, _ = self.exp_adder(e_x_le, neg_1023_le)
        
        # 2. 奇偶性 = E_real 的 LSB
        is_odd = e_real_le[..., 0:1]
        
        # 3. E_half = E_real >> 1 (算术右移)
        e_half_le = torch.cat([e_real_le[..., 1:], e_real_le[..., 10:11]], dim=-1)
        
        # 4. E_new = E_half + 1023
        bias_le = torch.cat([ones]*10 + [zeros], dim=-1)  # 1023 LSB first
        e_new_le, _ = self.exp_adder(e_half_le, bias_le)
        e_new = e_new_le.flip(-1)  # MSB first
        
        # 5. 尾数初始猜测 (向量化)
        # sqrt(2) 的 FP64 尾数部分: 0x6A09E667F3BCD
        sqrt2_mant_bits = 0x6A09E667F3BCD
        
        sqrt2_mant = torch.zeros(batch_shape + (52,), device=device)
        for i in range(52):
            bit_idx = 51 - i  # MSB first
            if (sqrt2_mant_bits >> bit_idx) & 1:
                sqrt2_mant[..., i] = 1.0
        
        zero_mant = torch.zeros(batch_shape + (52,), device=device)
        
        # 使用向量化MUX选择尾数
        m_new = self.vec_mux(is_odd.expand_as(sqrt2_mant), sqrt2_mant, zero_mant)
        
        return torch.cat([zeros, e_new, m_new], dim=-1)

    def reset(self):
        self.vec_not.reset()
        self.vec_xor.reset()
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_mux.reset()
        self.exp_adder.reset()


# ==============================================================================
# FP64 Sqrt 主模块 - 向量化版本
# ==============================================================================
class SpikeFP64Sqrt(nn.Module):
    """FP64 Square Root - Newton-Raphson Iteration (向量化版本)

    y_{n+1} = 0.5 * (y_n + x / y_n)

    使用 12 次迭代确保 FP64 精度。
    """
    def __init__(self, iterations=12, neuron_template=None):
        super().__init__()
        self.iterations = iterations
        nt = neuron_template

        # 预分配参数形状
        max_shape_64 = (64,)
        max_shape_52 = (52,)
        max_shape_11 = (11,)
        max_shape_1 = (1,)

        self.guess = SpikeFP64SqrtGuess(neuron_template=nt)

        # 迭代所需的运算单元 (已经向量化)
        self.divs = nn.ModuleList([SpikeFP64Divider(neuron_template=nt) for _ in range(iterations)])
        self.adds = nn.ModuleList([SpikeFP64Adder(neuron_template=nt) for _ in range(iterations)])
        self.muls = nn.ModuleList([SpikeFP64Multiplier(neuron_template=nt) for _ in range(iterations)])

        # 向量化门电路
        self.vec_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_64)

        # 独立实例的 Tree (不同输入大小需要独立实例)
        # 指数全1检测 (11-bit)
        self.vec_and_tree_exp = VecANDTree(neuron_template=nt, max_param_shape=max_shape_11)

        # 尾数非零检测 (52-bit)
        self.vec_or_tree_mant = VecORTree(neuron_template=nt, max_param_shape=max_shape_52)

        # 指数非零检测 (11-bit)
        self.vec_or_tree_exp = VecORTree(neuron_template=nt, max_param_shape=max_shape_11)
        
    def forward(self, x):
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        s_x = x[..., 0:1]
        e_x = x[..., 1:12]
        m_x = x[..., 12:64]

        # ===== 特殊值检测 (向量化) =====
        e_all_one = self.vec_and_tree_exp(e_x)
        m_any = self.vec_or_tree_mant(m_x)
        m_is_zero = self.vec_not(m_any)
        e_any = self.vec_or_tree_exp(e_x)
        e_is_zero = self.vec_not(e_any)
        
        # NaN: E=全1, M≠0
        is_nan = self.vec_and(e_all_one, m_any)
        
        # Inf: E=全1, M=0
        is_inf_base = self.vec_and(e_all_one, m_is_zero)
        not_sign = self.vec_not(s_x)
        is_pos_inf = self.vec_and(is_inf_base, not_sign)
        
        # 零: E=0, M=0
        is_zero = self.vec_and(e_is_zero, m_is_zero)
        
        # 负数 -> NaN
        is_negative = s_x
        
        # ===== Newton-Raphson 迭代 =====
        y = self.guess(x)
        
        const_0_5 = make_fp64_constant(0.5, batch_shape, device)
        
        for i in range(self.iterations):
            term = self.divs[i](x, y)
            y_sum = self.adds[i](y, term)
            y = self.muls[i](y_sum, const_0_5)
        
        result = y
        
        # ===== 特殊值处理 (向量化) =====
        nan_val = make_fp64_constant(float('nan'), batch_shape, device)
        inf_val = make_fp64_constant(float('inf'), batch_shape, device)
        zero_val = make_fp64_constant(0.0, batch_shape, device)
        
        # 负数 -> NaN
        result = self.vec_mux(is_negative.expand_as(result), nan_val, result)
        
        # NaN -> NaN
        result = self.vec_mux(is_nan.expand_as(result), nan_val, result)
        
        # +Inf -> +Inf
        result = self.vec_mux(is_pos_inf.expand_as(result), inf_val, result)
        
        # 零 -> 零
        result = self.vec_mux(is_zero.expand_as(result), zero_val, result)
        
        return result

    def reset(self):
        self.guess.reset()
        for m in self.divs: m.reset()
        for m in self.adds: m.reset()
        for m in self.muls: m.reset()
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_not.reset()
        self.vec_mux.reset()
        # Tree instances
        self.vec_and_tree_exp.reset()
        self.vec_or_tree_mant.reset()
        self.vec_or_tree_exp.reset()
