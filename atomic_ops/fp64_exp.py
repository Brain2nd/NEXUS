"""
FP64 指数函数 exp(x) - 100%纯SNN门电路实现
==============================================

用于高精度计算，确保FP32 exp达到bit-exact。

算法 (Cody-Waite Range Reduction):
为了避免在 r = x - z * (ln2/N) 步骤中的灾难性抵消，
我们将 ln2/N 分解为高位和低位两个常数:
    C_hi + C_lo ≈ ln2/N
其中 C_hi 只有高位有值，低位为0，保证 x - z*C_hi 无舍入误差。

流程:
1. z = round(x * N/ln2), N=64
2. k = z // 64, j = z % 64
3. r_hi = x - z * C_hi
4. r = r_hi - z * C_lo
5. T[j] = 2^(j/64) 查表
6. 多项式逼近: P(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
7. exp(x) = 2^k * T[j] * P(r)

核心原则:
- 禁止使用 ones - x, 必须用 NOTGate
- 禁止使用 a * b 逻辑AND, 必须用 ANDGate  
- 所有操作通过纯SNN门电路完成
"""
import torch
import torch.nn as nn
import struct
import numpy as np
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate,
                          FullAdder)
from .vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX, VecANDTree, VecORTree,
    VecFullAdder, VecAdder, VecSubtractor
)
from .fp64_mul import SpikeFP64Multiplier
from .fp64_adder import SpikeFP64Adder
from .fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
# fp64_div imported inside classes to avoid circular import


# ==============================================================================
# FP64 辅助函数
# ==============================================================================

def float64_to_bits(f):
    """将Python float转换为64位整数表示"""
    return struct.unpack('>Q', struct.pack('>d', f))[0]


def make_fp64_constant(val, batch_shape, device):
    """创建FP64常量脉冲"""
    bits = float64_to_bits(val)
    pulse = torch.zeros(batch_shape + (64,), device=device)
    for i in range(64):
        pulse[..., i] = float((bits >> (63 - i)) & 1)
    return pulse


# ==============================================================================
# FP64 Floor函数
# ==============================================================================
class SpikeFP64Floor(nn.Module):
    """FP64 向下取整 - 100%纯SNN门电路
    
    FP64格式: [S | E10..E0 | M51..M0], bias=1023
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # 指数减法器 (E - 1023)
        self.exp_sub = nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)])
        self.exp_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(11)])
        
        # E < 1023 检测
        self.lt_bias_not = NOTGate(neuron_template=nt)
        
        # E >= 1075 检测 (无小数)
        self.ge_no_frac = nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)])
        
        # 尾数清零 (52位)
        self.mant_clear_cmp = nn.ModuleList([nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)]) for _ in range(52)])
        self.mant_clear_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(52)])
        self.mant_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(52)])
        
        # 小数检测
        self.frac_and = nn.ModuleList([ANDGate(neuron_template=nt) for _ in range(52)])
        self.frac_or = nn.ModuleList([ORGate(neuron_template=nt) for _ in range(51)])
        
        # 负数处理 (floor = trunc - 1)
        self.neg_frac_and = ANDGate(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        
        # 结果选择
        self.lt_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(64)])
        self.ge_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(64)])
        self.neg_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(64)])
        
    def forward(self, x):
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        s_x = x[..., 0:1]
        e_x = x[..., 1:12]   # 11位指数
        m_x = x[..., 12:64]  # 52位尾数
        
        e_x_le = e_x.flip(-1)  # LSB first
        
        # 1. 计算 E - 1023
        # 1023 = 0b01111111111
        const_1023_le = torch.cat([ones]*10 + [zeros], dim=-1)  # LSB first
        not_1023_le = []
        for i in range(11):
            not_1023_le.append(self.exp_not[i](const_1023_le[..., i:i+1]))
        not_1023_le = torch.cat(not_1023_le, dim=-1)
        
        borrow = ones
        shift_bits = []
        for i in range(11):
            s, c = self.exp_sub[i](e_x_le[..., i:i+1], not_1023_le[..., i:i+1], borrow)
            shift_bits.append(s)
            borrow = c
        shift = torch.cat(shift_bits, dim=-1)  # LSB first
        
        # E < 1023: borrow_out=0 means E < 1023
        e_lt_1023 = self.lt_bias_not(borrow)
        
        # 2. E >= 1075 检测 (52 + 1023 = 1075, 无小数)
        # 1075 = 0b10000110011
        const_1075_le = torch.cat([ones, ones, zeros, zeros, ones, ones, zeros, zeros, zeros, zeros, ones], dim=-1)
        not_1075_le = []
        for i in range(11):
            not_1075_le.append(self.exp_not[i](const_1075_le[..., i:i+1]))
        not_1075_le = torch.cat(not_1075_le, dim=-1)
        
        borrow = ones
        for i in range(11):
            s, c = self.ge_no_frac[i](e_x_le[..., i:i+1], not_1075_le[..., i:i+1], borrow)
            borrow = c
        e_ge_1075 = borrow  # E >= 1075
        
        # 3. 清零小数位
        cleared_mant = []
        frac_bits = []
        for i in range(52):
            # threshold = 1024 + i (第i位是小数当 E < threshold)
            threshold = 1024 + i
            th_le = torch.zeros(batch_shape + (11,), device=device)
            for bit_idx in range(11):
                th_le[..., bit_idx] = float((threshold >> bit_idx) & 1)
            
            not_th_le = []
            for j in range(11):
                not_th_le.append(self.mant_clear_not[i](th_le[..., j:j+1]))
            not_th_le = torch.cat(not_th_le, dim=-1)
            
            borrow = ones
            for j in range(11):
                s, c = self.mant_clear_cmp[i][j](e_x_le[..., j:j+1], not_th_le[..., j:j+1], borrow)
                borrow = c
            
            is_frac = self.mant_clear_not[i](borrow)  # E < threshold means this bit is fractional
            frac_bits.append(is_frac)
            
            cleared_bit = self.mant_mux[i](is_frac, zeros, m_x[..., i:i+1])
            cleared_mant.append(cleared_bit)
        
        cleared_mant = torch.cat(cleared_mant, dim=-1)
        
        # 4. 检测是否有非零小数
        frac_and_vals = []
        for i in range(52):
            frac_and_val = self.frac_and[i](frac_bits[i], m_x[..., i:i+1])
            frac_and_vals.append(frac_and_val)
        
        has_frac = frac_and_vals[0]
        for i in range(1, 52):
            has_frac = self.frac_or[i-1](has_frac, frac_and_vals[i])
        
        trunc_result = torch.cat([s_x, e_x, cleared_mant], dim=-1)
        
        # 5. 负数有小数时: floor = trunc - 1
        neg_one = make_fp64_constant(-1.0, batch_shape, device)
        trunc_minus_one = self.fp64_adder(trunc_result, neg_one)
        
        neg_has_frac = self.neg_frac_and(s_x, has_frac)
        
        result_bits = []
        for i in range(64):
            bit = self.neg_mux[i](neg_has_frac, trunc_minus_one[..., i:i+1], trunc_result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        # 6. E < 1023: 返回0或-1
        zero_val = make_fp64_constant(0.0, batch_shape, device)
        neg_one_val = make_fp64_constant(-1.0, batch_shape, device)
        
        lt_result = []
        for i in range(64):
            bit = self.lt_mux[i](s_x, neg_one_val[..., i:i+1], zero_val[..., i:i+1])
            lt_result.append(bit)
        lt_result = torch.cat(lt_result, dim=-1)
        
        result_bits = []
        for i in range(64):
            bit = self.lt_mux[i](e_lt_1023, lt_result[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        # 7. E >= 1075: 返回原值
        result_bits = []
        for i in range(64):
            bit = self.ge_mux[i](e_ge_1075, x[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        return result
    
    def reset(self):
        for g in self.exp_sub: g.reset()
        for g in self.exp_not: g.reset()
        self.lt_bias_not.reset()
        for g in self.ge_no_frac: g.reset()
        for cmp_list in self.mant_clear_cmp:
            for g in cmp_list: g.reset()
        for g in self.mant_clear_not: g.reset()
        for g in self.mant_mux: g.reset()
        for g in self.frac_and: g.reset()
        for g in self.frac_or: g.reset()
        self.neg_frac_and.reset()
        self.fp64_adder.reset()
        for g in self.lt_mux: g.reset()
        for g in self.ge_mux: g.reset()
        for g in self.neg_mux: g.reset()


# ==============================================================================
# FP64 2^k 缩放
# ==============================================================================
class SpikeFP64ScaleBy2K(nn.Module):
    """FP64 乘以 2^k - 直接修改指数
    
    k是一个FP64整数，需要正确提取其整数值然后加到x的指数上。
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # E - 1023
        self.exp_sub_bias = nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)])
        self.exp_not_bias = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(11)])
        
        # Barrel Shifter Right (提取k的整数值, 需要6层处理最多64位移动)
        self.shift_layers = nn.ModuleList([nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(16)]) for _ in range(6)])
        self.shift_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(6)])
        
        # 符号处理
        self.neg_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(11)])
        self.neg_add = nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)])
        self.sign_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(11)])
        
        # 指数相加
        self.exp_add = nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)])
        
        # k=0检测
        self.k_zero_or = nn.ModuleList([ORGate(neuron_template=nt) for _ in range(63)])
        self.k_zero_not = NOTGate(neuron_template=nt)
        self.zero_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(64)])
        
    def forward(self, x, k):
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        s_x = x[..., 0:1]
        e_x = x[..., 1:12]
        m_x = x[..., 12:64]
        
        s_k = k[..., 0:1]
        e_k = k[..., 1:12]
        m_k = k[..., 12:64]
        
        # 1. 计算 shift = e_k - 1023 (k的指数偏移)
        e_k_le = e_k.flip(-1)  # LSB first
        const_1023_le = torch.cat([ones]*10 + [zeros], dim=-1)
        not_1023_le = []
        for i in range(11):
            not_1023_le.append(self.exp_not_bias[i](const_1023_le[..., i:i+1]))
        not_1023_le = torch.cat(not_1023_le, dim=-1)
        
        shift = []
        borrow = ones
        for i in range(11):
            s, c = self.exp_sub_bias[i](e_k_le[..., i:i+1], not_1023_le[..., i:i+1], borrow)
            shift.append(s)
            borrow = c
        shift = torch.cat(shift, dim=-1)  # LSB first, 这是指数偏移
        
        # 2. 构造完整的值: 1.m_k (16位: 隐藏位 + 15位尾数高位)
        val = torch.cat([ones, m_k[..., 0:15]], dim=-1)  # 16位, MSB first
        
        # 右移 (15 - shift) 位
        s_bar = []
        for i in range(4):
            s_bar.append(self.shift_not[i](shift[..., i:i+1]))
        
        current = val.flip(-1)  # 转LSB first方便处理
        for layer in range(4):
            shift_amt = 1 << layer
            next_val = []
            for i in range(16):
                if i + shift_amt < 16:
                    shifted = current[..., i + shift_amt:i + shift_amt + 1]
                else:
                    shifted = zeros
                bit = self.shift_layers[layer][i](s_bar[layer], shifted, current[..., i:i+1])
                next_val.append(bit)
            current = torch.cat(next_val, dim=-1)
        
        # 取低11位作为k的整数值
        k_abs_le = current[..., :11]  # LSB first
        
        # 3. 处理符号
        not_k = []
        for i in range(11):
            not_k.append(self.neg_not[i](k_abs_le[..., i:i+1]))
        not_k = torch.cat(not_k, dim=-1)
        
        neg_k = []
        carry = ones
        for i in range(11):
            s, c = self.neg_add[i](not_k[..., i:i+1], zeros, carry)
            neg_k.append(s)
            carry = c
        neg_k = torch.cat(neg_k, dim=-1)
        
        k_final = []
        for i in range(11):
            bit = self.sign_mux[i](s_k, neg_k[..., i:i+1], k_abs_le[..., i:i+1])
            k_final.append(bit)
        k_final = torch.cat(k_final, dim=-1)
        
        # 4. 指数相加: E_new = E_x + k
        e_x_le = e_x.flip(-1)
        e_new = []
        carry = zeros
        for i in range(11):
            s, c = self.exp_add[i](e_x_le[..., i:i+1], k_final[..., i:i+1], carry)
            e_new.append(s)
            carry = c
        e_new = torch.cat(e_new, dim=-1).flip(-1)  # 转回MSB first
        
        # 5. k=0检测 (所有非符号位都是0)
        k_any = k[..., 1:2]
        for i in range(2, 64):
            k_any = self.k_zero_or[i-2](k_any, k[..., i:i+1])
        k_is_zero = self.k_zero_not(k_any)
        
        result_scaled = torch.cat([s_x, e_new, m_x], dim=-1)
        
        final_bits = []
        for i in range(64):
            bit = self.zero_mux[i](k_is_zero, x[..., i:i+1], result_scaled[..., i:i+1])
            final_bits.append(bit)
        
        return torch.cat(final_bits, dim=-1)
    
    def reset(self):
        for g in self.exp_sub_bias: g.reset()
        for g in self.exp_not_bias: g.reset()
        for layer in self.shift_layers:
            for g in layer: g.reset()
        for g in self.shift_not: g.reset()
        for g in self.neg_not: g.reset()
        for g in self.neg_add: g.reset()
        for g in self.sign_mux: g.reset()
        for g in self.exp_add: g.reset()
        for g in self.k_zero_or: g.reset()
        self.k_zero_not.reset()
        for g in self.zero_mux: g.reset()


# ==============================================================================
# FP64 查表 (64项)
# ==============================================================================
class SpikeFP64LookupExp2(nn.Module):
    """查表模块: T[j] = 2^(j/64), 64项"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # 预计算64项查表值 (FP64精度)
        self.table_values = []
        for j in range(64):
            val = 2.0 ** (j / 64.0)
            self.table_values.append(float64_to_bits(val))
        
        # 6层 MUX 树 (2^6 = 64)
        self.mux_l0 = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(32 * 64)])
        self.mux_l1 = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(16 * 64)])
        self.mux_l2 = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(8 * 64)])
        self.mux_l3 = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(4 * 64)])
        self.mux_l4 = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(2 * 64)])
        self.mux_l5 = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(64)])
        
    def _make_constant(self, bits_val, batch_shape, device):
        pulse = torch.zeros(batch_shape + (64,), device=device)
        for i in range(64):
            pulse[..., i] = float((bits_val >> (63 - i)) & 1)
        return pulse

    def forward(self, idx_bits):
        """idx_bits: [..., 6] 6位索引 MSB first"""
        self.reset()
        batch_shape = idx_bits.shape[:-1]
        device = idx_bits.device
        
        consts = [self._make_constant(h, batch_shape, device) for h in self.table_values]
        
        b5, b4, b3, b2, b1, b0 = [idx_bits[..., i:i+1] for i in range(6)]
        
        # Layer 0 (b5): 64 -> 32
        out_l0 = []
        for i in range(32):
            val_bits = []
            A = consts[i]
            B = consts[i + 32]
            for bit in range(64):
                gate_idx = i * 64 + bit
                v = self.mux_l0[gate_idx](b5, B[..., bit:bit+1], A[..., bit:bit+1])
                val_bits.append(v)
            out_l0.append(torch.cat(val_bits, dim=-1))
        
        # Layer 1 (b4): 32 -> 16
        out_l1 = []
        for i in range(16):
            val_bits = []
            A = out_l0[i]
            B = out_l0[i + 16]
            for bit in range(64):
                gate_idx = i * 64 + bit
                v = self.mux_l1[gate_idx](b4, B[..., bit:bit+1], A[..., bit:bit+1])
                val_bits.append(v)
            out_l1.append(torch.cat(val_bits, dim=-1))
        
        # Layer 2 (b3): 16 -> 8
        out_l2 = []
        for i in range(8):
            val_bits = []
            A = out_l1[i]
            B = out_l1[i + 8]
            for bit in range(64):
                gate_idx = i * 64 + bit
                v = self.mux_l2[gate_idx](b3, B[..., bit:bit+1], A[..., bit:bit+1])
                val_bits.append(v)
            out_l2.append(torch.cat(val_bits, dim=-1))
        
        # Layer 3 (b2): 8 -> 4
        out_l3 = []
        for i in range(4):
            val_bits = []
            A = out_l2[i]
            B = out_l2[i + 4]
            for bit in range(64):
                gate_idx = i * 64 + bit
                v = self.mux_l3[gate_idx](b2, B[..., bit:bit+1], A[..., bit:bit+1])
                val_bits.append(v)
            out_l3.append(torch.cat(val_bits, dim=-1))
        
        # Layer 4 (b1): 4 -> 2
        out_l4 = []
        for i in range(2):
            val_bits = []
            A = out_l3[i]
            B = out_l3[i + 2]
            for bit in range(64):
                gate_idx = i * 64 + bit
                v = self.mux_l4[gate_idx](b1, B[..., bit:bit+1], A[..., bit:bit+1])
                val_bits.append(v)
            out_l4.append(torch.cat(val_bits, dim=-1))
        
        # Layer 5 (b0): 2 -> 1
        val_bits = []
        A = out_l4[0]
        B = out_l4[1]
        for bit in range(64):
            v = self.mux_l5[bit](b0, B[..., bit:bit+1], A[..., bit:bit+1])
            val_bits.append(v)
        
        return torch.cat(val_bits, dim=-1)

    def reset(self):
        for l in [self.mux_l0, self.mux_l1, self.mux_l2, self.mux_l3, self.mux_l4, self.mux_l5]:
            for g in l: g.reset()


# ==============================================================================
# FP64 提取低6位
# ==============================================================================
class SpikeFP64ExtractLow6(nn.Module):
    """从 [0, 63] 范围的 FP64 整数中提取 6-bit 二进制脉冲"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # E - 1023
        self.exp_sub = nn.ModuleList([FullAdder(neuron_template=nt) for _ in range(11)])
        self.exp_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(11)])
        
        # shift解码
        self.shift_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(11)])
        self.shift_and = nn.ModuleList([ANDGate(neuron_template=nt) for _ in range(20)])
        
        # 输出逻辑
        self.b_ands = nn.ModuleList([nn.ModuleList([ANDGate(neuron_template=nt) for _ in range(7)]) for _ in range(6)])
        self.b_ors = nn.ModuleList([nn.ModuleList([ORGate(neuron_template=nt) for _ in range(6)]) for _ in range(6)])
        
    def forward(self, x):
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        e_x = x[..., 1:12]
        m_x = x[..., 12:64]
        
        e_x_le = e_x.flip(-1)
        
        # E - 1023
        const_1023_le = torch.cat([ones]*10 + [zeros], dim=-1)
        not_1023_le = []
        for i in range(11):
            not_1023_le.append(self.exp_not[i](const_1023_le[..., i:i+1]))
        not_1023_le = torch.cat(not_1023_le, dim=-1)
        
        shift = []
        borrow = ones
        for i in range(11):
            s, c = self.exp_sub[i](e_x_le[..., i:i+1], not_1023_le[..., i:i+1], borrow)
            shift.append(s)
            borrow = c
        shift = torch.cat(shift, dim=-1)  # LSB first
        
        # 解码 shift (0..5)
        s_bits = [shift[..., i:i+1] for i in range(11)]
        s_inv = [self.shift_not[i](s_bits[i]) for i in range(11)]
        
        # 高8位必须为0
        hi_zero = s_inv[3]
        for i in range(4, 11):
            hi_zero = self.shift_and[i-4](hi_zero, s_inv[i])
        
        # 解码低3位
        is_0 = self.shift_and[7](hi_zero, self.shift_and[8](s_inv[0], self.shift_and[9](s_inv[1], s_inv[2])))
        is_1 = self.shift_and[10](hi_zero, self.shift_and[11](s_bits[0], self.shift_and[12](s_inv[1], s_inv[2])))
        is_2 = self.shift_and[13](hi_zero, self.shift_and[14](s_inv[0], self.shift_and[15](s_bits[1], s_inv[2])))
        is_3 = self.shift_and[16](hi_zero, self.shift_and[17](s_bits[0], self.shift_and[18](s_bits[1], s_inv[2])))
        is_4 = self.shift_and[19](hi_zero, self.shift_and[8](s_inv[0], self.shift_and[9](s_inv[1], s_bits[2])))
        is_5 = self.shift_and[10](hi_zero, self.shift_and[11](s_bits[0], self.shift_and[12](s_inv[1], s_bits[2])))
        
        is_s = [is_0, is_1, is_2, is_3, is_4, is_5]
        m = [m_x[..., i:i+1] for i in range(6)]
        
        # 构造输出
        final_bits = []
        for b in range(6):
            terms = []
            for s in range(6):
                if s == b:
                    term = self.b_ands[b][s](is_s[s], ones)
                    terms.append(term)
                elif s > b and s - 1 - b < 6:
                    term = self.b_ands[b][s](is_s[s], m[s - 1 - b])
                    terms.append(term)
            
            if not terms:
                res = self.shift_and[0](is_0, is_1)  # Always 0
            else:
                res = terms[0]
                for i in range(1, len(terms)):
                    res = self.b_ors[b][i-1](res, terms[i])
            final_bits.append(res)
        
        return torch.cat(list(reversed(final_bits)), dim=-1)  # [b5..b0]
    
    def reset(self):
        for g in self.exp_sub: g.reset()
        for g in self.exp_not: g.reset()
        for g in self.shift_not: g.reset()
        for g in self.shift_and: g.reset()
        for row in self.b_ands:
            for g in row: g.reset()
        for row in self.b_ors:
            for g in row: g.reset()


# ==============================================================================
# FP64 Exp 主函数
# ==============================================================================
class SpikeFP64Exp(nn.Module):
    """FP64 指数函数 exp(x) - 100%纯SNN门电路实现
    
    输入: FP64脉冲
    输出: FP64脉冲
    
    算法 (Cody-Waite Range Reduction):
    1. z = round(x * N/ln2), N=64
    2. k = z // 64, j = z % 64
    3. r = (x - z * C_hi) - z * C_lo
    4. T = lookup(j)
    5. P = 1 + r + r^2/2 + r^3/6 + r^4/24
    6. res = 2^k * T * P
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # 运算组件 - 增加了一个乘法器和一个减法器用于Cody-Waite，多项式增加到7阶需要更多
        self.mul = nn.ModuleList([SpikeFP64Multiplier(neuron_template=nt) for _ in range(13)]) # 13 -> 13
        self.add = nn.ModuleList([SpikeFP64Adder(neuron_template=nt) for _ in range(11)]) # 10 -> 11
        
        self.floor = SpikeFP64Floor(neuron_template=nt)
        self.floor_k = SpikeFP64Floor(neuron_template=nt)
        self.extract_j = SpikeFP64ExtractLow6(neuron_template=nt)
        self.lookup = SpikeFP64LookupExp2(neuron_template=nt)
        self.scale = SpikeFP64ScaleBy2K(neuron_template=nt)
        
        # 向量化门电路
        self.vec_not = VecNOT(neuron_template=nt)
        self.vec_and = VecAND(neuron_template=nt)
        self.vec_mux = VecMUX(neuron_template=nt)
        self.vec_and_tree = VecANDTree(neuron_template=nt)
        self.vec_or_tree = VecORTree(neuron_template=nt)

    def forward(self, x):
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        s_x = x[..., 0:1]
        e_x = x[..., 1:12]
        m_x = x[..., 12:64]
        
        # 特殊值检测 (向量化)
        e_all_one = self.vec_and_tree(e_x)
        m_any = self.vec_or_tree(m_x)
        m_is_zero = self.vec_not(m_any)
        
        is_nan = self.vec_and(e_all_one, m_any)
        is_inf_base = self.vec_and(e_all_one, m_is_zero)
        not_sign = self.vec_not(s_x)
        is_pos_inf = self.vec_and(is_inf_base, not_sign)
        is_neg_inf = self.vec_and(is_inf_base, s_x)
        
        # 常量 (FP64精度)
        # N = 64
        # N/ln2 = 92.33248261689366
        inv_ln2_n = make_fp64_constant(64.0 / np.log(2.0), batch_shape, device)
        half = make_fp64_constant(0.5, batch_shape, device)
        inv_64 = make_fp64_constant(1.0 / 64.0, batch_shape, device)
        const_64 = make_fp64_constant(64.0, batch_shape, device)
        
        # Cody-Waite Constants for ln2/64
        # LN2_N_HI + LN2_N_LO ≈ ln2/64
        # 0.010830424696249145
        # 分解策略: 取前40位作为HI
        val_ln2_n = np.log(2.0) / 64.0
        bits_ln2_n = float64_to_bits(val_ln2_n)
        # 掩码去掉低24位 (保留符号、指数和尾数高28位, 总共约40位有效精度)
        mask_hi = 0xFFFFFFFFFF000000
        bits_hi = bits_ln2_n & mask_hi
        
        # 构造 HI 值
        val_hi = struct.unpack('>d', struct.pack('>Q', bits_hi))[0]
        val_lo = val_ln2_n - val_hi
        
        ln2_n_hi = make_fp64_constant(val_hi, batch_shape, device)
        ln2_n_lo = make_fp64_constant(val_lo, batch_shape, device)
        
        # 多项式系数 (7阶)
        c1 = make_fp64_constant(1.0, batch_shape, device)
        c2 = make_fp64_constant(0.5, batch_shape, device)
        c3 = make_fp64_constant(1.0/6.0, batch_shape, device)
        c4 = make_fp64_constant(1.0/24.0, batch_shape, device)
        c5 = make_fp64_constant(1.0/120.0, batch_shape, device)
        c6 = make_fp64_constant(1.0/720.0, batch_shape, device)
        c7 = make_fp64_constant(1.0/5040.0, batch_shape, device)
        
        # 1. z = round(x * N/ln2)
        x_scaled = self.mul[0](x, inv_ln2_n)
        x_plus_half = self.add[0](x_scaled, half)
        z = self.floor(x_plus_half)
        
        # 2. k = floor(z / 64)
        z_div_64 = self.mul[1](z, inv_64)
        k = self.floor_k(z_div_64)
        
        # 3. j = z % 64
        k_times_64 = self.mul[2](k, const_64)
        neg_k64_s = self.vec_not(k_times_64[..., 0:1])
        neg_k64 = torch.cat([neg_k64_s, k_times_64[..., 1:]], dim=-1)
        j_float = self.add[1](z, neg_k64)
        j_bits = self.extract_j(j_float)
        
        # 4. r = (x - z * C_hi) - z * C_lo
        # r_hi = x - z * C_hi
        z_hi = self.mul[3](z, ln2_n_hi)
        neg_z_hi_s = self.vec_not(z_hi[..., 0:1])
        neg_z_hi = torch.cat([neg_z_hi_s, z_hi[..., 1:]], dim=-1)
        r_hi = self.add[2](x, neg_z_hi)
        
        # r = r_hi - z * C_lo
        z_lo = self.mul[4](z, ln2_n_lo)
        neg_z_lo_s = self.vec_not(z_lo[..., 0:1])
        neg_z_lo = torch.cat([neg_z_lo_s, z_lo[..., 1:]], dim=-1)
        r = self.add[3](r_hi, neg_z_lo)
        
        # 5. T = lookup(j)
        T = self.lookup(j_bits)
        
        # 6. P(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040
        # Horner: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r*(1/720 + r/5040))))))
        r_c7 = self.mul[5](r, c7)
        p0 = self.add[4](r_c7, c6)

        p0_r = self.mul[6](p0, r)
        p1 = self.add[5](p0_r, c5)
        
        p1_r = self.mul[7](p1, r)
        p2 = self.add[6](p1_r, c4)
        
        p2_r = self.mul[8](p2, r)
        p3 = self.add[7](p2_r, c3)
        
        p3_r = self.mul[9](p3, r)
        p4 = self.add[8](p3_r, c2)

        p4_r = self.mul[10](p4, r)
        p5 = self.add[9](p4_r, c1)

        p5_r = self.mul[11](p5, r)
        P = self.add[10](p5_r, c1)
        
        # 7. Result = 2^k * T * P
        TP = self.mul[12](T, P)
        result = self.scale(TP, k)
        
        # 特殊值输出 (向量化)
        nan_val = make_fp64_constant(float('nan'), batch_shape, device)
        inf_val = make_fp64_constant(float('inf'), batch_shape, device)
        zero_val = make_fp64_constant(0.0, batch_shape, device)
        
        # NaN
        result = self.vec_mux(is_nan.expand_as(result), nan_val, result)
        
        # -Inf -> 0
        result = self.vec_mux(is_neg_inf.expand_as(result), zero_val, result)
        
        # +Inf -> +Inf
        result = self.vec_mux(is_pos_inf.expand_as(result), inf_val, result)
        
        return result

    def reset(self):
        for m in self.mul: m.reset()
        for a in self.add: a.reset()
        self.floor.reset()
        self.floor_k.reset()
        self.extract_j.reset()
        self.lookup.reset()
        self.scale.reset()
        self.vec_not.reset()
        self.vec_and.reset()
        self.vec_mux.reset()
        self.vec_and_tree.reset()
        self.vec_or_tree.reset()


# ==============================================================================
# FP32 Exp 使用 FP64 内部精度
# ==============================================================================
class SpikeFP32ExpHighPrecision(nn.Module):
    """FP32 Exp 使用 FP64 内部精度实现 bit-exact
    
    流程: FP32输入 -> FP64 -> Exp计算 -> FP64 -> FP32输出
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        
    def forward(self, x):
        """
        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲
        """
        self.reset()
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)
        
        # FP64 Exp
        result_fp64 = self.fp64_exp(x_fp64)
        
        # FP64 -> FP32
        result_fp32 = self.fp64_to_fp32(result_fp64)
        
        return result_fp32
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_exp.reset()
        self.fp64_to_fp32.reset()


# ==============================================================================
# 全链路FP64高精度Sigmoid
# ==============================================================================
class SpikeFP32SigmoidFullFP64(nn.Module):
    """FP32 Sigmoid 全链路FP64内部精度
    
    sigmoid(x) = 1 / (1 + exp(-x))
    
    流程: FP32输入 -> FP64 -> 全部FP64计算 -> FP64 -> FP32输出
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        from .fp64_div import SpikeFP64Divider
        nt = neuron_template
        
        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        
        self.sign_not = NOTGate(neuron_template=nt)
        
    def forward(self, x):
        """
        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲
        """
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)
        
        # -x (翻转符号位)
        neg_x_sign = self.sign_not(x_fp64[..., 0:1])
        neg_x = torch.cat([neg_x_sign, x_fp64[..., 1:]], dim=-1)
        
        # exp(-x)
        exp_neg_x = self.fp64_exp(neg_x)
        
        # 1 + exp(-x)
        one_fp64 = make_fp64_constant(1.0, batch_shape, device)
        one_plus_exp = self.fp64_adder(one_fp64, exp_neg_x)
        
        # 1 / (1 + exp(-x))
        result_fp64 = self.fp64_divider(one_fp64, one_plus_exp)
        
        # FP64 -> FP32
        result_fp32 = self.fp64_to_fp32(result_fp64)
        
        return result_fp32
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_exp.reset()
        self.fp64_adder.reset()
        self.fp64_divider.reset()
        self.fp64_to_fp32.reset()
        self.sign_not.reset()


# ==============================================================================
# 全链路FP64高精度SiLU
# ==============================================================================
class SpikeFP32SiLUFullFP64(nn.Module):
    """FP32 SiLU 全链路FP64内部精度
    
    SiLU(x) = x * sigmoid(x)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        from .fp64_mul import SpikeFP64Multiplier
        from .fp64_div import SpikeFP64Divider
        nt = neuron_template
        
        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_mul = SpikeFP64Multiplier(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        
        self.sign_not = NOTGate(neuron_template=nt)
        
    def forward(self, x):
        self.reset()
        device = x.device
        batch_shape = x.shape[:-1]
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)
        
        # sigmoid(x) = 1 / (1 + exp(-x))
        neg_x_sign = self.sign_not(x_fp64[..., 0:1])
        neg_x = torch.cat([neg_x_sign, x_fp64[..., 1:]], dim=-1)
        exp_neg_x = self.fp64_exp(neg_x)
        
        one_fp64 = make_fp64_constant(1.0, batch_shape, device)
        one_plus_exp = self.fp64_adder(one_fp64, exp_neg_x)
        sigmoid_fp64 = self.fp64_divider(one_fp64, one_plus_exp)
        
        # x * sigmoid(x)
        result_fp64 = self.fp64_mul(x_fp64, sigmoid_fp64)
        
        # FP64 -> FP32
        result_fp32 = self.fp64_to_fp32(result_fp64)
        
        return result_fp32
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_exp.reset()
        self.fp64_adder.reset()
        self.fp64_divider.reset()
        self.fp64_mul.reset()
        self.fp64_to_fp32.reset()
        self.sign_not.reset()


# ==============================================================================
# 全链路FP64高精度Softmax
# ==============================================================================
class SpikeFP32SoftmaxFullFP64(nn.Module):
    """FP32 Softmax 全链路FP64内部精度
    
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        from .fp64_div import SpikeFP64Divider
        nt = neuron_template
        
        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        
    def forward(self, x):
        """
        输入: [..., N, 32] FP32脉冲
        输出: [..., N, 32] FP32脉冲
        """
        self.reset()
        
        batch_shape = x.shape[:-2]
        N = x.shape[-2]
        
        # FP32 -> FP64
        x_fp64 = self.fp32_to_fp64(x)  # [..., N, 64]
        
        # exp(x_i) for all i
        exp_x = self.fp64_exp(x_fp64)  # [..., N, 64]
        
        # sum(exp(x_j))
        sum_exp = exp_x[..., 0, :]  # [..., 64]
        for i in range(1, N):
            self.fp64_adder.reset()
            sum_exp = self.fp64_adder(sum_exp, exp_x[..., i, :])
        
        # exp(x_i) / sum for each i
        result_fp64 = []
        for i in range(N):
            self.fp64_divider.reset()
            softmax_i = self.fp64_divider(exp_x[..., i, :], sum_exp)
            result_fp64.append(softmax_i.unsqueeze(-2))
        
        result_fp64 = torch.cat(result_fp64, dim=-2)  # [..., N, 64]
        
        # FP64 -> FP32
        result_fp32 = self.fp64_to_fp32(result_fp64)  # [..., N, 32]
        
        return result_fp32
    
    def reset(self):
        self.fp32_to_fp64.reset()
        self.fp64_exp.reset()
        self.fp64_adder.reset()
        self.fp64_divider.reset()
        self.fp64_to_fp32.reset()

