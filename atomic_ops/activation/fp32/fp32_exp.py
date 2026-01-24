"""
FP32 指数函数 exp(x) - 100%纯SNN门电路实现 (glibc算法复刻版)
=============================================================

基于glibc e_expf.c算法复刻，以实现与PyTorch/CUDA的高精度匹配。

核心算法:
1. 范围缩减 + 查表:
   z = round(x * N/ln2), 其中 N=32
   k = z // 32 (整数部分)
   j = z % 32  (查表索引)
   
   x = (k + j/32) * ln2 + r
   r = x - z * (ln2/N)
   
   为了高精度，ln2被拆分为 ln2_hi + ln2_lo:
   r = x - z*ln2_hi/N - z*ln2_lo/N

2. 查表:
   T[j] = 2^(j/32) 使用纯SNN MUX树查找

3. 多项式逼近 (Minimax 3阶):
   P(r) = C0*r^3 + C1*r^2 + C2*r + 1

4. 结果重构:
   exp(x) = 2^k * T[j] * P(r)

核心原则:
- 禁止使用 ones - x, 必须用 NOTGate
- 禁止使用 a * b 逻辑AND, 必须用 ANDGate  
- 所有操作通过纯SNN门电路完成

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.core.logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate,
                          HalfAdder, FullAdder, RippleCarryAdder)
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX, VecORTree, VecANDTree,
    VecFullAdder, VecAdder, VecSubtractor
)
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder


# ==============================================================================
# FP32 辅助组件 - 位提取与查表
# ==============================================================================

class SpikeFP32ExtractLow5(nn.Module):
    """从 [0, 31] 范围的 FP32 整数中提取 5-bit 二进制脉冲
    
    输入: x (FP32脉冲, 值在0-31之间)
    输出: 5-bit 脉冲 [b4, b3, b2, b1, b0] (MSB first)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # E检测: E=127..131 - 单实例
        self.exp_sub = FullAdder(neuron_template=nt, max_param_shape=(1,))
        self.exp_not = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # 输出逻辑门 - 单实例
        self.b_and = ANDGate(neuron_template=nt, max_param_shape=(1,))
        self.b_or = ORGate(neuron_template=nt, max_param_shape=(1,))

        # shift 解码器 - 单实例
        self.shift_is_zero = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.shift_and = ANDGate(neuron_template=nt, max_param_shape=(1,))
        
    def forward(self, x):
        device = x.device
        e_x = x[..., 1:9]
        m_x = x[..., 9:32]
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        # 1. 计算 shift = E - 127 (vectorized NOT, sequential adder)
        # NOT(127) = 10000000
        const_not_127 = torch.zeros_like(e_x)
        const_not_127[..., 0] = 1.0

        shift = []
        borrow = ones
        for i in range(7, -1, -1):
            s, c = self.exp_sub(e_x[..., i:i+1], const_not_127[..., i:i+1], borrow)
            shift.insert(0, s)
            borrow = c
        shift = torch.cat(shift, dim=-1)

        # 2. 解码 shift (0..4) (vectorized NOT for all 8 bits)
        s_inv = self.shift_is_zero(shift)
        s_bits = [shift[..., i:i+1] for i in range(8)]
        s_inv_list = [s_inv[..., i:i+1] for i in range(8)]

        # High 5 bits must be zero (tree reduction)
        hi_zero = s_inv_list[0]
        for i in range(1, 5):
            hi_zero = self.shift_and(hi_zero, s_inv_list[i])

        # Low 3 bits decode (single instance AND gate)
        is_0 = self.shift_and(hi_zero, self.shift_and(s_inv_list[5], self.shift_and(s_inv_list[6], s_inv_list[7])))
        is_1 = self.shift_and(hi_zero, self.shift_and(s_inv_list[5], self.shift_and(s_inv_list[6], s_bits[7])))
        is_2 = self.shift_and(hi_zero, self.shift_and(s_inv_list[5], self.shift_and(s_bits[6], s_inv_list[7])))
        is_3 = self.shift_and(hi_zero, self.shift_and(s_inv_list[5], self.shift_and(s_bits[6], s_bits[7])))
        is_4 = self.shift_and(hi_zero, self.shift_and(s_bits[5], self.shift_and(s_inv_list[6], s_inv_list[7])))

        # 3. 构造输出 (single instance AND/OR gates)
        m = [m_x[..., i:i+1] for i in range(4)]  # m0..m3

        is_s = [is_0, is_1, is_2, is_3, is_4]

        final_bits = []
        for b in range(5):
            terms = []
            for s in range(5):
                if s == b:
                    term = self.b_and(is_s[s], ones)
                    terms.append(term)
                elif s > b:
                    m_idx = s - 1 - b
                    if m_idx < 4:
                        term = self.b_and(is_s[s], m[m_idx])
                        terms.append(term)

            # OR all terms (tree reduction)
            if not terms:
                # Hack: use AND(is_0, is_1) which is always 0
                res = self.shift_and(is_0, is_1)
            else:
                res = terms[0]
                for i in range(1, len(terms)):
                    res = self.b_or(res, terms[i])
            final_bits.append(res)

        return torch.cat(list(reversed(final_bits)), dim=-1)  # [b4..b0]

    def reset(self):
        self.exp_sub.reset()
        self.exp_not.reset()
        self.b_and.reset()
        self.b_or.reset()
        self.shift_is_zero.reset()
        self.shift_and.reset()


class SpikeFP32LookupExp2(nn.Module):
    """查表模块: T[j] = 2^(j/32)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self._neuron_template = neuron_template
        # 预计算的查表数据 (glibc EXP2F_TABLE_BITS=5)
        self.table_hex = [
            0x3f800000, 0x3f82cd87, 0x3f85aac3, 0x3f88980f,
            0x3f8b95c2, 0x3f8ea43a, 0x3f91c3d3, 0x3f94f4f0,
            0x3f9837f0, 0x3f9b8d3a, 0x3f9ef532, 0x3fa27043,
            0x3fa5fed7, 0x3fa9a15b, 0x3fad583f, 0x3fb123f6,
            0x3fb504f3, 0x3fb8fbaf, 0x3fbd08a4, 0x3fc12c4d,
            0x3fc5672a, 0x3fc9b9be, 0x3fce248c, 0x3fd2a81e,
            0x3fd744fd, 0x3fdbfbb8, 0x3fdbfbb8, 0x3fe0ccdf, # Index 25 duplicated in copy? check
            0x3fe5b907, 0x3feac0c7, 0x3fefe4ba, 0x3ff5257d,
            0x3ffa83b3
        ]
        # Fix: index 26 was missing/wrong in my quick copy above?
        # Let's use the accurate list from previous shell output
        self.table_hex = [
            0x3f800000, 0x3f82cd87, 0x3f85aac3, 0x3f88980f,
            0x3f8b95c2, 0x3f8ea43a, 0x3f91c3d3, 0x3f94f4f0,
            0x3f9837f0, 0x3f9b8d3a, 0x3f9ef532, 0x3fa27043,
            0x3fa5fed7, 0x3fa9a15b, 0x3fad583f, 0x3fb123f6,
            0x3fb504f3, 0x3fb8fbaf, 0x3fbd08a4, 0x3fc12c4d,
            0x3fc5672a, 0x3fc9b9be, 0x3fce248c, 0x3fd2a81e,
            0x3fd744fd, 0x3fdbfbb8, 0x3fe0ccdf, 0x3fe5b907,
            0x3feac0c7, 0x3fefe4ba, 0x3ff5257d, 0x3ffa83b3
        ]
        
        # 5层 MUX 树 - 向量化 (每层使用单个 VecMUX)
        # 最大形状: 16 * 32 = 512 位 (第一层)
        self.vec_mux = VecMUX(neuron_template=neuron_template, max_param_shape=(512,))
        
    def _make_constant(self, hex_val, batch_shape, device):
        pulse = torch.zeros(batch_shape + (32,), device=device)
        ival = int(hex_val, 16) if isinstance(hex_val, str) else hex_val
        for i in range(32):
            pulse[..., i] = float((ival >> (31 - i)) & 1)
        return pulse

    def forward(self, idx_bits):
        batch_shape = idx_bits.shape[:-1]
        device = idx_bits.device
        
        # 构建常量表
        consts = []
        for h in self.table_hex:
            consts.append(self._make_constant(h, batch_shape, device))
            
        b4, b3, b2, b1, b0 = idx_bits[..., 0:1], idx_bits[..., 1:2], idx_bits[..., 2:3], idx_bits[..., 3:4], idx_bits[..., 4:5]
        
        # Layer 0 (b4): 16对常量选择，每对32位 - 向量化
        out_l0 = []
        b4_exp = b4.expand(*batch_shape, 32)
        for i in range(16):
            A = consts[i]
            B = consts[i+16]
            out_l0.append(self.vec_mux(b4_exp, B, A))
            
        # Layer 1 (b3): 8对选择 - 向量化
        out_l1 = []
        b3_exp = b3.expand(*batch_shape, 32)
        for i in range(8):
            A = out_l0[i]
            B = out_l0[i+8]
            out_l1.append(self.vec_mux(b3_exp, B, A))
            
        # Layer 2 (b2): 4对选择 - 向量化
        out_l2 = []
        b2_exp = b2.expand(*batch_shape, 32)
        for i in range(4):
            A = out_l1[i]
            B = out_l1[i+4]
            out_l2.append(self.vec_mux(b2_exp, B, A))
            
        # Layer 3 (b1): 2对选择 - 向量化
        out_l3 = []
        b1_exp = b1.expand(*batch_shape, 32)
        for i in range(2):
            A = out_l2[i]
            B = out_l2[i+2]
            out_l3.append(self.vec_mux(b1_exp, B, A))
            
        # Layer 4 (b0): 1对选择 - 向量化
        b0_exp = b0.expand(*batch_shape, 32)
        result = self.vec_mux(b0_exp, out_l3[1], out_l3[0])
        
        return result

    def reset(self):
        self.vec_mux.reset()


# ==============================================================================
# FP32 Floor函数 - 向下取整 (使用FP32减法实现负数处理)
# ==============================================================================
class SpikeFP32Floor(nn.Module):
    """FP32 向下取整 - 100%纯SNN门电路"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 指数减法器 (E - 127) - 单实例
        self.exp_sub_127 = FullAdder(neuron_template=nt, max_param_shape=(1,))
        self.exp_not_127 = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # 比较器: E < 127 (即 x < 1) - 单实例
        self.lt_127_detect = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # 比较器: E >= 150 (即无小数) - 单实例
        self.ge_150_detect = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # 尾数清零 - 单实例
        self.mant_lt_cmp = FullAdder(neuron_template=nt, max_param_shape=(1,))
        self.mant_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.mant_lt_not = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.mant_borrow_not = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # 检测是否有小数部分 - 单实例
        self.frac_or = ORGate(neuron_template=nt, max_param_shape=(1,))
        self.frac_and_sign = ANDGate(neuron_template=nt, max_param_shape=(1,))
        self.frac_and_mant = ANDGate(neuron_template=nt, max_param_shape=(1,))

        # 使用FP32加法器实现减1
        self.fp32_adder = SpikeFP32Adder(neuron_template=nt)

        # 负数有小数的组合检测
        self.neg_has_frac_and = ANDGate(neuron_template=nt, max_param_shape=(1,))

        # 结果选择MUX - 单实例
        self.lt1_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.ge150_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.neg_frac_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        
    def _make_constant(self, bits, batch_shape, device):
        pulse = torch.zeros(batch_shape + (32,), device=device)
        for i in range(32):
            pulse[..., i] = float((bits >> (31 - i)) & 1)
        return pulse
        
    def forward(self, x):
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        s_x = x[..., 0:1]
        e_x = x[..., 1:9]
        m_x = x[..., 9:32]

        e_x_le = e_x.flip(-1)

        # 判断 E < 127 (vectorized NOT, sequential adder chain)
        const_127_le = torch.cat([ones, ones, ones, ones, ones, ones, ones, zeros], dim=-1)
        not_127_le = self.exp_not_127(const_127_le)

        borrow = ones
        for i in range(8):
            s, c = self.exp_sub_127(e_x_le[..., i:i+1], not_127_le[..., i:i+1], borrow)
            borrow = c
        e_lt_127 = self.lt_127_detect(borrow)

        # 判断 E >= 150 (vectorized NOT, sequential adder chain)
        const_150_le = torch.cat([zeros, ones, ones, zeros, ones, zeros, zeros, ones], dim=-1)
        not_150_le = self.ge_150_detect(const_150_le)

        borrow = ones
        for i in range(8):
            s, c = self.exp_sub_127(e_x_le[..., i:i+1], not_150_le[..., i:i+1], borrow)
            borrow = c
        e_ge_150 = borrow
        
        # 清零小数位 (single instance gates, sequential for carry dependency)
        cleared_mant = []
        frac_bits = []
        for i in range(23):
            threshold = 128 + i
            th_le = torch.zeros(batch_shape + (8,), device=device)
            for bit_idx in range(8):
                th_le[..., bit_idx] = float((threshold >> bit_idx) & 1)

            # Vectorized NOT
            not_th_le = self.mant_lt_not(th_le)

            # Sequential adder chain
            borrow = ones
            for j in range(8):
                s, c = self.mant_lt_cmp(e_x_le[..., j:j+1], not_th_le[..., j:j+1], borrow)
                borrow = c

            is_frac = self.mant_borrow_not(borrow)
            frac_bits.append(is_frac)

            cleared_bit = self.mant_mux(is_frac, zeros, m_x[..., i:i+1])
            cleared_mant.append(cleared_bit)

        cleared_mant = torch.cat(cleared_mant, dim=-1)

        # 检测是否存在非零小数 (tree reduction)
        frac_and_vals = []
        for i in range(23):
            frac_and_val = self.frac_and_mant(frac_bits[i], m_x[..., i:i+1])
            frac_and_vals.append(frac_and_val)

        has_frac = frac_and_vals[0]
        for i in range(1, 23):
            has_frac = self.frac_or(has_frac, frac_and_vals[i])
        
        trunc_result = torch.cat([s_x, e_x, cleared_mant], dim=-1)
        
        # 负数有小数时: floor = trunc - 1
        neg_one_const = self._make_constant(0xBF800000, batch_shape, device)
        trunc_minus_one = self.fp32_adder(trunc_result, neg_one_const)
        
        neg_has_frac = self.neg_has_frac_and(s_x, has_frac)

        # neg_frac_mux (vectorized)
        neg_has_frac_32 = neg_has_frac.expand_as(trunc_result)
        result = self.neg_frac_mux(neg_has_frac_32, trunc_minus_one, trunc_result)

        # E < 127 (vectorized)
        zero_val = torch.zeros(batch_shape + (32,), device=device)
        neg_one_val = self._make_constant(0xBF800000, batch_shape, device)

        s_x_32 = s_x.expand_as(neg_one_val)
        lt1_result = self.lt1_mux(s_x_32, neg_one_val, zero_val)

        e_lt_127_32 = e_lt_127.expand_as(result)
        result = self.lt1_mux(e_lt_127_32, lt1_result, result)

        # E >= 150 (vectorized)
        e_ge_150_32 = e_ge_150.expand_as(result)
        result = self.ge150_mux(e_ge_150_32, x, result)
        
        return result
    
    def reset(self):
        self.exp_sub_127.reset()
        self.exp_not_127.reset()
        self.lt_127_detect.reset()
        self.ge_150_detect.reset()
        self.mant_lt_cmp.reset()
        self.mant_lt_not.reset()
        self.mant_borrow_not.reset()
        self.mant_mux.reset()
        self.frac_or.reset()
        self.frac_and_sign.reset()
        self.frac_and_mant.reset()
        self.fp32_adder.reset()
        self.neg_has_frac_and.reset()
        self.lt1_mux.reset()
        self.ge150_mux.reset()
        self.neg_frac_mux.reset()


# ==============================================================================
# FP32 2^k 缩放 - 正确提取k的整数值
# ==============================================================================
class SpikeFP32ScaleBy2K(nn.Module):
    """FP32 乘以 2^k - 正确从FP32整数k中提取整数值"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # E - 127 - 单实例
        self.exp_sub_127 = FullAdder(neuron_template=nt, max_param_shape=(1,))
        self.exp_not_127 = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # k_mant << shift (Barrel Shifter) - 单实例
        self.shift_layer0 = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.shift_layer1 = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.shift_layer2 = MUXGate(neuron_template=nt, max_param_shape=(1,))

        # sign(k)处理 - 单实例
        self.neg_not = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.neg_add = FullAdder(neuron_template=nt, max_param_shape=(1,))
        self.sign_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))

        # 结果相加 (E + k_int) - 单实例
        self.exp_add = FullAdder(neuron_template=nt, max_param_shape=(1,))

        # 溢出处理 - 简单clamp - 单实例
        self.exp_not = NOTGate(neuron_template=nt, max_param_shape=(1,))

        # K是否为0检测 - 单实例
        self.k_exp_zero_or = ORGate(neuron_template=nt, max_param_shape=(1,))
        self.k_exp_zero_not = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.k_mant_zero_or = ORGate(neuron_template=nt, max_param_shape=(1,))
        self.k_mant_zero_not = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.k_is_zero_and = ANDGate(neuron_template=nt, max_param_shape=(1,))

        self.zero_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        
    def forward(self, x, k):
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        s_x = x[..., 0:1]
        e_x = x[..., 1:9]
        m_x = x[..., 9:32]

        s_k = k[..., 0:1]
        e_k = k[..., 1:9]
        m_k = k[..., 9:32]

        # 1. 计算 shift = e_k - 127 (vectorized NOT, sequential adder chain)
        e_k_le = e_k.flip(-1)
        const_127_le = torch.cat([ones, ones, ones, ones, ones, ones, ones, zeros], dim=-1)
        not_127_le = self.exp_not_127(const_127_le)

        shift = []
        borrow = ones
        for i in range(8):
            s, c = self.exp_sub_127(e_k_le[..., i:i+1], not_127_le[..., i:i+1], borrow)
            shift.append(s)
            borrow = c
        shift = torch.cat(shift, dim=-1)  # LSB first
        
        # 2. 提取 k_int 的绝对值
        # 构造完整尾数: 1.m_k
        val = torch.cat([ones, m_k[..., 0:7]], dim=-1)
        
        # Barrel Shifter (Right Shift by 7 - shift)
        # Target: k_abs = val >> (7 - shift)
        # Let amount = 7 - shift. Since shift <= 7, amount = NOT(shift) & 7
        
        # Invert shift bits for control (vectorized)
        s_bar_all = self.exp_not(shift[..., :3])
        s_bar = [s_bar_all[..., i:i+1] for i in range(3)]

        # L0 (amount[0], shift 1): Right Shift 1 (vectorized)
        zeros_8 = zeros.expand(batch_shape + (8,))
        val_shifted_1 = torch.cat([zeros, val[..., :-1]], dim=-1)
        s_bar_0_8 = s_bar[0].expand_as(val)
        l0_tensor = self.shift_layer0(s_bar_0_8, val_shifted_1, val)

        # L1 (amount[1], shift 2): Right Shift 2 (vectorized)
        l0_shifted_2 = torch.cat([zeros.expand(batch_shape + (2,)), l0_tensor[..., :-2]], dim=-1)
        s_bar_1_8 = s_bar[1].expand_as(l0_tensor)
        l1_tensor = self.shift_layer1(s_bar_1_8, l0_shifted_2, l0_tensor)

        # L2 (amount[2], shift 4): Right Shift 4 (vectorized)
        l1_shifted_4 = torch.cat([zeros.expand(batch_shape + (4,)), l1_tensor[..., :-4]], dim=-1)
        s_bar_2_8 = s_bar[2].expand_as(l1_tensor)
        l2_tensor = self.shift_layer2(s_bar_2_8, l1_shifted_4, l1_tensor)

        k_abs = l2_tensor  # [MSB...LSB]
        k_abs_le = k_abs.flip(-1)  # [LSB...MSB]
        
        # 3. 处理符号 (k < 0)
        # 如果 s_k=1, k_int = -k_abs
        # 二进制补码: NOT(k_abs) + 1 (vectorized NOT, sequential adder)
        not_k = self.neg_not(k_abs_le)

        neg_k = []
        carry = ones
        for i in range(8):  # +1
            s, c = self.neg_add(not_k[..., i:i+1], zeros, carry)
            neg_k.append(s)
            carry = c  # discard
        neg_k = torch.cat(neg_k, dim=-1)

        # sign_mux (vectorized)
        s_k_8 = s_k.expand_as(neg_k)
        k_final = self.sign_mux(s_k_8, neg_k, k_abs_le)
        
        # 4. 指数相加: E_new = E_old + k_int (sequential adder chain)
        e_x_le = e_x.flip(-1)
        e_new = []
        carry = zeros
        for i in range(8):
            s, c = self.exp_add(e_x_le[..., i:i+1], k_final[..., i:i+1], carry)
            e_new.append(s)
            carry = c
        e_new = torch.cat(e_new, dim=-1).flip(-1)  # MSB first

        # 5. 如果 k=0 (输入0), 直接返回x (tree reduction)
        # 检测k是否为0
        k_e_zero = self.k_exp_zero_or(e_k[..., 0:1], e_k[..., 1:2])
        for i in range(2, 8):
            k_e_zero = self.k_exp_zero_or(k_e_zero, e_k[..., i:i+1])
        k_e_zero_n = self.k_exp_zero_not(k_e_zero)  # E全是0

        k_m_zero = self.k_mant_zero_or(m_k[..., 0:1], m_k[..., 1:2])
        for i in range(2, 23):
            k_m_zero = self.k_mant_zero_or(k_m_zero, m_k[..., i:i+1])
        k_m_zero_n = self.k_mant_zero_not(k_m_zero)  # M全是0

        k_is_zero = self.k_is_zero_and(k_e_zero_n, k_m_zero_n)

        # 组装结果 (vectorized)
        result_scaled = torch.cat([s_x, e_new, m_x], dim=-1)

        k_is_zero_32 = k_is_zero.expand_as(result_scaled)
        return self.zero_mux(k_is_zero_32, x, result_scaled)
    
    def reset(self):
        self.exp_sub_127.reset()
        self.exp_not_127.reset()
        self.shift_layer0.reset()
        self.shift_layer1.reset()
        self.shift_layer2.reset()
        self.neg_not.reset()
        self.neg_add.reset()
        self.sign_mux.reset()
        self.exp_add.reset()
        self.exp_not.reset()
        self.k_exp_zero_or.reset()
        self.k_exp_zero_not.reset()
        self.k_mant_zero_or.reset()
        self.k_mant_zero_not.reset()
        self.k_is_zero_and.reset()
        self.zero_mux.reset()


# ==============================================================================
# FP32 Exp 主函数 (glibc算法复刻)
# ==============================================================================
class SpikeFP32Exp(nn.Module):
    """FP32 指数函数 exp(x) - 100%纯SNN门电路实现

    算法:
    1. z = round(x * N/ln2), N=32
    2. k = z // 32, j = z % 32
    3. r = x - z*ln2_hi/N - z*ln2_lo/N
    4. T = lookup(j)
    5. P = C1*r + C2*r^2 + C3*r^3 + 1
    6. res = 2^k * T * P

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式 (None/TrainingMode.STE/TrainingMode.TEMPORAL)（梯度流过）
    """
    def __init__(self, neuron_template=None, training_mode=None):
        super().__init__()
        self.training_mode = TrainingMode.validate(training_mode)
        nt = neuron_template
        
        # 运算组件
        self.mul_inv_ln2 = SpikeFP32Multiplier(neuron_template=nt)
        self.add_half = SpikeFP32Adder(neuron_template=nt)
        self.floor = SpikeFP32Floor(neuron_template=nt)
        self.extract_j = SpikeFP32ExtractLow5(neuron_template=nt)
        self.lookup = SpikeFP32LookupExp2(neuron_template=nt)
        
        # r 计算
        self.mul_z_ln2_hi = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_z_ln2_lo = SpikeFP32Multiplier(neuron_template=nt)
        self.sub_hi = SpikeFP32Adder(neuron_template=nt)
        self.sub_lo = SpikeFP32Adder(neuron_template=nt)
        self.sign_not_hi = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.sign_not_lo = NOTGate(neuron_template=nt, max_param_shape=(1,))
        
        # 多项式 P(r) = 1 + r + C2*r^2 + C3*r^3
        # Horner: 1 + r*(1 + r*(C2 + r*C3))
        self.mul_r2 = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_r3 = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_c2 = SpikeFP32Multiplier(neuron_template=nt)
        self.mul_c3 = SpikeFP32Multiplier(neuron_template=nt)
        self.add_1 = SpikeFP32Adder(neuron_template=nt)
        self.add_2 = SpikeFP32Adder(neuron_template=nt)
        self.add_3 = SpikeFP32Adder(neuron_template=nt)
        
        # 组合
        self.mul_tp = SpikeFP32Multiplier(neuron_template=nt)
        
        # 2^k 缩放
        # 需要计算 k = floor(z/32)
        # z/32 = z * 0.03125
        self.mul_1_32 = SpikeFP32Multiplier(neuron_template=nt)
        self.floor_k = SpikeFP32Floor(neuron_template=nt)
        
        # 计算 j = z % 32
        self.mul_32 = SpikeFP32Multiplier(neuron_template=nt)
        self.sign_not_j = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.sub_j = SpikeFP32Adder(neuron_template=nt)

        self.scale = SpikeFP32ScaleBy2K(neuron_template=nt)

        # 特殊值 - 单实例
        self.exp_all_one_and = ANDGate(neuron_template=nt, max_param_shape=(1,))
        self.mant_zero_or = ORGate(neuron_template=nt, max_param_shape=(1,))
        self.mant_zero_not = NOTGate(neuron_template=nt, max_param_shape=(1,))
        self.is_nan_and = ANDGate(neuron_template=nt, max_param_shape=(1,))
        self.is_pos_inf_and = ANDGate(neuron_template=nt, max_param_shape=(1,))
        self.is_neg_inf_and = ANDGate(neuron_template=nt, max_param_shape=(1,))
        self.sign_not2 = NOTGate(neuron_template=nt, max_param_shape=(1,))

        self.nan_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.inf_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        self.zero_mux = MUXGate(neuron_template=nt, max_param_shape=(1,))
        
    def _make_constant(self, hex_val, batch_shape, device):
        pulse = torch.zeros(batch_shape + (32,), device=device)
        ival = int(hex_val, 16) if isinstance(hex_val, str) else hex_val
        for i in range(32):
            pulse[..., i] = float((ival >> (31 - i)) & 1)
        return pulse

    def forward(self, x):
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            s_x = x[..., 0:1]
            e_x = x[..., 1:9]
            m_x = x[..., 9:32]

            # 特殊值检测 (tree reduction)
            e_all_one = e_x[..., 0:1]
            for i in range(1, 8):
                e_all_one = self.exp_all_one_and(e_all_one, e_x[..., i:i+1])

            m_any_one = m_x[..., 0:1]
            for i in range(1, 23):
                m_any_one = self.mant_zero_or(m_any_one, m_x[..., i:i+1])
            m_is_zero = self.mant_zero_not(m_any_one)

            is_nan = self.is_nan_and(e_all_one, m_any_one)
            not_sign = self.sign_not2(s_x)
            is_inf_base = self.is_pos_inf_and(e_all_one, m_is_zero)
            is_pos_inf = self.is_pos_inf_and(is_inf_base, not_sign)
            is_neg_inf = self.is_neg_inf_and(is_inf_base, s_x)

            # 常量
            # N/ln2 = 32 / 0.693147... = 46.16624
            inv_ln2_n = self._make_constant(0x4238aa3b, batch_shape, device)
            # 0.5
            half = self._make_constant(0x3f000000, batch_shape, device)
            # ln2_hi/N = 0.69314575 / 32 = 0.0216608
            ln2_hi_n = self._make_constant(0x3cb17200, batch_shape, device)
            # ln2_lo/N = 1.428e-6 / 32 = 4.46e-8
            ln2_lo_n = self._make_constant(0x333fbe8e, batch_shape, device)
            # 1/32
            inv_32 = self._make_constant(0x3d000000, batch_shape, device)
            # Coeffs
            c1 = self._make_constant(0x3f800000, batch_shape, device) # 1.0
            c2 = self._make_constant(0x3f000000, batch_shape, device) # 0.5
            c3 = self._make_constant(0x3e2aaaab, batch_shape, device) # 0.16666...

            # 1. z = round(x * N/ln2)
            x_scaled = self.mul_inv_ln2(x, inv_ln2_n)
            x_plus_half = self.add_half(x_scaled, half)
            z = self.floor(x_plus_half)

            # 2. 分解 z -> k, j
            # k = floor(z / 32)
            z_div_32 = self.mul_1_32(z, inv_32)
            k = self.floor_k(z_div_32)

            # j = z % 32
            # j_float = z - k * 32
            # k * 32.0 (FP32)
            const_32 = self._make_constant(0x42000000, batch_shape, device)
            k_times_32 = self.mul_32(k, const_32)

            # negate k_times_32
            neg_k32_s = self.sign_not_j(k_times_32[..., 0:1])
            neg_k32 = torch.cat([neg_k32_s, k_times_32[..., 1:]], dim=-1)

            # j_float = z + (-k*32)
            j_float = self.sub_j(z, neg_k32)

            # 提取低5位
            j_bits = self.extract_j(j_float)

            # 3. r = x - z*(ln2_hi/N) - z*(ln2_lo/N)
            z_ln2_hi = self.mul_z_ln2_hi(z, ln2_hi_n)
            z_ln2_lo = self.mul_z_ln2_lo(z, ln2_lo_n)

            # Negate z_ln2
            neg_hi_s = self.sign_not_hi(z_ln2_hi[..., 0:1])
            neg_hi = torch.cat([neg_hi_s, z_ln2_hi[..., 1:]], dim=-1)

            neg_lo_s = self.sign_not_lo(z_ln2_lo[..., 0:1])
            neg_lo = torch.cat([neg_lo_s, z_ln2_lo[..., 1:]], dim=-1)

            # x - hi - lo
            tmp = self.sub_hi(x, neg_hi)
            r = self.sub_lo(tmp, neg_lo)

            # 4. T = lookup(j)
            T = self.lookup(j_bits)

            # 5. P = 1 + r + C2*r^2 + C3*r^3
            # Horner: 1 + r * (1 + r * (C2 + r * C3))
            # P0 = C3*r + C2
            r_c3 = self.mul_c3(r, c3)
            p0 = self.add_1(r_c3, c2)

            # P1 = p0*r + 1
            p0_r = self.mul_c2(p0, r)
            p1 = self.add_2(p0_r, c1)

            # P2 = p1*r + 1
            p1_r = self.mul_r2(p1, r)
            P = self.add_3(p1_r, c1)

            # 6. Result = 2^k * T * P
            TP = self.mul_tp(T, P)
            out_pulse = self.scale(TP, k)

            # 特殊值输出 (vectorized)
            nan_val = self._make_constant(0x7FC00000, batch_shape, device)
            inf_val = self._make_constant(0x7F800000, batch_shape, device)
            zero_val = torch.cat([zeros] * 32, dim=-1)

            # NaN
            is_nan_32 = is_nan.expand_as(out_pulse)
            out_pulse = self.nan_mux(is_nan_32, nan_val, out_pulse)

            # -Inf -> 0
            is_neg_inf_32 = is_neg_inf.expand_as(out_pulse)
            out_pulse = self.zero_mux(is_neg_inf_32, zero_val, out_pulse)

            # +Inf -> +Inf
            is_pos_inf_32 = is_pos_inf.expand_as(out_pulse)
            out_pulse = self.inf_mux(is_pos_inf_32, inf_val, out_pulse)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_exp
            return ste_exp(x, out_pulse)

        return out_pulse

    def reset(self):
        self.mul_inv_ln2.reset()
        self.add_half.reset()
        self.floor.reset()
        self.extract_j.reset()
        self.lookup.reset()
        self.mul_z_ln2_hi.reset()
        self.mul_z_ln2_lo.reset()
        self.sub_hi.reset()
        self.sub_lo.reset()
        self.sign_not_hi.reset()
        self.sign_not_lo.reset()
        self.mul_r2.reset()
        self.mul_r3.reset()
        self.mul_c2.reset()
        self.mul_c3.reset()
        self.add_1.reset()
        self.add_2.reset()
        self.add_3.reset()
        self.mul_tp.reset()
        self.mul_1_32.reset()
        self.floor_k.reset()
        self.mul_32.reset()
        self.sign_not_j.reset()
        self.sub_j.reset()
        self.scale.reset()
        self.exp_all_one_and.reset()
        self.mant_zero_or.reset()
        self.mant_zero_not.reset()
        self.is_nan_and.reset()
        self.is_pos_inf_and.reset()
        self.is_neg_inf_and.reset()
        self.sign_not2.reset()
        self.nan_mux.reset()
        self.inf_mux.reset()
        self.zero_mux.reset()
