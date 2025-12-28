"""
FP16 空间编码加法器 - 100%纯SNN门电路实现
用于多精度累加：FP8 -> FP16累加 -> FP8

FP16 格式: [S | E4 E3 E2 E1 E0 | M9..M0], bias=15
内部精度: 16位尾数 (hidden + 10 mant + 5 guard)
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate, 
                          HalfAdder, FullAdder, RippleCarryAdder, ORTree)
from .vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX, VecORTree, VecANDTree,
    VecFullAdder, VecAdder, VecSubtractor
)
from .fp16_components import ComparatorNBit, SubtractorNBit, BarrelShifterRightNBit


# ==============================================================================
# FP16 专用组件
# ==============================================================================
class Comparator5Bit(ComparatorNBit):
    """5位比较器（FP16指数）"""
    def __init__(self, neuron_template=None):
        super().__init__(bits=5, neuron_template=neuron_template)


class Comparator11Bit(ComparatorNBit):
    """11位比较器（FP16尾数: hidden + 10 mant）"""
    def __init__(self, neuron_template=None):
        super().__init__(bits=11, neuron_template=neuron_template)


class Subtractor5Bit(SubtractorNBit):
    """5位减法器（FP16指数差）"""
    def __init__(self, neuron_template=None):
        super().__init__(bits=5, neuron_template=neuron_template)


class BarrelShifterRight16(BarrelShifterRightNBit):
    """16位桶形右移位器"""
    def __init__(self, neuron_template=None):
        super().__init__(data_bits=16, shift_bits=5, neuron_template=neuron_template)


class BarrelShifterLeft16(nn.Module):
    """16位桶形左移位器（归一化）- 向量化"""
    def __init__(self, neuron_template=None):
        super().__init__()
        # 向量化 MUX (单实例复用)
        self.vec_mux = VecMUX(neuron_template=neuron_template)
            
    def forward(self, X, shift):
        """X: [..., 16], shift: [..., 4] (MSB first, 只用低4位)"""
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        
        current = X
        for layer in range(4):
            shift_amt = 2 ** (3 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 左移操作 (向量化)
            zeros_pad = torch.zeros(batch_shape + (shift_amt,), device=device)
            shifted = torch.cat([current[..., shift_amt:], zeros_pad], dim=-1)
            s_bit_exp = s_bit.expand(*batch_shape, 16)
            current = self.vec_mux(s_bit_exp, shifted, current)
        
        return current
    
    def reset(self):
        self.vec_mux.reset()


class LeadingZeroDetector16(nn.Module):
    """16位前导零检测器 - 输出4位LZC (向量化SNN)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化门电路 (复用单实例)
        self.vec_not = VecNOT(neuron_template=nt)
        self.vec_and = VecAND(neuron_template=nt)
        self.vec_or = VecOR(neuron_template=nt)
        self.vec_mux = VecMUX(neuron_template=nt)
        self.vec_or_tree = VecORTree(neuron_template=nt)
        
    def forward(self, X):
        """X: [..., 16], returns: [..., 4] (LZC, MSB first)"""
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        # 初始化 lzc 为 0 (4位)
        lzc = torch.zeros(batch_shape + (4,), device=device)
        found = zeros.clone()
        
        for i in range(16):
            bit = X[..., i:i+1]
            not_found = self.vec_not(found)
            is_first = self.vec_and(bit, not_found)
            
            # 如果 is_first=1，设置 lzc 为当前位置 i 的二进制表示
            pos_bits = torch.tensor([
                (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1
            ], device=device, dtype=torch.float32)
            pos_bits = pos_bits.expand(*batch_shape, 4)
            
            # lzc = MUX(is_first, pos_bits, lzc)
            is_first_exp = is_first.expand(*batch_shape, 4)
            lzc = self.vec_mux(is_first_exp, pos_bits, lzc)
            
            found = self.vec_or(found, is_first)
        
        # 全零: lzc = 15 (1111)
        any_one = self.vec_or_tree(X)
        all_zero = self.vec_not(any_one)
        lzc_15 = torch.ones(batch_shape + (4,), device=device)
        all_zero_exp = all_zero.expand(*batch_shape, 4)
        lzc = self.vec_mux(all_zero_exp, lzc_15, lzc)
        
        return lzc
    
    def reset(self):
        self.vec_not.reset()
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_mux.reset()
        self.vec_or_tree.reset()


class RippleCarryAdder16(nn.Module):
    """16位加法器 - 向量化"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.vec_adder = VecAdder(16, neuron_template=neuron_template)
        
    def forward(self, A, B, Cin=None):
        """A, B: [..., 16] LSB first"""
        return self.vec_adder(A, B, Cin)
    
    def reset(self):
        self.vec_adder.reset()


class Subtractor16Bit(SubtractorNBit):
    """16位减法器"""
    def __init__(self, neuron_template=None):
        super().__init__(bits=16, neuron_template=neuron_template)


# ==============================================================================
# FP16 加法器主体
# ==============================================================================
class SpikeFP16Adder(nn.Module):
    """FP16 空间编码加法器 - 100%纯SNN门电路
    
    FP16 格式: [S | E4..E0 | M9..M0], bias=15
    内部使用 16 位尾数精度
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # ===== 指数比较 =====
        self.exp_cmp = Comparator5Bit(neuron_template=nt)
        self.mantissa_cmp = Comparator11Bit(neuron_template=nt)  # hidden + 10 mant
        
        # ===== 指数差 =====
        self.exp_sub_ab = Subtractor5Bit(neuron_template=nt)
        self.exp_sub_ba = Subtractor5Bit(neuron_template=nt)
        self.exp_diff_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        
        # ===== 绝对值比较 =====
        self.abs_eq_and = ANDGate(neuron_template=nt)
        self.mant_ge_or = ORGate(neuron_template=nt)
        self.abs_ge_and = ANDGate(neuron_template=nt)
        self.abs_ge_or = ORGate(neuron_template=nt)
        
        # ===== E=0 检测 =====
        self.e_zero_or = nn.ModuleList([ORGate(neuron_template=nt) for _ in range(8)])
        self.e_zero_not = nn.ModuleList([NOTGate(neuron_template=nt) for _ in range(2)])
        self.subnorm_exp_mux_a = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        self.subnorm_exp_mux_b = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        
        # ===== 对齐移位器 =====
        self.align_shifter = BarrelShifterRight16(neuron_template=nt)
        
        # ===== 尾数运算 =====
        self.mantissa_adder = RippleCarryAdder16(neuron_template=nt)
        self.mantissa_sub = Subtractor16Bit(neuron_template=nt)
        
        # ===== 符号处理 =====
        self.sign_xor = XORGate(neuron_template=nt)
        self.exact_cancel_and = ANDGate(neuron_template=nt)
        
        # ===== 交换逻辑 =====
        self.swap_mux_s = MUXGate(neuron_template=nt)
        self.swap_mux_e = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        self.swap_mux_m = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(32)])  # 2 * 16
        
        # ===== 结果选择 (Step 5: sum vs diff) =====
        self.result_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(17)])  # 16 + carry
        
        # ===== 尾数路径选择 (Step 7: overflow vs normal) =====
        self.mant_path_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(12)])  # 10 mant + round + sticky
        
        # ===== 归一化 =====
        self.lzd = LeadingZeroDetector16(neuron_template=nt)
        self.norm_shifter = BarrelShifterLeft16(neuron_template=nt)
        self.exp_adj_sub = Subtractor5Bit(neuron_template=nt)
        
        # ===== 溢出处理 =====
        self.exp_overflow_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        self.post_round_exp_inc = RippleCarryAdder(bits=5, neuron_template=nt)
        
        # ===== 下溢处理 =====
        self.underflow_cmp = Comparator5Bit(neuron_template=nt)
        self.underflow_or = ORGate(neuron_template=nt)
        self.underflow_not = NOTGate(neuron_template=nt)  # NOT(is_underflow) 用于舍入控制
        self.and_do_round = ANDGate(neuron_template=nt)  # do_round AND not_underflow (纯SNN)
        self.underflow_mux_e = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        self.underflow_mux_m = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(10)])
        
        # ===== 舍入 =====
        self.round_or = ORGate(neuron_template=nt)
        self.round_and = ANDGate(neuron_template=nt)
        self.round_adder = RippleCarryAdder(bits=11, neuron_template=nt)  # hidden + 10 mant
        
        # ===== 最终选择 =====
        self.cancel_mux_s = MUXGate(neuron_template=nt)
        self.cancel_mux_e = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        self.cancel_mux_m = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(10)])
        
        # ===== 溢出到NaN =====
        self.exp_overflow_and = ANDGate(neuron_template=nt)
        self.nan_mux_s = MUXGate(neuron_template=nt)
        self.nan_mux_e = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        self.nan_mux_m = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(10)])
        
        # ===== 舍入溢出处理 =====
        self.not_round_carry = NOTGate(neuron_template=nt)
        self.mant_clear_and = nn.ModuleList([ANDGate(neuron_template=nt) for _ in range(10)])
        self.round_exp_inc = RippleCarryAdder(bits=5, neuron_template=nt)
        self.round_exp_mux = nn.ModuleList([MUXGate(neuron_template=nt) for _ in range(5)])
        
        # ===== Sticky bit 计算（纯SNN OR门链）=====
        # sticky_overflow: OR(bits 11-15) = 5 bits
        self.sticky_overflow_or = nn.ModuleList([ORGate(neuron_template=nt) for _ in range(4)])  # 4个OR门组成链
        # sticky_norm: OR(bits 12-15) = 4 bits
        self.sticky_norm_or = nn.ModuleList([ORGate(neuron_template=nt) for _ in range(3)])  # 3个OR门组成链
        
    def forward(self, A, B):
        """
        Args:
            A, B: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            [..., 16] FP16 脉冲
        """
        self.reset()
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取各部分
        s_a = A[..., 0:1]
        e_a = A[..., 1:6]  # [E4..E0] MSB first
        m_a_raw = A[..., 6:16]  # [M9..M0]
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:6]
        m_b_raw = B[..., 6:16]
        
        # ===== E=0 检测与隐藏位 =====
        e_a_or_01 = self.e_zero_or[0](e_a[..., 3:4], e_a[..., 4:5])
        e_a_or_23 = self.e_zero_or[1](e_a[..., 1:2], e_a[..., 2:3])
        e_a_or_all = self.e_zero_or[2](e_a_or_01, e_a_or_23)
        e_a_nonzero = self.e_zero_or[3](e_a_or_all, e_a[..., 0:1])
        e_a_is_zero = self.e_zero_not[0](e_a_nonzero)
        hidden_a = e_a_nonzero
        
        e_b_or_01 = self.e_zero_or[4](e_b[..., 3:4], e_b[..., 4:5])
        e_b_or_23 = self.e_zero_or[5](e_b[..., 1:2], e_b[..., 2:3])
        e_b_or_all = self.e_zero_or[6](e_b_or_01, e_b_or_23)
        e_b_nonzero = self.e_zero_or[7](e_b_or_all, e_b[..., 0:1])
        e_b_is_zero = self.e_zero_not[1](e_b_nonzero)
        hidden_b = e_b_nonzero
        
        # 有效指数（subnormal时E=1）
        e_a_eff = []
        for i in range(5):
            if i == 4:  # LSB
                e_a_eff.append(self.subnorm_exp_mux_a[i](e_a_is_zero, ones, e_a[..., i:i+1]))
            else:
                e_a_eff.append(self.subnorm_exp_mux_a[i](e_a_is_zero, zeros, e_a[..., i:i+1]))
        e_a_eff = torch.cat(e_a_eff, dim=-1)
        
        e_b_eff = []
        for i in range(5):
            if i == 4:
                e_b_eff.append(self.subnorm_exp_mux_b[i](e_b_is_zero, ones, e_b[..., i:i+1]))
            else:
                e_b_eff.append(self.subnorm_exp_mux_b[i](e_b_is_zero, zeros, e_b[..., i:i+1]))
        e_b_eff = torch.cat(e_b_eff, dim=-1)
        
        # 16位尾数: hidden(1) + M(10) + guard(5)
        m_a = torch.cat([hidden_a, m_a_raw, zeros, zeros, zeros, zeros, zeros], dim=-1)
        m_b = torch.cat([hidden_b, m_b_raw, zeros, zeros, zeros, zeros, zeros], dim=-1)
        
        # ===== Step 1: 比较绝对值 =====
        a_exp_gt, a_exp_eq = self.exp_cmp(e_a_eff, e_b_eff)
        m_a_with_hidden = torch.cat([hidden_a, m_a_raw], dim=-1)
        m_b_with_hidden = torch.cat([hidden_b, m_b_raw], dim=-1)
        a_mant_gt, a_mant_eq = self.mantissa_cmp(m_a_with_hidden, m_b_with_hidden)
        
        a_abs_eq_b = self.abs_eq_and(a_exp_eq, a_mant_eq)
        a_mant_ge = self.mant_ge_or(a_mant_gt, a_mant_eq)
        exp_eq_and_mant_ge = self.abs_ge_and(a_exp_eq, a_mant_ge)
        a_ge_b = self.abs_ge_or(a_exp_gt, exp_eq_and_mant_ge)
        
        # ===== Step 2: 指数差（转LSB first进行减法）=====
        e_a_lsb = e_a_eff.flip(-1)
        e_b_lsb = e_b_eff.flip(-1)
        diff_ab_lsb, _ = self.exp_sub_ab(e_a_lsb, e_b_lsb)
        diff_ba_lsb, _ = self.exp_sub_ba(e_b_lsb, e_a_lsb)
        diff_ab = diff_ab_lsb.flip(-1)
        diff_ba = diff_ba_lsb.flip(-1)
        
        exp_diff = []
        for i in range(5):
            d = self.exp_diff_mux[i](a_ge_b, diff_ab[..., i:i+1], diff_ba[..., i:i+1])
            exp_diff.append(d)
        exp_diff = torch.cat(exp_diff, dim=-1)
        
        # e_max
        e_max = []
        for i in range(5):
            e = self.swap_mux_e[i](a_ge_b, e_a_eff[..., i:i+1], e_b_eff[..., i:i+1])
            e_max.append(e)
        e_max = torch.cat(e_max, dim=-1)
        
        # ===== Step 3: 尾数对齐（16位）=====
        m_large = []
        m_small_unshifted = []
        for i in range(16):
            ml = self.swap_mux_m[i](a_ge_b, m_a[..., i:i+1], m_b[..., i:i+1])
            ms = self.swap_mux_m[i+16](a_ge_b, m_b[..., i:i+1], m_a[..., i:i+1])
            m_large.append(ml)
            m_small_unshifted.append(ms)
        m_large = torch.cat(m_large, dim=-1)
        m_small_unshifted = torch.cat(m_small_unshifted, dim=-1)
        
        m_small = self.align_shifter(m_small_unshifted, exp_diff)
        
        # ===== Step 4: 符号处理 =====
        is_diff_sign = self.sign_xor(s_a, s_b)
        exact_cancel = self.exact_cancel_and(is_diff_sign, a_abs_eq_b)
        s_large = self.swap_mux_s(a_ge_b, s_a, s_b)
        
        # ===== Step 5: 尾数运算（16位，LSB first）=====
        m_large_lsb = m_large.flip(-1)
        m_small_lsb = m_small.flip(-1)
        
        sum_result_lsb, sum_carry = self.mantissa_adder(m_large_lsb, m_small_lsb)
        diff_result_lsb, _ = self.mantissa_sub(m_large_lsb, m_small_lsb)  # 也使用 LSB first
        
        sum_result = sum_result_lsb.flip(-1)
        diff_result = diff_result_lsb.flip(-1)  # 转回 MSB first
        
        mantissa_result = []
        for i in range(16):
            r = self.result_mux[i](is_diff_sign, diff_result[..., i:i+1], sum_result[..., i:i+1])
            mantissa_result.append(r)
        mantissa_result = torch.cat(mantissa_result, dim=-1)
        result_carry = self.result_mux[16](is_diff_sign, zeros, sum_carry)
        
        # ===== Step 6: 归一化 =====
        lzc = self.lzd(mantissa_result)
        lzc_5bit = torch.cat([zeros, lzc], dim=-1)
        
        # 检测下溢
        lzc_gt_emax, lzc_eq_emax = self.underflow_cmp(lzc_5bit, e_max)
        is_underflow = self.underflow_or(lzc_gt_emax, lzc_eq_emax)
        
        # 归一化
        norm_mantissa = self.norm_shifter(mantissa_result, lzc)
        e_after_norm_lsb, _ = self.exp_adj_sub(e_max.flip(-1), lzc_5bit.flip(-1))
        e_after_norm = e_after_norm_lsb.flip(-1)
        
        # 溢出路径：E + 1
        one_5bit = torch.cat([zeros, zeros, zeros, zeros, ones], dim=-1)
        e_inc_lsb, exp_inc_carry = self.post_round_exp_inc(e_max.flip(-1), one_5bit.flip(-1))
        e_plus_one = e_inc_lsb.flip(-1)
        
        # 选择指数
        e_normal = []
        for i in range(5):
            e_sel = self.underflow_mux_e[i](is_underflow, zeros, e_after_norm[..., i:i+1])
            e_normal.append(e_sel)
        e_normal = torch.cat(e_normal, dim=-1)
        
        final_e_pre = []
        for i in range(5):
            e_sel = self.exp_overflow_mux[i](result_carry, e_plus_one[..., i:i+1], e_normal[..., i:i+1])
            final_e_pre.append(e_sel)
        final_e_pre = torch.cat(final_e_pre, dim=-1)
        
        # ===== Step 7: 提取尾数并舍入 =====
        # 溢出情况（result_carry=1）：结果形如 1x.xxxxx
        # 取位0-9作为尾数（隐藏位在位-1），位10是round，位11-15是sticky
        m_overflow = mantissa_result[..., 0:10]
        round_overflow = mantissa_result[..., 10:11]
        # sticky_overflow = OR(bits 11-15) 使用纯SNN门电路
        s_ov_01 = self.sticky_overflow_or[0](mantissa_result[..., 11:12], mantissa_result[..., 12:13])
        s_ov_23 = self.sticky_overflow_or[1](mantissa_result[..., 13:14], mantissa_result[..., 14:15])
        s_ov_0123 = self.sticky_overflow_or[2](s_ov_01, s_ov_23)
        sticky_overflow = self.sticky_overflow_or[3](s_ov_0123, mantissa_result[..., 15:16])
        
        # 正常归一化情况：归一化后位0是隐藏位=1，取位1-10作为尾数
        # 位11是round，位12-15是sticky
        m_norm = norm_mantissa[..., 1:11]
        round_norm = norm_mantissa[..., 11:12]
        # sticky_norm = OR(bits 12-15) 使用纯SNN门电路
        s_nm_01 = self.sticky_norm_or[0](norm_mantissa[..., 12:13], norm_mantissa[..., 13:14])
        s_nm_23 = self.sticky_norm_or[1](norm_mantissa[..., 14:15], norm_mantissa[..., 15:16])
        sticky_norm = self.sticky_norm_or[2](s_nm_01, s_nm_23)
        
        # 下溢情况（subnormal）：没有隐藏位，取位0-9作为尾数
        m_subnorm = mantissa_result[..., 0:10]
        
        # 选择尾数和舍入位
        # 1. 先根据 result_carry 选择溢出 vs 正常路径
        # 2. 再根据 is_underflow 选择下溢路径
        
        # 溢出 vs 正常归一化选择（使用单独的 MUX 门，避免状态污染）
        m_pre = []
        for i in range(10):
            m_sel = self.mant_path_mux[i](result_carry, m_overflow[..., i:i+1], m_norm[..., i:i+1])
            m_pre.append(m_sel)
        m_pre = torch.cat(m_pre, dim=-1)
        
        round_pre = self.mant_path_mux[10](result_carry, round_overflow, round_norm)
        sticky_pre = self.mant_path_mux[11](result_carry, sticky_overflow, sticky_norm)
        
        # 下溢选择（subnormal不需要舍入，直接截断）
        m_selected = []
        for i in range(10):
            m_sel = self.underflow_mux_m[i](is_underflow, m_subnorm[..., i:i+1], m_pre[..., i:i+1])
            m_selected.append(m_sel)
        m_selected = torch.cat(m_selected, dim=-1)
        
        # RNE舍入（仅对非下溢情况）
        L = m_selected[..., 9:10]  # LSB of result
        R = round_pre
        S = sticky_pre
        sticky_or_L = self.round_or(S, L)
        do_round = self.round_and(R, sticky_or_L)
        
        # 下溢时不舍入 (纯SNN AND门)
        not_underflow = self.underflow_not(is_underflow)
        do_round = self.and_do_round(do_round, not_underflow)
        
        # 尾数+1（LSB first）
        m_selected_lsb = m_selected.flip(-1)
        m_11bit_lsb = torch.cat([m_selected_lsb, zeros], dim=-1)  # [M0..M9, 0]
        round_inc = torch.cat([do_round] + [zeros]*10, dim=-1)
        m_rounded_lsb, round_carry = self.round_adder(m_11bit_lsb, round_inc)
        m_rounded = m_rounded_lsb[..., :10].flip(-1)  # 取低10位并转回MSB first
        
        # 舍入溢出处理（如果进位，尾数变0）
        not_round_c = self.not_round_carry(round_carry)
        m_final = []
        for i in range(10):
            m_bit = self.mant_clear_and[i](not_round_c, m_rounded[..., i:i+1])
            m_final.append(m_bit)
        m_final = torch.cat(m_final, dim=-1)
        
        # 指数调整
        exp_round_inc = torch.cat([zeros, zeros, zeros, zeros, round_carry], dim=-1)
        e_rounded_lsb, _ = self.round_exp_inc(final_e_pre.flip(-1), exp_round_inc.flip(-1))
        e_rounded = e_rounded_lsb.flip(-1)
        
        computed_e = []
        for i in range(5):
            e_sel = self.round_exp_mux[i](round_carry, e_rounded[..., i:i+1], final_e_pre[..., i:i+1])
            computed_e.append(e_sel)
        computed_e = torch.cat(computed_e, dim=-1)
        
        # ===== Step 8: 符号 =====
        computed_s = s_large  # 简化：取较大数的符号
        
        # ===== 完全抵消 =====
        cancel_s = self.cancel_mux_s(exact_cancel, zeros, computed_s)
        cancel_e = []
        for i in range(5):
            e_sel = self.cancel_mux_e[i](exact_cancel, zeros, computed_e[..., i:i+1])
            cancel_e.append(e_sel)
        cancel_e = torch.cat(cancel_e, dim=-1)
        
        cancel_m = []
        for i in range(10):
            m_sel = self.cancel_mux_m[i](exact_cancel, zeros, m_final[..., i:i+1])
            cancel_m.append(m_sel)
        cancel_m = torch.cat(cancel_m, dim=-1)
        
        # ===== NaN处理 =====
        is_exp_overflow = self.exp_overflow_and(result_carry, exp_inc_carry)
        
        final_s = self.nan_mux_s(is_exp_overflow, computed_s, cancel_s)
        
        final_e = []
        for i in range(5):
            e_sel = self.nan_mux_e[i](is_exp_overflow, ones, cancel_e[..., i:i+1])
            final_e.append(e_sel)
        final_e = torch.cat(final_e, dim=-1)
        
        final_m = []
        for i in range(10):
            m_sel = self.nan_mux_m[i](is_exp_overflow, ones, cancel_m[..., i:i+1])
            final_m.append(m_sel)
        final_m = torch.cat(final_m, dim=-1)
        
        return torch.cat([final_s, final_e, final_m], dim=-1)
    
    def reset(self):
        self.exp_cmp.reset()
        self.mantissa_cmp.reset()
        self.exp_sub_ab.reset()
        self.exp_sub_ba.reset()
        for m in self.exp_diff_mux: m.reset()
        self.abs_eq_and.reset()
        self.mant_ge_or.reset()
        self.abs_ge_and.reset()
        self.abs_ge_or.reset()
        for g in self.e_zero_or: g.reset()
        for g in self.e_zero_not: g.reset()
        for m in self.subnorm_exp_mux_a: m.reset()
        for m in self.subnorm_exp_mux_b: m.reset()
        self.align_shifter.reset()
        self.mantissa_adder.reset()
        self.mantissa_sub.reset()
        self.sign_xor.reset()
        self.exact_cancel_and.reset()
        self.swap_mux_s.reset()
        for m in self.swap_mux_e: m.reset()
        for m in self.swap_mux_m: m.reset()
        for m in self.result_mux: m.reset()
        for m in self.mant_path_mux: m.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.exp_adj_sub.reset()
        for m in self.exp_overflow_mux: m.reset()
        self.post_round_exp_inc.reset()
        self.underflow_cmp.reset()
        self.underflow_or.reset()
        self.underflow_not.reset()
        self.and_do_round.reset()
        for m in self.underflow_mux_e: m.reset()
        for m in self.underflow_mux_m: m.reset()
        self.round_or.reset()
        self.round_and.reset()
        self.round_adder.reset()
        self.cancel_mux_s.reset()
        for m in self.cancel_mux_e: m.reset()
        for m in self.cancel_mux_m: m.reset()
        self.exp_overflow_and.reset()
        self.nan_mux_s.reset()
        for m in self.nan_mux_e: m.reset()
        for m in self.nan_mux_m: m.reset()
        self.not_round_carry.reset()
        for g in self.mant_clear_and: g.reset()
        self.round_exp_inc.reset()
        for m in self.round_exp_mux: m.reset()
        for g in self.sticky_overflow_or: g.reset()
        for g in self.sticky_norm_or: g.reset()

