"""
FP16 空间编码加法器 - 100%纯SNN门电路实现
用于多精度累加：FP8 -> FP16累加 -> FP8

FP16 格式: [S | E4 E3 E2 E1 E0 | M9..M0], bias=15
内部精度: 16位尾数 (hidden + 10 mant + 5 guard)
"""
import torch
import torch.nn as nn
from atomic_ops.core.logic_gates import (HalfAdder, FullAdder, ORTree)
# 单比特门改用 Vec* 版本（支持 max_param_shape）
# 注意：使用 VecAdder 代替旧的 RippleCarryAdder（支持 max_param_shape）
from atomic_ops.core.vec_logic_gates import (
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
    MAX_BITS = 16

    def __init__(self, neuron_template=None):
        super().__init__()
        # 向量化 MUX (单实例复用)
        self.vec_mux = VecMUX(neuron_template=neuron_template, max_param_shape=(self.MAX_BITS,))
            
    def forward(self, X, shift):
        """X: [..., 16], shift: [..., 4] (MSB first, 只用低4位)"""
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
    MAX_BITS = 16

    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        max_shape_16 = (self.MAX_BITS,)
        max_shape_4 = (4,)
        max_shape_1 = (1,)
        # 向量化门电路 (复用单实例)
        self.vec_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_4)
        self.vec_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_16)
        
    def forward(self, X):
        """X: [..., 16], returns: [..., 4] (LZC, MSB first)"""
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
    MAX_BITS = 16

    def __init__(self, neuron_template=None):
        super().__init__()
        # 预分配参数形状
        max_shape = (self.MAX_BITS,)
        self.vec_adder = VecAdder(16, neuron_template=neuron_template, max_param_shape=max_shape)
        
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
        # 单实例 (动态扩展机制支持不同位宽)
        self.exp_diff_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 绝对值比较 =====
        self.abs_eq_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.mant_ge_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.abs_ge_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.abs_ge_or = VecOR(neuron_template=nt, max_param_shape=(1,))

        # ===== E=0 检测 =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.e_zero_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_zero_not = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.subnorm_exp_mux_a = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.subnorm_exp_mux_b = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 对齐移位器 =====
        self.align_shifter = BarrelShifterRight16(neuron_template=nt)
        
        # ===== 尾数运算 =====
        self.mantissa_adder = RippleCarryAdder16(neuron_template=nt)
        self.mantissa_sub = Subtractor16Bit(neuron_template=nt)
        
        # ===== 符号处理 =====
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.exact_cancel_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 交换逻辑 =====
        self.swap_mux_s = VecMUX(neuron_template=nt, max_param_shape=(1,))
        # 单实例 (动态扩展机制支持不同位宽)
        self.swap_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.swap_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 结果选择 (Step 5: sum vs diff) =====
        self.result_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 尾数路径选择 (Step 7: overflow vs normal) =====
        self.mant_path_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 归一化 =====
        self.lzd = LeadingZeroDetector16(neuron_template=nt)
        self.norm_shifter = BarrelShifterLeft16(neuron_template=nt)
        self.exp_adj_sub = Subtractor5Bit(neuron_template=nt)
        
        # ===== 溢出处理 =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.exp_overflow_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.post_round_exp_inc = VecAdder(bits=5, neuron_template=nt, max_param_shape=(5,))

        # ===== 下溢处理 =====
        self.underflow_cmp = Comparator5Bit(neuron_template=nt)
        self.underflow_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.underflow_not = VecNOT(neuron_template=nt, max_param_shape=(1,))  # NOT(is_underflow) 用于舍入控制
        self.and_do_round = VecAND(neuron_template=nt, max_param_shape=(1,))  # do_round AND not_underflow (纯SNN)
        # 单实例 (动态扩展机制支持不同位宽)
        self.underflow_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.underflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 舍入 =====
        self.round_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.round_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.round_adder = VecAdder(bits=11, neuron_template=nt, max_param_shape=(11,))  # hidden + 10 mant
        
        # ===== 最终选择 =====
        self.cancel_mux_s = VecMUX(neuron_template=nt, max_param_shape=(1,))
        # 单实例 (动态扩展机制支持不同位宽)
        self.cancel_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.cancel_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 溢出到NaN =====
        self.exp_overflow_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.nan_mux_s = VecMUX(neuron_template=nt, max_param_shape=(1,))
        # 单实例 (动态扩展机制支持不同位宽)
        self.nan_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.nan_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 舍入溢出处理 =====
        self.not_round_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        # 单实例 (动态扩展机制支持不同位宽)
        self.mant_clear_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.round_exp_inc = VecAdder(bits=5, neuron_template=nt, max_param_shape=(5,))
        self.round_exp_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== Sticky bit 计算（纯SNN OR门链）=====
        # 单实例 (动态扩展机制支持不同位宽)
        self.sticky_overflow_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_norm_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
    def forward(self, A, B):
        """
        Args:
            A, B: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            [..., 16] FP16 脉冲
        """
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
        e_a_or_01 = self.e_zero_or(e_a[..., 3:4], e_a[..., 4:5])
        e_a_or_23 = self.e_zero_or(e_a[..., 1:2], e_a[..., 2:3])
        e_a_or_all = self.e_zero_or(e_a_or_01, e_a_or_23)
        e_a_nonzero = self.e_zero_or(e_a_or_all, e_a[..., 0:1])
        e_a_is_zero = self.e_zero_not(e_a_nonzero)
        hidden_a = e_a_nonzero

        e_b_or_01 = self.e_zero_or(e_b[..., 3:4], e_b[..., 4:5])
        e_b_or_23 = self.e_zero_or(e_b[..., 1:2], e_b[..., 2:3])
        e_b_or_all = self.e_zero_or(e_b_or_01, e_b_or_23)
        e_b_nonzero = self.e_zero_or(e_b_or_all, e_b[..., 0:1])
        e_b_is_zero = self.e_zero_not(e_b_nonzero)
        hidden_b = e_b_nonzero

        # 有效指数（subnormal时E=1）- 向量化
        subnorm_val_a = torch.cat([zeros, zeros, zeros, zeros, ones], dim=-1)
        e_a_is_zero_5 = e_a_is_zero.expand_as(e_a)
        e_a_eff = self.subnorm_exp_mux_a(e_a_is_zero_5, subnorm_val_a, e_a)

        subnorm_val_b = torch.cat([zeros, zeros, zeros, zeros, ones], dim=-1)
        e_b_is_zero_5 = e_b_is_zero.expand_as(e_b)
        e_b_eff = self.subnorm_exp_mux_b(e_b_is_zero_5, subnorm_val_b, e_b)
        
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
        
        # 向量化: exp_diff
        a_ge_b_5 = a_ge_b.expand_as(diff_ab)
        exp_diff = self.exp_diff_mux(a_ge_b_5, diff_ab, diff_ba)

        # 向量化: e_max
        e_max = self.swap_mux_e(a_ge_b_5, e_a_eff, e_b_eff)

        # ===== Step 3: 尾数对齐（16位）- 向量化 =====
        a_ge_b_16 = a_ge_b.expand_as(m_a)
        m_large = self.swap_mux_m(a_ge_b_16, m_a, m_b)
        m_small_unshifted = self.swap_mux_m(a_ge_b_16, m_b, m_a)
        
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
        
        # 向量化: mantissa_result
        is_diff_sign_16 = is_diff_sign.expand_as(sum_result)
        mantissa_result = self.result_mux(is_diff_sign_16, diff_result, sum_result)
        result_carry = self.result_mux(is_diff_sign, zeros, sum_carry)
        
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
        
        # 向量化: 选择指数
        zeros_5 = torch.cat([zeros] * 5, dim=-1)
        is_underflow_5 = is_underflow.expand_as(e_after_norm)
        e_normal = self.underflow_mux_e(is_underflow_5, zeros_5, e_after_norm)

        result_carry_5 = result_carry.expand_as(e_normal)
        final_e_pre = self.exp_overflow_mux(result_carry_5, e_plus_one, e_normal)
        
        # ===== Step 7: 提取尾数并舍入 =====
        # 溢出情况（result_carry=1）：结果形如 1x.xxxxx
        # 取位0-9作为尾数（隐藏位在位-1），位10是round，位11-15是sticky
        m_overflow = mantissa_result[..., 0:10]
        round_overflow = mantissa_result[..., 10:11]
        # sticky_overflow = OR(bits 11-15) 使用纯SNN门电路
        s_ov_01 = self.sticky_overflow_or(mantissa_result[..., 11:12], mantissa_result[..., 12:13])
        s_ov_23 = self.sticky_overflow_or(mantissa_result[..., 13:14], mantissa_result[..., 14:15])
        s_ov_0123 = self.sticky_overflow_or(s_ov_01, s_ov_23)
        sticky_overflow = self.sticky_overflow_or(s_ov_0123, mantissa_result[..., 15:16])

        # 正常归一化情况：归一化后位0是隐藏位=1，取位1-10作为尾数
        # 位11是round，位12-15是sticky
        m_norm = norm_mantissa[..., 1:11]
        round_norm = norm_mantissa[..., 11:12]
        # sticky_norm = OR(bits 12-15) 使用纯SNN门电路
        s_nm_01 = self.sticky_norm_or(norm_mantissa[..., 12:13], norm_mantissa[..., 13:14])
        s_nm_23 = self.sticky_norm_or(norm_mantissa[..., 14:15], norm_mantissa[..., 15:16])
        sticky_norm = self.sticky_norm_or(s_nm_01, s_nm_23)
        
        # 下溢情况（subnormal）：没有隐藏位，取位0-9作为尾数
        m_subnorm = mantissa_result[..., 0:10]
        
        # 选择尾数和舍入位
        # 1. 先根据 result_carry 选择溢出 vs 正常路径
        # 2. 再根据 is_underflow 选择下溢路径
        
        # 溢出 vs 正常归一化选择 - 向量化
        result_carry_10 = result_carry.expand_as(m_overflow)
        m_pre = self.mant_path_mux(result_carry_10, m_overflow, m_norm)
        round_pre = self.mant_path_mux(result_carry, round_overflow, round_norm)
        sticky_pre = self.mant_path_mux(result_carry, sticky_overflow, sticky_norm)

        # 下溢选择 - 向量化
        is_underflow_10 = is_underflow.expand_as(m_pre)
        m_selected = self.underflow_mux_m(is_underflow_10, m_subnorm, m_pre)
        
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
        
        # 舍入溢出处理 - 向量化
        not_round_c = self.not_round_carry(round_carry)
        not_round_c_10 = not_round_c.expand_as(m_rounded)
        m_final = self.mant_clear_and(not_round_c_10, m_rounded)

        # 指数调整
        exp_round_inc = torch.cat([zeros, zeros, zeros, zeros, round_carry], dim=-1)
        e_rounded_lsb, _ = self.round_exp_inc(final_e_pre.flip(-1), exp_round_inc.flip(-1))
        e_rounded = e_rounded_lsb.flip(-1)

        # 向量化
        round_carry_5 = round_carry.expand_as(e_rounded)
        computed_e = self.round_exp_mux(round_carry_5, e_rounded, final_e_pre)
        
        # ===== Step 8: 符号 =====
        computed_s = s_large  # 简化：取较大数的符号
        
        # ===== 完全抵消 - 向量化 =====
        cancel_s = self.cancel_mux_s(exact_cancel, zeros, computed_s)
        exact_cancel_5 = exact_cancel.expand_as(computed_e)
        cancel_e = self.cancel_mux_e(exact_cancel_5, zeros_5, computed_e)

        zeros_10 = torch.cat([zeros] * 10, dim=-1)
        exact_cancel_10 = exact_cancel.expand_as(m_final)
        cancel_m = self.cancel_mux_m(exact_cancel_10, zeros_10, m_final)

        # ===== NaN处理 - 向量化 =====
        is_exp_overflow = self.exp_overflow_and(result_carry, exp_inc_carry)

        final_s = self.nan_mux_s(is_exp_overflow, computed_s, cancel_s)

        ones_5 = torch.cat([ones] * 5, dim=-1)
        is_exp_overflow_5 = is_exp_overflow.expand_as(cancel_e)
        final_e = self.nan_mux_e(is_exp_overflow_5, ones_5, cancel_e)

        ones_10 = torch.cat([ones] * 10, dim=-1)
        is_exp_overflow_10 = is_exp_overflow.expand_as(cancel_m)
        final_m = self.nan_mux_m(is_exp_overflow_10, ones_10, cancel_m)
        
        return torch.cat([final_s, final_e, final_m], dim=-1)
    
    def reset(self):
        self.exp_cmp.reset()
        self.mantissa_cmp.reset()
        self.exp_sub_ab.reset()
        self.exp_sub_ba.reset()
        self.exp_diff_mux.reset()
        self.abs_eq_and.reset()
        self.mant_ge_or.reset()
        self.abs_ge_and.reset()
        self.abs_ge_or.reset()
        self.e_zero_or.reset()
        self.e_zero_not.reset()
        self.subnorm_exp_mux_a.reset()
        self.subnorm_exp_mux_b.reset()
        self.align_shifter.reset()
        self.mantissa_adder.reset()
        self.mantissa_sub.reset()
        self.sign_xor.reset()
        self.exact_cancel_and.reset()
        self.swap_mux_s.reset()
        self.swap_mux_e.reset()
        self.swap_mux_m.reset()
        self.result_mux.reset()
        self.mant_path_mux.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.exp_adj_sub.reset()
        self.exp_overflow_mux.reset()
        self.post_round_exp_inc.reset()
        self.underflow_cmp.reset()
        self.underflow_or.reset()
        self.underflow_not.reset()
        self.and_do_round.reset()
        self.underflow_mux_e.reset()
        self.underflow_mux_m.reset()
        self.round_or.reset()
        self.round_and.reset()
        self.round_adder.reset()
        self.cancel_mux_s.reset()
        self.cancel_mux_e.reset()
        self.cancel_mux_m.reset()
        self.exp_overflow_and.reset()
        self.nan_mux_s.reset()
        self.nan_mux_e.reset()
        self.nan_mux_m.reset()
        self.not_round_carry.reset()
        self.mant_clear_and.reset()
        self.round_exp_inc.reset()
        self.round_exp_mux.reset()
        self.sticky_overflow_or.reset()
        self.sticky_norm_or.reset()

