"""
FP32 空间编码加法器 - 100%纯SNN门电路实现 (向量化版本)
用于对齐朴素 PyTorch nn.Linear（FP32 累加）

FP32 格式: [S | E7..E0 | M22..M0], bias=127
内部精度: 28位尾数 (hidden + 23 mant + 4 guard)

向量化原则:
1. 使用 VecAND, VecOR, VecXOR, VecNOT, VecMUX 代替 ModuleList
2. 可并行操作一次处理所有位
3. 串行依赖仍保留循环
"""
import torch
import torch.nn as nn
from .vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecORTree, VecANDTree, VecAdder, VecSubtractor
)
from .logic_gates import RippleCarryAdder
from .fp32_components import (
    Comparator8Bit, Comparator24Bit, Subtractor8Bit,
    RippleCarryAdder28Bit, Subtractor28Bit,
    BarrelShifterRight28WithSticky,
    BarrelShifterLeft28, LeadingZeroDetector28
)


class SpikeFP32Adder(nn.Module):
    """FP32 空间编码加法器 - 100%纯SNN门电路 (向量化版本)
    
    FP32 格式: [S | E7..E0 | M22..M0], bias=127
    内部使用 28 位尾数精度 (hidden + 23 mant + 4 guard)
    """
    def __init__(self):
        super().__init__()
        
        # ===== 向量化基础门电路 =====
        self.vec_and = VecAND()
        self.vec_or = VecOR()
        self.vec_xor = VecXOR()
        self.vec_not = VecNOT()
        self.vec_mux = VecMUX()
        self.vec_or_tree = VecORTree()
        self.vec_and_tree = VecANDTree()
        
        # ===== 比较器 =====
        self.exp_cmp = Comparator8Bit()
        self.mantissa_cmp = Comparator24Bit()
        
        # ===== 指数差 =====
        self.exp_sub_ab = Subtractor8Bit()
        self.exp_sub_ba = Subtractor8Bit()
        
        # ===== 绝对值比较门 =====
        self.abs_eq_and = VecAND()
        self.mant_ge_or = VecOR()
        self.abs_ge_and = VecAND()
        self.abs_ge_or = VecOR()
        
        # ===== 对齐移位器 =====
        self.align_shifter = BarrelShifterRight28WithSticky()
        
        # ===== 尾数运算 =====
        self.mantissa_adder = RippleCarryAdder28Bit()
        self.mantissa_sub = Subtractor28Bit()
        self.sub_one = Subtractor28Bit()
        
        # ===== 归一化 =====
        self.lzd = LeadingZeroDetector28()
        self.norm_shifter = BarrelShifterLeft28()
        self.exp_adj_sub = Subtractor8Bit()
        
        # ===== 溢出/下溢处理 =====
        self.underflow_cmp = Comparator8Bit()
        self.post_round_exp_inc = RippleCarryAdder(bits=8)
        self.round_exp_inc = RippleCarryAdder(bits=8)
        
        # ===== 舍入 =====
        self.round_adder = RippleCarryAdder(bits=24)
        
    def forward(self, A, B):
        """
        Args:
            A, B: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        Returns:
            [..., 32] FP32 脉冲
        """
        self.reset()
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取各部分
        s_a = A[..., 0:1]
        e_a = A[..., 1:9]   # [E7..E0] MSB first
        m_a_raw = A[..., 9:32]  # [M22..M0]
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:9]
        m_b_raw = B[..., 9:32]
        
        # ===== E=0 检测与隐藏位 (向量化) =====
        e_a_nonzero = self.vec_or_tree(e_a)
        e_a_is_zero = self.vec_not(e_a_nonzero)
        hidden_a = e_a_nonzero
        
        e_b_nonzero = self.vec_or_tree(e_b)
        e_b_is_zero = self.vec_not(e_b_nonzero)
        hidden_b = e_b_nonzero
        
        # 有效指数 (subnormal时E=1)
        e_one = torch.cat([zeros]*7 + [ones], dim=-1)
        e_a_eff = self.vec_mux(e_a_is_zero.expand_as(e_a), e_one, e_a)
        e_b_eff = self.vec_mux(e_b_is_zero.expand_as(e_b), e_one, e_b)
        
        # 28位尾数: hidden(1) + M(23) + guard(4)
        m_a = torch.cat([hidden_a, m_a_raw, zeros, zeros, zeros, zeros], dim=-1)
        m_b = torch.cat([hidden_b, m_b_raw, zeros, zeros, zeros, zeros], dim=-1)
        
        # ===== Step 1: 比较绝对值 =====
        a_exp_gt, a_exp_eq = self.exp_cmp(e_a_eff, e_b_eff)
        m_a_with_hidden = torch.cat([hidden_a, m_a_raw], dim=-1)
        m_b_with_hidden = torch.cat([hidden_b, m_b_raw], dim=-1)
        a_mant_gt, a_mant_eq = self.mantissa_cmp(m_a_with_hidden, m_b_with_hidden)
        
        a_abs_eq_b = self.abs_eq_and(a_exp_eq, a_mant_eq)
        a_mant_ge = self.mant_ge_or(a_mant_gt, a_mant_eq)
        exp_eq_and_mant_ge = self.abs_ge_and(a_exp_eq, a_mant_ge)
        a_ge_b = self.abs_ge_or(a_exp_gt, exp_eq_and_mant_ge)
        
        # ===== Step 2: 指数差 (LSB first) =====
        e_a_lsb = e_a_eff.flip(-1)
        e_b_lsb = e_b_eff.flip(-1)
        diff_ab_lsb, _ = self.exp_sub_ab(e_a_lsb, e_b_lsb)
        diff_ba_lsb, _ = self.exp_sub_ba(e_b_lsb, e_a_lsb)
        diff_ab = diff_ab_lsb.flip(-1)
        diff_ba = diff_ba_lsb.flip(-1)
        
        # 向量化MUX选择指数差
        exp_diff = self.vec_mux(a_ge_b.expand_as(diff_ab), diff_ab, diff_ba)
        
        # ===== 检测大指数差 (exp_diff >= 28) =====
        high_bits = exp_diff[..., 0:3]  # bit7,6,5
        high_bits_any = self.vec_or_tree(high_bits)
        bit4_and_bit3 = self.vec_and(exp_diff[..., 3:4], exp_diff[..., 4:5])
        is_big_diff = self.vec_or(high_bits_any, bit4_and_bit3)
        
        # 取低5位作为移位量
        shift_amt = exp_diff[..., 3:8]  # 最多移31位
        
        # e_max (向量化MUX)
        e_max = self.vec_mux(a_ge_b.expand_as(e_a_eff), e_a_eff, e_b_eff)
        
        # ===== Step 3: 尾数对齐 (向量化) =====
        m_large = self.vec_mux(a_ge_b.expand_as(m_a), m_a, m_b)
        m_small_unshifted = self.vec_mux(a_ge_b.expand_as(m_a), m_b, m_a)
        
        m_small, shift_sticky = self.align_shifter(m_small_unshifted, shift_amt)
        
        # ===== Step 4: 符号处理 =====
        is_diff_sign = self.vec_xor(s_a, s_b)
        exact_cancel = self.vec_and(is_diff_sign, a_abs_eq_b)
        s_large = self.vec_mux(a_ge_b, s_a, s_b)
        
        # ===== Step 5: 尾数运算 (28位, LSB first) =====
        m_large_lsb = m_large.flip(-1)
        m_small_lsb = m_small.flip(-1)
        
        sum_result_lsb, sum_carry = self.mantissa_adder(m_large_lsb, m_small_lsb)
        diff_result_lsb, _ = self.mantissa_sub(m_large_lsb, m_small_lsb)
        
        sum_result = sum_result_lsb.flip(-1)
        diff_result = diff_result_lsb.flip(-1)
        
        # ===== 减法时的sticky补偿 =====
        need_sub_one = self.vec_and(is_diff_sign, shift_sticky)
        one_28bit = torch.cat([ones] + [zeros]*27, dim=-1)
        diff_minus_one_lsb, _ = self.sub_one(diff_result_lsb, one_28bit)
        diff_minus_one = diff_minus_one_lsb.flip(-1)
        
        diff_final = self.vec_mux(need_sub_one.expand_as(diff_result), diff_minus_one, diff_result)
        
        # 选择加减结果
        mantissa_result = self.vec_mux(is_diff_sign.expand_as(diff_final), diff_final, sum_result)
        result_carry = self.vec_mux(is_diff_sign, zeros, sum_carry)
        
        # ===== Step 6: 归一化 =====
        lzc = self.lzd(mantissa_result)
        lzc_8bit = torch.cat([zeros, zeros, zeros, lzc], dim=-1)
        
        # 检测下溢
        lzc_gt_emax, lzc_eq_emax = self.underflow_cmp(lzc_8bit, e_max)
        is_underflow = self.vec_or(lzc_gt_emax, lzc_eq_emax)
        
        # 归一化
        norm_mantissa = self.norm_shifter(mantissa_result, lzc)
        e_after_norm_lsb, _ = self.exp_adj_sub(e_max.flip(-1), lzc_8bit.flip(-1))
        e_after_norm = e_after_norm_lsb.flip(-1)
        
        # 溢出路径: E + 1
        one_8bit = torch.cat([zeros]*7 + [ones], dim=-1)
        e_inc_lsb, exp_inc_carry = self.post_round_exp_inc(e_max.flip(-1), one_8bit.flip(-1))
        e_plus_one = e_inc_lsb.flip(-1)
        
        # 选择指数 (向量化)
        zero_8bit = torch.cat([zeros]*8, dim=-1)
        e_normal = self.vec_mux(is_underflow.expand_as(e_after_norm), zero_8bit, e_after_norm)
        final_e_pre = self.vec_mux(result_carry.expand_as(e_plus_one), e_plus_one, e_normal)
        
        # ===== Step 7: 提取尾数并舍入 =====
        # 溢出情况: 取位0-22
        m_overflow = mantissa_result[..., 0:23]
        round_overflow = mantissa_result[..., 23:24]
        sticky_overflow_bits = mantissa_result[..., 24:28]
        sticky_overflow = self.vec_or_tree(sticky_overflow_bits)
        
        # 正常归一化情况: 取位1-23
        m_norm = norm_mantissa[..., 1:24]
        round_norm = norm_mantissa[..., 24:25]
        sticky_norm_bits = norm_mantissa[..., 25:28]
        sticky_norm = self.vec_or_tree(sticky_norm_bits)
        
        # 下溢情况
        m_subnorm = mantissa_result[..., 0:23]
        
        # 选择尾数 (向量化)
        m_pre = self.vec_mux(result_carry.expand_as(m_overflow), m_overflow, m_norm)
        round_pre = self.vec_mux(result_carry, round_overflow, round_norm)
        sticky_pre_raw = self.vec_mux(result_carry, sticky_overflow, sticky_norm)
        
        # 合并shift_sticky
        not_diff = self.vec_not(is_diff_sign)
        add_shift_sticky = self.vec_and(not_diff, shift_sticky)
        sticky_pre = self.vec_or(sticky_pre_raw, add_shift_sticky)
        
        # 下溢选择
        m_selected = self.vec_mux(is_underflow.expand_as(m_subnorm), m_subnorm, m_pre)
        
        # RNE舍入
        L = m_selected[..., 22:23]
        R = round_pre
        S = sticky_pre
        sticky_or_L = self.vec_or(S, L)
        do_round = self.vec_and(R, sticky_or_L)
        
        # 下溢时不舍入
        not_underflow = self.vec_not(is_underflow)
        do_round = self.vec_and(do_round, not_underflow)
        
        # 尾数+1 (LSB first)
        m_selected_lsb = m_selected.flip(-1)
        m_24bit_lsb = torch.cat([m_selected_lsb, zeros], dim=-1)
        round_inc = torch.cat([do_round] + [zeros]*23, dim=-1)
        m_rounded_lsb, round_carry = self.round_adder(m_24bit_lsb, round_inc)
        m_rounded = m_rounded_lsb[..., :23].flip(-1)
        
        # 舍入溢出处理 (向量化)
        not_round_c = self.vec_not(round_carry)
        m_final = self.vec_and(not_round_c.expand_as(m_rounded), m_rounded)
        
        # 指数调整
        exp_round_inc = torch.cat([zeros]*7 + [round_carry], dim=-1)
        e_rounded_lsb, _ = self.round_exp_inc(final_e_pre.flip(-1), exp_round_inc.flip(-1))
        e_rounded = e_rounded_lsb.flip(-1)
        
        computed_e = self.vec_mux(round_carry.expand_as(e_rounded), e_rounded, final_e_pre)
        
        # ===== Step 8: 符号 =====
        computed_s = s_large
        
        # ===== 完全抵消 =====
        zero_23bit = torch.cat([zeros]*23, dim=-1)
        cancel_s = self.vec_mux(exact_cancel, zeros, computed_s)
        cancel_e = self.vec_mux(exact_cancel.expand_as(computed_e), zero_8bit, computed_e)
        cancel_m = self.vec_mux(exact_cancel.expand_as(m_final), zero_23bit, m_final)
        
        # ===== Inf/NaN处理 =====
        computed_e_all_one = self.vec_and_tree(computed_e)
        
        one_8bit_val = torch.cat([ones]*8, dim=-1)
        final_s = self.vec_mux(computed_e_all_one, computed_s, cancel_s)
        final_e = self.vec_mux(computed_e_all_one.expand_as(cancel_e), one_8bit_val, cancel_e)
        final_m = self.vec_mux(computed_e_all_one.expand_as(cancel_m), zero_23bit, cancel_m)
        
        # ===== 大指数差处理: 直接返回较大的数 =====
        larger_input = self.vec_mux(a_ge_b.expand_as(A), A, B)
        
        # 组装正常结果
        normal_result = torch.cat([final_s, final_e, final_m], dim=-1)
        
        # 根据is_big_diff选择最终结果
        result = self.vec_mux(is_big_diff.expand_as(normal_result), larger_input, normal_result)
        
        return result
    
    def reset(self):
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_xor.reset()
        self.vec_not.reset()
        self.vec_mux.reset()
        self.vec_or_tree.reset()
        self.vec_and_tree.reset()
        self.exp_cmp.reset()
        self.mantissa_cmp.reset()
        self.exp_sub_ab.reset()
        self.exp_sub_ba.reset()
        self.abs_eq_and.reset()
        self.mant_ge_or.reset()
        self.abs_ge_and.reset()
        self.abs_ge_or.reset()
        self.align_shifter.reset()
        self.mantissa_adder.reset()
        self.mantissa_sub.reset()
        self.sub_one.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.exp_adj_sub.reset()
        self.underflow_cmp.reset()
        self.post_round_exp_inc.reset()
        self.round_exp_inc.reset()
        self.round_adder.reset()
