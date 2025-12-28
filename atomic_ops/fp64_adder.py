"""
FP64 空间编码加法器 - 100%纯SNN门电路实现 (向量化版本)
用于高精度计算 (支持Exp等超越函数的精确实现)

FP64 格式: [S | E10..E0 | M51..M0], bias=1023
内部精度: 57位尾数 (hidden + 52 mant + 4 guard)

向量化原则:
1. 使用 VecAND, VecOR, VecXOR, VecNOT, VecMUX 代替 ModuleList
2. 可并行操作一次处理所有位
3. 串行依赖（如进位链）仍保留循环
"""
import torch
import torch.nn as nn
from .vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecORTree, VecANDTree, VecAdder, VecSubtractor, VecComparator
)
from .fp64_components import (
    Comparator11Bit, Comparator53Bit, Subtractor11Bit,
    RippleCarryAdder57Bit, Subtractor57Bit,
    BarrelShifterRight57WithSticky,
    BarrelShifterLeft57, LeadingZeroDetector57
)
from .logic_gates import RippleCarryAdder


class SpikeFP64Adder(nn.Module):
    """FP64 空间编码加法器 - 100%纯SNN门电路 (向量化版本)
    
    FP64 格式: [S | E10..E0 | M51..M0], bias=1023
    内部使用 57 位尾数精度 (hidden + 52 mant + 4 guard)
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # ===== 向量化基础门电路 (复用) =====
        self.vec_and = VecAND(neuron_template=nt)
        self.vec_or = VecOR(neuron_template=nt)
        self.vec_xor = VecXOR(neuron_template=nt)
        self.vec_not = VecNOT(neuron_template=nt)
        self.vec_mux = VecMUX(neuron_template=nt)
        self.vec_or_tree = VecORTree(neuron_template=nt)
        self.vec_and_tree = VecANDTree(neuron_template=nt)
        
        # ===== 比较器 =====
        self.exp_cmp = Comparator11Bit(neuron_template=nt)
        self.mantissa_cmp = Comparator53Bit(neuron_template=nt)
        
        # ===== 指数差 =====
        self.exp_sub_ab = Subtractor11Bit(neuron_template=nt)
        self.exp_sub_ba = Subtractor11Bit(neuron_template=nt)
        
        # ===== 绝对值比较门 =====
        self.abs_eq_and = VecAND(neuron_template=nt)
        self.mant_ge_or = VecOR(neuron_template=nt)
        self.abs_ge_and = VecAND(neuron_template=nt)
        self.abs_ge_or = VecOR(neuron_template=nt)
        
        # ===== 对齐移位器 =====
        self.align_shifter = BarrelShifterRight57WithSticky(neuron_template=nt)
        
        # ===== 尾数运算 =====
        self.mantissa_adder = RippleCarryAdder57Bit(neuron_template=nt)
        self.mantissa_sub = Subtractor57Bit(neuron_template=nt)
        self.sub_one = Subtractor57Bit(neuron_template=nt)
        
        # ===== 归一化 =====
        self.lzd = LeadingZeroDetector57(neuron_template=nt)
        self.norm_shifter = BarrelShifterLeft57(neuron_template=nt)
        self.exp_adj_sub = Subtractor11Bit(neuron_template=nt)
        
        # ===== 溢出/下溢处理 =====
        self.underflow_cmp = Comparator11Bit(neuron_template=nt)
        self.post_round_exp_inc = RippleCarryAdder(bits=11, neuron_template=nt)
        self.round_exp_inc = RippleCarryAdder(bits=11, neuron_template=nt)
        
        # ===== 舍入 =====
        self.round_adder = RippleCarryAdder(bits=53, neuron_template=nt)
        
    def forward(self, A, B):
        """
        Args:
            A, B: [..., 64] FP64 脉冲 [S, E10..E0, M51..M0]
        Returns:
            [..., 64] FP64 脉冲
        """
        self.reset()
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取各部分
        s_a = A[..., 0:1]
        e_a = A[..., 1:12]   # [E10..E0] MSB first
        m_a_raw = A[..., 12:64]  # [M51..M0]
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:12]
        m_b_raw = B[..., 12:64]
        
        # ===== E=0 检测与隐藏位 =====
        e_a_nonzero = self.vec_or_tree(e_a)
        e_a_is_zero = self.vec_not(e_a_nonzero)
        hidden_a = e_a_nonzero
        
        e_b_nonzero = self.vec_or_tree(e_b)
        e_b_is_zero = self.vec_not(e_b_nonzero)
        hidden_b = e_b_nonzero
        
        # 有效指数 (subnormal时E=1)
        # 创建常量 E=1 (00000000001)
        e_one = torch.cat([zeros]*10 + [ones], dim=-1)
        e_a_eff = self.vec_mux(e_a_is_zero.expand_as(e_a), e_one, e_a)
        e_b_eff = self.vec_mux(e_b_is_zero.expand_as(e_b), e_one, e_b)
        
        # 57位尾数: hidden(1) + M(52) + guard(4)
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
        
        # 选择指数差 (向量化MUX一次处理所有位)
        exp_diff = self.vec_mux(a_ge_b.expand_as(diff_ab), diff_ab, diff_ba)
        
        # ===== 检测大指数差 (exp_diff >= 57) =====
        high_bits = exp_diff[..., 0:5]  # bit10-6
        high_bits_any = self.vec_or_tree(high_bits)
        bit543 = exp_diff[..., 5:8]  # bit5,4,3
        bit543_all = self.vec_and_tree(bit543)
        is_big_diff = self.vec_or(high_bits_any, bit543_all)
        
        # 取低6位作为移位量
        shift_amt = exp_diff[..., 5:11]  # 最多移63位
        
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
        
        # ===== Step 5: 尾数运算 (57位, LSB first) =====
        m_large_lsb = m_large.flip(-1)
        m_small_lsb = m_small.flip(-1)
        
        sum_result_lsb, sum_carry = self.mantissa_adder(m_large_lsb, m_small_lsb)
        diff_result_lsb, _ = self.mantissa_sub(m_large_lsb, m_small_lsb)
        
        sum_result = sum_result_lsb.flip(-1)
        diff_result = diff_result_lsb.flip(-1)
        
        # ===== 减法时的sticky补偿 =====
        need_sub_one = self.vec_and(is_diff_sign, shift_sticky)
        one_57bit = torch.cat([ones] + [zeros]*56, dim=-1)  # 1 in LSB-first
        diff_minus_one_lsb, _ = self.sub_one(diff_result_lsb, one_57bit)
        diff_minus_one = diff_minus_one_lsb.flip(-1)
        
        # 选择差结果 (向量化MUX)
        diff_final = self.vec_mux(need_sub_one.expand_as(diff_result), diff_minus_one, diff_result)
        
        # 选择加减结果
        mantissa_result = self.vec_mux(is_diff_sign.expand_as(diff_final), diff_final, sum_result)
        result_carry = self.vec_mux(is_diff_sign, zeros, sum_carry)
        
        # ===== Step 6: 归一化 =====
        lzc = self.lzd(mantissa_result)
        lzc_11bit = torch.cat([zeros, zeros, zeros, zeros, zeros, lzc], dim=-1)
        
        # 检测下溢
        lzc_gt_emax, lzc_eq_emax = self.underflow_cmp(lzc_11bit, e_max)
        is_underflow = self.vec_or(lzc_gt_emax, lzc_eq_emax)
        
        # 归一化
        norm_mantissa = self.norm_shifter(mantissa_result, lzc)
        e_after_norm_lsb, _ = self.exp_adj_sub(e_max.flip(-1), lzc_11bit.flip(-1))
        e_after_norm = e_after_norm_lsb.flip(-1)
        
        # 溢出路径: E + 1
        one_11bit = torch.cat([zeros]*10 + [ones], dim=-1)
        e_inc_lsb, exp_inc_carry = self.post_round_exp_inc(e_max.flip(-1), one_11bit.flip(-1))
        e_plus_one = e_inc_lsb.flip(-1)
        
        # 选择指数 (向量化)
        zero_11bit = torch.cat([zeros]*11, dim=-1)
        e_normal = self.vec_mux(is_underflow.expand_as(e_after_norm), zero_11bit, e_after_norm)
        final_e_pre = self.vec_mux(result_carry.expand_as(e_plus_one), e_plus_one, e_normal)
        
        # ===== Step 7: 提取尾数并舍入 =====
        # 溢出情况 (result_carry=1): 取位0-51
        m_overflow = mantissa_result[..., 0:52]
        round_overflow = mantissa_result[..., 52:53]
        sticky_overflow_bits = mantissa_result[..., 53:57]
        sticky_overflow = self.vec_or_tree(sticky_overflow_bits)
        
        # 正常归一化情况: 取位1-52
        m_norm = norm_mantissa[..., 1:53]
        round_norm = norm_mantissa[..., 53:54]
        sticky_norm_bits = norm_mantissa[..., 54:57]
        sticky_norm = self.vec_or_tree(sticky_norm_bits)
        
        # 下溢情况
        m_subnorm = mantissa_result[..., 0:52]
        
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
        L = m_selected[..., 51:52]
        R = round_pre
        S = sticky_pre
        sticky_or_L = self.vec_or(S, L)
        do_round = self.vec_and(R, sticky_or_L)
        
        # 下溢时不舍入
        not_underflow = self.vec_not(is_underflow)
        do_round = self.vec_and(do_round, not_underflow)
        
        # 尾数+1 (LSB first)
        m_selected_lsb = m_selected.flip(-1)
        m_53bit_lsb = torch.cat([m_selected_lsb, zeros], dim=-1)
        round_inc = torch.cat([do_round] + [zeros]*52, dim=-1)
        m_rounded_lsb, round_carry = self.round_adder(m_53bit_lsb, round_inc)
        m_rounded = m_rounded_lsb[..., :52].flip(-1)
        
        # 舍入溢出处理 (向量化)
        not_round_c = self.vec_not(round_carry)
        m_final = self.vec_and(not_round_c.expand_as(m_rounded), m_rounded)
        
        # 指数调整
        exp_round_inc = torch.cat([zeros]*10 + [round_carry], dim=-1)
        e_rounded_lsb, _ = self.round_exp_inc(final_e_pre.flip(-1), exp_round_inc.flip(-1))
        e_rounded = e_rounded_lsb.flip(-1)
        
        computed_e = self.vec_mux(round_carry.expand_as(e_rounded), e_rounded, final_e_pre)
        
        # ===== Step 8: 符号 =====
        computed_s = s_large
        
        # ===== 完全抵消 =====
        zero_52bit = torch.cat([zeros]*52, dim=-1)
        cancel_s = self.vec_mux(exact_cancel, zeros, computed_s)
        cancel_e = self.vec_mux(exact_cancel.expand_as(computed_e), zero_11bit, computed_e)
        cancel_m = self.vec_mux(exact_cancel.expand_as(m_final), zero_52bit, m_final)
        
        # ===== Inf/NaN处理 =====
        computed_e_all_one = self.vec_and_tree(computed_e)
        
        one_11bit_val = torch.cat([ones]*11, dim=-1)
        final_s = self.vec_mux(computed_e_all_one, computed_s, cancel_s)
        final_e = self.vec_mux(computed_e_all_one.expand_as(cancel_e), one_11bit_val, cancel_e)
        final_m = self.vec_mux(computed_e_all_one.expand_as(cancel_m), zero_52bit, cancel_m)
        
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
