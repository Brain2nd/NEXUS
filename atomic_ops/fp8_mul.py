import torch
import torch.nn as nn
# 使用向量化基础门以保持与 FP16/FP32 组件的一致性
from .vec_logic_gates import (
    VecAND as ANDGate,
    VecOR as ORGate,
    VecXOR as XORGate,
    VecNOT as NOTGate,
    VecMUX as MUXGate,
    VecHalfAdder as HalfAdder,
    VecFullAdder as FullAdder,
    VecAdder as RippleCarryAdder
)
# 保留专用组件（这些内部也需要向量化，但接口兼容）
from .logic_gates import (OR3Gate, ArrayMultiplier4x4_Strict,
                          NewNormalizationUnit, Denormalizer)

class SpikeFP8Multiplier(nn.Module):
    """纯脉冲驱动的FP8乘法器
    输入输出都是8位脉冲序列 [..., 8]
    全程使用脉冲神经元门电路，无实数域计算
    
    支持：
    - 正常数乘法
    - 指数下溢处理（结果变为subnormal或零）
    - 指数上溢处理（结果变为NaN）
    - 完整的subnormal输出处理
    
    100%纯SNN：所有操作仅使用IF神经元门电路
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template  # 简写
        
        # 符号XOR
        self.sign_xor = XORGate(neuron_template=nt)
        
        # 指数加法器 (4-bit + 4-bit)
        self.exp_adder = RippleCarryAdder(bits=5, neuron_template=nt)  # 5位防溢出
        
        # 减bias用的加法器 (加-7的补码 = 加11001)
        self.bias_sub_adder = RippleCarryAdder(bits=5, neuron_template=nt)
        
        # 尾数乘法器
        self.mantissa_mul = ArrayMultiplier4x4_Strict(neuron_template=nt)
        
        # 规格化单元 (替代原来的简单溢出检测)
        self.norm_unit = NewNormalizationUnit(neuron_template=nt)
        
        # 指数修正加法器 (E_raw + Shift)
        self.exp_norm_adder = RippleCarryAdder(bits=5, neuron_template=nt)
        
        # Denormalization components
        self.denormalizer = Denormalizer(neuron_template=nt)
        self.e_inv_gate = NOTGate(neuron_template=nt)
        self.shift_calc_adder = RippleCarryAdder(bits=5, neuron_template=nt)
        self.mux_p_final = MUXGate(neuron_template=nt)
        self.mux_e_final = MUXGate(neuron_template=nt)
        self.denorm_or = ORGate(neuron_template=nt)
        self.denorm_not = NOTGate(neuron_template=nt)
        self.denorm_check_zero = OR3Gate(neuron_template=nt) # Check if E=0 (bits 0,1,2)
        self.denorm_check_zero2 = ORGate(neuron_template=nt) # Check bits 3,4
        self.denorm_is_zero = NOTGate(neuron_template=nt)
        
        # 尾数溢出后的MUX
        self.mux_m0_final = MUXGate(neuron_template=nt)
        self.mux_m1_final = MUXGate(neuron_template=nt)
        self.mux_m2_final = MUXGate(neuron_template=nt)
        
        # Sticky OR
        self.sticky_or_overflow = OR3Gate(neuron_template=nt)
        self.sticky_or_no_overflow = ORGate(neuron_template=nt)
        self.sticky_or_extra = ORGate(neuron_template=nt)  # 合并来自 NormUnit 的 sticky_extra
        self.mux_sticky = MUXGate(neuron_template=nt)
        self.sub_sticky_or = ORGate(neuron_template=nt)   # round_bit OR sticky
        self.sub_sticky_or2 = ORGate(neuron_template=nt)  # m2_raw OR sub_sticky_base
        
        # RNE逻辑: do_round = round_bit AND (sticky OR m2)
        self.rne_or = ORGate(neuron_template=nt)
        self.rne_and = ANDGate(neuron_template=nt)
        
        # Subnormal RNE 逻辑 (独立的门)
        self.sub_rne_or = ORGate(neuron_template=nt)
        self.sub_rne_and = ANDGate(neuron_template=nt)
        self.sub_round_mux = MUXGate(neuron_template=nt)  # 选择 sub_round: overflow ? m2_normal : sticky_extra
        
        # E=-1 RNE 逻辑
        self.em1_rne_or = ORGate(neuron_template=nt)
        self.em1_rne_and = ANDGate(neuron_template=nt)
        self.em1_sticky_or = ORGate(neuron_template=nt)  # m2 OR round_bit
        self.em1_sticky_or2 = ORGate(neuron_template=nt)  # (m2 OR round_bit) OR sticky
        
        # E=-2 RNE 逻辑
        self.em2_rne_or = ORGate(neuron_template=nt)
        self.em2_rne_and = ANDGate(neuron_template=nt)
        self.em2_sticky_or = ORGate(neuron_template=nt)  # m1 OR m2
        self.em2_sticky_or2 = ORGate(neuron_template=nt)  # (m1 OR m2) OR round_bit
        
        # E=-3 RNE 逻辑 (sticky 需要包含 round_bit)
        self.em3_sticky_or2 = ORGate(neuron_template=nt)  # (m0 OR m1 OR m2) OR round_bit
        
        # 舍入加法器 (3-bit mantissa + round_bit)
        self.round_ha0 = HalfAdder(neuron_template=nt)
        self.round_ha1 = HalfAdder(neuron_template=nt)
        self.round_ha2 = HalfAdder(neuron_template=nt)
        
        # 指数+1加法器 (用于m_carry进位)
        self.exp_inc_adder = RippleCarryAdder(bits=5, neuron_template=nt)
        
        # ===== 输入零检测 (纯SNN) =====
        # 真正的零: E=0 AND M=0
        # E=0检测：~E[0] AND ~E[1] AND ~E[2] AND ~E[3]
        self.a_not_e0 = NOTGate(neuron_template=nt)
        self.a_not_e1 = NOTGate(neuron_template=nt)
        self.a_not_e2 = NOTGate(neuron_template=nt)
        self.a_not_e3 = NOTGate(neuron_template=nt)
        self.a_and_e01 = ANDGate(neuron_template=nt)
        self.a_and_e23 = ANDGate(neuron_template=nt)
        self.a_e_is_zero = ANDGate(neuron_template=nt)
        # M=0检测：~M[0] AND ~M[1] AND ~M[2]
        self.a_not_m0 = NOTGate(neuron_template=nt)
        self.a_not_m1 = NOTGate(neuron_template=nt)
        self.a_not_m2 = NOTGate(neuron_template=nt)
        self.a_and_m01 = ANDGate(neuron_template=nt)
        self.a_m_is_zero = ANDGate(neuron_template=nt)
        # 真正的零: E=0 AND M=0
        self.a_is_true_zero = ANDGate(neuron_template=nt)
        
        self.b_not_e0 = NOTGate(neuron_template=nt)
        self.b_not_e1 = NOTGate(neuron_template=nt)
        self.b_not_e2 = NOTGate(neuron_template=nt)
        self.b_not_e3 = NOTGate(neuron_template=nt)
        self.b_and_e01 = ANDGate(neuron_template=nt)
        self.b_and_e23 = ANDGate(neuron_template=nt)
        self.b_e_is_zero = ANDGate(neuron_template=nt)
        # M=0检测
        self.b_not_m0 = NOTGate(neuron_template=nt)
        self.b_not_m1 = NOTGate(neuron_template=nt)
        self.b_not_m2 = NOTGate(neuron_template=nt)
        self.b_and_m01 = ANDGate(neuron_template=nt)
        self.b_m_is_zero = ANDGate(neuron_template=nt)
        # 真正的零
        self.b_is_true_zero = ANDGate(neuron_template=nt)
        
        self.input_zero_or = ORGate(neuron_template=nt)  # A=0 OR B=0
        
        # ===== 输入subnormal检测 (纯SNN) =====
        # subnormal = E=0 AND M≠0
        self.a_not_m_is_zero = NOTGate(neuron_template=nt)
        self.a_is_subnormal = ANDGate(neuron_template=nt)  # E=0 AND NOT(M=0)
        self.b_not_m_is_zero = NOTGate(neuron_template=nt)
        self.b_is_subnormal = ANDGate(neuron_template=nt)
        
        # 尾数前导位选择 (subnormal用0, normal用1)
        self.mux_a_leading = MUXGate(neuron_template=nt)  # NOT(a_is_subnormal) ? 1 : 0
        self.mux_b_leading = MUXGate(neuron_template=nt)
        
        # 指数修正：subnormal的有效指数是1而不是0
        # E_eff = subnormal ? 1 : E
        self.mux_a_e0 = MUXGate(neuron_template=nt)
        self.mux_a_e1 = MUXGate(neuron_template=nt)
        self.mux_a_e2 = MUXGate(neuron_template=nt)
        self.mux_a_e3 = MUXGate(neuron_template=nt)
        self.mux_b_e0 = MUXGate(neuron_template=nt)
        self.mux_b_e1 = MUXGate(neuron_template=nt)
        self.mux_b_e2 = MUXGate(neuron_template=nt)
        self.mux_b_e3 = MUXGate(neuron_template=nt)
        
        # ===== 输出E=0检测 (e_final_5) =====
        self.out_not_e0 = NOTGate(neuron_template=nt)
        self.out_not_e1 = NOTGate(neuron_template=nt)
        self.out_not_e2 = NOTGate(neuron_template=nt)
        self.out_not_e3 = NOTGate(neuron_template=nt)
        self.out_and_01 = ANDGate(neuron_template=nt)
        self.out_and_23 = ANDGate(neuron_template=nt)
        self.out_e_is_zero = ANDGate(neuron_template=nt)
        
        # ===== e_normalized == 0 检测 =====
        self.norm_not_e0 = NOTGate(neuron_template=nt)
        self.norm_not_e1 = NOTGate(neuron_template=nt)
        self.norm_not_e2 = NOTGate(neuron_template=nt)
        self.norm_not_e3 = NOTGate(neuron_template=nt)
        self.norm_and_01 = ANDGate(neuron_template=nt)
        self.norm_and_23 = ANDGate(neuron_template=nt)
        self.norm_e_is_zero = ANDGate(neuron_template=nt)
        self.e_both_zero_and = ANDGate(neuron_template=nt)  # e_final=0 AND e_norm=0
        
        # 下溢检测：符号位=1表示负数
        self.underflow_not = NOTGate(neuron_template=nt)
        
        # is_subnormal = e_both_zero AND (NOT is_negative_norm)
        self.is_subnormal_and = ANDGate(neuron_template=nt)
        
        # subnormal或underflow的OR
        self.sub_or_under = ORGate(neuron_template=nt)
        
        # ===== Subnormal尾数舍入 (纯SNN HalfAdder) =====
        self.sub_round_ha0 = HalfAdder(neuron_template=nt)  # m2 + round_bit
        self.sub_round_ha1 = HalfAdder(neuron_template=nt)  # m1 + carry
        self.sub_round_ha2 = HalfAdder(neuron_template=nt)  # m0 + carry
        
        # 当E=0舍入产生进位时，饱和到M=7
        self.sub_saturate_m0 = MUXGate(neuron_template=nt)
        self.sub_saturate_m1 = MUXGate(neuron_template=nt)
        self.sub_saturate_m2 = MUXGate(neuron_template=nt)
        
        # Subnormal 溢出时 E=1 的 MUX
        self.sub_overflow_and = ANDGate(neuron_template=nt)  # is_subnormal AND sub_c2
        self.mux_sub_overflow_e0 = MUXGate(neuron_template=nt)  # 溢出时 e0=1
        
        # ===== E=-1检测 (纯SNN) =====
        # E=-1 = 11111 (5位补码)
        self.minus1_and_01 = ANDGate(neuron_template=nt)
        self.minus1_and_23 = ANDGate(neuron_template=nt)
        self.minus1_and_0123 = ANDGate(neuron_template=nt)
        self.minus1_and_all = ANDGate(neuron_template=nt)
        
        # ===== E=-1尾数舍入 (纯SNN HalfAdder) =====
        self.e_m1_round_ha0 = HalfAdder(neuron_template=nt)
        self.e_m1_round_ha1 = HalfAdder(neuron_template=nt)
        self.e_m1_round_ha2 = HalfAdder(neuron_template=nt)  # 用于 m0 进位
        
        # ===== E=-2检测 (纯SNN) =====
        # E=-2 = 11110 (5位补码)
        # E[4]=1, E[3]=1, E[2]=1, E[1]=1, E[0]=0
        self.minus2_not_e0 = NOTGate(neuron_template=nt)
        self.minus2_and_01 = ANDGate(neuron_template=nt)
        self.minus2_and_23 = ANDGate(neuron_template=nt)
        self.minus2_and_0123 = ANDGate(neuron_template=nt)
        self.minus2_and_all = ANDGate(neuron_template=nt)
        
        # ===== E=-2尾数舍入 (纯SNN HalfAdder) =====
        self.e_m2_round_ha0 = HalfAdder(neuron_template=nt)
        self.e_m2_round_ha1 = HalfAdder(neuron_template=nt)
        self.e_m2_round_ha2 = HalfAdder(neuron_template=nt)  # 用于 m0 进位
        
        # ===== E=-3检测 (纯SNN) =====
        # E=-3 = 11101 (5位补码)
        # E[4]=1, E[3]=1, E[2]=1, E[1]=0, E[0]=1
        self.minus3_not_e1 = NOTGate(neuron_template=nt)
        self.minus3_and_01 = ANDGate(neuron_template=nt)
        self.minus3_and_23 = ANDGate(neuron_template=nt)
        self.minus3_and_0123 = ANDGate(neuron_template=nt)
        self.minus3_and_all = ANDGate(neuron_template=nt)
        
        # E=-3 RNE 逻辑
        self.em3_sticky_or = OR3Gate(neuron_template=nt)
        self.em3_rne_or = ORGate(neuron_template=nt)
        self.em3_rne_and = ANDGate(neuron_template=nt)
        
        # E=-3 尾数舍入
        self.e_m3_round_ha0 = HalfAdder(neuron_template=nt)
        
        # ===== E=-2 vs E=-1 vs 更深选择 =====
        self.mux_m2_m0 = MUXGate(neuron_template=nt)
        self.mux_m2_m1 = MUXGate(neuron_template=nt)
        self.mux_m2_m2 = MUXGate(neuron_template=nt)
        
        # ===== E=-1 vs (E=-2或更深)选择 =====
        self.mux_under_m0 = MUXGate(neuron_template=nt)
        self.mux_under_m1 = MUXGate(neuron_template=nt)
        self.mux_under_m2 = MUXGate(neuron_template=nt)
        
        # ===== subnormal vs underflow选择 (纯SNN MUX) =====
        self.mux_sub_m0 = MUXGate(neuron_template=nt)
        self.mux_sub_m1 = MUXGate(neuron_template=nt)
        self.mux_sub_m2 = MUXGate(neuron_template=nt)
        
        # ===== 最终结果选择MUX =====
        self.mux_final_e0 = MUXGate(neuron_template=nt)
        self.mux_final_e1 = MUXGate(neuron_template=nt)
        self.mux_final_e2 = MUXGate(neuron_template=nt)
        self.mux_final_e3 = MUXGate(neuron_template=nt)
        self.mux_final_m0 = MUXGate(neuron_template=nt)
        self.mux_final_m1 = MUXGate(neuron_template=nt)
        self.mux_final_m2 = MUXGate(neuron_template=nt)

        # ===== 上溢检测 (sum_e >= 22，即 raw_e >= 15) =====
        # sum_e >= 22 = 10110:
        # (sum_e[4]=1 且 sum_e[3]=1) 或
        # (sum_e[4]=1 且 sum_e[3]=0 且 sum_e[2]=1 且 sum_e[1]=1)
        self.overflow_43 = ANDGate(neuron_template=nt)  # sum_e[4] AND sum_e[3]
        self.overflow_not_3 = NOTGate(neuron_template=nt)  # NOT(sum_e[3])
        self.overflow_21 = ANDGate(neuron_template=nt)  # sum_e[2] AND sum_e[1]
        self.overflow_210 = ANDGate(neuron_template=nt)  # (sum_e[2] AND sum_e[1]) AND sum_e[0]
        self.overflow_4n321 = ANDGate(neuron_template=nt)  # sum_e[4] AND NOT(sum_e[3])
        self.overflow_4n321_21 = ANDGate(neuron_template=nt)  # above AND (sum_e[2] AND sum_e[1])
        self.overflow_final = ORGate(neuron_template=nt)  # 43 OR 4n321_21
        self.overflow_and_msb = ANDGate(neuron_template=nt)  # sum_e>=22 AND e_final[4]

        # 上溢时的 NaN 输出选择
        self.mux_overflow_e0 = MUXGate(neuron_template=nt)
        self.mux_overflow_e1 = MUXGate(neuron_template=nt)
        self.mux_overflow_e2 = MUXGate(neuron_template=nt)
        self.mux_overflow_e3 = MUXGate(neuron_template=nt)
        self.mux_overflow_m0 = MUXGate(neuron_template=nt)
        self.mux_overflow_m1 = MUXGate(neuron_template=nt)
        self.mux_overflow_m2 = MUXGate(neuron_template=nt)

        # ===== 输入为零时清零 (纯SNN AND with NOT) =====
        self.zero_not = NOTGate(neuron_template=nt)  # NOT(input_has_zero)
        self.zero_and_e0 = ANDGate(neuron_template=nt)
        self.zero_and_e1 = ANDGate(neuron_template=nt)
        self.zero_and_e2 = ANDGate(neuron_template=nt)
        self.zero_and_e3 = ANDGate(neuron_template=nt)
        self.zero_and_m0 = ANDGate(neuron_template=nt)
        self.zero_and_m1 = ANDGate(neuron_template=nt)
        self.zero_and_m2 = ANDGate(neuron_template=nt)

        # ===== sticky_extra 用途修正 =====
        # Pre-shift 总是把 P[0] 移到 sticky_extra，但 P[0] 的正确角色取决于 shift_amt：
        # - shift = 0-2: P[0] 是 sticky
        # - shift = 3: P[0] 是 round_bit
        # - shift >= 4: P[0] 是 m2

        # shift >= 4 修正 (m2)
        self.shift_ge4_and = ANDGate(neuron_template=nt)  # shift_amt[2] AND sticky_extra
        self.m2_raw_fix_or = ORGate(neuron_template=nt)  # p_norm[3] OR (shift_ge4 AND sticky_extra)
        self.shift_lt4_not = NOTGate(neuron_template=nt)  # NOT(shift_amt[2])

        # shift = 3 修正 (round_bit)
        # shift = 3: shift_amt = 011, 即 NOT(shift_amt[2]) AND shift_amt[1] AND shift_amt[0]
        self.shift_eq3_and_10 = ANDGate(neuron_template=nt)  # shift_amt[1] AND shift_amt[0]
        self.shift_eq3_and = ANDGate(neuron_template=nt)  # (NOT shift_amt[2]) AND (shift_amt[1] AND shift_amt[0])
        self.shift_eq3_sticky = ANDGate(neuron_template=nt)  # shift_eq3 AND sticky_extra
        self.round_bit_fix_or = ORGate(neuron_template=nt)  # round_bit OR (shift_eq3 AND sticky_extra)

        # sticky 修正：只有 shift <= 2 时才包含 sticky_extra
        # shift <= 2: NOT(shift_amt[2]) AND NOT(shift_amt[1] AND shift_amt[0])
        # 简化：NOT(shift >= 3) = NOT(shift_amt[2] OR (shift_amt[1] AND shift_amt[0]))
        self.shift_ge3_or = ORGate(neuron_template=nt)  # shift_amt[2] OR (shift_amt[1] AND shift_amt[0])
        self.shift_lt3_not = NOTGate(neuron_template=nt)  # NOT(shift >= 3)
        self.sticky_extra_masked = ANDGate(neuron_template=nt)  # sticky_extra AND (shift < 3)

        self.sub_sticky_base = ORGate(neuron_template=nt)  # round_bit OR (p1 OR p0)
        self.sub_sticky_final = ORGate(neuron_template=nt)  # base OR masked_sticky_extra
        
    def forward(self, A, B):
        # A, B: [..., 8] = S(1) E(4) M(3)
        # 支持广播：A和B可以有不同的前导维度

        # 高层组件统一reset所有内部门电路
        self.reset_all()

        A, B = torch.broadcast_tensors(A, B)
        
        # ===== 1. 符号 =====
        s_a = A[..., 0:1]
        s_b = B[..., 0:1]
        s_out = self.sign_xor(s_a, s_b)
        
        # 广播后的形状用于创建辅助张量
        zeros = torch.zeros_like(s_out)
        ones = torch.ones_like(s_out)
        
        # ===== 检测输入是否为真正的零 (纯SNN) =====
        # 真正的零: E=0 AND M=0 (不包括subnormal)
        e_a_be = A[..., 1:5]  # 大端
        e_b_be = B[..., 1:5]
        m_a_be = A[..., 5:8]
        m_b_be = B[..., 5:8]
        
        # A的E=0检测
        a_not_e0 = self.a_not_e0(e_a_be[..., 0:1])
        a_not_e1 = self.a_not_e1(e_a_be[..., 1:2])
        a_not_e2 = self.a_not_e2(e_a_be[..., 2:3])
        a_not_e3 = self.a_not_e3(e_a_be[..., 3:4])
        a_and_e01 = self.a_and_e01(a_not_e0, a_not_e1)
        a_and_e23 = self.a_and_e23(a_not_e2, a_not_e3)
        a_e_is_zero = self.a_e_is_zero(a_and_e01, a_and_e23)
        
        # A的M=0检测
        a_not_m0 = self.a_not_m0(m_a_be[..., 0:1])
        a_not_m1 = self.a_not_m1(m_a_be[..., 1:2])
        a_not_m2 = self.a_not_m2(m_a_be[..., 2:3])
        a_and_m01 = self.a_and_m01(a_not_m0, a_not_m1)
        a_m_is_zero = self.a_m_is_zero(a_and_m01, a_not_m2)
        
        # A是真正的零: E=0 AND M=0
        a_is_true_zero = self.a_is_true_zero(a_e_is_zero, a_m_is_zero)
        
        # B的E=0检测
        b_not_e0 = self.b_not_e0(e_b_be[..., 0:1])
        b_not_e1 = self.b_not_e1(e_b_be[..., 1:2])
        b_not_e2 = self.b_not_e2(e_b_be[..., 2:3])
        b_not_e3 = self.b_not_e3(e_b_be[..., 3:4])
        b_and_e01 = self.b_and_e01(b_not_e0, b_not_e1)
        b_and_e23 = self.b_and_e23(b_not_e2, b_not_e3)
        b_e_is_zero = self.b_e_is_zero(b_and_e01, b_and_e23)
        
        # B的M=0检测
        b_not_m0 = self.b_not_m0(m_b_be[..., 0:1])
        b_not_m1 = self.b_not_m1(m_b_be[..., 1:2])
        b_not_m2 = self.b_not_m2(m_b_be[..., 2:3])
        b_and_m01 = self.b_and_m01(b_not_m0, b_not_m1)
        b_m_is_zero = self.b_m_is_zero(b_and_m01, b_not_m2)
        
        # B是真正的零
        b_is_true_zero = self.b_is_true_zero(b_e_is_zero, b_m_is_zero)
        
        # 任一输入为零
        input_has_zero = self.input_zero_or(a_is_true_zero, b_is_true_zero)
        
        # ===== 输入subnormal检测 =====
        # subnormal = E=0 AND M≠0
        a_not_m_zero = self.a_not_m_is_zero(a_m_is_zero)
        a_is_subnormal = self.a_is_subnormal(a_e_is_zero, a_not_m_zero)
        
        b_not_m_zero = self.b_not_m_is_zero(b_m_is_zero)
        b_is_subnormal = self.b_is_subnormal(b_e_is_zero, b_not_m_zero)
        
        # ===== 2. 指数处理 =====
        # subnormal的有效指数是1（因为value = M/8 × 2^(1-bias)）
        # 对于指数计算：使用1而不是0
        e_a_le = e_a_be.flip(-1)  # 转小端
        e_b_le = e_b_be.flip(-1)
        
        zeros_a = torch.zeros_like(A[..., 0:1])
        zeros_b = torch.zeros_like(B[..., 0:1])
        
        # 如果是subnormal，E=1（小端: 1000）
        e_a_0 = self.mux_a_e0(a_is_subnormal, ones, e_a_le[..., 0:1])
        e_a_1 = self.mux_a_e1(a_is_subnormal, zeros, e_a_le[..., 1:2])
        e_a_2 = self.mux_a_e2(a_is_subnormal, zeros, e_a_le[..., 2:3])
        e_a_3 = self.mux_a_e3(a_is_subnormal, zeros, e_a_le[..., 3:4])
        e_a_corrected = torch.cat([e_a_0, e_a_1, e_a_2, e_a_3], dim=-1)
        
        e_b_0 = self.mux_b_e0(b_is_subnormal, ones, e_b_le[..., 0:1])
        e_b_1 = self.mux_b_e1(b_is_subnormal, zeros, e_b_le[..., 1:2])
        e_b_2 = self.mux_b_e2(b_is_subnormal, zeros, e_b_le[..., 2:3])
        e_b_3 = self.mux_b_e3(b_is_subnormal, zeros, e_b_le[..., 3:4])
        e_b_corrected = torch.cat([e_b_0, e_b_1, e_b_2, e_b_3], dim=-1)
        
        e_a_5 = torch.cat([e_a_corrected, zeros], dim=-1)  # 扩展到5位
        e_b_5 = torch.cat([e_b_corrected, zeros], dim=-1)
        
        sum_e, _ = self.exp_adder(e_a_5, e_b_5)
        
        # 减bias(7): 加-7的5位补码 = 11001 (小端: 10011)
        zeros_sum = zeros
        ones_sum = ones
        neg_bias = torch.cat([ones_sum, zeros_sum, zeros_sum, ones_sum, ones_sum], dim=-1)
        raw_e, _ = self.bias_sub_adder(sum_e, neg_bias)
        
        # ===== 3. 尾数乘法 =====
        m_a_frac = A[..., 5:8]
        m_b_frac = B[..., 5:8]
        
        # 尾数前导位：normal用1，subnormal用0
        # MUX(is_subnormal, 0, 1): 当subnormal时选0，否则选1
        leading_a = self.mux_a_leading(a_is_subnormal, zeros, ones)
        leading_b = self.mux_b_leading(b_is_subnormal, zeros, ones)
        
        m_a = torch.cat([leading_a, m_a_frac], dim=-1).flip(-1)  # 小端
        m_b = torch.cat([leading_b, m_b_frac], dim=-1).flip(-1)
        
        p_le = self.mantissa_mul(m_a, m_b)  # 8位乘积
        p_be = p_le.flip(-1)  # 大端
        
        # ===== 4. 规格化与舍入 =====
        # 使用规格化单元处理所有情况 (Normal/Subnormal乘积)
        # P_norm: [P0...P7] aligned such that P6 is implicit 1 (if possible)
        # exp_adj: 5-bit 2's complement adjustment
        # sticky_extra: 被 Pre-shift 丢弃的 P[0]
        # overflow: P[7]，表示乘积是否溢出到 2.xxx
        # shift_amt: 左移量，用于 Subnormal 路径修正
        p_norm, exp_adj, sticky_extra, overflow, shift_amt = self.norm_unit(p_le)
        
        # 提取尾数位 (p_norm是Little Endian)
        # 权重: p7(2^1), p6(2^0), p5(2^-1), p4(2^-2), p3(2^-3), p2(2^-4), p1(2^-5), p0(2^-6)
        # 规格化后: 1.M0M1M2... 即 p6=1, p5=M0, p4=M1, p3=M2
        m0_raw = p_norm[..., 5:6]
        m1_raw = p_norm[..., 4:5]
        m2_raw_uncorrected = p_norm[..., 3:4]
        round_bit_uncorrected = p_norm[..., 2:3]
        
        # ===== sticky_extra 用途修正 =====
        # Pre-shift 把 P[0] 移到 sticky_extra，其正确角色取决于 shift_amt：
        # - shift = 0-2: P[0] 是 sticky
        # - shift = 3: P[0] 是 round_bit
        # - shift >= 4: P[0] 是 m2
        
        shift_ge4 = shift_amt[..., 2:3]
        shift_10 = self.shift_eq3_and_10(shift_amt[..., 1:2], shift_amt[..., 0:1])
        shift_lt4 = self.shift_lt4_not(shift_ge4)
        
        # shift = 3 修正 (round_bit)
        shift_eq3 = self.shift_eq3_and(shift_lt4, shift_10)
        shift_eq3_correction = self.shift_eq3_sticky(shift_eq3, sticky_extra)
        round_bit = self.round_bit_fix_or(round_bit_uncorrected, shift_eq3_correction)
        
        # shift >= 4 修正 (m2)
        shift_ge4_correction = self.shift_ge4_and(shift_ge4, sticky_extra)
        m2_raw = self.m2_raw_fix_or(m2_raw_uncorrected, shift_ge4_correction)
        
        # Sticky bit: 只有 shift <= 2 时才包含 sticky_extra
        # shift >= 3: shift_amt[2] OR (shift_amt[1] AND shift_amt[0])
        shift_ge3 = self.shift_ge3_or(shift_ge4, shift_10)
        shift_lt3 = self.shift_lt3_not(shift_ge3)
        sticky_p1_p0 = self.sticky_or_overflow(p_norm[..., 1:2], p_norm[..., 0:1], zeros)
        sticky_extra_for_sticky = self.sticky_extra_masked(sticky_extra, shift_lt3)
        sticky = self.sticky_or_extra(sticky_p1_p0, sticky_extra_for_sticky)
        
        # RNE舍入
        sticky_or_m2 = self.rne_or(sticky, m2_raw)
        do_round = self.rne_and(round_bit, sticky_or_m2)
        
        m2_round, c0 = self.round_ha0(m2_raw, do_round)
        m1_round, c1 = self.round_ha1(m1_raw, c0)
        m0_round, m_carry = self.round_ha2(m0_raw, c1)
        
        # 舍入溢出处理 (P_rounded)
        m0_normal = self.mux_m0_final(m_carry, zeros, m0_round)
        m1_normal = self.mux_m1_final(m_carry, zeros, m1_round)
        m2_normal = self.mux_m2_final(m_carry, zeros, m2_round)
        
        # 构造 P_rounded (Little Endian)
        # High bits (p7, p6, p5, p4, p3)
        # p7: m_carry
        # p6: NOT(m_carry) (因为 Normalized implies p6=1. 如果溢出 p6=0)
        # p5: m0_normal
        # p4: m1_normal
        # p3: m2_normal
        # p2, p1, p0: 0
        
        not_carry = self.denorm_not(m_carry) # Reuse NOT gate or define new one
        # I should define a new one or reuse self.a_not_e0? Let's use `self.denorm_not` which I added.
        
        p_rounded_7 = m_carry
        p_rounded_6 = not_carry
        p_rounded_5 = m0_normal
        p_rounded_4 = m1_normal
        p_rounded_3 = m2_normal
        p_rounded_2 = zeros
        p_rounded_1 = zeros
        p_rounded_0 = zeros
        
        p_rounded = torch.cat([p_rounded_0, p_rounded_1, p_rounded_2, p_rounded_3, 
                               p_rounded_4, p_rounded_5, p_rounded_6, p_rounded_7], dim=-1)
        
        # ===== 5. 指数调整 =====
        # E_norm = raw_e + exp_adj (Both are 5-bit Little Endian)
        e_normalized, exp_carry = self.exp_norm_adder(raw_e, exp_adj)
        
        # 处理舍入进位 (m_carry)
        inc_carry = torch.cat([m_carry, zeros, zeros, zeros, zeros], dim=-1)
        e_final_5, _ = self.exp_inc_adder(e_normalized, inc_carry)
        
        # ===== 6. Subnormal/下溢处理 (Denormalization) =====
        # 如果 E_final <= 0 (MSB=1 OR E=0)，则需要右移
        # Shift = 1 - E_final
        # 1 - E = 1 + (~E + 1) = ~E + 2.
        
        # Calc ~E
        e_inv = self.e_inv_gate(e_final_5)
        
        # Calc ~E + 2. (2 is 00010)
        two = torch.cat([zeros, ones, zeros, zeros, zeros], dim=-1)
        shift_denorm_5, _ = self.shift_calc_adder(e_inv, two)
        
        # 取 Shift 的低3位 [S0, S1, S2]
        shift_denorm = shift_denorm_5[..., 0:3]
        
        # 检测是否 Denorm: E <= 0
        # E=0: Bits 0-4 are 0.
        # E<0: Bit 4 (MSB) is 1.
        e_msb = e_final_5[..., 4:5]
        
        # Check E==0
        # OR(e0, e1, e2, e3, e4) -> If 0, then E=0.
        # Wait, OR all bits.
        e_any = self.denorm_check_zero(e_final_5[..., 0:1], e_final_5[..., 1:2], e_final_5[..., 2:3])
        e_any2 = self.denorm_check_zero2(e_final_5[..., 3:4], e_final_5[..., 4:5]) # Use ORGate
        e_is_nonzero = self.denorm_or(e_any, e_any2)
        e_is_zero = self.denorm_is_zero(e_is_nonzero)
        
        is_denorm = self.denorm_or(e_msb, e_is_zero)
        
        # 右移
        p_denorm = self.denormalizer(p_rounded, shift_denorm)
        
        # 选择最终 P 和 E
        # 如果 is_denorm: P=p_denorm, E=0
        # 否则: P=p_rounded, E=e_final
        
        # MUX P
        p_final_0 = self.mux_p_final(is_denorm, p_denorm[..., 0:1], p_rounded[..., 0:1])
        p_final_1 = self.mux_p_final(is_denorm, p_denorm[..., 1:2], p_rounded[..., 1:2])
        p_final_2 = self.mux_p_final(is_denorm, p_denorm[..., 2:3], p_rounded[..., 2:3])
        p_final_3 = self.mux_p_final(is_denorm, p_denorm[..., 3:4], p_rounded[..., 3:4])
        p_final_4 = self.mux_p_final(is_denorm, p_denorm[..., 4:5], p_rounded[..., 4:5])
        p_final_5 = self.mux_p_final(is_denorm, p_denorm[..., 5:6], p_rounded[..., 5:6])
        p_final_6 = self.mux_p_final(is_denorm, p_denorm[..., 6:7], p_rounded[..., 6:7])
        p_final_7 = self.mux_p_final(is_denorm, p_denorm[..., 7:8], p_rounded[..., 7:8])
        
        # MUX E
        # 如果 denorm, E=0.
        e_final_0 = self.mux_e_final(is_denorm, zeros, e_final_5[..., 0:1])
        e_final_1 = self.mux_e_final(is_denorm, zeros, e_final_5[..., 1:2])
        e_final_2 = self.mux_e_final(is_denorm, zeros, e_final_5[..., 2:3])
        e_final_3 = self.mux_e_final(is_denorm, zeros, e_final_5[..., 3:4])
        # e4 (sign) also 0
        
        # 提取最终 M (p5, p4, p3)
        # Subnormal format: 0.M. p6=0. M=p5,p4,p3.
        # Normal format: 1.M. p6=1. M=p5,p4,p3.
        # So bits match!
        m0_out = p_final_5
        m1_out = p_final_4
        m2_out = p_final_3
        
        e0_out = e_final_0
        e1_out = e_final_1
        e2_out = e_final_2
        e3_out = e_final_3
        
        # ===== 7. 符号与输出 =====
        # e_final_5 用于最终输出的指数
        # e_normalized 用于 E=-1/-2/-3 检测（因为 m_carry 只对 Normal 输出有意义）
        is_negative_norm = e_normalized[..., 4:5]  # 用于 underflow 路径检测
        e_norm_le = e_normalized[..., 0:4]  # 用于 E=-1/-2/-3 检测
        
        is_negative = e_final_5[..., 4:5]  # 符号位（用于输出）
        e_final_le = e_final_5[..., 0:4]  # 用于最终输出
        
        # 检测 e_final_5 == 0
        out_not0 = self.out_not_e0(e_final_le[..., 0:1])
        out_not1 = self.out_not_e1(e_final_le[..., 1:2])
        out_not2 = self.out_not_e2(e_final_le[..., 2:3])
        out_not3 = self.out_not_e3(e_final_le[..., 3:4])
        out_and01 = self.out_and_01(out_not0, out_not1)
        out_and23 = self.out_and_23(out_not2, out_not3)
        e_final_is_zero = self.out_e_is_zero(out_and01, out_and23)
        
        # 检测 e_normalized == 0 (需要新的门)
        norm_not0 = self.norm_not_e0(e_norm_le[..., 0:1])
        norm_not1 = self.norm_not_e1(e_norm_le[..., 1:2])
        norm_not2 = self.norm_not_e2(e_norm_le[..., 2:3])
        norm_not3 = self.norm_not_e3(e_norm_le[..., 3:4])
        norm_and01 = self.norm_and_01(norm_not0, norm_not1)
        norm_and23 = self.norm_and_23(norm_not2, norm_not3)
        e_norm_is_zero = self.norm_e_is_zero(norm_and01, norm_and23)
        
        # is_subnormal = (e_final_5 == 0) AND (e_normalized == 0) AND (e_normalized >= 0)
        # 关键修复：必须同时满足 e_final=0 且 e_normalized=0
        # 这样当 e_normalized=-1 且 m_carry=1 时 (e_final=0 但 e_normalized≠0)，不会被误判
        not_negative_norm = self.underflow_not(is_negative_norm)
        e_both_zero = self.e_both_zero_and(e_final_is_zero, e_norm_is_zero)
        is_subnormal = self.is_subnormal_and(e_both_zero, not_negative_norm)
        
        # is_special = is_negative_norm OR is_subnormal
        # 关键修复：用 is_negative_norm (基于 e_normalized) 而非 is_negative (基于 e_final_5)
        # 这样当 e_normalized < 0 时，即使 e_final_5 >= 0，也会走 special 路径
        is_special = self.sub_or_under(is_negative_norm, is_subnormal)
        
        # ===== 7. Subnormal尾数计算 (纯SNN) =====
        # E=0时：1.m0m1m2 右移1位 → 0.1m0m1m2
        # 取3位: [1, m0, m1]，舍入位是m2，sticky是原始舍入后的低位
        # 使用 raw 值而非 normal 值（避免被 Normal 路径舍入影响）
        m0_sub_raw = ones  # 隐含1变成显式
        m1_sub_raw = m0_raw
        m2_sub_raw = m1_raw
        # Subnormal 路径的舍入:
        # 规格化后格式: 1.m0m1m2 round_bit [sticky]
        # Subnormal 右移1位: 0.1m0m1 [m2] [round_bit OR sticky]
        # 注意：m2_raw 已经被修正过（当 shift >= 4 时包含 sticky_extra）
        # 所以 sub_round = m2_raw（已修正）
        # sub_sticky 使用已经排除 sticky_extra 的 sticky
        sub_round = m2_raw  # 使用已修正的 m2_raw
        sub_sticky = self.sub_sticky_or(round_bit, sticky)  # sticky 已经正确处理过
        
        # RNE 舍入：do_round = sub_round AND (sub_sticky OR m2_sub_raw)
        sub_sticky_or_lsb = self.sub_rne_or(sub_sticky, m2_sub_raw)
        sub_do_round = self.sub_rne_and(sub_round, sub_sticky_or_lsb)
        
        # 舍入：用HalfAdder链
        m2_sub_temp, sub_c0 = self.sub_round_ha0(m2_sub_raw, sub_do_round)
        m1_sub_temp, sub_c1 = self.sub_round_ha1(m1_sub_raw, sub_c0)
        m0_sub_temp, sub_c2 = self.sub_round_ha2(m0_sub_raw, sub_c1)
        
        # 当进位sub_c2=1时，Subnormal 0.111 溢出到 Normal 1.000
        # 此时应该设置 M=0, E=1（而不是饱和到 M=7）
        # 修正：sub_c2=1 时选择 zeros (M=0)
        m0_subnormal = self.sub_saturate_m0(sub_c2, zeros, m0_sub_temp)
        m1_subnormal = self.sub_saturate_m1(sub_c2, zeros, m1_sub_temp)
        m2_subnormal = self.sub_saturate_m2(sub_c2, zeros, m2_sub_temp)
        
        # 记录 Subnormal 溢出标志，用于在输出时设置 E=1
        sub_overflow_flag = sub_c2  # 保存以便后续使用
        
        # ===== 8. E=-1检测 (纯SNN) =====
        # 5位补码 -1 = 11111
        # 使用 e_norm_le (不含 m_carry) 进行检测
        m1_and_01 = self.minus1_and_01(e_norm_le[..., 0:1], e_norm_le[..., 1:2])
        m1_and_23 = self.minus1_and_23(e_norm_le[..., 2:3], e_norm_le[..., 3:4])
        m1_and_0123 = self.minus1_and_0123(m1_and_01, m1_and_23)
        e_is_minus1 = self.minus1_and_all(m1_and_0123, is_negative_norm)
        
        # E=-1时：1.m0m1m2 右移2位 → 0.01m0m1m2
        # 取3位: [0, 1, m0]，舍入位是m1, sticky = m2 OR round_bit OR original_sticky
        # 使用 raw 值而非 normal 值（避免被之前的舍入影响）
        m0_e_minus1_raw = zeros
        m1_e_minus1_raw = ones
        m2_e_minus1_raw = m0_raw
        e_m1_round = m1_raw
        e_m1_sticky_t = self.em1_sticky_or(m2_raw, round_bit)
        e_m1_sticky = self.em1_sticky_or2(e_m1_sticky_t, sticky)
        
        # RNE: do_round = round AND (sticky OR lsb)
        # lsb = m2_e_minus1_raw = m0_normal
        em1_sticky_or_lsb = self.em1_rne_or(e_m1_sticky, m2_e_minus1_raw)
        em1_do_round = self.em1_rne_and(e_m1_round, em1_sticky_or_lsb)
        
        m2_e_minus1, em1_c0 = self.e_m1_round_ha0(m2_e_minus1_raw, em1_do_round)
        m1_e_minus1, em1_c1 = self.e_m1_round_ha1(m1_e_minus1_raw, em1_c0)
        m0_e_minus1, _ = self.e_m1_round_ha2(m0_e_minus1_raw, em1_c1)
        
        # ===== 9. E=-2检测 (纯SNN) =====
        # 5位补码 -2 = 11110
        # 使用 e_norm_le (不含 m_carry) 进行检测
        not_e0 = self.minus2_not_e0(e_norm_le[..., 0:1])
        m2_and_01 = self.minus2_and_01(not_e0, e_norm_le[..., 1:2])
        m2_and_23 = self.minus2_and_23(e_norm_le[..., 2:3], e_norm_le[..., 3:4])
        m2_and_0123 = self.minus2_and_0123(m2_and_01, m2_and_23)
        e_is_minus2 = self.minus2_and_all(m2_and_0123, is_negative_norm)
        
        # E=-2时：1.m0m1m2 右移3位 → 0.001m0m1m2
        # 取3位: [0, 0, 1]，舍入位是m0, sticky = m1 OR m2 OR round_bit OR original_sticky
        # 使用 raw 值而非 normal 值
        m0_e_minus2_raw = zeros
        m1_e_minus2_raw = zeros
        m2_e_minus2_raw = ones
        e_m2_round = m0_raw
        e_m2_sticky_t = self.em2_sticky_or(m1_raw, m2_raw)
        e_m2_sticky_t2 = self.em2_sticky_or2(e_m2_sticky_t, round_bit)
        e_m2_sticky = self.sticky_or_no_overflow(e_m2_sticky_t2, sticky)
        
        # RNE: do_round = round AND (sticky OR lsb)
        # lsb = m2_e_minus2_raw = 1 (常数)
        em2_sticky_or_lsb = self.em2_rne_or(e_m2_sticky, m2_e_minus2_raw)
        em2_do_round = self.em2_rne_and(e_m2_round, em2_sticky_or_lsb)
        
        m2_e_minus2, em2_c0 = self.e_m2_round_ha0(m2_e_minus2_raw, em2_do_round)
        m1_e_minus2, em2_c1 = self.e_m2_round_ha1(m1_e_minus2_raw, em2_c0)
        m0_e_minus2, _ = self.e_m2_round_ha2(m0_e_minus2_raw, em2_c1)
        
        # ===== 9.5 E=-3检测 (纯SNN) =====
        # 5位补码 -3 = 11101
        # 使用 e_norm_le (不含 m_carry) 进行检测
        not_e1 = self.minus3_not_e1(e_norm_le[..., 1:2])
        m3_and_01 = self.minus3_and_01(e_norm_le[..., 0:1], not_e1)
        m3_and_23 = self.minus3_and_23(e_norm_le[..., 2:3], e_norm_le[..., 3:4])
        m3_and_0123 = self.minus3_and_0123(m3_and_01, m3_and_23)
        e_is_minus3 = self.minus3_and_all(m3_and_0123, is_negative_norm)
        
        # E=-3时：1.m0m1m2 右移4位 → 0.0001m0m1m2
        # 取3位: [0, 0, 0]，舍入位是隐含1（常数1），sticky = m0 OR m1 OR m2 OR round_bit OR original
        # 使用 raw 值而非 normal 值
        m0_e_minus3_raw = zeros
        m1_e_minus3_raw = zeros
        m2_e_minus3_raw = zeros
        e_m3_round = ones  # 隐含1变成舍入位
        e_m3_sticky = self.em3_sticky_or(m0_raw, m1_raw, m2_raw)
        e_m3_sticky_t2 = self.em3_sticky_or2(e_m3_sticky, round_bit)
        e_m3_sticky_full = self.sticky_or_no_overflow(e_m3_sticky_t2, sticky)
        
        # RNE: do_round = round AND (sticky OR lsb)
        # lsb = m2_e_minus3_raw = 0，所以只看 sticky
        em3_sticky_or_lsb = self.em3_rne_or(e_m3_sticky_full, m2_e_minus3_raw)
        em3_do_round = self.em3_rne_and(e_m3_round, em3_sticky_or_lsb)
        
        # 舍入：如果进位，M = 001，否则 M = 000
        m2_e_minus3, _ = self.e_m3_round_ha0(m2_e_minus3_raw, em3_do_round)
        m1_e_minus3 = m1_e_minus3_raw
        m0_e_minus3 = m0_e_minus3_raw
        
        # 更深下溢 (E<-3)：下溢到零 M=000
        m0_deep = zeros
        m1_deep = zeros
        m2_deep = zeros
        
        # ===== 10. 选择underflow尾数 (E=-1 vs E=-2 vs E=-3 vs 更深) =====
        # 先选E=-3还是更深
        m0_m3_or_deep = self.mux_m2_m0(e_is_minus3, m0_e_minus3, m0_deep)
        m1_m3_or_deep = self.mux_m2_m1(e_is_minus3, m1_e_minus3, m1_deep)
        m2_m3_or_deep = self.mux_m2_m2(e_is_minus3, m2_e_minus3, m2_deep)
        
        # 再选E=-2还是(E=-3或更深)
        m0_m2_or_deep = self.mux_m2_m0(e_is_minus2, m0_e_minus2, m0_m3_or_deep)
        m1_m2_or_deep = self.mux_m2_m1(e_is_minus2, m1_e_minus2, m1_m3_or_deep)
        m2_m2_or_deep = self.mux_m2_m2(e_is_minus2, m2_e_minus2, m2_m3_or_deep)
        
        # 再选E=-1还是(E=-2或更深)
        m0_underflow = self.mux_under_m0(e_is_minus1, m0_e_minus1, m0_m2_or_deep)
        m1_underflow = self.mux_under_m1(e_is_minus1, m1_e_minus1, m1_m2_or_deep)
        m2_underflow = self.mux_under_m2(e_is_minus1, m2_e_minus1, m2_m2_or_deep)
        
        # ===== 11. 选择special尾数 (subnormal vs underflow) =====
        m0_special = self.mux_sub_m0(is_subnormal, m0_subnormal, m0_underflow)
        m1_special = self.mux_sub_m1(is_subnormal, m1_subnormal, m1_underflow)
        m2_special = self.mux_sub_m2(is_subnormal, m2_subnormal, m2_underflow)
        
        # ===== 12. 选择最终结果 =====
        e_normal = e_final_le.flip(-1)  # 大端
        
        e0_out = self.mux_final_e0(is_special, zeros, e_normal[..., 0:1])
        e1_out = self.mux_final_e1(is_special, zeros, e_normal[..., 1:2])
        e2_out = self.mux_final_e2(is_special, zeros, e_normal[..., 2:3])
        e3_out = self.mux_final_e3(is_special, zeros, e_normal[..., 3:4])
        
        # Subnormal 溢出处理：当 is_subnormal=1 且 sub_overflow_flag=1 时，E=1 (0001)
        # e_out 是大端格式 [E3, E2, E1, E0]，E=1 意味着 E0=1（即 e3_out=1）
        sub_overflow = self.sub_overflow_and(is_subnormal, sub_overflow_flag)
        e3_out = self.mux_sub_overflow_e0(sub_overflow, ones, e3_out)
        
        m0_out = self.mux_final_m0(is_special, m0_special, m0_normal)
        m1_out = self.mux_final_m1(is_special, m1_special, m1_normal)
        m2_out = self.mux_final_m2(is_special, m2_special, m2_normal)
        
        # ===== 12.5 上溢检测 =====
        # 正确的 overflow 条件：e_final >= 16
        # 在 5 位补码中，e_final_5[4]=1 且 sum_e >= 22 时是 overflow
        # （因为 sum_e >= 22 保证 e_normalized 不可能是真正的负数）
        # sum_e >= 22 = 10110:
        # (sum_e[4]=1 且 sum_e[3]=1) 或
        # (sum_e[4]=1 且 sum_e[3]=0 且 sum_e[2]=1 且 sum_e[1]=1)
        of_43 = self.overflow_43(sum_e[..., 4:5], sum_e[..., 3:4])
        of_not_3 = self.overflow_not_3(sum_e[..., 3:4])
        of_21 = self.overflow_21(sum_e[..., 2:3], sum_e[..., 1:2])
        of_4n3 = self.overflow_4n321(sum_e[..., 4:5], of_not_3)
        of_4n321 = self.overflow_4n321_21(of_4n3, of_21)
        sum_e_ge_22 = self.overflow_final(of_43, of_4n321)
        # 只有当 sum_e >= 22 且 e_final_5[4] = 1 时才是真正的 overflow
        is_overflow = self.overflow_and_msb(sum_e_ge_22, e_final_5[..., 4:5])
        
        # 上溢时输出 NaN: E=15 (1111), M=7 (111)
        e0_out = self.mux_overflow_e0(is_overflow, ones, e0_out)
        e1_out = self.mux_overflow_e1(is_overflow, ones, e1_out)
        e2_out = self.mux_overflow_e2(is_overflow, ones, e2_out)
        e3_out = self.mux_overflow_e3(is_overflow, ones, e3_out)
        m0_out = self.mux_overflow_m0(is_overflow, ones, m0_out)
        m1_out = self.mux_overflow_m1(is_overflow, ones, m1_out)
        m2_out = self.mux_overflow_m2(is_overflow, ones, m2_out)
        
        # ===== 13. 输入为零时清零 (纯SNN) =====
        not_zero = self.zero_not(input_has_zero)
        e0_out = self.zero_and_e0(e0_out, not_zero)
        e1_out = self.zero_and_e1(e1_out, not_zero)
        e2_out = self.zero_and_e2(e2_out, not_zero)
        e3_out = self.zero_and_e3(e3_out, not_zero)
        m0_out = self.zero_and_m0(m0_out, not_zero)
        m1_out = self.zero_and_m1(m1_out, not_zero)
        m2_out = self.zero_and_m2(m2_out, not_zero)
        
        # ===== 14. 组装输出 =====
        e_out = torch.cat([e0_out, e1_out, e2_out, e3_out], dim=-1)
        m_out = torch.cat([m0_out, m1_out, m2_out], dim=-1)
        return torch.cat([s_out, e_out, m_out], dim=-1)

    def reset(self):
        self.sign_xor.reset()
        self.exp_adder.reset()
        self.bias_sub_adder.reset()
        self.mantissa_mul.reset()
        self.norm_unit.reset()
        self.exp_norm_adder.reset()
        self.denormalizer.reset()
        self.e_inv_gate.reset()
        self.shift_calc_adder.reset()
        self.mux_p_final.reset()
        self.mux_e_final.reset()
        self.denorm_or.reset()
        self.denorm_not.reset()
        self.denorm_check_zero.reset()
        self.denorm_check_zero2.reset()
        self.denorm_is_zero.reset()
        self.mux_m0_final.reset()
        self.mux_m1_final.reset()
        self.mux_m2_final.reset()
        self.sticky_or_overflow.reset()
        self.sticky_or_no_overflow.reset()
        self.sticky_or_extra.reset()
        self.mux_sticky.reset()
        self.sub_sticky_or.reset()
        self.sub_sticky_or2.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.sub_rne_or.reset()
        self.sub_rne_and.reset()
        self.sub_round_mux.reset()
        self.em1_rne_or.reset()
        self.em1_rne_and.reset()
        self.em1_sticky_or.reset()
        self.em1_sticky_or2.reset()
        self.em2_rne_or.reset()
        self.em2_rne_and.reset()
        self.em2_sticky_or.reset()
        self.em2_sticky_or2.reset()
        self.em3_sticky_or2.reset()
        self.round_ha0.reset()
        self.round_ha1.reset()
        self.round_ha2.reset()
        self.exp_inc_adder.reset()
        
        # 输入零检测
        self.a_not_e0.reset()
        self.a_not_e1.reset()
        self.a_not_e2.reset()
        self.a_not_e3.reset()
        self.a_and_e01.reset()
        self.a_and_e23.reset()
        self.a_e_is_zero.reset()
        self.a_not_m0.reset()
        self.a_not_m1.reset()
        self.a_not_m2.reset()
        self.a_and_m01.reset()
        self.a_m_is_zero.reset()
        self.a_is_true_zero.reset()
        self.b_not_e0.reset()
        self.b_not_e1.reset()
        self.b_not_e2.reset()
        self.b_not_e3.reset()
        self.b_and_e01.reset()
        self.b_and_e23.reset()
        self.b_e_is_zero.reset()
        self.b_not_m0.reset()
        self.b_not_m1.reset()
        self.b_not_m2.reset()
        self.b_and_m01.reset()
        self.b_m_is_zero.reset()
        self.b_is_true_zero.reset()
        self.input_zero_or.reset()
        
        # 输入subnormal检测
        self.a_not_m_is_zero.reset()
        self.a_is_subnormal.reset()
        self.b_not_m_is_zero.reset()
        self.b_is_subnormal.reset()
        self.mux_a_leading.reset()
        self.mux_b_leading.reset()
        self.mux_a_e0.reset()
        self.mux_a_e1.reset()
        self.mux_a_e2.reset()
        self.mux_a_e3.reset()
        self.mux_b_e0.reset()
        self.mux_b_e1.reset()
        self.mux_b_e2.reset()
        self.mux_b_e3.reset()
        
        # 输出E=0检测
        self.out_not_e0.reset()
        self.out_not_e1.reset()
        self.out_not_e2.reset()
        self.out_not_e3.reset()
        self.out_and_01.reset()
        self.out_and_23.reset()
        self.out_e_is_zero.reset()
        # e_normalized == 0 检测
        self.norm_not_e0.reset()
        self.norm_not_e1.reset()
        self.norm_not_e2.reset()
        self.norm_not_e3.reset()
        self.norm_and_01.reset()
        self.norm_and_23.reset()
        self.norm_e_is_zero.reset()
        self.e_both_zero_and.reset()
        self.underflow_not.reset()
        self.is_subnormal_and.reset()
        self.sub_or_under.reset()
        
        # Subnormal舍入
        self.sub_round_ha0.reset()
        self.sub_round_ha1.reset()
        self.sub_round_ha2.reset()
        self.sub_saturate_m0.reset()
        self.sub_saturate_m1.reset()
        self.sub_saturate_m2.reset()
        self.sub_overflow_and.reset()
        self.mux_sub_overflow_e0.reset()
        
        # E=-1检测
        self.minus1_and_01.reset()
        self.minus1_and_23.reset()
        self.minus1_and_0123.reset()
        self.minus1_and_all.reset()
        
        # E=-1舍入
        self.e_m1_round_ha0.reset()
        self.e_m1_round_ha1.reset()
        self.e_m1_round_ha2.reset()
        
        # E=-2检测
        self.minus2_not_e0.reset()
        self.minus2_and_01.reset()
        self.minus2_and_23.reset()
        self.minus2_and_0123.reset()
        self.minus2_and_all.reset()
        
        # E=-2舍入
        self.e_m2_round_ha0.reset()
        self.e_m2_round_ha1.reset()
        self.e_m2_round_ha2.reset()
        
        # E=-3检测
        self.minus3_not_e1.reset()
        self.minus3_and_01.reset()
        self.minus3_and_23.reset()
        self.minus3_and_0123.reset()
        self.minus3_and_all.reset()
        self.em3_sticky_or.reset()
        self.em3_rne_or.reset()
        self.em3_rne_and.reset()
        self.e_m3_round_ha0.reset()
        
        # MUX选择
        self.mux_m2_m0.reset()
        self.mux_m2_m1.reset()
        self.mux_m2_m2.reset()
        self.mux_under_m0.reset()
        self.mux_under_m1.reset()
        self.mux_under_m2.reset()
        self.mux_sub_m0.reset()
        self.mux_sub_m1.reset()
        self.mux_sub_m2.reset()
        self.mux_final_e0.reset()
        self.mux_final_e1.reset()
        self.mux_final_e2.reset()
        self.mux_final_e3.reset()
        self.mux_final_m0.reset()
        self.mux_final_m1.reset()
        self.mux_final_m2.reset()
        
        # 上溢检测门
        self.overflow_43.reset()
        self.overflow_not_3.reset()
        self.overflow_21.reset()
        self.overflow_210.reset()
        self.overflow_4n321.reset()
        self.overflow_4n321_21.reset()
        self.overflow_final.reset()
        self.overflow_and_msb.reset()
        self.mux_overflow_e0.reset()
        self.mux_overflow_e1.reset()
        self.mux_overflow_e2.reset()
        self.mux_overflow_e3.reset()
        self.mux_overflow_m0.reset()
        self.mux_overflow_m1.reset()
        self.mux_overflow_m2.reset()
        
        # 清零门
        self.zero_not.reset()
        self.zero_and_e0.reset()
        self.zero_and_e1.reset()
        self.zero_and_e2.reset()
        self.zero_and_e3.reset()
        self.zero_and_m0.reset()
        self.zero_and_m1.reset()
        self.zero_and_m2.reset()
        
        # sticky_extra 用途修正门
        self.shift_ge4_and.reset()
        self.m2_raw_fix_or.reset()
        self.shift_lt4_not.reset()
        self.shift_eq3_and_10.reset()
        self.shift_eq3_and.reset()
        self.shift_eq3_sticky.reset()
        self.round_bit_fix_or.reset()
        self.shift_ge3_or.reset()
        self.shift_lt3_not.reset()
        self.sticky_extra_masked.reset()
        self.sub_sticky_base.reset()
        self.sub_sticky_final.reset()

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()
