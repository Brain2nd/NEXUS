"""
FP32 除法器 - 100%纯SNN门电路实现
======================================

FP32 格式: [S | E7..E0 | M22..M0], bias=127

核心算法 (恢复余数除法):
1. 符号: Sr = Sa XOR Sb
2. 指数: Er = Ea - Eb + 127
3. 尾数除法: 26次迭代 (24位商 + 2位舍入)
   - 每次迭代: 尝试减法，根据结果选择商位
4. RNE舍入

特殊情况:
- x / 0 = Inf (x != 0)
- 0 / 0 = NaN
- Inf / x = Inf (x != Inf, x != 0)
- x / Inf = 0 (x != Inf)
- NaN / x = NaN, x / NaN = NaN

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.logic_gates import (HalfAdder, FullAdder, ORTree)
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX, VecORTree, VecANDTree,
    VecFullAdder, VecAdder, VecSubtractor
)


# ==============================================================================
# 25位减法器 (用于SNN原生逐位除法)
# ==============================================================================
class Subtractor25Bit(nn.Module):
    """25位减法器 - 纯SNN (LSB first)
    
    用于逐位除法: R最大24位，左移后25位
    A - B = A + (~B) + 1
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 25
        nt = neuron_template
        # 单实例 (动态扩展机制支持不同位宽)
        self.not_gate = VecNOT(neuron_template=nt, max_param_shape=None)
        self.adder = FullAdder(neuron_template=nt, max_param_shape=None)
        self.borrow_not = VecNOT(neuron_template=nt, max_param_shape=None)

    def forward(self, A, B):
        """A - B, LSB first
        Returns: (差, 借位符号)
        借位符号=0表示A>=B, 借位符号=1表示A<B
        """
        # ~B
        not_b = []
        for i in range(self.bits):
            not_b.append(self.not_gate(B[..., i:i+1]))
        not_b = torch.cat(not_b, dim=-1)

        # A + ~B + 1
        c = torch.ones_like(A[..., 0:1])
        sum_bits = []
        for i in range(self.bits):
            s, c = self.adder(A[..., i:i+1], not_b[..., i:i+1], c)
            sum_bits.append(s)

        result = torch.cat(sum_bits, dim=-1)
        borrow = self.borrow_not(c)  # c=1→borrow=0 (A>=B)

        return result, borrow

    def reset(self):
        self.not_gate.reset()
        self.adder.reset()
        self.borrow_not.reset()


# ==============================================================================
# 48位减法器 (保留用于兼容)
# ==============================================================================
class Subtractor48Bit(nn.Module):
    """48位减法器 - 纯SNN (LSB first)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 48
        nt = neuron_template
        # 单实例 (动态扩展机制支持不同位宽)
        self.not_gate = VecNOT(neuron_template=nt, max_param_shape=None)
        self.adder = FullAdder(neuron_template=nt, max_param_shape=None)
        self.borrow_not = VecNOT(neuron_template=nt, max_param_shape=None)

    def forward(self, A, B):
        not_b = []
        for i in range(self.bits):
            not_b.append(self.not_gate(B[..., i:i+1]))
        not_b = torch.cat(not_b, dim=-1)

        c = torch.ones_like(A[..., 0:1])
        sum_bits = []
        for i in range(self.bits):
            s, c = self.adder(A[..., i:i+1], not_b[..., i:i+1], c)
            sum_bits.append(s)

        result = torch.cat(sum_bits, dim=-1)
        borrow = self.borrow_not(c)
        return result, borrow

    def reset(self):
        self.not_gate.reset()
        self.adder.reset()
        self.borrow_not.reset()


# ==============================================================================
# SNN原生逐位除法迭代单元 (25位)
# ==============================================================================
class DivisionIteration25(nn.Module):
    """SNN原生逐位除法迭代 - 25位余数

    符合IF神经元"累积-阈值-复位"动力学:
    - R = 余数 (膜电位)
    - D = 除数 (阈值)
    - R >= D 时发放脉冲(Q=1)并复位(R = R - D)

    输入: R (余数, 25位 LSB first), D (除数, 25位 LSB first, 高位补0)
    输出: Q_bit (商位), R_next (新余数)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        self.sub = Subtractor25Bit(neuron_template=nt)
        self.not_borrow = VecNOT(neuron_template=nt, max_param_shape=None)
        # 单实例 (动态扩展机制支持不同位宽)
        self.mux_r = VecMUX(neuron_template=nt, max_param_shape=None)

    def forward(self, R, D):
        """
        R: [..., 25] 当前余数 (LSB first)
        D: [..., 25] 除数 (LSB first, 24位有效+1位0)
        Returns: Q_bit [..., 1], R_next [..., 25]
        """
        # 尝试减法 R - D
        R_trial, borrow = self.sub(R, D)

        # Q = NOT(borrow) = 1 当 R >= D
        Q_bit = self.not_borrow(borrow)

        # 选择余数 - 向量化: borrow=0时R_trial, borrow=1时R(恢复)
        borrow_25 = borrow.expand_as(R)
        R_next = self.mux_r(borrow_25, R, R_trial)

        return Q_bit, R_next

    def reset(self):
        self.sub.reset()
        self.not_borrow.reset()
        self.mux_r.reset()


# ==============================================================================
# 旧版48位除法迭代单元 (保留兼容)
# ==============================================================================
class DivisionIteration(nn.Module):
    """48位除法迭代 - 恢复余数算法"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        self.sub = Subtractor48Bit(neuron_template=nt)
        self.not_borrow = VecNOT(neuron_template=nt, max_param_shape=None)
        # 单实例 (动态扩展机制支持不同位宽)
        self.mux_r = VecMUX(neuron_template=nt, max_param_shape=None)

    def forward(self, R, D_extended):
        """
        R: [..., 48] 当前余数 (LSB first)
        D_extended: [..., 48] 扩展的除数 (LSB first)
        Returns: Q_bit [..., 1], R_next [..., 48]
        """
        # 尝试减法
        R_trial, borrow = self.sub(R, D_extended)

        # Q_bit = NOT(borrow) = 1 当 A>=B
        Q_bit = self.not_borrow(borrow)

        # 选择余数 - 向量化: borrow=0时选R_trial, borrow=1时选R
        borrow_48 = borrow.expand_as(R)
        R_next = self.mux_r(borrow_48, R, R_trial)

        return Q_bit, R_next

    def reset(self):
        self.sub.reset()
        self.not_borrow.reset()
        self.mux_r.reset()


# ==============================================================================
# FP32 除法器主类
# ==============================================================================
class SpikeFP32Divider(nn.Module):
    """FP32 除法器 - 100%纯SNN门电路实现
    
    输入: A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0]
    输出: [..., 32] FP32脉冲 (A / B)
    
    使用恢复余数除法，26次迭代获得24位商+2位舍入。
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # ===== 符号 =====
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=None)

        # ===== 指数运算 (10位防止溢出) =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.exp_sub = FullAdder(neuron_template=nt, max_param_shape=None)
        self.exp_not = VecNOT(neuron_template=nt, max_param_shape=None)
        # 加bias: + 127 (10位)
        self.exp_add_bias = VecAdder(bits=10, neuron_template=nt, max_param_shape=(10,))
        # 指数调整 (归一化时减1) (10位)
        self.exp_adjust = VecAdder(bits=10, neuron_template=nt, max_param_shape=(10,))
        self.exp_adjust_not = VecNOT(neuron_template=nt, max_param_shape=None)

        # ===== 溢出/下溢检测 =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.overflow_check_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.overflow_or = VecOR(neuron_template=nt, max_param_shape=None)
        self.overflow_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.overflow_not = VecNOT(neuron_template=nt, max_param_shape=None)
        # 下溢
        self.underflow_not = VecNOT(neuron_template=nt, max_param_shape=None)
        self.underflow_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.underflow_or = VecOR(neuron_template=nt, max_param_shape=None)
        # 溢出/下溢结果MUX - 单实例
        self.overflow_mux_e = VecMUX(neuron_template=nt, max_param_shape=None)
        self.overflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=None)
        self.underflow_mux_e = VecMUX(neuron_template=nt, max_param_shape=None)
        self.underflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=None)

        # ===== Subnormal输出处理 =====
        from .fp32_mul import BarrelShifterRight48
        self.subnormal_shifter = BarrelShifterRight48(neuron_template=nt)
        self.shift_add_one = VecAdder(bits=10, neuron_template=nt, max_param_shape=(10,))
        # 单实例 (动态扩展机制支持不同位宽)
        self.subnorm_sticky_or = VecOR(neuron_template=nt, max_param_shape=None)
        self.subnorm_rne_or = VecOR(neuron_template=nt, max_param_shape=None)
        self.subnorm_rne_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.subnorm_round_adder = VecAdder(bits=24, neuron_template=nt, max_param_shape=(24,))
        self.subnorm_mux_m = VecMUX(neuron_template=nt, max_param_shape=None)
        self.is_subnormal_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.not_very_underflow = VecNOT(neuron_template=nt, max_param_shape=None)

        # ===== 归一化 =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.normalize_mux = VecMUX(neuron_template=nt, max_param_shape=None)
        self.normalize_exp_mux = VecMUX(neuron_template=nt, max_param_shape=None)
        self.q0_not = VecNOT(neuron_template=nt, max_param_shape=None)

        # ===== 特殊值检测 =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.exp_all_one_and_a = VecAND(neuron_template=nt, max_param_shape=None)
        self.exp_all_one_and_b = VecAND(neuron_template=nt, max_param_shape=None)

        # 指数全0检测
        self.exp_zero_or_a = VecOR(neuron_template=nt, max_param_shape=None)
        self.exp_zero_or_b = VecOR(neuron_template=nt, max_param_shape=None)
        self.exp_zero_not_a = VecNOT(neuron_template=nt, max_param_shape=None)
        self.exp_zero_not_b = VecNOT(neuron_template=nt, max_param_shape=None)

        # 尾数全0检测
        self.mant_zero_or_a = VecOR(neuron_template=nt, max_param_shape=None)
        self.mant_zero_or_b = VecOR(neuron_template=nt, max_param_shape=None)
        self.mant_zero_not_a = VecNOT(neuron_template=nt, max_param_shape=None)
        self.mant_zero_not_b = VecNOT(neuron_template=nt, max_param_shape=None)

        # 零检测: E=0 AND M=0
        self.a_is_zero_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.b_is_zero_and = VecAND(neuron_template=nt, max_param_shape=None)

        # Inf检测: E=FF AND M=0
        self.a_is_inf_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.b_is_inf_and = VecAND(neuron_template=nt, max_param_shape=None)

        # NaN检测: E=FF AND M!=0
        self.a_mant_nonzero_not = VecNOT(neuron_template=nt, max_param_shape=None)
        self.b_mant_nonzero_not = VecNOT(neuron_template=nt, max_param_shape=None)
        self.a_is_nan_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.b_is_nan_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.either_nan_or = VecOR(neuron_template=nt, max_param_shape=None)

        # x/0 = Inf, 0/0 = NaN
        self.div_by_zero_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.zero_div_zero_and = VecAND(neuron_template=nt, max_param_shape=None)

        # 结果特殊值组合
        self.result_is_nan_or = VecOR(neuron_template=nt, max_param_shape=None)
        self.result_is_inf_or = VecOR(neuron_template=nt, max_param_shape=None)
        self.result_is_zero_or = VecOR(neuron_template=nt, max_param_shape=None)

        # ===== SNN原生逐位除法 (49次迭代) =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.div_iteration = DivisionIteration25(neuron_template=nt)
        
        # ===== 比较器 (mant_a >= mant_b) =====
        self.mant_cmp_sub = Subtractor25Bit(neuron_template=nt)  # 24位比较用25位减法器
        
        # ===== Sticky OR树 (检测25位余数是否非零) - 向量化 =====
        self.vec_or_tree = VecORTree(neuron_template=nt, max_param_shape=(25,))

        # ===== RNE舍入 =====
        self.rne_or = VecOR(neuron_template=nt, max_param_shape=None)
        self.rne_and = VecAND(neuron_template=nt, max_param_shape=None)
        self.round_adder = VecAdder(bits=23, neuron_template=nt, max_param_shape=(23,))

        # ===== 输出选择 - 向量化 =====
        # 分离指数和尾数的 VecMUX，避免不同位宽导致的形状冲突
        self.vec_mux_exp = VecMUX(neuron_template=nt, max_param_shape=(8,))  # 8-bit 指数
        self.vec_mux_mant = VecMUX(neuron_template=nt, max_param_shape=(23,))  # 23-bit 尾数
        
        # 纯SNN NOT门
        self.not_a_is_zero = VecNOT(neuron_template=nt, max_param_shape=None)
        self.not_b_is_inf = VecNOT(neuron_template=nt, max_param_shape=None)
        self.not_result_is_nan = VecNOT(neuron_template=nt, max_param_shape=None)
        self.not_result_is_inf = VecNOT(neuron_template=nt, max_param_shape=None)
        self.inf_and_not_nan = VecAND(neuron_template=nt, max_param_shape=None)
        self.zero_and_not_nan = VecAND(neuron_template=nt, max_param_shape=None)
        self.zero_and_not_inf = VecAND(neuron_template=nt, max_param_shape=None)
        
    def forward(self, A, B):
        """
        A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0] MSB first
        Returns: [..., 32] FP32脉冲 (A / B)
        """
        A, B = torch.broadcast_tensors(A, B)
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # ===== 1. 提取各部分 =====
        s_a = A[..., 0:1]
        e_a = A[..., 1:9]  # MSB first
        m_a = A[..., 9:32]  # MSB first
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:9]
        m_b = B[..., 9:32]
        
        # ===== 2. 符号 =====
        s_out = self.sign_xor(s_a, s_b)
        
        # ===== 3. 特殊值检测 =====
        # 指数全1检测 (Inf/NaN)
        e_a_all_one = e_a[..., 0:1]
        for i in range(1, 8):
            e_a_all_one = self.exp_all_one_and_a(e_a_all_one, e_a[..., i:i+1])

        e_b_all_one = e_b[..., 0:1]
        for i in range(1, 8):
            e_b_all_one = self.exp_all_one_and_b(e_b_all_one, e_b[..., i:i+1])

        # 指数全0检测
        e_a_any_one = e_a[..., 0:1]
        for i in range(1, 8):
            e_a_any_one = self.exp_zero_or_a(e_a_any_one, e_a[..., i:i+1])
        e_a_is_zero = self.exp_zero_not_a(e_a_any_one)

        e_b_any_one = e_b[..., 0:1]
        for i in range(1, 8):
            e_b_any_one = self.exp_zero_or_b(e_b_any_one, e_b[..., i:i+1])
        e_b_is_zero = self.exp_zero_not_b(e_b_any_one)

        # 尾数全0检测
        m_a_any_one = m_a[..., 0:1]
        for i in range(1, 23):
            m_a_any_one = self.mant_zero_or_a(m_a_any_one, m_a[..., i:i+1])
        m_a_is_zero = self.mant_zero_not_a(m_a_any_one)
        m_a_nonzero = m_a_any_one

        m_b_any_one = m_b[..., 0:1]
        for i in range(1, 23):
            m_b_any_one = self.mant_zero_or_b(m_b_any_one, m_b[..., i:i+1])
        m_b_is_zero = self.mant_zero_not_b(m_b_any_one)
        m_b_nonzero = m_b_any_one
        
        # 零检测: E=0 AND M=0
        a_is_zero = self.a_is_zero_and(e_a_is_zero, m_a_is_zero)
        b_is_zero = self.b_is_zero_and(e_b_is_zero, m_b_is_zero)
        
        # Inf检测: E=FF AND M=0
        a_is_inf = self.a_is_inf_and(e_a_all_one, m_a_is_zero)
        b_is_inf = self.b_is_inf_and(e_b_all_one, m_b_is_zero)
        
        # NaN检测: E=FF AND M!=0
        a_is_nan = self.a_is_nan_and(e_a_all_one, m_a_nonzero)
        b_is_nan = self.b_is_nan_and(e_b_all_one, m_b_nonzero)
        either_nan = self.either_nan_or(a_is_nan, b_is_nan)
        
        # x/0 (x!=0) = Inf
        not_a_is_zero = self.not_a_is_zero(a_is_zero)
        div_by_zero = self.div_by_zero_and(not_a_is_zero, b_is_zero)
        
        # 0/0 = NaN
        zero_div_zero = self.zero_div_zero_and(a_is_zero, b_is_zero)
        
        # Inf/Inf = NaN
        inf_div_inf = self.a_is_inf_and(a_is_inf, b_is_inf)  # 复用门
        
        # 结果是NaN的情况
        result_is_nan = self.result_is_nan_or(either_nan, zero_div_zero)
        result_is_nan = self.either_nan_or(result_is_nan, inf_div_inf)  # 复用门
        
        # 结果是Inf的情况: x/0 或 Inf/y (y!=0, y!=Inf)
        not_b_is_inf = self.not_b_is_inf(b_is_inf)
        inf_div_y = self.a_is_inf_and(a_is_inf, not_b_is_inf)  # 复用门
        result_is_inf = self.result_is_inf_or(div_by_zero, inf_div_y)
        
        # 结果是零的情况: 0/x (x!=0) 或 x/Inf (x!=Inf)
        not_b_zero = self.not_a_is_zero(b_is_zero)  # 复用门
        zero_div_x = self.a_is_zero_and(a_is_zero, not_b_zero)  # 复用门
        not_a_inf = self.not_b_is_inf(a_is_inf)  # 复用门
        x_div_inf = self.b_is_inf_and(not_a_inf, b_is_inf)  # 复用门
        result_is_zero = self.result_is_zero_or(zero_div_x, x_div_inf)
        
        # ===== 4. 指数处理 (10位防止溢出) =====
        e_a_le = e_a.flip(-1)  # 转LSB first
        e_b_le = e_b.flip(-1)

        # Ea - Eb: A + ~B + 1 (10位计算)
        e_b_not = []
        for i in range(8):
            e_b_not.append(self.exp_not(e_b_le[..., i:i+1]))
        e_b_not = torch.cat(e_b_not, dim=-1)

        # 扩展到10位 (符号扩展)
        e_a_10 = torch.cat([e_a_le, zeros, zeros], dim=-1)  # 正数扩展
        e_b_not_10 = torch.cat([e_b_not, ones, ones], dim=-1)  # 负数补码扩展

        c = ones  # 初始进位=1 (补码减法)
        exp_diff = []
        for i in range(10):
            s, c = self.exp_sub(e_a_10[..., i:i+1], e_b_not_10[..., i:i+1], c)
            exp_diff.append(s)
        exp_diff = torch.cat(exp_diff, dim=-1)
        
        # + 127 (10位)
        const_127 = torch.cat([ones, ones, ones, ones, ones, ones, ones, zeros, zeros, zeros], dim=-1)
        exp_result_10, _ = self.exp_add_bias(exp_diff, const_127)
        
        # 保存10位指数用于溢出/下溢检测
        exp_for_check = exp_result_10
        
        # ===== 5. SNN原生逐位尾数除法 =====
        # 构建24位尾数 (带隐藏位) - MSB first
        dividend_msb = torch.cat([ones, m_a], dim=-1)  # [24] MSB first: [1, M22, ..., M0]
        divisor_le = torch.cat([m_b.flip(-1), ones], dim=-1)  # [24] LSB first: [M0, ..., M22, 1]
        
        # 扩展除数到25位用于比较 (LSB first, 高位补0)
        D_25 = torch.cat([divisor_le, zeros], dim=-1)  # [25] LSB first
        
        # ===== 逐位除法: (dividend << 25) / divisor =====
        # 虚拟49位被除数 = dividend || 00...0 (24位 + 25个0)
        # R初始 = 0 (余数寄存器，25位足够因为R<D恒成立)
        # 每次迭代: 取被除数1位加入R，比较，产生商位
        
        R = torch.zeros(batch_shape + (25,), device=device)  # 25位余数 LSB first
        
        Q_bits = []
        for i in range(49):
            # 取虚拟被除数的第i位 (MSB first)
            # i < 24: 取 dividend_msb[i]
            # i >= 24: 取 0
            if i < 24:
                bit = dividend_msb[..., i:i+1]
            else:
                bit = zeros
            
            # R左移1位，新位加入最低位
            # LSB first: R[0]是最低位，左移=向高位移动
            R = torch.cat([bit, R[..., :-1]], dim=-1)
            
            # 比较 R 与 D，决定商位 (IF神经元: 累积-阈值-复位)
            q_bit, R = self.div_iteration(R, D_25)
            Q_bits.append(q_bit)
        
        # 商: Q[0..48]，Q[0]是最高位商 (MSB first)
        Q = torch.cat(Q_bits, dim=-1)  # [49] MSB first
        
        # ===== 判断 mant_a >= mant_b =====
        # 使用25位减法器比较
        mant_a_25 = torch.cat([m_a.flip(-1), ones, zeros], dim=-1)  # [25] LSB first
        mant_b_25 = torch.cat([m_b.flip(-1), ones, zeros], dim=-1)  # [25] LSB first
        _, borrow = self.mant_cmp_sub(mant_a_25, mant_b_25)
        mant_a_ge_b = self.q0_not(borrow)  # borrow=0 → mant_a >= mant_b
        
        # ===== 商位提取 =====
        # 49位商 Q = (mant_a << 25) / mant_b，以MSB first存储
        # Q[i] 对应 bit (48-i)
        # 
        # Q = 0x3000000 示例 (26位有效):
        # - 首个1在bit25 → Q[23]
        # - bit24 → Q[24]
        # - ...
        # 
        # 如果 mant_a >= mant_b (商>=1):
        #   隐藏位在Q[23] (bit25)
        #   23位尾数在Q[24:47] (bit24-bit2)
        #   round在Q[47] (bit1)
        #   sticky = Q[48] (bit0) | 余数
        #
        # 如果 mant_a < mant_b (商<1):
        #   隐藏位在Q[24] (bit24)
        #   23位尾数在Q[25:48] (bit23-bit1)
        #   round在Q[48] (bit0)
        #   sticky = 余数
        #   指数需要-1
        
        # 检查余数是否非零
        # 使用向量化 OR 树检测余数是否非零
        remainder_nonzero = self.vec_or_tree(R)
        
        # 正常路径 (mant_a >= mant_b)
        mant_normal = Q[..., 24:47]  # 23位尾数 (bit24-bit2)
        round_normal = Q[..., 47:48]  # bit1
        sticky_normal = self.rne_or(Q[..., 48:49], remainder_nonzero)  # bit0 | 余数
        
        # 归一化路径 (mant_a < mant_b)
        mant_shifted = Q[..., 25:48]  # 23位尾数 (bit23-bit1)
        round_shifted = Q[..., 48:49]  # bit0
        sticky_shifted = remainder_nonzero  # 只有余数
        
        # 使用MUX选择尾数 - 向量化
        mant_a_ge_b_23 = mant_a_ge_b.expand_as(mant_normal)
        mant_23 = self.normalize_mux(mant_a_ge_b_23, mant_normal, mant_shifted)
        
        # 选择round (用简化逻辑: MUX)
        # round_bit = mant_a_ge_b ? round_normal : round_shifted
        round_bit_normal_sel = self.rne_and(mant_a_ge_b, round_normal)
        round_bit_shifted_sel = self.rne_and(self.q0_not(mant_a_ge_b), round_shifted)
        round_bit = self.rne_or(round_bit_normal_sel, round_bit_shifted_sel)
        
        # 选择sticky
        sticky_normal_sel = self.rne_and(mant_a_ge_b, sticky_normal)
        sticky_shifted_sel = self.rne_and(self.q0_not(mant_a_ge_b), sticky_shifted)
        sticky_bit = self.rne_or(sticky_normal_sel, sticky_shifted_sel)
        
        # ===== 6. RNE舍入 =====
        lsb = mant_23[..., 22:23]  # 尾数最低位 (MSB first, 索引22)
        s_or_l = self.rne_or(sticky_bit, lsb)
        round_up = self.rne_and(round_bit, s_or_l)
        
        # 尾数+1 (如果需要舍入)
        mant_23_le = mant_23.flip(-1)  # 转LSB first
        round_inc = torch.cat([round_up] + [zeros]*22, dim=-1)
        mant_rounded_le, carry = self.round_adder(mant_23_le, round_inc)
        mant_final = mant_rounded_le.flip(-1)  # 转回MSB first
        
        # ===== 归一化: 指数调整 (10位) =====
        need_normalize = self.q0_not(mant_a_ge_b)  # mant_a < mant_b 时需要指数-1
        
        # 指数调整: 如果需要归一化，指数-1 (10位计算)
        # exp - 1 = exp + NOT(1) + 1
        const_1_10 = torch.cat([ones] + [zeros]*9, dim=-1)  # 0x001 (10位)
        not_1_10 = []
        for i in range(10):
            not_1_10.append(self.exp_adjust_not(const_1_10[..., i:i+1]))
        not_1_10 = torch.cat(not_1_10, dim=-1)  # NOT(1) (10位)

        # exp + NOT(1) + 1
        c = ones  # +1
        exp_minus1_10 = []
        for i in range(10):
            s, c = self.exp_sub(exp_result_10[..., i:i+1], not_1_10[..., i:i+1], c)
            exp_minus1_10.append(s)
        exp_minus1_10 = torch.cat(exp_minus1_10, dim=-1)

        # 选择10位指数 - 向量化 (低8位)
        need_normalize_8 = need_normalize.expand(*batch_shape, 8)
        exp_minus1_low8 = exp_minus1_10[..., :8]
        exp_result_low8 = exp_result_10[..., :8]
        exp_adjusted_low8 = self.normalize_exp_mux(need_normalize_8, exp_minus1_low8, exp_result_low8)

        # 高2位: 使用简单逻辑
        high2_bits = []
        for i in range(8, 10):
            bit = self.underflow_and(need_normalize, exp_minus1_10[..., i:i+1])
            not_norm = self.underflow_not(need_normalize)
            bit2 = self.underflow_and(not_norm, exp_result_10[..., i:i+1])
            bit = self.underflow_or(bit, bit2)
            high2_bits.append(bit)
        exp_adjusted_10 = torch.cat([exp_adjusted_low8] + high2_bits, dim=-1)
        
        # ===== 溢出/下溢检测 =====
        # 溢出: 10位有符号值 >= 255
        # 即: bit9=0 (正数) 且 低9位>=255 (bit8=1 或 低8位全1)
        exp_bit9 = exp_adjusted_10[..., 9:10]  # 符号位
        exp_bit8 = exp_adjusted_10[..., 8:9]
        
        # 检测低8位全1
        exp_all_255 = exp_adjusted_10[..., 0:1]
        for i in range(1, 8):
            exp_all_255 = self.overflow_check_and(exp_all_255, exp_adjusted_10[..., i:i+1])
        
        # low9 >= 255: bit8=1 或 低8位全1
        low9_ge_255 = self.overflow_or(exp_bit8, exp_all_255)
        
        # 溢出: bit9=0 且 low9>=255
        not_bit9 = self.overflow_not(exp_bit9)
        is_overflow = self.overflow_and(not_bit9, low9_ge_255)
        
        # 下溢: 10位有符号值 <= 0
        # bit9=1 (负数) 或 全10位=0
        not_bits = []
        for i in range(10):
            not_bits.append(self.underflow_not(exp_adjusted_10[..., i:i+1]))

        exp_is_zero = not_bits[0]
        for i in range(1, 10):
            exp_is_zero = self.underflow_and(exp_is_zero, not_bits[i])

        is_underflow_or_subnormal = self.underflow_or(exp_bit9, exp_is_zero)

        # ===== Subnormal输出处理 =====
        # 当exp <= 0 但 > -150时，产生subnormal
        # 计算移位量: shift = 1 - exp = ~exp + 1 + 1 = ~exp + 2

        # 取反exp_adjusted_10
        exp_neg = []
        for i in range(10):
            exp_neg.append(self.underflow_not(exp_adjusted_10[..., i:i+1]))
        exp_neg_10 = torch.cat(exp_neg, dim=-1)

        # 加2得到移位量 (因为 1 - exp = ~exp + 2 对于补码)
        const_2_10 = torch.cat([zeros, ones] + [zeros]*8, dim=-1)  # 0x002
        shift_full_10, _ = self.shift_add_one(exp_neg_10, const_2_10)

        # 检测移位量是否过大 (>= 32)
        shift_bit5_or_higher = shift_full_10[..., 5:6]
        for i in range(6, 10):
            shift_bit5_or_higher = self.subnorm_sticky_or(shift_bit5_or_higher, shift_full_10[..., i:i+1])

        # 移位量合理则是subnormal，否则完全下溢
        is_reasonable_shift = self.not_very_underflow(shift_bit5_or_higher)
        is_subnormal = self.is_subnormal_and(is_underflow_or_subnormal, is_reasonable_shift)
        is_underflow = self.underflow_and(is_underflow_or_subnormal, shift_bit5_or_higher)
        
        # 构建48位尾数用于subnormal移位 (使用舍入前的数据)
        # 从mant_23 (MSB first, 23位舍入前) 扩展到48位
        # 格式: [1, mant_23, round_bit, sticky_bit, 0...0]
        # 注意：sticky_bit作为单独一位，其他位填充0
        mant_48_msb = torch.cat([ones, mant_23, round_bit, sticky_bit] + [zeros]*22, dim=-1)
        
        # 移位
        shift_final = shift_full_10[..., :6]  # 低6位
        shift_final_be = shift_final.flip(-1)  # MSB first
        shifted_mant, shift_sticky = self.subnormal_shifter(mant_48_msb, shift_final_be)
        
        # 提取subnormal尾数
        subnorm_mant = shifted_mant[..., 1:24]  # 23位
        subnorm_round = shifted_mant[..., 24:25]
        
        # Sticky
        subnorm_sticky = shifted_mant[..., 25:26]
        for i in range(26, 48):
            if i - 26 < 22:
                subnorm_sticky = self.subnorm_sticky_or(subnorm_sticky, shifted_mant[..., i:i+1])
        subnorm_sticky = self.subnorm_sticky_or(subnorm_sticky, shift_sticky)
        
        # RNE舍入
        subnorm_lsb = subnorm_mant[..., 22:23]
        s_or_l_sub = self.subnorm_rne_or(subnorm_sticky, subnorm_lsb)
        round_up_sub = self.subnorm_rne_and(subnorm_round, s_or_l_sub)
        
        # 尾数+1
        subnorm_mant_le = subnorm_mant.flip(-1)
        subnorm_mant_24_le = torch.cat([subnorm_mant_le, zeros], dim=-1)
        round_inc_sub = torch.cat([round_up_sub] + [zeros]*23, dim=-1)
        subnorm_mant_rounded, _ = self.subnorm_round_adder(subnorm_mant_24_le, round_inc_sub)
        subnorm_mant_final = subnorm_mant_rounded[..., :23].flip(-1)
        
        # 取低8位作为最终指数
        exp_final_le = exp_adjusted_10[..., :8]
        exp_final = exp_final_le.flip(-1)  # 转MSB first
        
        # ===== 7. 输出选择 =====
        # NaN: E=FF, M=非零
        nan_exp = torch.cat([ones]*8, dim=-1)
        nan_mant = torch.cat([ones] + [zeros]*22, dim=-1)
        
        # Inf: E=FF, M=0
        inf_exp = torch.cat([ones]*8, dim=-1)
        inf_mant = torch.cat([zeros]*23, dim=-1)
        
        # Zero: E=0, M=0
        zero_exp = torch.cat([zeros]*8, dim=-1)
        zero_mant = torch.cat([zeros]*23, dim=-1)
        
        # 先应用NaN (向量化)
        result_is_nan_e = result_is_nan.expand(*batch_shape, 8)
        result_is_nan_m = result_is_nan.expand(*batch_shape, 23)
        e_out = self.vec_mux_exp(result_is_nan_e, nan_exp, exp_final)
        m_out = self.vec_mux_mant(result_is_nan_m, nan_mant, mant_final)
        
        # 应用Inf (向量化)
        not_nan = self.not_result_is_nan(result_is_nan)
        inf_and_not_nan = self.inf_and_not_nan(result_is_inf, not_nan)
        inf_and_not_nan_e = inf_and_not_nan.expand(*batch_shape, 8)
        inf_and_not_nan_m = inf_and_not_nan.expand(*batch_shape, 23)
        e_out = self.vec_mux_exp(inf_and_not_nan_e, inf_exp, e_out)
        m_out = self.vec_mux_mant(inf_and_not_nan_m, inf_mant, m_out)
        
        # 应用Zero (向量化)
        not_inf = self.not_result_is_inf(result_is_inf)
        zero_and_not_nan = self.zero_and_not_nan(result_is_zero, not_nan)
        zero_and_not_inf = self.zero_and_not_inf(zero_and_not_nan, not_inf)
        zero_and_not_inf_e = zero_and_not_inf.expand(*batch_shape, 8)
        zero_and_not_inf_m = zero_and_not_inf.expand(*batch_shape, 23)
        e_out = self.vec_mux_exp(zero_and_not_inf_e, zero_exp, e_out)
        m_out = self.vec_mux_mant(zero_and_not_inf_m, zero_mant, m_out)
        
        # 应用计算溢出 (exp >= 255 → Inf) (向量化)
        overflow_and_valid = self.overflow_and(is_overflow, not_nan)
        overflow_e = overflow_and_valid.expand(*batch_shape, 8)
        overflow_m = overflow_and_valid.expand(*batch_shape, 23)
        e_out = self.vec_mux_exp(overflow_e, inf_exp, e_out)
        m_out = self.vec_mux_mant(overflow_m, inf_mant, m_out)
        
        # 应用Subnormal (exp <= 0 但 > -150，且非零输入结果) (向量化)
        not_overflow = self.overflow_not(is_overflow)
        not_result_zero = self.overflow_not(result_is_zero)  # 复用
        subnorm_temp = self.is_subnormal_and(is_subnormal, not_nan)
        subnorm_and_valid = self.underflow_and(subnorm_temp, not_result_zero)  # 复用
        subnorm_e = subnorm_and_valid.expand(*batch_shape, 8)
        subnorm_m = subnorm_and_valid.expand(*batch_shape, 23)
        e_out = self.vec_mux_exp(subnorm_e, zero_exp, e_out)
        m_out = self.vec_mux_mant(subnorm_m, subnorm_mant_final, m_out)

        # 应用完全下溢 (exp < -150 → 0，且非零输入结果) (向量化)
        underflow_and_valid = self.underflow_and(is_underflow, not_nan)
        underflow_temp = self.underflow_and(underflow_and_valid, not_overflow)
        underflow_and_not_overflow = self.underflow_and(underflow_temp, not_result_zero)  # 复用
        underflow_e = underflow_and_not_overflow.expand(*batch_shape, 8)
        underflow_m = underflow_and_not_overflow.expand(*batch_shape, 23)
        e_out = self.vec_mux_exp(underflow_e, zero_exp, e_out)
        m_out = self.vec_mux_mant(underflow_m, zero_mant, m_out)
        
        # 组装输出
        result = torch.cat([s_out, e_out, m_out], dim=-1)
        
        return result
    
    def reset(self):
        self.sign_xor.reset()
        self.exp_sub.reset()
        self.exp_not.reset()
        self.exp_add_bias.reset()
        self.exp_adjust.reset()
        self.exp_all_one_and_a.reset()
        self.exp_all_one_and_b.reset()
        self.exp_zero_or_a.reset()
        self.exp_zero_or_b.reset()
        self.exp_zero_not_a.reset()
        self.exp_zero_not_b.reset()
        self.mant_zero_or_a.reset()
        self.mant_zero_or_b.reset()
        self.mant_zero_not_a.reset()
        self.mant_zero_not_b.reset()
        self.a_is_zero_and.reset()
        self.b_is_zero_and.reset()
        self.a_is_inf_and.reset()
        self.b_is_inf_and.reset()
        self.a_mant_nonzero_not.reset()
        self.b_mant_nonzero_not.reset()
        self.a_is_nan_and.reset()
        self.b_is_nan_and.reset()
        self.either_nan_or.reset()
        self.div_by_zero_and.reset()
        self.zero_div_zero_and.reset()
        self.result_is_nan_or.reset()
        self.result_is_inf_or.reset()
        self.result_is_zero_or.reset()
        self.div_iteration.reset()
        self.mant_cmp_sub.reset()
        self.vec_or_tree.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.round_adder.reset()
        self.vec_mux_exp.reset()
        self.vec_mux_mant.reset()
        # 溢出/下溢检测
        self.overflow_check_and.reset()
        self.overflow_or.reset()
        self.overflow_and.reset()
        self.overflow_not.reset()
        self.underflow_not.reset()
        self.underflow_and.reset()
        self.underflow_or.reset()
        self.overflow_mux_e.reset()
        self.overflow_mux_m.reset()
        self.underflow_mux_e.reset()
        self.underflow_mux_m.reset()
        # Subnormal处理
        self.subnormal_shifter.reset()
        self.shift_add_one.reset()
        self.subnorm_sticky_or.reset()
        self.subnorm_rne_or.reset()
        self.subnorm_rne_and.reset()
        self.subnorm_round_adder.reset()
        self.subnorm_mux_m.reset()
        self.is_subnormal_and.reset()
        self.not_very_underflow.reset()
        self.not_a_is_zero.reset()
        self.not_b_is_inf.reset()
        self.not_result_is_nan.reset()
        self.not_result_is_inf.reset()
        self.inf_and_not_nan.reset()
        self.zero_and_not_nan.reset()
        self.zero_and_not_inf.reset()
        # 归一化相关
        self.exp_adjust_not.reset()
        self.normalize_mux.reset()
        self.normalize_exp_mux.reset()
        self.q0_not.reset()

