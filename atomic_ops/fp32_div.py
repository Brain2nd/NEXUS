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

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate,
                          HalfAdder, FullAdder, RippleCarryAdder, ORTree)
from .vec_logic_gates import (
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
    def __init__(self):
        super().__init__()
        self.bits = 25
        self.not_gates = nn.ModuleList([NOTGate() for _ in range(25)])
        self.adders = nn.ModuleList([FullAdder() for _ in range(25)])
        self.borrow_not = NOTGate()
        
    def forward(self, A, B):
        """A - B, LSB first
        Returns: (差, 借位符号)
        借位符号=0表示A>=B, 借位符号=1表示A<B
        """
        # ~B
        not_b = []
        for i in range(self.bits):
            not_b.append(self.not_gates[i](B[..., i:i+1]))
        not_b = torch.cat(not_b, dim=-1)
        
        # A + ~B + 1
        c = torch.ones_like(A[..., 0:1])
        sum_bits = []
        for i in range(self.bits):
            s, c = self.adders[i](A[..., i:i+1], not_b[..., i:i+1], c)
            sum_bits.append(s)
        
        result = torch.cat(sum_bits, dim=-1)
        borrow = self.borrow_not(c)  # c=1→borrow=0 (A>=B)
        
        return result, borrow
    
    def reset(self):
        for g in self.not_gates: g.reset()
        for g in self.adders: g.reset()
        self.borrow_not.reset()


# ==============================================================================
# 48位减法器 (保留用于兼容)
# ==============================================================================
class Subtractor48Bit(nn.Module):
    """48位减法器 - 纯SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.bits = 48
        self.not_gates = nn.ModuleList([NOTGate() for _ in range(48)])
        self.adders = nn.ModuleList([FullAdder() for _ in range(48)])
        
    def forward(self, A, B):
        not_b = []
        for i in range(self.bits):
            not_b.append(self.not_gates[i](B[..., i:i+1]))
        not_b = torch.cat(not_b, dim=-1)
        
        c = torch.ones_like(A[..., 0:1])
        sum_bits = []
        for i in range(self.bits):
            s, c = self.adders[i](A[..., i:i+1], not_b[..., i:i+1], c)
            sum_bits.append(s)
        
        result = torch.cat(sum_bits, dim=-1)
        borrow = self.not_gates[0](c)
        return result, borrow
    
    def reset(self):
        for g in self.not_gates: g.reset()
        for g in self.adders: g.reset()


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
    def __init__(self):
        super().__init__()
        self.sub = Subtractor25Bit()
        self.not_borrow = NOTGate()
        self.mux_r = nn.ModuleList([MUXGate() for _ in range(25)])
        
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
        
        # 选择余数: borrow=0时R_trial, borrow=1时R(恢复)
        R_next = []
        for i in range(25):
            r_bit = self.mux_r[i](borrow, R[..., i:i+1], R_trial[..., i:i+1])
            R_next.append(r_bit)
        R_next = torch.cat(R_next, dim=-1)
        
        return Q_bit, R_next
    
    def reset(self):
        self.sub.reset()
        self.not_borrow.reset()
        for mux in self.mux_r: mux.reset()


# ==============================================================================
# 旧版48位除法迭代单元 (保留兼容)
# ==============================================================================
class DivisionIteration(nn.Module):
    """48位除法迭代 - 恢复余数算法"""
    def __init__(self):
        super().__init__()
        self.sub = Subtractor48Bit()
        self.not_borrow = NOTGate()
        self.mux_r = nn.ModuleList([MUXGate() for _ in range(48)])
        
    def forward(self, R, D_extended):
        """
        R: [..., 48] 当前余数 (LSB first)
        D_extended: [..., 48] 扩展的除数 (LSB first)
        Returns: Q_bit [..., 1], R_next [..., 48]
        """
        # 尝试减法
        R_trial, borrow = self.sub(R, D_extended)
        
        # 如果无借位(borrow=0)，使用R_trial; 否则恢复R
        # Q_bit = NOT(borrow) = 1 当 A>=B
        # borrow=0时表示成功(A>=B)，borrow=1时表示失败(A<B)
        # Q_bit = NOT(borrow) - 使用纯SNN NOT门
        Q_bit = self.not_borrow(borrow)
        
        # 选择余数: borrow=0时选R_trial, borrow=1时选R
        R_next = []
        for i in range(48):
            r_bit = self.mux_r[i](borrow, R[..., i:i+1], R_trial[..., i:i+1])
            R_next.append(r_bit)
        R_next = torch.cat(R_next, dim=-1)
        
        return Q_bit, R_next
    
    def reset(self):
        self.sub.reset()
        self.not_borrow.reset()
        for mux in self.mux_r:
            mux.reset()


# ==============================================================================
# FP32 除法器主类
# ==============================================================================
class SpikeFP32Divider(nn.Module):
    """FP32 除法器 - 100%纯SNN门电路实现
    
    输入: A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0]
    输出: [..., 32] FP32脉冲 (A / B)
    
    使用恢复余数除法，26次迭代获得24位商+2位舍入。
    """
    def __init__(self):
        super().__init__()
        
        # ===== 符号 =====
        self.sign_xor = XORGate()
        
        # ===== 指数运算 (10位防止溢出) =====
        # 指数减法: Ea - Eb (10位)
        self.exp_sub = nn.ModuleList([FullAdder() for _ in range(10)])
        self.exp_not = nn.ModuleList([NOTGate() for _ in range(10)])
        # 加bias: + 127 (10位)
        self.exp_add_bias = RippleCarryAdder(bits=10)
        # 指数调整 (归一化时减1) (10位)
        self.exp_adjust = RippleCarryAdder(bits=10)
        self.exp_adjust_not = nn.ModuleList([NOTGate() for _ in range(10)])
        
        # ===== 溢出/下溢检测 =====
        # 溢出: 10位有符号 >= 255
        self.overflow_check_and = nn.ModuleList([ANDGate() for _ in range(8)])
        self.overflow_or = ORGate()
        self.overflow_and = ANDGate()
        self.overflow_not = NOTGate()
        # 下溢: 10位有符号 <= 0
        self.underflow_not = nn.ModuleList([NOTGate() for _ in range(10)])
        self.underflow_and = nn.ModuleList([ANDGate() for _ in range(9)])
        self.underflow_or = ORGate()
        # 溢出/下溢结果MUX
        self.overflow_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.overflow_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        self.underflow_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.underflow_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        
        # ===== Subnormal输出处理 =====
        # 需要对尾数进行额外右移
        from .fp32_mul import BarrelShifterRight48
        self.subnormal_shifter = BarrelShifterRight48()
        self.shift_add_one = RippleCarryAdder(bits=10)
        self.subnorm_sticky_or = nn.ModuleList([ORGate() for _ in range(24)])
        self.subnorm_rne_or = ORGate()
        self.subnorm_rne_and = ANDGate()
        self.subnorm_round_adder = RippleCarryAdder(bits=24)
        self.subnorm_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        self.is_subnormal_and = ANDGate()
        self.not_very_underflow = NOTGate()
        
        # ===== 归一化 =====
        # 当Q0=0时需要归一化: 尾数左移1位, 指数-1
        self.normalize_mux = nn.ModuleList([MUXGate() for _ in range(23)])  # 尾数选择
        self.normalize_exp_mux = nn.ModuleList([MUXGate() for _ in range(8)])  # 指数选择
        self.q0_not = NOTGate()  # Q0=0检测
        
        # ===== 特殊值检测 =====
        # 指数全1检测 (Inf/NaN)
        self.exp_all_one_and_a = nn.ModuleList([ANDGate() for _ in range(7)])
        self.exp_all_one_and_b = nn.ModuleList([ANDGate() for _ in range(7)])
        
        # 指数全0检测 (Zero/Subnormal)
        self.exp_zero_or_a = nn.ModuleList([ORGate() for _ in range(7)])
        self.exp_zero_or_b = nn.ModuleList([ORGate() for _ in range(7)])
        self.exp_zero_not_a = NOTGate()
        self.exp_zero_not_b = NOTGate()
        
        # 尾数全0检测
        self.mant_zero_or_a = nn.ModuleList([ORGate() for _ in range(22)])
        self.mant_zero_or_b = nn.ModuleList([ORGate() for _ in range(22)])
        self.mant_zero_not_a = NOTGate()
        self.mant_zero_not_b = NOTGate()
        
        # 零检测: E=0 AND M=0
        self.a_is_zero_and = ANDGate()
        self.b_is_zero_and = ANDGate()
        
        # Inf检测: E=FF AND M=0
        self.a_is_inf_and = ANDGate()
        self.b_is_inf_and = ANDGate()
        
        # NaN检测: E=FF AND M!=0
        self.a_mant_nonzero_not = NOTGate()
        self.b_mant_nonzero_not = NOTGate()
        self.a_is_nan_and = ANDGate()
        self.b_is_nan_and = ANDGate()
        self.either_nan_or = ORGate()
        
        # x/0 = Inf, 0/0 = NaN
        self.div_by_zero_and = ANDGate()  # a_not_zero AND b_is_zero
        self.zero_div_zero_and = ANDGate()  # a_is_zero AND b_is_zero
        
        # 结果特殊值组合
        self.result_is_nan_or = ORGate()
        self.result_is_inf_or = ORGate()
        self.result_is_zero_or = ORGate()
        
        # ===== SNN原生逐位除法 (49次迭代) =====
        # 使用25位迭代单元，符合IF神经元动力学
        self.div_iterations = nn.ModuleList([DivisionIteration25() for _ in range(49)])
        
        # ===== 比较器 (mant_a >= mant_b) =====
        self.mant_cmp_sub = Subtractor25Bit()  # 24位比较用25位减法器
        
        # ===== Sticky OR树 (检测25位余数是否非零) - 向量化 =====
        self.vec_or_tree = VecORTree()
        
        # ===== RNE舍入 =====
        self.rne_or = ORGate()
        self.rne_and = ANDGate()
        self.round_adder = RippleCarryAdder(bits=24)
        
        # ===== 输出选择 - 向量化 =====
        self.vec_mux = VecMUX()
        
        # 纯SNN NOT门
        self.not_a_is_zero = NOTGate()
        self.not_b_is_inf = NOTGate()
        self.not_result_is_nan = NOTGate()
        self.not_result_is_inf = NOTGate()
        self.inf_and_not_nan = ANDGate()
        self.zero_and_not_nan = ANDGate()
        self.zero_and_not_inf = ANDGate()
        
    def forward(self, A, B):
        """
        A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0] MSB first
        Returns: [..., 32] FP32脉冲 (A / B)
        """
        self.reset()
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
            e_a_all_one = self.exp_all_one_and_a[i-1](e_a_all_one, e_a[..., i:i+1])
        
        e_b_all_one = e_b[..., 0:1]
        for i in range(1, 8):
            e_b_all_one = self.exp_all_one_and_b[i-1](e_b_all_one, e_b[..., i:i+1])
        
        # 指数全0检测
        e_a_any_one = e_a[..., 0:1]
        for i in range(1, 8):
            e_a_any_one = self.exp_zero_or_a[i-1](e_a_any_one, e_a[..., i:i+1])
        e_a_is_zero = self.exp_zero_not_a(e_a_any_one)
        
        e_b_any_one = e_b[..., 0:1]
        for i in range(1, 8):
            e_b_any_one = self.exp_zero_or_b[i-1](e_b_any_one, e_b[..., i:i+1])
        e_b_is_zero = self.exp_zero_not_b(e_b_any_one)
        
        # 尾数全0检测
        m_a_any_one = m_a[..., 0:1]
        for i in range(1, 23):
            m_a_any_one = self.mant_zero_or_a[i-1](m_a_any_one, m_a[..., i:i+1])
        m_a_is_zero = self.mant_zero_not_a(m_a_any_one)
        m_a_nonzero = m_a_any_one
        
        m_b_any_one = m_b[..., 0:1]
        for i in range(1, 23):
            m_b_any_one = self.mant_zero_or_b[i-1](m_b_any_one, m_b[..., i:i+1])
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
            e_b_not.append(self.exp_not[i](e_b_le[..., i:i+1]))
        e_b_not = torch.cat(e_b_not, dim=-1)
        
        # 扩展到10位 (符号扩展)
        e_a_10 = torch.cat([e_a_le, zeros, zeros], dim=-1)  # 正数扩展
        e_b_not_10 = torch.cat([e_b_not, ones, ones], dim=-1)  # 负数补码扩展
        
        c = ones  # 初始进位=1 (补码减法)
        exp_diff = []
        for i in range(10):
            s, c = self.exp_sub[i](e_a_10[..., i:i+1], e_b_not_10[..., i:i+1], c)
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
            q_bit, R = self.div_iterations[i](R, D_25)
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
        
        # 使用MUX选择尾数
        mant_23_list = []
        for i in range(23):
            bit = self.normalize_mux[i](mant_a_ge_b, mant_normal[..., i:i+1], mant_shifted[..., i:i+1])
            mant_23_list.append(bit)
        mant_23 = torch.cat(mant_23_list, dim=-1)  # MSB first
        
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
            not_1_10.append(self.exp_adjust_not[i](const_1_10[..., i:i+1]))
        not_1_10 = torch.cat(not_1_10, dim=-1)  # NOT(1) (10位)
        
        # exp + NOT(1) + 1
        c = ones  # +1
        exp_minus1_10 = []
        for i in range(10):
            s, c = self.exp_sub[i](exp_result_10[..., i:i+1], not_1_10[..., i:i+1], c)
            exp_minus1_10.append(s)
        exp_minus1_10 = torch.cat(exp_minus1_10, dim=-1)
        
        # 选择10位指数
        exp_adjusted_10 = []
        for i in range(10):
            if i < 8:
                bit = self.normalize_exp_mux[i](need_normalize,
                                                exp_minus1_10[..., i:i+1],
                                                exp_result_10[..., i:i+1])
            else:
                # 高2位: 使用简单逻辑
                bit = self.underflow_and[i-8](need_normalize, exp_minus1_10[..., i:i+1])
                not_norm = self.underflow_not[i](need_normalize)
                bit2 = self.underflow_and[i-8](not_norm, exp_result_10[..., i:i+1])
                bit = self.underflow_or(bit, bit2)
            exp_adjusted_10.append(bit)
        exp_adjusted_10 = torch.cat(exp_adjusted_10, dim=-1)
        
        # ===== 溢出/下溢检测 =====
        # 溢出: 10位有符号值 >= 255
        # 即: bit9=0 (正数) 且 低9位>=255 (bit8=1 或 低8位全1)
        exp_bit9 = exp_adjusted_10[..., 9:10]  # 符号位
        exp_bit8 = exp_adjusted_10[..., 8:9]
        
        # 检测低8位全1
        exp_all_255 = exp_adjusted_10[..., 0:1]
        for i in range(1, 8):
            exp_all_255 = self.overflow_check_and[i-1](exp_all_255, exp_adjusted_10[..., i:i+1])
        
        # low9 >= 255: bit8=1 或 低8位全1
        low9_ge_255 = self.overflow_or(exp_bit8, exp_all_255)
        
        # 溢出: bit9=0 且 low9>=255
        not_bit9 = self.overflow_not(exp_bit9)
        is_overflow = self.overflow_and(not_bit9, low9_ge_255)
        
        # 下溢: 10位有符号值 <= 0
        # bit9=1 (负数) 或 全10位=0
        not_bits = []
        for i in range(10):
            not_bits.append(self.underflow_not[i](exp_adjusted_10[..., i:i+1]))
        
        exp_is_zero = not_bits[0]
        for i in range(1, 10):
            exp_is_zero = self.underflow_and[min(i-1, 7)](exp_is_zero, not_bits[i])
        
        is_underflow_or_subnormal = self.underflow_or(exp_bit9, exp_is_zero)
        
        # ===== Subnormal输出处理 =====
        # 当exp <= 0 但 > -150时，产生subnormal
        # 计算移位量: shift = 1 - exp = ~exp + 1 + 1 = ~exp + 2
        
        # 取反exp_adjusted_10
        exp_neg = []
        for i in range(10):
            exp_neg.append(self.underflow_not[i](exp_adjusted_10[..., i:i+1]))
        exp_neg_10 = torch.cat(exp_neg, dim=-1)
        
        # 加2得到移位量 (因为 1 - exp = ~exp + 2 对于补码)
        const_2_10 = torch.cat([zeros, ones] + [zeros]*8, dim=-1)  # 0x002
        shift_full_10, _ = self.shift_add_one(exp_neg_10, const_2_10)
        
        # 检测移位量是否过大 (>= 32)
        shift_bit5_or_higher = shift_full_10[..., 5:6]
        for i in range(6, 10):
            shift_bit5_or_higher = self.subnorm_sticky_or[i-6](shift_bit5_or_higher, shift_full_10[..., i:i+1])
        
        # 移位量合理则是subnormal，否则完全下溢
        is_reasonable_shift = self.not_very_underflow(shift_bit5_or_higher)
        is_subnormal = self.is_subnormal_and(is_underflow_or_subnormal, is_reasonable_shift)
        is_underflow = self.underflow_and[8](is_underflow_or_subnormal, shift_bit5_or_higher)
        
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
                subnorm_sticky = self.subnorm_sticky_or[i-26](subnorm_sticky, shifted_mant[..., i:i+1])
        subnorm_sticky = self.subnorm_sticky_or[22](subnorm_sticky, shift_sticky)
        
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
        e_out = self.vec_mux(result_is_nan_e, nan_exp, exp_final)
        m_out = self.vec_mux(result_is_nan_m, nan_mant, mant_final)
        
        # 应用Inf (向量化)
        not_nan = self.not_result_is_nan(result_is_nan)
        inf_and_not_nan = self.inf_and_not_nan(result_is_inf, not_nan)
        inf_and_not_nan_e = inf_and_not_nan.expand(*batch_shape, 8)
        inf_and_not_nan_m = inf_and_not_nan.expand(*batch_shape, 23)
        e_out = self.vec_mux(inf_and_not_nan_e, inf_exp, e_out)
        m_out = self.vec_mux(inf_and_not_nan_m, inf_mant, m_out)
        
        # 应用Zero (向量化)
        not_inf = self.not_result_is_inf(result_is_inf)
        zero_and_not_nan = self.zero_and_not_nan(result_is_zero, not_nan)
        zero_and_not_inf = self.zero_and_not_inf(zero_and_not_nan, not_inf)
        zero_and_not_inf_e = zero_and_not_inf.expand(*batch_shape, 8)
        zero_and_not_inf_m = zero_and_not_inf.expand(*batch_shape, 23)
        e_out = self.vec_mux(zero_and_not_inf_e, zero_exp, e_out)
        m_out = self.vec_mux(zero_and_not_inf_m, zero_mant, m_out)
        
        # 应用计算溢出 (exp >= 255 → Inf) (向量化)
        overflow_and_valid = self.overflow_and(is_overflow, not_nan)
        overflow_e = overflow_and_valid.expand(*batch_shape, 8)
        overflow_m = overflow_and_valid.expand(*batch_shape, 23)
        e_out = self.vec_mux(overflow_e, inf_exp, e_out)
        m_out = self.vec_mux(overflow_m, inf_mant, m_out)
        
        # 应用Subnormal (exp <= 0 但 > -150，且非零输入结果) (向量化)
        not_overflow = self.overflow_not(is_overflow)
        not_result_zero = self.overflow_not(result_is_zero)  # 复用
        subnorm_temp = self.is_subnormal_and(is_subnormal, not_nan)
        subnorm_and_valid = self.underflow_and[4](subnorm_temp, not_result_zero)  # 复用
        subnorm_e = subnorm_and_valid.expand(*batch_shape, 8)
        subnorm_m = subnorm_and_valid.expand(*batch_shape, 23)
        e_out = self.vec_mux(subnorm_e, zero_exp, e_out)
        m_out = self.vec_mux(subnorm_m, subnorm_mant_final, m_out)
        
        # 应用完全下溢 (exp < -150 → 0，且非零输入结果) (向量化)
        underflow_and_valid = self.underflow_and[7](is_underflow, not_nan)
        underflow_temp = self.underflow_and[6](underflow_and_valid, not_overflow)
        underflow_and_not_overflow = self.underflow_and[5](underflow_temp, not_result_zero)  # 复用
        underflow_e = underflow_and_not_overflow.expand(*batch_shape, 8)
        underflow_m = underflow_and_not_overflow.expand(*batch_shape, 23)
        e_out = self.vec_mux(underflow_e, zero_exp, e_out)
        m_out = self.vec_mux(underflow_m, zero_mant, m_out)
        
        # 组装输出
        result = torch.cat([s_out, e_out, m_out], dim=-1)
        
        return result
    
    def reset(self):
        self.sign_xor.reset()
        for g in self.exp_sub: g.reset()
        for g in self.exp_not: g.reset()
        self.exp_add_bias.reset()
        self.exp_adjust.reset()
        for g in self.exp_all_one_and_a: g.reset()
        for g in self.exp_all_one_and_b: g.reset()
        for g in self.exp_zero_or_a: g.reset()
        for g in self.exp_zero_or_b: g.reset()
        self.exp_zero_not_a.reset()
        self.exp_zero_not_b.reset()
        for g in self.mant_zero_or_a: g.reset()
        for g in self.mant_zero_or_b: g.reset()
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
        for div in self.div_iterations: div.reset()
        self.mant_cmp_sub.reset()
        self.vec_or_tree.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.round_adder.reset()
        self.vec_mux.reset()
        # 溢出/下溢检测
        for g in self.overflow_check_and: g.reset()
        self.overflow_or.reset()
        self.overflow_and.reset()
        self.overflow_not.reset()
        for g in self.underflow_not: g.reset()
        for g in self.underflow_and: g.reset()
        self.underflow_or.reset()
        # overflow/underflow MUX 已合并到 vec_mux
        # Subnormal处理
        self.subnormal_shifter.reset()
        self.shift_add_one.reset()
        for g in self.subnorm_sticky_or: g.reset()
        self.subnorm_rne_or.reset()
        self.subnorm_rne_and.reset()
        self.subnorm_round_adder.reset()
        # subnorm_mux_m 已合并到 vec_mux
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
        for g in self.exp_adjust_not: g.reset()
        # normalize_mux 保留（未向量化部分）
        for mux in self.normalize_mux: mux.reset()
        for mux in self.normalize_exp_mux: mux.reset()
        self.q0_not.reset()

