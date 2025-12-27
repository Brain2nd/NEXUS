"""
FP32 乘法器 - 100%纯SNN门电路实现
======================================

FP32 格式: [S | E7..E0 | M22..M0], bias=127

核心算法:
1. 符号: Sr = Sa XOR Sb
2. 指数: Er = Ea + Eb - 127
3. 尾数: 24x24 阵列乘法 (隐藏位 + 23位尾数)
4. 规格化: LZD + 桶形移位
5. RNE舍入: Guard/Round/Sticky位处理

特殊情况:
- 零: 任一操作数为零 → 结果为零
- 无穷大: 非零 × Inf → Inf, 0 × Inf → NaN
- NaN: 任一为NaN → NaN
- Subnormal: 完整支持

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
# 48位加法器 (用于尾数乘积) - 向量化
# ==============================================================================
class RippleCarryAdder48Bit(nn.Module):
    """48位加法器 - 向量化SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.bits = 48
        self.vec_adder = VecAdder(48)
        
    def forward(self, A, B, Cin=None):
        """A + B, LSB first"""
        return self.vec_adder(A, B, Cin)
    
    def reset(self):
        self.vec_adder.reset()


# ==============================================================================
# 24x24 阵列乘法器 (Wallace Tree压缩) - 向量化
# ==============================================================================
class ArrayMultiplier24x24(nn.Module):
    """24x24位阵列乘法器 - 向量化SNN实现
    
    使用部分积累加方式:
    - 24个部分积，每个24位
    - 逐级累加生成48位结果
    
    输入: A, B: [..., 24] (LSB first)
    输出: P: [..., 48] (LSB first)
    """
    def __init__(self):
        super().__init__()
        
        # 部分积生成: 向量化AND门 (单实例处理所有576个AND)
        self.vec_and = VecAND()
        
        # 累加器: 向量化48位加法器 (复用单实例)
        self.vec_adder = VecAdder(48)
        
    def forward(self, A, B):
        """
        A, B: [..., 24] LSB first
        Returns: [..., 48] LSB first
        """
        self.reset()
        device = A.device
        batch_shape = A.shape[:-1]
        zeros_24 = torch.zeros(batch_shape + (24,), device=device)
        
        # 生成所有部分积 (向量化)
        partial_products = []
        for i in range(24):
            # 第i个部分积: A & B[i] (广播)
            b_i = B[..., i:i+1].expand(*batch_shape, 24)  # [..., 24]
            pp = self.vec_and(A, b_i)  # [..., 24]
            
            # 扩展到48位，低位补零（移位）
            if i > 0:
                low_zeros = torch.zeros(batch_shape + (i,), device=device)
                pp_48 = torch.cat([low_zeros, pp, zeros_24[..., :24-i]], dim=-1)
            else:
                pp_48 = torch.cat([pp, zeros_24], dim=-1)
            partial_products.append(pp_48)
        
        # 累加所有部分积 (复用加法器)
        result = partial_products[0]
        for i in range(1, 24):
            result, _ = self.vec_adder(result, partial_products[i])
        
        return result
    
    def reset(self):
        self.vec_and.reset()
        self.vec_adder.reset()


# ==============================================================================
# 48位前导零检测器
# ==============================================================================
class LeadingZeroDetector48(nn.Module):
    """48位前导零检测器 - 输出6位LZC (100%纯SNN门电路)
    
    输入: X[47:0] MSB first
    输出: LZC[5:0] 前导零个数 (MSB first)
    
    实现: 使用纯门电路，禁止 * 和 + 操作脉冲值
    """
    def __init__(self):
        super().__init__()
        # 向量化门电路 (复用单实例)
        self.vec_not = VecNOT()
        self.vec_and = VecAND()
        self.vec_or = VecOR()
        self.vec_mux = VecMUX()
        self.vec_or_tree = VecORTree()
        
    def forward(self, X):
        """X: [..., 48] MSB first, returns: [..., 6] LZC MSB first"""
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 初始化lzc为全0 (6位)
        lzc = torch.zeros(batch_shape + (6,), device=device)
        
        # found = 是否已找到第一个1
        found = zeros.clone()
        
        for i in range(48):
            bit = X[..., i:i+1]
            
            # not_found = NOT(found)
            not_found = self.vec_not(found)
            
            # is_first = bit AND not_found
            is_first = self.vec_and(bit, not_found)
            
            # 如果is_first=1，设置lzc为当前位置i的二进制表示
            # i的6位二进制 (扩展到batch)
            pos_bits = torch.tensor([
                (i >> 5) & 1, (i >> 4) & 1, (i >> 3) & 1,
                (i >> 2) & 1, (i >> 1) & 1, i & 1
            ], device=device, dtype=torch.float32)
            pos_bits = pos_bits.expand(*batch_shape, 6)
            
            # lzc = MUX(is_first, pos_bits, lzc)
            is_first_exp = is_first.expand(*batch_shape, 6)
            lzc = self.vec_mux(is_first_exp, pos_bits, lzc)
            
            # found = found OR is_first
            found = self.vec_or(found, is_first)
        
        # 检测是否全零 (使用向量化 OR 树)
        any_one = self.vec_or_tree(X)
        all_zero = self.vec_not(any_one)
        
        # 如果全零，lzc = 48 = 0b110000
        lzc_48 = torch.tensor([1, 1, 0, 0, 0, 0], device=device, dtype=torch.float32)
        lzc_48 = lzc_48.expand(*batch_shape, 6)
        all_zero_exp = all_zero.expand(*batch_shape, 6)
        lzc = self.vec_mux(all_zero_exp, lzc_48, lzc)
        
        return lzc
    
    def reset(self):
        self.vec_not.reset()
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_mux.reset()
        self.vec_or_tree.reset()


# ==============================================================================
# 48位桶形右移位器 (用于subnormal处理) - 向量化
# ==============================================================================
class BarrelShifterRight48(nn.Module):
    """48位桶形右移位器 (向量化SNN) - 输出sticky位"""
    def __init__(self):
        super().__init__()
        self.data_bits = 48
        self.shift_bits = 6  # 最多移63位
        
        # 向量化门电路 (单实例复用)
        self.vec_mux = VecMUX()
        self.vec_or = VecOR()
        self.vec_or_tree = VecORTree()
            
    def forward(self, X, shift):
        """
        X: [..., 48] MSB first
        shift: [..., 6] MSB first
        Returns: (shifted_data, sticky_bit)
        """
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        sticky_accum = zeros.clone()
        
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 计算被移出位的OR (sticky)
            if shift_amt <= self.data_bits:
                start_idx = self.data_bits - shift_amt
                shifted_out_bits = current[..., start_idx:]
                layer_sticky = self.vec_or_tree(shifted_out_bits)
            else:
                layer_sticky = zeros
            
            # 只有当s_bit=1时才累积sticky
            sticky_contrib = self.vec_mux(s_bit, layer_sticky, zeros)
            sticky_accum = self.vec_or(sticky_accum, sticky_contrib)
            
            # 右移操作 (向量化)
            zeros_pad = torch.zeros(batch_shape + (shift_amt,), device=device)
            shifted = torch.cat([zeros_pad, current[..., :-shift_amt]], dim=-1)
            s_bit_exp = s_bit.expand(*batch_shape, self.data_bits)
            current = self.vec_mux(s_bit_exp, shifted, current)
        
        return current, sticky_accum
    
    def reset(self):
        self.vec_mux.reset()
        self.vec_or.reset()
        self.vec_or_tree.reset()


# ==============================================================================
# 48位桶形左移位器 - 向量化
# ==============================================================================
class BarrelShifterLeft48(nn.Module):
    """48位桶形左移位器 (向量化SNN)"""
    def __init__(self):
        super().__init__()
        self.data_bits = 48
        self.shift_bits = 6  # 最多移63位
        
        # 向量化门电路 (单实例复用)
        self.vec_mux = VecMUX()
            
    def forward(self, X, shift):
        """X: [..., 48], shift: [..., 6] (MSB first)"""
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        
        current = X
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 左移操作 (向量化)
            zeros_pad = torch.zeros(batch_shape + (shift_amt,), device=device)
            shifted = torch.cat([current[..., shift_amt:], zeros_pad], dim=-1)
            s_bit_exp = s_bit.expand(*batch_shape, self.data_bits)
            current = self.vec_mux(s_bit_exp, shifted, current)
        
        return current
    
    def reset(self):
        self.vec_mux.reset()


# ==============================================================================
# 9位加法器 (指数运算) - 向量化
# ==============================================================================
class RippleCarryAdder9Bit(nn.Module):
    """9位加法器 - 向量化SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.vec_adder = VecAdder(9)
        
    def forward(self, A, B, Cin=None):
        return self.vec_adder(A, B, Cin)
    
    def reset(self):
        self.vec_adder.reset()


# ==============================================================================
# 10位加法器 (指数运算 - 用于正确检测溢出/下溢) - 向量化
# ==============================================================================
class RippleCarryAdder10Bit(nn.Module):
    """10位加法器 - 向量化SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.vec_adder = VecAdder(10)
        
    def forward(self, A, B, Cin=None):
        return self.vec_adder(A, B, Cin)
    
    def reset(self):
        self.vec_adder.reset()


# ==============================================================================
# 10位减法器 - 向量化
# ==============================================================================
class Subtractor10Bit(nn.Module):
    """10位减法器 - 向量化SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.vec_subtractor = VecSubtractor(10)
        
    def forward(self, A, B, Bin=None):
        """A - B, LSB first (Bin 参数保留用于接口兼容，未使用)"""
        return self.vec_subtractor(A, B)
    
    def reset(self):
        self.vec_subtractor.reset()


# ==============================================================================
# 9位减法器 - 向量化
# ==============================================================================
class Subtractor9Bit(nn.Module):
    """9位减法器 - 向量化SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.vec_subtractor = VecSubtractor(9)
        
    def forward(self, A, B, Bin=None):
        """A - B, LSB first (Bin 参数保留用于接口兼容，未使用)"""
        return self.vec_subtractor(A, B)
    
    def reset(self):
        self.vec_subtractor.reset()


# ==============================================================================
# FP32 乘法器主类
# ==============================================================================
class SpikeFP32Multiplier(nn.Module):
    """FP32 乘法器 - 100%纯SNN门电路实现
    
    输入: A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0]
    输出: [..., 32] FP32脉冲
    
    特殊情况处理:
    - 零 × 任何 = 零
    - Inf × 非零 = Inf
    - NaN × 任何 = NaN
    - 0 × Inf = NaN
    - Subnormal: 完整支持
    """
    def __init__(self):
        super().__init__()
        
        # ===== 符号 =====
        self.sign_xor = XORGate()
        
        # ===== 指数运算 (使用10位确保正确检测溢出/下溢) =====
        # 指数加法: Ea + Eb (10位)
        self.exp_adder = RippleCarryAdder10Bit()
        # 减bias: - 127
        self.bias_sub = Subtractor10Bit()
        # 指数+1 (规格化调整)
        self.exp_inc = RippleCarryAdder10Bit()
        # 指数减法 (LZC调整)
        self.exp_lzc_sub = Subtractor10Bit()
        
        # ===== 尾数乘法 =====
        self.mantissa_mul = ArrayMultiplier24x24()
        
        # ===== 规格化 =====
        self.lzd = LeadingZeroDetector48()
        self.norm_shifter = BarrelShifterLeft48()
        
        # ===== RNE舍入 =====
        self.rne_or = ORGate()
        self.rne_and = ANDGate()
        self.round_adder = RippleCarryAdder(bits=24)
        
        # ===== Sticky bit OR树 =====
        self.sticky_or = nn.ModuleList([ORGate() for _ in range(22)])
        
        # ===== 特殊值检测 =====
        # 指数全1检测 (Inf/NaN)
        self.exp_all_one_or = nn.ModuleList([ORGate() for _ in range(7)])
        self.exp_all_one_and = nn.ModuleList([ANDGate() for _ in range(7)])
        
        # 指数全0检测 (Zero/Subnormal)
        self.exp_zero_or_a = nn.ModuleList([ORGate() for _ in range(7)])
        self.exp_zero_not_a = NOTGate()
        self.exp_zero_or_b = nn.ModuleList([ORGate() for _ in range(7)])
        self.exp_zero_not_b = NOTGate()
        
        # 尾数全0检测
        self.mant_zero_or_a = nn.ModuleList([ORGate() for _ in range(22)])
        self.mant_zero_not_a = NOTGate()
        self.mant_zero_or_b = nn.ModuleList([ORGate() for _ in range(22)])
        self.mant_zero_not_b = NOTGate()
        
        # ===== 零检测 =====
        self.a_is_zero_and = ANDGate()
        self.b_is_zero_and = ANDGate()
        self.either_zero_or = ORGate()
        
        # ===== Inf检测 =====
        self.a_is_inf_and = ANDGate()
        self.b_is_inf_and = ANDGate()
        self.either_inf_or = ORGate()
        
        # ===== NaN检测 =====
        self.a_mant_nonzero_not = NOTGate()
        self.b_mant_nonzero_not = NOTGate()
        self.a_is_nan_and = ANDGate()
        self.b_is_nan_and = ANDGate()
        self.either_nan_or = ORGate()
        
        # ===== 0 × Inf = NaN =====
        self.zero_times_inf_and = ANDGate()
        self.result_is_nan_or = ORGate()
        
        # ===== Subnormal检测 =====
        self.a_is_subnormal_and = ANDGate()
        self.b_is_subnormal_and = ANDGate()
        
        # ===== 尾数前导位选择 =====
        self.mux_a_leading = MUXGate()
        self.mux_b_leading = MUXGate()
        
        # ===== 指数修正 (subnormal有效指数=1) =====
        self.mux_a_exp = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_b_exp = nn.ModuleList([MUXGate() for _ in range(8)])
        
        # ===== 溢出/下溢处理 =====
        # 溢出检测: exp >= 255 (使用10位计算)
        # 需要检测9位指数的第9位(溢出位)和低8位>=255
        self.overflow_bit8_and_positive = ANDGate()  # bit8=1 且 非负数 → 溢出
        self.overflow_255_check = nn.ModuleList([ANDGate() for _ in range(8)])  # 检测低8位>=255
        
        # 下溢检测: exp <= 0 (9位有符号)
        # bit8=1表示负数(下溢)，或低9位全0
        self.underflow_not = nn.ModuleList([NOTGate() for _ in range(9)])
        self.underflow_and = nn.ModuleList([ANDGate() for _ in range(8)])
        self.underflow_or = ORGate()  # 负数OR全零
        
        # 溢出结果选择MUX
        self.overflow_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.overflow_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        self.not_overflow = NOTGate()
        
        # 下溢结果选择MUX  
        self.underflow_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.underflow_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        self.not_underflow = NOTGate()
        
        # ===== Subnormal处理 =====
        # subnormal检测: exp <= 0 但 exp > -150 (大约)
        # 需要额外右移 (1 - exp) 位
        self.subnormal_shifter = BarrelShifterRight48()  # 对尾数进行额外右移
        
        # 移位量计算: shift = 1 - exp = 1 + (-exp)
        # 需要将10位负指数转换为正的移位量
        self.shift_not = nn.ModuleList([NOTGate() for _ in range(10)])  # 取反
        self.shift_add_one = RippleCarryAdder10Bit()  # 加1得到补码的绝对值
        self.shift_add_one_const = RippleCarryAdder(bits=6)  # 加1得到移位量
        
        # subnormal舍入
        self.subnorm_sticky_or = nn.ModuleList([ORGate() for _ in range(24)])  # 合并sticky
        self.subnorm_rne_or = ORGate()
        self.subnorm_rne_and = ANDGate()
        self.subnorm_round_adder = RippleCarryAdder(bits=24)
        
        # subnormal结果选择
        self.subnorm_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        self.is_subnormal_and = ANDGate()  # exp<=0 且 exp>-150
        self.not_very_underflow = NOTGate()  # 非完全下溢
        
        # ===== 结果选择MUX =====
        # NaN输出
        self.nan_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.nan_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        # Inf输出
        self.inf_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.inf_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        # 零输出
        self.zero_mux_e = nn.ModuleList([MUXGate() for _ in range(8)])
        self.zero_mux_m = nn.ModuleList([MUXGate() for _ in range(23)])
        
        # ===== 舍入进位处理 =====
        self.round_carry_not = NOTGate()
        self.mant_clear_and = nn.ModuleList([ANDGate() for _ in range(23)])
        self.exp_round_inc = RippleCarryAdder(bits=8)
        self.exp_round_mux = nn.ModuleList([MUXGate() for _ in range(8)])
        
        # ===== 乘积溢出检测 (P[47]=1) =====
        self.prod_overflow_mux_m = nn.ModuleList([MUXGate() for _ in range(24)])
        
        # ===== 特殊值选择 (纯SNN NOT/AND门) =====
        self.not_result_is_nan = NOTGate()
        self.not_either_inf = NOTGate()
        self.inf_and_not_nan_gate = ANDGate()
        self.zero_and_not_nan_gate = ANDGate()
        self.zero_only_gate = ANDGate()
        
    def forward(self, A, B):
        """
        A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0] MSB first
        Returns: [..., 32] FP32脉冲
        """
        self.reset()
        A, B = torch.broadcast_tensors(A, B)
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # ===== 1. 提取各部分 =====
        s_a = A[..., 0:1]
        e_a = A[..., 1:9]  # [E7..E0] MSB first
        m_a = A[..., 9:32]  # [M22..M0] MSB first
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:9]
        m_b = B[..., 9:32]
        
        # ===== 2. 符号 =====
        s_out = self.sign_xor(s_a, s_b)
        
        # ===== 3. 特殊值检测 =====
        # 指数全1检测 (Inf/NaN)
        e_a_all_one = e_a[..., 0:1]
        for i in range(1, 8):
            e_a_all_one = self.exp_all_one_and[i-1](e_a_all_one, e_a[..., i:i+1])
        
        e_b_all_one = e_b[..., 0:1]
        for i in range(1, 8):
            e_b_all_one = self.exp_all_one_and[i-1](e_b_all_one, e_b[..., i:i+1])
        
        # 指数全0检测 (Zero/Subnormal)
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
        
        m_b_any_one = m_b[..., 0:1]
        for i in range(1, 23):
            m_b_any_one = self.mant_zero_or_b[i-1](m_b_any_one, m_b[..., i:i+1])
        m_b_is_zero = self.mant_zero_not_b(m_b_any_one)
        
        # 零检测: E=0 AND M=0
        a_is_zero = self.a_is_zero_and(e_a_is_zero, m_a_is_zero)
        b_is_zero = self.b_is_zero_and(e_b_is_zero, m_b_is_zero)
        either_zero = self.either_zero_or(a_is_zero, b_is_zero)
        
        # Inf检测: E=全1 AND M=0
        a_is_inf = self.a_is_inf_and(e_a_all_one, m_a_is_zero)
        b_is_inf = self.b_is_inf_and(e_b_all_one, m_b_is_zero)
        either_inf = self.either_inf_or(a_is_inf, b_is_inf)
        
        # NaN检测: E=全1 AND M≠0
        m_a_nonzero = self.a_mant_nonzero_not(m_a_is_zero)
        m_b_nonzero = self.b_mant_nonzero_not(m_b_is_zero)
        a_is_nan = self.a_is_nan_and(e_a_all_one, m_a_nonzero)
        b_is_nan = self.b_is_nan_and(e_b_all_one, m_b_nonzero)
        either_nan = self.either_nan_or(a_is_nan, b_is_nan)
        
        # 0 × Inf = NaN
        zero_times_inf = self.zero_times_inf_and(either_zero, either_inf)
        result_is_nan = self.result_is_nan_or(either_nan, zero_times_inf)
        
        # Subnormal检测: E=0 AND M≠0
        a_is_subnormal = self.a_is_subnormal_and(e_a_is_zero, m_a_nonzero)
        b_is_subnormal = self.b_is_subnormal_and(e_b_is_zero, m_b_nonzero)
        
        # ===== 4. 指数处理 =====
        # Subnormal有效指数=1
        e_a_le = e_a.flip(-1)  # 转LSB first
        e_b_le = e_b.flip(-1)
        
        # 如果subnormal，使用E=1 (LSB first: [1,0,0,0,0,0,0,0])
        e_a_corrected = []
        for i in range(8):
            if i == 0:
                e_bit = self.mux_a_exp[i](a_is_subnormal, ones, e_a_le[..., i:i+1])
            else:
                e_bit = self.mux_a_exp[i](a_is_subnormal, zeros, e_a_le[..., i:i+1])
            e_a_corrected.append(e_bit)
        e_a_corrected = torch.cat(e_a_corrected, dim=-1)
        
        e_b_corrected = []
        for i in range(8):
            if i == 0:
                e_bit = self.mux_b_exp[i](b_is_subnormal, ones, e_b_le[..., i:i+1])
            else:
                e_bit = self.mux_b_exp[i](b_is_subnormal, zeros, e_b_le[..., i:i+1])
            e_b_corrected.append(e_bit)
        e_b_corrected = torch.cat(e_b_corrected, dim=-1)
        
        # 扩展到10位 (用于正确检测溢出/下溢)
        e_a_10 = torch.cat([e_a_corrected, zeros, zeros], dim=-1)
        e_b_10 = torch.cat([e_b_corrected, zeros, zeros], dim=-1)
        
        # Ea + Eb (10位)
        sum_e_10, _ = self.exp_adder(e_a_10, e_b_10)
        
        # - 127: 使用减法器直接减去 127
        # 127 = 0b0001111111, LSB first: [1,1,1,1,1,1,1,0,0,0]
        const_127 = torch.cat([ones, ones, ones, ones, ones, ones, ones, zeros, zeros, zeros], dim=-1)
        raw_e_10, _ = self.bias_sub(sum_e_10, const_127)
        
        # ===== 5. 尾数乘法 =====
        # 恢复隐藏位
        leading_a = self.mux_a_leading(a_is_subnormal, zeros, ones)
        leading_b = self.mux_b_leading(b_is_subnormal, zeros, ones)
        
        # 24位尾数 (LSB first)
        m_a_le = m_a.flip(-1)  # [M0..M22]
        m_b_le = m_b.flip(-1)
        m_a_24 = torch.cat([m_a_le, leading_a], dim=-1)  # [M0..M22, 1]
        m_b_24 = torch.cat([m_b_le, leading_b], dim=-1)
        
        # 24x24乘法
        product_48 = self.mantissa_mul(m_a_24, m_b_24)  # LSB first
        
        # ===== 6. 规格化 =====
        # 检测乘积是否溢出 (P[47]=1表示 >= 2.0)
        prod_overflow = product_48[..., 47:48]
        
        # 转MSB first用于LZD
        product_48_be = product_48.flip(-1)
        
        # 前导零检测
        lzc = self.lzd(product_48_be)
        
        # 规格化移位
        product_norm = self.norm_shifter(product_48_be, lzc)
        
        # 调整指数
        # 如果prod_overflow=1，不需要额外移位，但指数+1
        # 否则根据LZC调整指数
        
        # 正常情况：乘积在[1,4)范围
        # P[47]=1: 乘积在[2,4)，需要指数+1
        # P[47]=0: 乘积在[1,2)，无需调整
        
        # 首先处理乘积溢出情况
        # 使用10位常量
        one_10 = torch.cat([ones, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        exp_overflow, _ = self.exp_inc(raw_e_10, one_10)
        
        # 根据LZC调整指数
        # 对于normal乘法，LZC应该是0或1
        # LZC=0: P[47]=1, 指数+1
        # LZC=1: P[47]=0, 指数不变
        # LZC>1: 结果是subnormal
        
        # lzc 是6位 MSB first，需要转成 LSB first 然后扩展到10位
        lzc_le = lzc.flip(-1)  # 转 LSB first
        lzc_10 = torch.cat([lzc_le, zeros, zeros, zeros, zeros], dim=-1)  # 高位补零到10位
        exp_adjusted, _ = self.exp_lzc_sub(raw_e_10, lzc_10)
        
        # 再加1（因为正常情况下product在bit46位置，需要+1）
        exp_final_pre, _ = self.exp_inc(exp_adjusted, one_10)
        
        # ===== 7. 提取尾数和舍入位 =====
        # 规格化后格式: [1, M22, M21, ..., M0, Round, Sticky...]
        # product_norm: MSB first
        # 位23是隐藏1，位0-22是尾数，后面是舍入/粘滞位
        
        # 处理溢出情况
        # 如果prod_overflow=1: 取bit[1:24]作为尾数，bit[24]是round，后面是sticky
        # 如果prod_overflow=0: 取bit[1:24]作为尾数，bit[24]是round，后面是sticky
        # 实际上规格化后格式一致
        
        mant_norm = product_norm[..., 1:24]  # 23位尾数 MSB first
        round_bit = product_norm[..., 24:25]
        
        # Sticky = OR(bit[25:48])
        sticky = product_norm[..., 25:26]
        for i in range(26, 48):
            if i - 26 < 22:
                sticky = self.sticky_or[i-26](sticky, product_norm[..., i:i+1])
        
        # RNE舍入
        lsb = mant_norm[..., 22:23]  # 最低位
        s_or_l = self.rne_or(sticky, lsb)
        round_up = self.rne_and(round_bit, s_or_l)
        
        # 尾数+1 (LSB first)
        mant_le = mant_norm.flip(-1)
        mant_24_le = torch.cat([mant_le, zeros], dim=-1)  # 扩展1位检测进位
        round_inc = torch.cat([round_up] + [zeros]*23, dim=-1)
        mant_rounded, _ = self.round_adder(mant_24_le, round_inc)
        
        # 进位检测
        mant_carry = mant_rounded[..., 23:24]
        
        # 如果进位，尾数清零
        not_carry = self.round_carry_not(mant_carry)
        mant_final_le = []
        for i in range(23):
            m_bit = self.mant_clear_and[i](not_carry, mant_rounded[..., i:i+1])
            mant_final_le.append(m_bit)
        mant_final_le = torch.cat(mant_final_le, dim=-1)
        mant_final = mant_final_le.flip(-1)  # 转MSB first
        
        # 如果进位，指数+1
        # exp_final_pre 是 LSB first，取低8位
        exp_8_le = exp_final_pre[..., :8]  # LSB first
        carry_inc = torch.cat([mant_carry, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        exp_after_round_le, _ = self.exp_round_inc(exp_8_le, carry_inc)  # LSB first
        
        # 选择最终指数 (LSB first)
        exp_final_le = []
        for i in range(8):
            e_sel = self.exp_round_mux[i](mant_carry, exp_after_round_le[..., i:i+1], exp_8_le[..., i:i+1])
            exp_final_le.append(e_sel)
        exp_final_le = torch.cat(exp_final_le, dim=-1)
        exp_final = exp_final_le.flip(-1)  # 转 MSB first 用于输出
        
        # ===== 8. 溢出/下溢处理 =====
        # exp_final_pre 是10位 LSB first (有符号数)
        # 10位有符号范围: -512 到 +511，足以表示乘法指数范围
        # 对于乘法: exp = exp_a + exp_b - 127 + adjustment
        #   最大: 254 + 254 - 127 + 1 = 382
        #   最小: 1 + 1 - 127 = -125
        
        # 溢出: 10位有符号值 >= 255
        #   即: bit9=0 (正数) 且 低9位>=255
        # 下溢: 10位有符号值 <= 0
        #   即: bit9=1 (负数) 或 全10位=0
        
        exp_bit9 = exp_final_pre[..., 9:10]  # 符号位 (bit9)
        
        # 检测低9位是否 >= 255
        # >= 255 意味着: bit8=1 或 (低8位全1)
        exp_bit8 = exp_final_pre[..., 8:9]
        exp_all_255 = exp_final_pre[..., 0:1]
        for i in range(1, 8):
            exp_all_255 = self.overflow_255_check[i-1](exp_all_255, exp_final_pre[..., i:i+1])
        
        # low9 >= 255: bit8=1 或 低8位全1
        low9_ge_255 = self.underflow_or(exp_bit8, exp_all_255)
        
        # 溢出条件: bit9=0 且 low9>=255
        not_bit9 = self.underflow_not[0](exp_bit9)  # 复用underflow_not[0]
        is_overflow = self.overflow_bit8_and_positive(not_bit9, low9_ge_255)
        
        # 下溢: 10位有符号值 <= 0
        # bit9=1 (负数) 或 全10位=0
        
        # 检测10位全0
        not_bits = []
        for i in range(10):
            if i < 9:
                not_bits.append(self.underflow_not[i](exp_final_pre[..., i:i+1]))
            else:
                not_bits.append(not_bit9)
        
        exp_is_zero = self.underflow_and[0](not_bits[0], not_bits[1])
        for i in range(2, 9):
            exp_is_zero = self.underflow_and[i-1](exp_is_zero, not_bits[i])
        exp_is_zero = self.overflow_255_check[7](exp_is_zero, not_bits[9])
        
        # 下溢条件: bit9=1 (负数) 或 全零
        is_underflow_or_subnormal = self.underflow_or(exp_bit9, exp_is_zero)
        
        # ===== 8.5 Subnormal处理 =====
        # subnormal: exp <= 0 但 exp > -150 (大约)
        # 需要额外右移 (1 - exp) 位，然后exp=0
        
        # 计算移位量: shift = 1 - exp
        # 对于负的exp: shift = 1 + |exp| = 1 + (补码取反+1)
        # 简化: 直接使用 1 - exp (10位减法)
        
        # 首先判断是否是subnormal情况 (而非完全下溢)
        # 完全下溢: exp < -149 (大约)
        # 对于10位补码: -149 = 0b1101101011
        # 检测方式: bit9=1 (负数) 且 bit8=1 且 bit7=1 且 bit6=1 (大约 < -64)
        # 更精确: 检测 exp < -126 (因为subnormal最大移位是23位，对应exp=-23)
        # 但实际上需要检测 exp > -150
        
        # 简化判断: 如果exp在 [-126, 0] 范围内，则是subnormal
        # -126 的10位补码: 1110000010
        # 检测 exp >= -126: 不是那么负
        # 即: NOT(bit9=1 且 bit8=1 且 bit7=1 且 bit6=0 且 低6位表示<=-126)
        
        # 更简单的方法: 检测移位量是否 <= 25 (subnormal有效)
        # 移位量 = 1 - exp, 如果exp在[-24, 0]，移位量在[1, 25]
        # 如果exp < -24, 移位量 > 25，基本上会下溢到0
        
        # 计算 -exp (取反+1)
        exp_neg = []
        for i in range(10):
            exp_neg.append(self.shift_not[i](exp_final_pre[..., i:i+1]))
        exp_neg = torch.cat(exp_neg, dim=-1)
        
        one_10 = torch.cat([ones, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        neg_exp_10, _ = self.shift_add_one(exp_neg, one_10)  # -exp = ~exp + 1
        
        # 移位量 = 1 + (-exp) = 1 - exp (因为exp是负的或0)
        # 对于subnormal，我们需要移位 (1 - exp) 位
        # 如果exp=0, shift=1; 如果exp=-1, shift=2; 等等
        
        # 移位量 = neg_exp_10 + 1 (10位完整计算)
        shift_full_10, _ = self.shift_add_one(neg_exp_10, one_10)
        
        # 检测移位量是否过大 (>= 32，即bit5或更高位非零)
        # 如果bit5-9中任意一位为1，则移位量>=32，完全下溢
        shift_bit5_or_higher = shift_full_10[..., 5:6]
        for i in range(6, 10):
            shift_bit5_or_higher = self.subnorm_sticky_or[i-6](shift_bit5_or_higher, shift_full_10[..., i:i+1])
        
        # 提取低6位作为移位量
        shift_final = shift_full_10[..., :6]  # LSB first
        shift_final_be = shift_final.flip(-1)  # 转MSB first给移位器
        
        # 对规格化后的乘积进行额外右移
        # product_norm 是48位 MSB first，已经规格化
        # 右移后需要重新计算舍入
        
        shifted_product, shift_sticky = self.subnormal_shifter(product_norm, shift_final_be)
        
        # 提取移位后的尾数和舍入位
        # 移位后格式 (MSB first):
        # shifted_product[0]: 总是0（因为右移了至少1位）
        # shifted_product[1:24]: 23位subnormal尾数
        # shifted_product[24]: round位
        # shifted_product[25:]: sticky位（加上shift_sticky）
        
        subnorm_mant = shifted_product[..., 1:24]  # 23位尾数
        subnorm_round = shifted_product[..., 24:25]
        
        # Sticky = OR(shifted_product[25:48]) OR shift_sticky
        subnorm_sticky = shifted_product[..., 25:26]
        for i in range(26, 48):
            if i - 26 < 22:
                subnorm_sticky = self.subnorm_sticky_or[i-26](subnorm_sticky, shifted_product[..., i:i+1])
        subnorm_sticky = self.subnorm_sticky_or[22](subnorm_sticky, shift_sticky)
        
        # RNE舍入
        subnorm_lsb = subnorm_mant[..., 22:23]  # 尾数最低位
        s_or_l_sub = self.subnorm_rne_or(subnorm_sticky, subnorm_lsb)
        round_up_sub = self.subnorm_rne_and(subnorm_round, s_or_l_sub)
        
        # 尾数+1 (LSB first)
        subnorm_mant_le = subnorm_mant.flip(-1)
        subnorm_mant_24_le = torch.cat([subnorm_mant_le, zeros], dim=-1)
        round_inc_sub = torch.cat([round_up_sub] + [zeros]*23, dim=-1)
        subnorm_mant_rounded, _ = self.subnorm_round_adder(subnorm_mant_24_le, round_inc_sub)
        subnorm_mant_final = subnorm_mant_rounded[..., :23].flip(-1)  # 转MSB first
        
        # 判断是subnormal还是完全下溢
        # 如果移位量太大(>=32)，则完全下溢
        # shift_bit5_or_higher = 1 表示移位量>=32
        
        # 移位量合理: bit5到bit9都是0
        is_reasonable_shift = self.not_very_underflow(shift_bit5_or_higher)
        
        # subnormal条件: exp<=0 且 移位量合理 (<32)
        is_subnormal = self.is_subnormal_and(is_underflow_or_subnormal, is_reasonable_shift)
        
        # 完全下溢: exp<=0 且 移位量不合理 (>=32)
        is_underflow = self.underflow_and[7](is_underflow_or_subnormal, shift_bit5_or_higher)
        
        # ===== 9. 特殊值输出 =====
        # NaN: E=FF, M=非零 (使用0x7FC00000 = quiet NaN)
        nan_exp = torch.cat([ones]*8, dim=-1)
        nan_mant = torch.cat([ones, zeros, zeros] + [zeros]*20, dim=-1)  # M22=1
        
        # Inf: E=FF, M=0
        inf_exp = torch.cat([ones]*8, dim=-1)
        inf_mant = torch.cat([zeros]*23, dim=-1)
        
        # Zero: E=0, M=0
        zero_exp = torch.cat([zeros]*8, dim=-1)
        zero_mant = torch.cat([zeros]*23, dim=-1)
        
        # ===== 10. 选择最终结果 =====
        # 先应用NaN
        e_out = []
        m_out = []
        for i in range(8):
            e_bit = self.nan_mux_e[i](result_is_nan, nan_exp[..., i:i+1], exp_final[..., i:i+1])
            e_out.append(e_bit)
        for i in range(23):
            m_bit = self.nan_mux_m[i](result_is_nan, nan_mant[..., i:i+1], mant_final[..., i:i+1])
            m_out.append(m_bit)
        
        e_out = torch.cat(e_out, dim=-1)
        m_out = torch.cat(m_out, dim=-1)
        
        # 应用Inf (非NaN的Inf情况) - 纯SNN门电路
        not_nan = self.not_result_is_nan(result_is_nan)
        inf_and_not_nan = self.inf_and_not_nan_gate(either_inf, not_nan)
        e_out2 = []
        m_out2 = []
        for i in range(8):
            e_bit = self.inf_mux_e[i](inf_and_not_nan, inf_exp[..., i:i+1], e_out[..., i:i+1])
            e_out2.append(e_bit)
        for i in range(23):
            m_bit = self.inf_mux_m[i](inf_and_not_nan, inf_mant[..., i:i+1], m_out[..., i:i+1])
            m_out2.append(m_bit)
        
        e_out = torch.cat(e_out2, dim=-1)
        m_out = torch.cat(m_out2, dim=-1)
        
        # 应用Zero (非NaN非Inf的零情况) - 纯SNN门电路
        not_either_inf = self.not_either_inf(either_inf)
        zero_and_not_nan = self.zero_and_not_nan_gate(either_zero, not_nan)
        zero_only = self.zero_only_gate(zero_and_not_nan, not_either_inf)
        e_out3 = []
        m_out3 = []
        for i in range(8):
            e_bit = self.zero_mux_e[i](zero_only, zero_exp[..., i:i+1], e_out[..., i:i+1])
            e_out3.append(e_bit)
        for i in range(23):
            m_bit = self.zero_mux_m[i](zero_only, zero_mant[..., i:i+1], m_out[..., i:i+1])
            m_out3.append(m_bit)
        
        e_out = torch.cat(e_out3, dim=-1)
        m_out = torch.cat(m_out3, dim=-1)
        
        # ===== 11. 应用计算溢出 (非特殊值情况下exp>=255) =====
        # 溢出 → Inf (E=FF, M=0)
        not_special = self.not_overflow(result_is_nan)  # 复用gate检测非特殊值
        # 这里简化: 如果溢出且非NaN，输出Inf
        overflow_and_valid = self.overflow_bit8_and_positive(is_overflow, not_nan)
        e_out4 = []
        m_out4 = []
        for i in range(8):
            e_bit = self.overflow_mux_e[i](overflow_and_valid, inf_exp[..., i:i+1], e_out[..., i:i+1])
            e_out4.append(e_bit)
        for i in range(23):
            m_bit = self.overflow_mux_m[i](overflow_and_valid, inf_mant[..., i:i+1], m_out[..., i:i+1])
            m_out4.append(m_bit)
        
        e_out = torch.cat(e_out4, dim=-1)
        m_out = torch.cat(m_out4, dim=-1)
        
        # ===== 12. 应用Subnormal (exp<=0 但 >-150，且非零输入) =====
        # subnormal: exp=0, mant=subnorm_mant_final
        not_overflow_flag = self.not_underflow(is_overflow)
        not_either_zero = self.not_overflow(either_zero)  # 复用gate
        subnorm_temp = self.is_subnormal_and(is_subnormal, not_nan)  # 复用gate
        subnorm_and_valid = self.underflow_and[6](subnorm_temp, not_either_zero)  # 复用gate
        
        e_out5 = []
        m_out5 = []
        for i in range(8):
            e_bit = self.underflow_mux_e[i](subnorm_and_valid, zero_exp[..., i:i+1], e_out[..., i:i+1])
            e_out5.append(e_bit)
        for i in range(23):
            m_bit = self.subnorm_mux_m[i](subnorm_and_valid, subnorm_mant_final[..., i:i+1], m_out[..., i:i+1])
            m_out5.append(m_bit)
        
        e_out = torch.cat(e_out5, dim=-1)
        m_out = torch.cat(m_out5, dim=-1)
        
        # ===== 13. 应用完全下溢 (exp < -150左右 → 0, 且非零输入) =====
        underflow_only = self.overflow_bit8_and_positive(is_underflow, not_overflow_flag)  # 复用gate
        underflow_temp = self.zero_and_not_nan_gate(underflow_only, not_nan)  # 复用gate
        underflow_final = self.underflow_and[5](underflow_temp, not_either_zero)  # 复用gate
        e_out6 = []
        m_out6 = []
        for i in range(8):
            e_bit = self.overflow_mux_e[i](underflow_final, zero_exp[..., i:i+1], e_out[..., i:i+1])  # 复用
            e_out6.append(e_bit)
        for i in range(23):
            m_bit = self.overflow_mux_m[i](underflow_final, zero_mant[..., i:i+1], m_out[..., i:i+1])  # 复用
            m_out6.append(m_bit)
        
        e_out = torch.cat(e_out6, dim=-1)
        m_out = torch.cat(m_out6, dim=-1)
        
        # ===== 14. 组装输出 =====
        result = torch.cat([s_out, e_out, m_out], dim=-1)
        
        return result
    
    def reset(self):
        self.sign_xor.reset()
        self.exp_adder.reset()
        self.bias_sub.reset()
        self.exp_inc.reset()
        self.exp_lzc_sub.reset()
        self.mantissa_mul.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.round_adder.reset()
        for g in self.sticky_or: g.reset()
        for g in self.exp_all_one_or: g.reset()
        for g in self.exp_all_one_and: g.reset()
        for g in self.exp_zero_or_a: g.reset()
        self.exp_zero_not_a.reset()
        for g in self.exp_zero_or_b: g.reset()
        self.exp_zero_not_b.reset()
        for g in self.mant_zero_or_a: g.reset()
        self.mant_zero_not_a.reset()
        for g in self.mant_zero_or_b: g.reset()
        self.mant_zero_not_b.reset()
        self.a_is_zero_and.reset()
        self.b_is_zero_and.reset()
        self.either_zero_or.reset()
        self.a_is_inf_and.reset()
        self.b_is_inf_and.reset()
        self.either_inf_or.reset()
        self.a_mant_nonzero_not.reset()
        self.b_mant_nonzero_not.reset()
        self.a_is_nan_and.reset()
        self.b_is_nan_and.reset()
        self.either_nan_or.reset()
        self.zero_times_inf_and.reset()
        self.result_is_nan_or.reset()
        self.a_is_subnormal_and.reset()
        self.b_is_subnormal_and.reset()
        self.mux_a_leading.reset()
        self.mux_b_leading.reset()
        for mux in self.mux_a_exp: mux.reset()
        for mux in self.mux_b_exp: mux.reset()
        # 溢出/下溢检测
        self.overflow_bit8_and_positive.reset()
        for g in self.overflow_255_check: g.reset()
        for g in self.underflow_not: g.reset()
        for g in self.underflow_and: g.reset()
        self.underflow_or.reset()
        # 溢出/下溢结果选择
        for mux in self.overflow_mux_e: mux.reset()
        for mux in self.overflow_mux_m: mux.reset()
        self.not_overflow.reset()
        for mux in self.underflow_mux_e: mux.reset()
        for mux in self.underflow_mux_m: mux.reset()
        self.not_underflow.reset()
        # Subnormal处理
        self.subnormal_shifter.reset()
        for g in self.shift_not: g.reset()
        self.shift_add_one.reset()
        self.shift_add_one_const.reset()
        for g in self.subnorm_sticky_or: g.reset()
        self.subnorm_rne_or.reset()
        self.subnorm_rne_and.reset()
        self.subnorm_round_adder.reset()
        for mux in self.subnorm_mux_m: mux.reset()
        self.is_subnormal_and.reset()
        self.not_very_underflow.reset()
        for mux in self.nan_mux_e: mux.reset()
        for mux in self.nan_mux_m: mux.reset()
        for mux in self.inf_mux_e: mux.reset()
        for mux in self.inf_mux_m: mux.reset()
        for mux in self.zero_mux_e: mux.reset()
        for mux in self.zero_mux_m: mux.reset()
        self.round_carry_not.reset()
        for g in self.mant_clear_and: g.reset()
        self.exp_round_inc.reset()
        for mux in self.exp_round_mux: mux.reset()
        for mux in self.prod_overflow_mux_m: mux.reset()
        # 特殊值选择门
        self.not_result_is_nan.reset()
        self.not_either_inf.reset()
        self.inf_and_not_nan_gate.reset()
        self.zero_and_not_nan_gate.reset()
        self.zero_only_gate.reset()

