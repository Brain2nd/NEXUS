"""
FP32 相关组件 - 100%纯SNN门电路实现
用于对齐朴素 PyTorch nn.Linear（FP32 累加 -> FP8）

FP32 格式: [S | E7..E0 | M22..M0], bias=127
FP8 E4M3:  [S | E3..E0 | M2..M0], bias=7

转换关系: FP32_exp = FP8_exp + 120 (对于 normal)
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate, 
                          HalfAdder, FullAdder, RippleCarryAdder, ORTree, ANDTree)
from .vec_logic_gates import (VecAND, VecOR, VecXOR, VecNOT, VecMUX,
                               VecORTree, VecANDTree, VecAdder, VecSubtractor)


# ==============================================================================
# 参数化组件（扩展到更大位宽）
# ==============================================================================
class ComparatorNBit(nn.Module):
    """N位比较器 - 向量化SNN实现
    
    使用向量化门电路：一次处理所有位的独立操作
    """
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        
        # 向量化门电路：单个实例处理所有位
        self.vec_xor = VecXOR()      # 计算 A XOR B (所有位并行)
        self.vec_not_xor = VecNOT()  # 计算 NOT(A XOR B) = EQ (所有位并行)
        self.vec_not_b = VecNOT()    # 计算 NOT(B) (所有位并行)
        self.vec_gt_and = VecAND()   # 计算 A AND NOT(B) = GT (所有位并行)
        
        # 前缀逻辑有依赖，使用树归约
        self.eq_tree = VecANDTree()  # 最终 a_eq_b
        
        # 用于 a_gt_b 的逐位计算（有依赖，需要循环）
        self.eq_prefix_and = nn.ModuleList([VecAND() for _ in range(max(1, bits - 1))])
        self.result_and = VecAND()
        self.result_or = VecOR()
        
    def forward(self, A, B):
        """A, B: [..., bits] (MSB first)"""
        self.reset()
        n = self.bits
        
        # 向量化计算：一次处理所有 n 位
        xor_all = self.vec_xor(A, B)                    # [..., n]
        eq_all = self.vec_not_xor(xor_all)             # [..., n] eq[i] = NOT(xor[i])
        not_b_all = self.vec_not_b(B)                  # [..., n]
        gt_all = self.vec_gt_and(A, not_b_all)         # [..., n] gt[i] = A[i] AND NOT(B[i])
        
        if n == 1:
            return gt_all[..., 0:1], eq_all[..., 0:1]
        
        # a_eq_b = AND(所有 eq 位) - 使用 AND 树
        a_eq_b = self.eq_tree(eq_all)
        
        # a_gt_b 计算（有依赖，需要循环）
        # a_gt_b = gt[0] OR (eq[0] AND gt[1]) OR (eq[0] AND eq[1] AND gt[2]) ...
        eq_prefix = eq_all[..., 0:1]  # eq[0]
        a_gt_b = gt_all[..., 0:1]     # gt[0]
        
        for i in range(1, n):
            # term = eq_prefix AND gt[i]
            term = self.result_and(eq_prefix, gt_all[..., i:i+1])
            a_gt_b = self.result_or(a_gt_b, term)
            
            # 更新 eq_prefix = eq_prefix AND eq[i] (除了最后一位)
            if i < n - 1:
                eq_prefix = self.eq_prefix_and[i-1](eq_prefix, eq_all[..., i:i+1])
        
        return a_gt_b, a_eq_b
    
    def reset(self):
        self.vec_xor.reset()
        self.vec_not_xor.reset()
        self.vec_not_b.reset()
        self.vec_gt_and.reset()
        self.eq_tree.reset()
        for g in self.eq_prefix_and: g.reset()
        self.result_and.reset()
        self.result_or.reset()


class Comparator8Bit(ComparatorNBit):
    """8位比较器（FP32指数）"""
    def __init__(self):
        super().__init__(bits=8)


class Comparator24Bit(ComparatorNBit):
    """24位比较器（FP32尾数: hidden + 23 mant）"""
    def __init__(self):
        super().__init__(bits=24)


class SubtractorNBit(nn.Module):
    """N位减法器 - 向量化SNN实现 (LSB first)
    
    使用向量化门电路复用，减少实例数量。
    借位链有依赖，需要循环但复用门电路。
    """
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        
        # 向量化门电路：复用实例处理串行计算
        self.vec_xor1 = VecXOR()   # a XOR b
        self.vec_xor2 = VecXOR()   # t1 XOR borrow
        self.vec_not_a = VecNOT()  # NOT(a)
        self.vec_and1 = VecAND()   # NOT(a) AND b
        self.vec_and2 = VecAND()   # NOT(a) AND borrow
        self.vec_and3 = VecAND()   # b AND borrow
        self.vec_or1 = VecOR()     # term1 OR term2
        self.vec_or2 = VecOR()     # t12 OR term3
        
    def forward(self, A, B, Bin=None):
        """A - B, LSB first (index 0 = LSB)"""
        self.reset()
        n = self.bits
        
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin
            
        diffs = []
        for i in range(n):
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            # 复用向量化门电路
            t1 = self.vec_xor1(a_i, b_i)
            diff = self.vec_xor2(t1, borrow)
            
            not_a_i = self.vec_not_a(a_i)
            term1 = self.vec_and1(not_a_i, b_i)
            term2 = self.vec_and2(not_a_i, borrow)
            term3 = self.vec_and3(b_i, borrow)
            t12 = self.vec_or1(term1, term2)
            new_borrow = self.vec_or2(t12, term3)
            
            diffs.append(diff)
            borrow = new_borrow
        
        result = torch.cat(diffs, dim=-1)
        return result, borrow
    
    def reset(self):
        self.vec_xor1.reset()
        self.vec_xor2.reset()
        self.vec_not_a.reset()
        self.vec_and1.reset()
        self.vec_and2.reset()
        self.vec_and3.reset()
        self.vec_or1.reset()
        self.vec_or2.reset()


class Subtractor8Bit(SubtractorNBit):
    """8位减法器（FP32指数差）"""
    def __init__(self):
        super().__init__(bits=8)


class RippleCarryAdderNBit(nn.Module):
    """N位加法器 - 向量化SNN实现 (LSB first)
    
    使用 VecAdder 实现（进位链有依赖，但门电路复用）
    """
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.vec_adder = VecAdder(bits)
        
    def forward(self, A, B, Cin=None):
        """A + B, LSB first"""
        self.reset()
        result, cout = self.vec_adder(A, B)
        return result, cout
    
    def reset(self):
        self.vec_adder.reset()


class RippleCarryAdder28Bit(RippleCarryAdderNBit):
    """28位加法器（FP32内部精度）"""
    def __init__(self):
        super().__init__(bits=28)


class Subtractor28Bit(SubtractorNBit):
    """28位减法器（FP32内部精度）"""
    def __init__(self):
        super().__init__(bits=28)


# ==============================================================================
# 桶形移位器
# ==============================================================================
class BarrelShifterRight28(nn.Module):
    """28位桶形右移位器 - 向量化SNN实现
    
    每层 MUX 操作并行处理所有 28 位
    """
    def __init__(self):
        super().__init__()
        self.data_bits = 28
        self.shift_bits = 5  # 最多移31位
        
        # 每层使用一个 VecMUX 处理所有位
        self.vec_mux_layers = nn.ModuleList([VecMUX() for _ in range(self.shift_bits)])
            
    def forward(self, X, shift):
        """X: [..., 28], shift: [..., 5] (MSB first)"""
        self.reset()
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 构建移位后的版本
            if shift_amt < self.data_bits:
                # 右移: 高位补零，低位移入
                shifted = torch.cat([
                    zeros.expand(*zeros.shape[:-1], shift_amt),  # 高位补零
                    current[..., :-shift_amt]                    # 剩余位
                ], dim=-1)
            else:
                shifted = zeros.expand(*zeros.shape[:-1], self.data_bits)
            
            # 扩展 s_bit 到所有位
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)
            
            # 并行 MUX: 选择移位或保持
            current = self.vec_mux_layers[layer](s_bit_expanded, shifted, current)
        
        return current
    
    def reset(self):
        for mux in self.vec_mux_layers:
            mux.reset()


class BarrelShifterRight28WithSticky(nn.Module):
    """28位桶形右移位器 - 向量化SNN实现 - 输出sticky bit
    
    sticky = OR(所有被移出的位)
    """
    def __init__(self):
        super().__init__()
        self.data_bits = 28
        self.shift_bits = 5
        
        # 每层使用一个 VecMUX 处理所有位
        self.vec_mux_layers = nn.ModuleList([VecMUX() for _ in range(self.shift_bits)])
        
        # Sticky 计算：使用 VecORTree 替代串行 OR 链
        self.sticky_or_tree = VecORTree()
        self.sticky_mux = nn.ModuleList([VecMUX() for _ in range(self.shift_bits)])
        self.sticky_accum_or = VecOR()
        
    def forward(self, X, shift):
        """X: [..., 28], shift: [..., 5] (MSB first，bit0是MSB，bit27是LSB)
        Returns: (shifted_X, sticky)
        """
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        sticky_accum = zeros
        
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 计算被移出位的 OR (使用 VecORTree 并行归约)
            if shift_amt <= self.data_bits:
                start_idx = self.data_bits - shift_amt
                shifted_out_bits = current[..., start_idx:]  # [..., shift_amt]
                layer_sticky = self.sticky_or_tree(shifted_out_bits)
            else:
                layer_sticky = zeros
            
            # 如果这层移位了，累积 sticky
            layer_sticky_selected = self.sticky_mux[layer](s_bit, layer_sticky, zeros)
            sticky_accum = self.sticky_accum_or(sticky_accum, layer_sticky_selected)
            
            # 构建移位后的版本 (并行)
            if shift_amt < self.data_bits:
                shifted = torch.cat([
                    zeros.expand(*zeros.shape[:-1], shift_amt),
                    current[..., :-shift_amt]
                ], dim=-1)
            else:
                shifted = zeros.expand(*zeros.shape[:-1], self.data_bits)
            
            # 扩展 s_bit 到所有位
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)
            
            # 并行 MUX
            current = self.vec_mux_layers[layer](s_bit_expanded, shifted, current)
        
        return current, sticky_accum
    
    def reset(self):
        for mux in self.vec_mux_layers: mux.reset()
        self.sticky_or_tree.reset()
        for mux in self.sticky_mux: mux.reset()
        self.sticky_accum_or.reset()


class BarrelShifterLeft28(nn.Module):
    """28位桶形左移位器 - 向量化SNN实现
    
    每层 MUX 操作并行处理所有 28 位
    """
    def __init__(self):
        super().__init__()
        self.data_bits = 28
        self.shift_bits = 5
        
        # 每层使用一个 VecMUX 处理所有位
        self.vec_mux_layers = nn.ModuleList([VecMUX() for _ in range(self.shift_bits)])
            
    def forward(self, X, shift):
        """X: [..., 28], shift: [..., 5] (MSB first)"""
        self.reset()
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 构建左移后的版本
            if shift_amt < self.data_bits:
                # 左移: 低位补零，高位移出
                shifted = torch.cat([
                    current[..., shift_amt:],                     # 剩余位
                    zeros.expand(*zeros.shape[:-1], shift_amt)    # 低位补零
                ], dim=-1)
            else:
                shifted = zeros.expand(*zeros.shape[:-1], self.data_bits)
            
            # 扩展 s_bit 到所有位
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)
            
            # 并行 MUX
            current = self.vec_mux_layers[layer](s_bit_expanded, shifted, current)
        
        return current
    
    def reset(self):
        for mux in self.vec_mux_layers:
            mux.reset()


# ==============================================================================
# 前导零检测器
# ==============================================================================
class LeadingZeroDetector28(nn.Module):
    """28位前导零检测器 - 向量化SNN实现
    
    使用向量化门电路复用（有依赖需循环，但门电路实例复用）
    """
    def __init__(self):
        super().__init__()
        # 向量化门电路复用
        self.vec_not_found = VecNOT()
        self.vec_not_all_zero = VecNOT()
        self.vec_and_first = VecAND()   # bit AND NOT(found)
        self.vec_or_found = VecOR()     # found OR is_first
        self.vec_or_lzc = VecOR()       # lzc 累积
        self.vec_or_final = VecOR()     # 最终 OR
        
    def forward(self, X):
        """X: [..., 28], returns: [..., 5] (LZC, MSB first)"""
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        lzc = [zeros.clone() for _ in range(5)]
        found = zeros.clone()
        
        for i in range(28):
            bit = X[..., i:i+1]
            not_found = self.vec_not_found(found)
            is_first = self.vec_and_first(bit, not_found)
            
            pos = i
            # 将位置编码为5位
            for b in range(5):
                pos_bit = (pos >> (4 - b)) & 1
                if pos_bit:
                    lzc[b] = self.vec_or_lzc(lzc[b], is_first)
            
            # found = found OR is_first
            found = self.vec_or_found(found, is_first)
        
        # 全零情况: 返回 28 (二进制 11100)
        all_zero = self.vec_not_all_zero(found)
        # 28 = 0b11100 = [1, 1, 1, 0, 0]
        all_zero_bits = [1, 1, 1, 0, 0]
        for b in range(5):
            if all_zero_bits[b]:
                lzc[b] = self.vec_or_final(lzc[b], all_zero)
        
        return torch.cat(lzc, dim=-1)
    
    def reset(self):
        self.vec_not_found.reset()
        self.vec_not_all_zero.reset()
        self.vec_and_first.reset()
        self.vec_or_found.reset()
        self.vec_or_lzc.reset()
        self.vec_or_final.reset()


# ==============================================================================
# FP8 -> FP32 转换器
# ==============================================================================
class FP8ToFP32Converter(nn.Module):
    """FP8 E4M3 -> FP32 转换器（100%纯SNN门电路）
    
    FP8 E4M3: [S | E3 E2 E1 E0 | M2 M1 M0], bias=7
    FP32:     [S | E7..E0 | M22..M0], bias=127
    
    转换规则（Normal）：
    - sign: 直接复制
    - exp: FP32_exp = FP8_exp + 120 (bias差 = 127 - 7 = 120)
    - mant: 高位对齐，低位补0
    
    转换规则（Subnormal/Zero）：
    - FP8 E=0 时保持 FP32 E=0
    """
    def __init__(self):
        super().__init__()
        
        # 检测 FP8 E=0
        self.e_or_01 = ORGate()
        self.e_or_23 = ORGate()
        self.e_or_all = ORGate()
        self.e_is_zero_not = NOTGate()
        
        # 检测 M≠0 (纯 SNN OR 门)
        self.m_or_01 = ORGate()
        self.m_or_all = ORGate()
        
        # 8位加法器: FP8_exp (4位扩展) + 120
        # 120 = 0b01111000
        self.exp_adder = RippleCarryAdder(bits=8)
        
        # E=0时选择0指数 - 向量化
        self.vec_exp_mux = VecMUX()
        
        # 纯SNN NOT门 - 向量化
        self.vec_not = VecNOT()
        
        # 纯SNN AND门 - 向量化
        self.vec_and = VecAND()
        
        # 纯SNN OR门 - 向量化
        self.vec_or = VecOR()
        
        # 纯SNN MUX门 - 向量化
        self.vec_mux = VecMUX()
        
    def forward(self, fp8_pulse):
        """
        Args:
            fp8_pulse: [..., 8] FP8 脉冲 [S, E3, E2, E1, E0, M2, M1, M0]
        Returns:
            fp32_pulse: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        self.reset()
        device = fp8_pulse.device
        batch_shape = fp8_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取 FP8 各部分
        s = fp8_pulse[..., 0:1]
        e3 = fp8_pulse[..., 1:2]  # MSB
        e2 = fp8_pulse[..., 2:3]
        e1 = fp8_pulse[..., 3:4]
        e0 = fp8_pulse[..., 4:5]  # LSB
        m2 = fp8_pulse[..., 5:6]
        m1 = fp8_pulse[..., 6:7]
        m0 = fp8_pulse[..., 7:8]
        
        # 检测 E=0
        e_or_01 = self.e_or_01(e0, e1)
        e_or_23 = self.e_or_23(e2, e3)
        e_nonzero = self.e_or_all(e_or_01, e_or_23)
        e_is_zero = self.e_is_zero_not(e_nonzero)
        
        # 检测 M≠0 (纯 SNN OR 门)
        m_or_01 = self.m_or_01(m0, m1)
        m_nonzero = self.m_or_all(m_or_01, m2)
        
        # subnormal 检测: E=0 AND M≠0 (向量化门电路)
        is_subnormal = self.vec_and(e_is_zero, m_nonzero)
        
        # FP8 exp 扩展到 8 位 (LSB first for adder)
        fp8_exp_8bit_lsb = torch.cat([e0, e1, e2, e3, zeros, zeros, zeros, zeros], dim=-1)
        
        # +120 = 0b01111000, LSB first: [0, 0, 0, 1, 1, 1, 1, 0]
        const_120_lsb = torch.cat([zeros, zeros, zeros, ones, ones, ones, ones, zeros], dim=-1)
        
        # 加法 (LSB first)
        fp32_exp_raw_lsb, _ = self.exp_adder(fp8_exp_8bit_lsb, const_120_lsb)
        
        # 转回 MSB first
        fp32_exp_raw = fp32_exp_raw_lsb.flip(-1)
        
        # Subnormal 处理: 根据 M 的值计算 FP32 指数和尾数
        # M=1 (001): FP32_E=118, M=0
        # M=2 (010): FP32_E=119, M=0
        # M=3 (011): FP32_E=119, M=1000...
        # M=4 (100): FP32_E=120, M=0
        # M=5 (101): FP32_E=120, M=0100...
        # M=6 (110): FP32_E=120, M=1000...
        # M=7 (111): FP32_E=120, M=1100...
        
        # 检测 M 的最高位 (使用向量化门电路)
        not_m2 = self.vec_not(m2)
        not_m1 = self.vec_not(m1)
        m_is_1xx = m2  # M >= 4
        m_is_01x = self.vec_and(not_m2, m1)  # M in [2, 3]
        temp_001 = self.vec_and(not_m2, not_m1)
        m_is_001 = self.vec_and(temp_001, m0)  # M = 1
        
        # Subnormal FP32 指数 (MSB first)
        # M >= 4: E = 120 = 0b01111000
        # M in [2,3]: E = 119 = 0b01110111  
        # M = 1: E = 118 = 0b01110110
        sub_exp_120 = torch.cat([zeros, ones, ones, ones, ones, zeros, zeros, zeros], dim=-1)
        sub_exp_119 = torch.cat([zeros, ones, ones, ones, zeros, ones, ones, ones], dim=-1)
        sub_exp_118 = torch.cat([zeros, ones, ones, ones, zeros, ones, ones, zeros], dim=-1)
        
        # 使用向量化 MUX 门选择 subnormal 指数
        # 扩展选择信号到 8 位
        m_is_01x_exp = m_is_01x.expand(*m_is_01x.shape[:-1], 8)
        m_is_1xx_exp = m_is_1xx.expand(*m_is_1xx.shape[:-1], 8)
        # 01x 时选 119, 001 时选 118
        sel_01x_or_001 = self.vec_mux(m_is_01x_exp, sub_exp_119, sub_exp_118)
        # 1xx 时选 120, 否则选上面的结果
        sub_exp = self.vec_mux(m_is_1xx_exp, sub_exp_120, sel_01x_or_001)
        
        # Subnormal FP32 尾数 (23位, MSB first)
        # M=1: 0000... (M22..M0 = 0)
        # M=2: 0000...
        # M=3: 1000... (M22=1)
        # M=4: 0000...
        # M=5: 0100... (M21=1) 
        # M=6: 1000... (M22=1)
        # M=7: 1100... (M22=1, M21=1)
        # 使用向量化门电路
        sub_m22_part1 = self.vec_and(m_is_01x, m0)  # M=3
        sub_m22_part2 = self.vec_and(m_is_1xx, m1)  # M>=6
        sub_m22 = self.vec_or(sub_m22_part1, sub_m22_part2)
        sub_m21 = self.vec_and(m_is_1xx, m0)  # M=5 or M=7
        sub_mant = torch.cat([sub_m22, sub_m21] + [zeros] * 21, dim=-1)
        
        # 如果 E=0 且 M=0 (真零)，保持 FP32 E=0, M=0
        # 否则使用 subnormal 结果或 normal 结果
        zero_exp = torch.cat([zeros] * 8, dim=-1)
        zero_mant = torch.cat([zeros] * 23, dim=-1)
        
        # 向量化 NOT 和 AND 操作
        not_subnormal = self.vec_not(is_subnormal)
        not_m_nonzero = self.vec_not(m_nonzero)
        is_true_zero = self.vec_and(e_is_zero, not_m_nonzero)
        not_is_true_zero = self.vec_not(is_true_zero)
        
        # 选择指数 (使用向量化 MUX 门)
        is_subnormal_exp = is_subnormal.expand(*is_subnormal.shape[:-1], 8)
        is_true_zero_exp = is_true_zero.expand(*is_true_zero.shape[:-1], 8)
        # subnormal 时使用 sub_exp，normal 时使用 fp32_exp_raw
        e_normal_or_sub = self.vec_mux(is_subnormal_exp, sub_exp, fp32_exp_raw)
        # 真零时使用 0
        fp32_exp = self.vec_mux(is_true_zero_exp, zero_exp, e_normal_or_sub)
        
        # 选择尾数 (使用向量化 MUX 门)
        normal_mant = torch.cat([m2, m1, m0] + [zeros] * 20, dim=-1)
        is_subnormal_mant = is_subnormal.expand(*is_subnormal.shape[:-1], 23)
        is_true_zero_mant = is_true_zero.expand(*is_true_zero.shape[:-1], 23)
        # subnormal 时使用 sub_mant，normal 时使用 normal_mant
        m_normal_or_sub = self.vec_mux(is_subnormal_mant, sub_mant, normal_mant)
        # 真零时使用 0
        fp32_mant = self.vec_mux(is_true_zero_mant, zero_mant, m_normal_or_sub)
        
        # 组装 FP32: [S, E7..E0, M22..M0]
        fp32_pulse = torch.cat([s, fp32_exp, fp32_mant], dim=-1)
        
        return fp32_pulse
    
    def reset(self):
        self.e_or_01.reset()
        self.e_or_23.reset()
        self.e_or_all.reset()
        self.e_is_zero_not.reset()
        self.m_or_01.reset()
        self.m_or_all.reset()
        self.exp_adder.reset()
        self.vec_exp_mux.reset()
        self.vec_not.reset()
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_mux.reset()


# ==============================================================================
# FP32 -> FP8 转换器
# ==============================================================================
class FP32ToFP8Converter(nn.Module):
    """FP32 -> FP8 E4M3 转换器（100%纯SNN门电路）
    
    需要处理：
    - 指数转换: FP8_exp = FP32_exp - 120
    - 尾数截断 + RNE舍入: 23位 -> 3位
    - 溢出/下溢处理
    """
    def __init__(self):
        super().__init__()
        
        # 指数减法 (LSB first)
        self.exp_sub = Subtractor8Bit()
        
        # 溢出检测: FP32_exp > 134 (FP8_exp > 14)
        self.overflow_cmp = Comparator8Bit()
        
        # 下溢检测: FP32_exp < 120 (FP8_exp < 0)
        self.underflow_cmp = Comparator8Bit()
        
        # RNE 舍入
        self.rne_or = ORGate()
        self.rne_and = ANDGate()
        
        # Sticky bit OR 树
        self.sticky_or = nn.ModuleList([ORGate() for _ in range(19)])
        
        # 尾数 +1 (4位: 包含进位检测)
        self.mant_inc = RippleCarryAdder(bits=4)
        
        # 尾数清零（舍入进位时）
        self.not_carry = NOTGate()
        self.mant_clear_and = nn.ModuleList([ANDGate() for _ in range(3)])
        
        # 指数+1（舍入进位时）
        self.exp_inc = RippleCarryAdder(bits=4)
        
        # 溢出/下溢/正常结果选择
        self.overflow_mux_e = nn.ModuleList([MUXGate() for _ in range(4)])
        self.overflow_mux_m = nn.ModuleList([MUXGate() for _ in range(3)])
        self.underflow_mux_e = nn.ModuleList([MUXGate() for _ in range(4)])
        self.underflow_mux_m = nn.ModuleList([MUXGate() for _ in range(3)])
        
        # 舍入后指数选择
        self.round_exp_mux = nn.ModuleList([MUXGate() for _ in range(4)])
        
        # Subnormal 输出处理
        # 检测 FP32_exp ∈ [118, 121): subnormal 区间
        self.subnorm_cmp_low = Comparator8Bit()  # exp >= 118
        self.subnorm_cmp_high = Comparator8Bit() # exp < 121
        
        # Subnormal 尾数选择 MUX
        self.subnorm_mux_m = nn.ModuleList([MUXGate() for _ in range(3)])
        
        # ===== 纯 SNN 门电路替换比较运算 =====
        # Subnormal 处理的 sticky bit OR 树
        # sticky_120: OR of 20 bits (m19..m0)
        self.sticky_120_or = nn.ModuleList([ORGate() for _ in range(19)])
        # sticky_119: OR of 21 bits (m20..m0)
        self.sticky_119_or = nn.ModuleList([ORGate() for _ in range(20)])
        # sticky_118: OR of 22 bits (m21..m0)
        self.sticky_118_or = nn.ModuleList([ORGate() for _ in range(21)])
        # mant_is_zero_117: 检测 23 位尾数是否全 0
        self.mant_117_or = nn.ModuleList([ORGate() for _ in range(22)])
        self.mant_117_not = NOTGate()
        
        # 进位检测 (A+B 的进位 = A AND B)
        self.sub_m2_120_carry_and = ANDGate()
        self.sub_m1_120_carry_and = ANDGate()
        self.sub_m2_119_carry_and = ANDGate()
        self.sub_m1_119_carry_and = ANDGate()
        self.sub_m2_118_carry_and = ANDGate()
        
        # XOR 门用于 sum % 2 (A + B - 2*A*B = A XOR B)
        self.sub_m2_120_xor = XORGate()
        self.sub_m1_120_xor = XORGate()
        self.sub_m2_119_xor = XORGate()
        self.sub_m1_119_xor = XORGate()
        self.sub_m2_118_xor = XORGate()
        
        # ===== 纯 SNN NOT 门 (替换 ones - x) =====
        self.not_exp_ge_117 = NOTGate()
        self.not_exp_lsb0 = NOTGate()
        self.not_exp_lsb1 = NOTGate()
        self.not_exp_lsb2 = NOTGate()
        self.not_mant_is_zero_117 = NOTGate()
        self.not_subnorm_overflow = NOTGate()
        
        # ===== 纯 SNN OR 门 (替换 a+b-a*b) =====
        self.s_or_l_120_or = ORGate()
        self.s_or_l_119_or = ORGate()
        self.s_or_l_118_or = ORGate()
        self.exp_ge_or_eq_117_or = ORGate()
        
        # ===== 纯 SNN AND 门 (替换 a*b) =====
        self.and_is_subnormal = ANDGate()  # is_below_normal AND exp_ge_117
        self.and_is_underflow = ANDGate()  # is_below_normal AND not_exp_ge_117
        self.and_subnorm_exp_val = ANDGate()  # for subnorm_overflow selection
        
        # ===== 纯 SNN MUX 门 (选择 subnorm 指数) =====
        self.subnorm_exp_mux = nn.ModuleList([MUXGate() for _ in range(4)])
        
        # ===== 纯 SNN AND 门 (is_exp_117/118/119/120 计算) =====
        self.and_exp_117_a = ANDGate()  # exp_lsb0 AND not_exp_lsb1
        self.and_exp_117_b = ANDGate()  # (exp_lsb0 AND not_exp_lsb1) AND exp_lsb2
        self.and_exp_118_a = ANDGate()  # not_exp_lsb0 AND exp_lsb1
        self.and_exp_118_b = ANDGate()  # (not_exp_lsb0 AND exp_lsb1) AND exp_lsb2
        self.and_exp_119_a = ANDGate()  # exp_lsb0 AND exp_lsb1
        self.and_exp_119_b = ANDGate()  # (exp_lsb0 AND exp_lsb1) AND exp_lsb2
        self.and_exp_120_a = ANDGate()  # not_exp_lsb0 AND not_exp_lsb1
        self.and_exp_120_b = ANDGate()  # above AND not_exp_lsb2
        self.and_exp_120_c = ANDGate()  # above AND exp_lsb3
        
        # ===== 纯 SNN AND 门 (round_up 计算) =====
        self.and_round_up_120 = ANDGate()  # round_120 AND s_or_l_120
        self.and_round_up_119 = ANDGate()  # round_119 AND s_or_l_119
        self.and_round_up_118 = ANDGate()  # round_118 AND s_or_l_118
        
        # ===== 纯 SNN AND 门 (subnorm_overflow 计算) =====
        self.and_subnorm_overflow = ANDGate()  # is_exp_120 AND sub_to_normal_120
        
        # ===== 纯 SNN MUX 门 (subnorm_m 选择) =====
        self.mux_subnorm_m0 = nn.ModuleList([MUXGate() for _ in range(4)])  # 4路选择器
        self.mux_subnorm_m1 = nn.ModuleList([MUXGate() for _ in range(4)])
        self.mux_subnorm_m2 = nn.ModuleList([MUXGate() for _ in range(4)])
        
        # ===== AND 门用于 subnorm_m 的最终选择 =====
        self.and_subnorm_m0 = ANDGate()
        self.and_subnorm_m1 = ANDGate()
        self.and_subnorm_m2 = ANDGate()
        
        # ===== 纯 SNN AND 门 (subnorm 尾数清零) =====
        self.and_subnorm_m0 = ANDGate()
        self.and_subnorm_m1 = ANDGate()
        self.and_subnorm_m2 = ANDGate()
        
    def forward(self, fp32_pulse):
        """
        Args:
            fp32_pulse: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        Returns:
            fp8_pulse: [..., 8] FP8 脉冲 [S, E3..E0, M2..M0]
        """
        self.reset()
        device = fp32_pulse.device
        batch_shape = fp32_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取 FP32 各部分 (MSB first)
        s = fp32_pulse[..., 0:1]
        fp32_exp = fp32_pulse[..., 1:9]  # [E7..E0] MSB first
        fp32_mant = fp32_pulse[..., 9:32]  # [M22..M0] MSB first
        
        # 指数转换: FP8_exp = FP32_exp - 120 (LSB first)
        fp32_exp_lsb = fp32_exp.flip(-1)
        # 120 = 0b01111000, LSB first = [0, 0, 0, 1, 1, 1, 1, 0]
        const_120_lsb = torch.cat([zeros, zeros, zeros, ones, ones, ones, ones, zeros], dim=-1)
        fp8_exp_raw_lsb, _ = self.exp_sub(fp32_exp_lsb, const_120_lsb)
        
        # 取低4位 (LSB first)
        fp8_exp_lsb = fp8_exp_raw_lsb[..., :4]
        fp8_exp = fp8_exp_lsb.flip(-1)  # 转回 MSB first
        
        # 溢出检测: FP32_exp > 134 (0b10000110)
        const_134 = torch.cat([ones, zeros, zeros, zeros, zeros, ones, ones, zeros], dim=-1)
        is_overflow, _ = self.overflow_cmp(fp32_exp, const_134)
        
        # 下溢检测: FP32_exp < 121 (最小 normal = 121)
        const_121 = torch.cat([zeros, ones, ones, ones, ones, zeros, zeros, ones], dim=-1)
        is_below_normal, _ = self.underflow_cmp(const_121, fp32_exp)  # 121 > FP32_exp
        
        # Subnormal 区间检测: 117 <= FP32_exp < 121
        # exp=117 的值可能舍入到最小 subnormal (M=1)
        const_117 = torch.cat([zeros, ones, ones, ones, zeros, ones, zeros, ones], dim=-1)
        exp_ge_117, _ = self.subnorm_cmp_low(fp32_exp, const_117)  # exp >= 117
        _, exp_eq_117 = self.subnorm_cmp_low(fp32_exp, const_117)
        # exp >= 117 用纯 SNN OR 门实现: gt OR eq
        exp_ge_or_eq_117 = self.exp_ge_or_eq_117_or(exp_ge_117, exp_eq_117)
        is_subnormal = self.and_is_subnormal(is_below_normal, exp_ge_or_eq_117)  # 纯SNN AND
        
        # 完全下溢: FP32_exp < 117 (纯SNN NOT + AND)
        not_exp_ge_117 = self.not_exp_ge_117(exp_ge_or_eq_117)
        is_underflow = self.and_is_underflow(is_below_normal, not_exp_ge_117)  # 纯SNN AND
        
        # 尾数截断 + RNE 舍入
        # FP32 mant [M22..M0] MSB first
        # 取高3位 [M22, M21, M20] 作为 FP8 mant
        # Round bit = M19, Sticky = OR(M18..M0)
        m2 = fp32_mant[..., 0:1]  # M22
        m1 = fp32_mant[..., 1:2]  # M21
        m0 = fp32_mant[..., 2:3]  # M20
        L = m0  # Least significant bit of result
        R = fp32_mant[..., 3:4]  # M19 (Round bit)
        
        # Sticky = OR(M18..M0) - 19 bits
        sticky_bits = fp32_mant[..., 4:23]
        S = sticky_bits[..., 0:1]
        for i in range(1, 19):
            S = self.sticky_or[i-1](S, sticky_bits[..., i:i+1])
        
        # RNE: round_up = R AND (S OR L)
        s_or_l = self.rne_or(S, L)
        round_up = self.rne_and(R, s_or_l)
        
        # 尾数 +1 (LSB first)
        # 使用 4 位：[m0, m1, m2, 0]，如果结果的第4位=1，则表示进位
        mant_4bit_lsb = torch.cat([m0, m1, m2, zeros], dim=-1)
        round_inc_lsb = torch.cat([round_up, zeros, zeros, zeros], dim=-1)
        mant_rounded_lsb, _ = self.mant_inc(mant_4bit_lsb, round_inc_lsb)
        
        # 进位检测：第4位（index 3）变为1
        mant_carry = mant_rounded_lsb[..., 3:4]
        
        # 转回 MSB first，取低3位
        fp8_mant = torch.cat([
            mant_rounded_lsb[..., 2:3],  # m2
            mant_rounded_lsb[..., 1:2],  # m1
            mant_rounded_lsb[..., 0:1],  # m0
        ], dim=-1)
        
        # 如果尾数进位，尾数清零
        not_carry = self.not_carry(mant_carry)
        fp8_mant_final = torch.cat([
            self.mant_clear_and[0](not_carry, fp8_mant[..., 0:1]),
            self.mant_clear_and[1](not_carry, fp8_mant[..., 1:2]),
            self.mant_clear_and[2](not_carry, fp8_mant[..., 2:3]),
        ], dim=-1)
        
        # 如果尾数进位，指数+1 (LSB first)
        exp_inc_lsb = torch.cat([mant_carry, zeros, zeros, zeros], dim=-1)
        exp_after_round_lsb, _ = self.exp_inc(fp8_exp_lsb, exp_inc_lsb)
        exp_after_round = exp_after_round_lsb.flip(-1)
        
        # 选择最终指数
        final_exp_pre = []
        for i in range(4):
            e_sel = self.round_exp_mux[i](mant_carry, exp_after_round[..., i:i+1], fp8_exp[..., i:i+1])
            final_exp_pre.append(e_sel)
        final_exp_pre = torch.cat(final_exp_pre, dim=-1)
        
        # 溢出时设置为 NaN: E=1111, M=111
        nan_exp = torch.cat([ones, ones, ones, ones], dim=-1)
        nan_mant = torch.cat([ones, ones, ones], dim=-1)
        
        # 下溢时设置为 0
        zero_exp = torch.cat([zeros, zeros, zeros, zeros], dim=-1)
        zero_mant = torch.cat([zeros, zeros, zeros], dim=-1)
        
        # Subnormal 尾数计算
        # FP8 subnormal: E=0, value = M/8 × 2^(-6)
        # FP32 -> FP8 subnormal:
        # 对于 FP32_exp = 120 (unbiased=-7): 右移 1 位
        #   FP32 = 1.xxx × 2^(-7) → FP8 = 0.1xxx × 2^(-6) → M 取 [1, m22, m21]
        # 对于 FP32_exp = 119 (unbiased=-8): 右移 2 位
        #   FP32 = 1.xxx × 2^(-8) → FP8 = 0.01xxx × 2^(-6) → M 取 [0, 1, m22]
        # 对于 FP32_exp = 118 (unbiased=-9): 右移 3 位
        #   FP32 = 1.xxx × 2^(-9) → FP8 = 0.001xxx × 2^(-6) → M 取 [0, 0, 1]
        
        # 检测 exp (低 4 位: 117=0101, 118=0110, 119=0111, 120=1000)
        exp_lsb0 = fp32_exp[..., 7:8]  # LSB
        exp_lsb1 = fp32_exp[..., 6:7]
        exp_lsb2 = fp32_exp[..., 5:6]
        exp_lsb3 = fp32_exp[..., 4:5]
        
        # 纯SNN NOT 操作
        not_exp_lsb0 = self.not_exp_lsb0(exp_lsb0)
        not_exp_lsb1 = self.not_exp_lsb1(exp_lsb1)
        not_exp_lsb2 = self.not_exp_lsb2(exp_lsb2)
        
        # exp=117 (0b01110101): lsb0=1, lsb1=0, lsb2=1 (纯SNN AND门)
        is_exp_117_a = self.and_exp_117_a(exp_lsb0, not_exp_lsb1)
        is_exp_117 = self.and_exp_117_b(is_exp_117_a, exp_lsb2)
        # exp=118 (0b01110110): lsb0=0, lsb1=1, lsb2=1 (纯SNN AND门)
        is_exp_118_a = self.and_exp_118_a(not_exp_lsb0, exp_lsb1)
        is_exp_118 = self.and_exp_118_b(is_exp_118_a, exp_lsb2)
        # exp=119 (0b01110111): lsb0=1, lsb1=1, lsb2=1 (纯SNN AND门)
        is_exp_119_a = self.and_exp_119_a(exp_lsb0, exp_lsb1)
        is_exp_119 = self.and_exp_119_b(is_exp_119_a, exp_lsb2)
        # exp=120 (0b01111000): lsb0=0, lsb1=0, lsb2=0, lsb3=1 (纯SNN AND门)
        is_exp_120_a = self.and_exp_120_a(not_exp_lsb0, not_exp_lsb1)
        is_exp_120_b = self.and_exp_120_b(is_exp_120_a, not_exp_lsb2)
        is_exp_120 = self.and_exp_120_c(is_exp_120_b, exp_lsb3)
        
        # Subnormal 尾数计算 (考虑 FP32 尾数和舍入)
        # 
        # FP32 = 1.m22m21m20... × 2^(E-127)
        # FP8 subnormal = M/8 × 2^(-6)
        #
        # exp=120 (unbiased=-7): 右移1位
        #   FP8_M = round(1.m22m21m20... / 2) × 8 = round(4 + 2*m22 + m21 + m20/2 + ...)
        #   M = [1, m22, round(m21 + m20/2 + sticky)]
        # exp=119 (unbiased=-8): 右移2位
        #   FP8_M = round(1.m22m21m20... / 4) × 8 = round(2 + m22 + m21/2 + ...)
        #   M = [0, 1, round(m22 + m21/2 + sticky)]
        # exp=118 (unbiased=-9): 右移3位
        #   FP8_M = round(1.m22m21m20... / 8) × 8 = round(1 + m22/2 + ...)
        #   M = [0, 0, 1] (因为 1.xxx / 8 × 8 ≈ 1)
        
        # ===== 对于 exp=120: 舍入位=m20, LSB=m21, sticky=OR(m19..m0) =====
        round_120 = m0  # m20
        lsb_120 = m1    # m21
        # sticky_120: OR(m19..m0) = OR of 20 bits，使用纯 SNN OR 门链
        sticky_120 = fp32_mant[..., 3:4]  # m19 (R)
        for i in range(19):
            sticky_120 = self.sticky_120_or[i](sticky_120, fp32_mant[..., 4+i:5+i])
        # s_or_l_120 = sticky_120 OR lsb_120 (纯 SNN OR 门)
        s_or_l_120 = self.s_or_l_120_or(sticky_120, lsb_120)
        round_up_120 = self.and_round_up_120(round_120, s_or_l_120)  # 纯 SNN AND 门
        sub_m2_120_pre = m1  # m21
        # 进位检测: carry = AND(sub_m2_120_pre, round_up_120)
        sub_m2_120_carry = self.sub_m2_120_carry_and(sub_m2_120_pre, round_up_120)
        # sum % 2 = XOR(sub_m2_120_pre, round_up_120)
        sub_m2_120 = self.sub_m2_120_xor(sub_m2_120_pre, round_up_120)
        # sub_m1_120 = m2 + sub_m2_120_carry
        sub_m1_120_carry = self.sub_m1_120_carry_and(m2, sub_m2_120_carry)
        sub_m1_120 = self.sub_m1_120_xor(m2, sub_m2_120_carry)
        # 如果 m0 进位（从 1xx 变成 10xx），说明 M > 7，需要进位到 normal E=1
        sub_m0_120 = ones
        # 检测是否需要进位到 normal: 当 M 的 m0 位进位时
        sub_to_normal_120 = sub_m1_120_carry
        
        # ===== 对于 exp=119: 舍入位=m21, LSB=m22, sticky=OR(m20..m0) =====
        round_119 = m1  # m21
        lsb_119 = m2    # m22
        # sticky_119: OR(m20..m0) = OR of 21 bits
        sticky_119 = fp32_mant[..., 2:3]  # m20
        for i in range(20):
            sticky_119 = self.sticky_119_or[i](sticky_119, fp32_mant[..., 3+i:4+i])
        s_or_l_119 = self.s_or_l_119_or(sticky_119, lsb_119)  # 纯 SNN OR
        round_up_119 = self.and_round_up_119(round_119, s_or_l_119)  # 纯 SNN AND 门
        sub_m2_119_pre = m2  # m22
        # 进位和 XOR
        sub_m2_119_carry = self.sub_m2_119_carry_and(sub_m2_119_pre, round_up_119)
        sub_m2_119 = self.sub_m2_119_xor(sub_m2_119_pre, round_up_119)
        # sub_m1_119 = ones + sub_m2_119_carry，隐藏位始终为1
        # 当 carry=1 时，1+1=2，进位；当 carry=0 时，1+0=1，不进位
        sub_m1_119_carry = self.sub_m1_119_carry_and(ones, sub_m2_119_carry)  # = sub_m2_119_carry
        sub_m1_119 = self.sub_m1_119_xor(ones, sub_m2_119_carry)  # = NOT(sub_m2_119_carry)
        sub_m0_119 = sub_m1_119_carry
        
        # ===== 对于 exp=118: 舍入位=m22, LSB=隐藏1, sticky=OR(m21..m0) =====
        round_118 = m2  # m22
        lsb_118 = ones  # 隐藏位
        # sticky_118: OR(m21..m0) = OR of 22 bits
        sticky_118 = fp32_mant[..., 1:2]  # m21
        for i in range(21):
            sticky_118 = self.sticky_118_or[i](sticky_118, fp32_mant[..., 2+i:3+i])
        # s_or_l_118 = sticky_118 OR lsb_118 = sticky_118 OR 1 = 1 (纯 SNN OR)
        s_or_l_118 = self.s_or_l_118_or(sticky_118, lsb_118)  # 始终为 1
        round_up_118 = self.and_round_up_118(round_118, s_or_l_118)  # = m22 (纯 SNN AND 门)
        # sub_m2_118 = ones + round_up_118
        sub_m2_118_carry = self.sub_m2_118_carry_and(ones, round_up_118)  # = round_up_118
        sub_m2_118 = self.sub_m2_118_xor(ones, round_up_118)  # = NOT(round_up_118)
        sub_m1_118 = sub_m2_118_carry
        sub_m0_118 = zeros
        
        # ===== 对于 exp=117: 值 = 1.xxx × 2^(-10) =====
        # FP8 最小 subnormal = 2^(-9) = 0.001953125
        # 中点 = 2^(-10) = 0.0009765625
        # 如果 value > 中点，舍入到 M=1
        # 如果 value = 中点（尾数全0），RNE 舍入到偶数 0
        # 
        # 检测尾数是否全 0: 使用 NOT(OR(all_bits))
        mant_any_117 = fp32_mant[..., 0:1]
        for i in range(22):
            mant_any_117 = self.mant_117_or[i](mant_any_117, fp32_mant[..., 1+i:2+i])
        mant_is_zero_117 = self.mant_117_not(mant_any_117)  # 1 if all zeros
        
        # 如果尾数是 0（恰好是中点），舍入到 0；否则舍入到 M=1 (纯SNN NOT)
        not_mant_is_zero_117 = self.not_mant_is_zero_117(mant_is_zero_117)
        sub_m2_117 = not_mant_is_zero_117
        sub_m1_117 = zeros
        sub_m0_117 = zeros
        
        # 检测 subnormal 进位到 normal 的情况 (纯 SNN AND 门)
        # 当 exp=120 且 M 舍入进位时，结果应该是 E=1, M=0 (最小 normal)
        subnorm_overflow = self.and_subnorm_overflow(is_exp_120, sub_to_normal_120)
        
        # 选择正确的 subnormal 尾数 (使用纯 SNN MUX 门级联)
        # MUX 级联: exp_117 -> exp_118 -> exp_119 -> exp_120
        # subnorm_m0
        subnorm_m0_t1 = self.mux_subnorm_m0[0](is_exp_117, sub_m0_117, zeros)  # 117 或 默认0
        subnorm_m0_t2 = self.mux_subnorm_m0[1](is_exp_118, sub_m0_118, subnorm_m0_t1)  # 118 或 上一步
        subnorm_m0_t3 = self.mux_subnorm_m0[2](is_exp_119, sub_m0_119, subnorm_m0_t2)  # 119 或 上一步
        subnorm_m0 = self.mux_subnorm_m0[3](is_exp_120, sub_m0_120, subnorm_m0_t3)  # 120 或 上一步
        # subnorm_m1
        subnorm_m1_t1 = self.mux_subnorm_m1[0](is_exp_117, sub_m1_117, zeros)
        subnorm_m1_t2 = self.mux_subnorm_m1[1](is_exp_118, sub_m1_118, subnorm_m1_t1)
        subnorm_m1_t3 = self.mux_subnorm_m1[2](is_exp_119, sub_m1_119, subnorm_m1_t2)
        subnorm_m1 = self.mux_subnorm_m1[3](is_exp_120, sub_m1_120, subnorm_m1_t3)
        # subnorm_m2
        subnorm_m2_t1 = self.mux_subnorm_m2[0](is_exp_117, sub_m2_117, zeros)
        subnorm_m2_t2 = self.mux_subnorm_m2[1](is_exp_118, sub_m2_118, subnorm_m2_t1)
        subnorm_m2_t3 = self.mux_subnorm_m2[2](is_exp_119, sub_m2_119, subnorm_m2_t2)
        subnorm_m2 = self.mux_subnorm_m2[3](is_exp_120, sub_m2_120, subnorm_m2_t3)
        
        # 纯SNN NOT 操作
        not_subnorm_overflow = self.not_subnorm_overflow(subnorm_overflow)
        
        # 如果 subnormal 进位，使用 E=1, M=0 (纯SNN AND门)
        subnorm_m0_final = self.and_subnorm_m0(subnorm_m0, not_subnorm_overflow)
        subnorm_m1_final = self.and_subnorm_m1(subnorm_m1, not_subnorm_overflow)
        subnorm_m2_final = self.and_subnorm_m2(subnorm_m2, not_subnorm_overflow)
        subnorm_mant = torch.cat([subnorm_m0_final, subnorm_m1_final, subnorm_m2_final], dim=-1)
        
        # subnormal 进位时的指数应该是 E=0001 (1)
        subnorm_overflow_exp = torch.cat([zeros, zeros, zeros, ones], dim=-1)
        
        # 选择最终结果 (使用纯SNN MUX门)
        final_exp = []
        for i in range(4):
            e_ov = self.overflow_mux_e[i](is_overflow, nan_exp[..., i:i+1], final_exp_pre[..., i:i+1])
            # Subnormal 时 E=0，但如果 subnormal 进位则 E=1 (纯SNN MUX)
            e_subnormal_val = self.subnorm_exp_mux[i](subnorm_overflow, subnorm_overflow_exp[..., i:i+1], zero_exp[..., i:i+1])
            e_sub = self.underflow_mux_e[i](is_subnormal, e_subnormal_val, e_ov)
            e_final = self.underflow_mux_e[i](is_underflow, zero_exp[..., i:i+1], e_sub)
            final_exp.append(e_final)
        final_exp = torch.cat(final_exp, dim=-1)
        
        final_mant = []
        for i in range(3):
            m_ov = self.overflow_mux_m[i](is_overflow, nan_mant[..., i:i+1], fp8_mant_final[..., i:i+1])
            # Subnormal 时使用 subnorm_mant
            m_sub = self.subnorm_mux_m[i](is_subnormal, subnorm_mant[..., i:i+1], m_ov)
            m_final = self.underflow_mux_m[i](is_underflow, zero_mant[..., i:i+1], m_sub)
            final_mant.append(m_final)
        final_mant = torch.cat(final_mant, dim=-1)
        
        # 组装 FP8
        fp8_pulse = torch.cat([s, final_exp, final_mant], dim=-1)
        
        return fp8_pulse
    
    def reset(self):
        self.exp_sub.reset()
        self.overflow_cmp.reset()
        self.underflow_cmp.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        for g in self.sticky_or: g.reset()
        self.mant_inc.reset()
        self.not_carry.reset()
        for g in self.mant_clear_and: g.reset()
        self.exp_inc.reset()
        for mux in self.overflow_mux_e: mux.reset()
        for mux in self.overflow_mux_m: mux.reset()
        for mux in self.underflow_mux_e: mux.reset()
        for mux in self.underflow_mux_m: mux.reset()
        for mux in self.round_exp_mux: mux.reset()
        self.subnorm_cmp_low.reset()
        self.subnorm_cmp_high.reset()
        for mux in self.subnorm_mux_m: mux.reset()
        # 纯 SNN 门电路
        for g in self.sticky_120_or: g.reset()
        for g in self.sticky_119_or: g.reset()
        for g in self.sticky_118_or: g.reset()
        for g in self.mant_117_or: g.reset()
        self.mant_117_not.reset()
        self.sub_m2_120_carry_and.reset()
        self.sub_m1_120_carry_and.reset()
        self.sub_m2_119_carry_and.reset()
        self.sub_m1_119_carry_and.reset()
        self.sub_m2_118_carry_and.reset()
        self.sub_m2_120_xor.reset()
        self.sub_m1_120_xor.reset()
        self.sub_m2_119_xor.reset()
        self.sub_m1_119_xor.reset()
        self.sub_m2_118_xor.reset()
        # 纯 SNN NOT 门
        self.not_exp_ge_117.reset()
        self.not_exp_lsb0.reset()
        self.not_exp_lsb1.reset()
        self.not_exp_lsb2.reset()
        self.not_mant_is_zero_117.reset()
        self.not_subnorm_overflow.reset()
        # 纯 SNN OR 门
        self.s_or_l_120_or.reset()
        self.s_or_l_119_or.reset()
        self.s_or_l_118_or.reset()
        self.exp_ge_or_eq_117_or.reset()
        # 新增的纯 SNN 门电路
        self.and_exp_117_a.reset()
        self.and_exp_117_b.reset()
        self.and_exp_118_a.reset()
        self.and_exp_118_b.reset()
        self.and_exp_119_a.reset()
        self.and_exp_119_b.reset()
        self.and_exp_120_a.reset()
        self.and_exp_120_b.reset()
        self.and_exp_120_c.reset()
        self.and_round_up_120.reset()
        self.and_round_up_119.reset()
        self.and_round_up_118.reset()
        self.and_subnorm_overflow.reset()
        for mux in self.mux_subnorm_m0: mux.reset()
        for mux in self.mux_subnorm_m1: mux.reset()
        for mux in self.mux_subnorm_m2: mux.reset()
        self.and_subnorm_m0.reset()
        self.and_subnorm_m1.reset()
        self.and_subnorm_m2.reset()


# ==============================================================================
# FP32 -> FP16 转换器
# ==============================================================================
class FP32ToFP16Converter(nn.Module):
    """FP32 -> FP16 转换器（100%纯SNN门电路）
    
    FP32: [S | E7..E0 | M22..M0], bias=127, 32位
    FP16: [S | E4..E0 | M9..M0], bias=15, 16位
    
    转换规则:
    - sign: 直接复制
    - exp: FP16_exp = FP32_exp - 112 (bias差 = 127 - 15 = 112)
    - mant: 23位截断为10位 + RNE舍入
    """
    def __init__(self):
        super().__init__()
        
        # 指数减法
        self.exp_sub = Subtractor8Bit()
        
        # 溢出/下溢检测
        self.overflow_cmp = Comparator8Bit()
        self.underflow_cmp = Comparator8Bit()
        
        # NaN/Inf检测
        self.nan_exp_and = nn.ModuleList([ANDGate() for _ in range(7)])
        self.nan_mant_or = nn.ModuleList([ORGate() for _ in range(22)])
        self.is_nan_and = ANDGate()
        self.mant_zero_not = NOTGate()
        self.is_inf_and = ANDGate()
        
        # RNE舍入
        self.rne_or = ORGate()
        self.rne_and = ANDGate()
        self.sticky_or = nn.ModuleList([ORGate() for _ in range(11)])
        
        # 尾数+1
        self.mant_inc = RippleCarryAdderNBit(bits=11)
        self.not_carry = NOTGate()
        self.mant_clear_and = nn.ModuleList([ANDGate() for _ in range(10)])
        
        # 指数+1
        self.exp_inc = RippleCarryAdderNBit(bits=5)
        
        # 结果选择MUX
        self.overflow_mux = nn.ModuleList([MUXGate() for _ in range(16)])
        self.underflow_mux = nn.ModuleList([MUXGate() for _ in range(16)])
        self.nan_mux = nn.ModuleList([MUXGate() for _ in range(16)])
        self.inf_mux = nn.ModuleList([MUXGate() for _ in range(16)])
        self.round_exp_mux = nn.ModuleList([MUXGate() for _ in range(5)])
        
    def forward(self, fp32_pulse):
        self.reset()
        device = fp32_pulse.device
        batch_shape = fp32_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        s = fp32_pulse[..., 0:1]
        fp32_exp = fp32_pulse[..., 1:9]
        fp32_mant = fp32_pulse[..., 9:32]
        
        # NaN检测
        e_all_one = fp32_exp[..., 0:1]
        for i in range(1, 8):
            e_all_one = self.nan_exp_and[i-1](e_all_one, fp32_exp[..., i:i+1])
        
        m_any_one = fp32_mant[..., 0:1]
        for i in range(1, 23):
            m_any_one = self.nan_mant_or[i-1](m_any_one, fp32_mant[..., i:i+1])
        
        is_nan = self.is_nan_and(e_all_one, m_any_one)
        m_is_zero = self.mant_zero_not(m_any_one)
        is_inf = self.is_inf_and(e_all_one, m_is_zero)
        
        # 指数转换
        fp32_exp_lsb = fp32_exp.flip(-1)
        const_112_lsb = torch.cat([zeros, zeros, zeros, zeros, ones, ones, ones, zeros], dim=-1)
        fp16_exp_raw_lsb, _ = self.exp_sub(fp32_exp_lsb, const_112_lsb)
        fp16_exp_lsb = fp16_exp_raw_lsb[..., :5]
        fp16_exp = fp16_exp_lsb.flip(-1)
        
        # 溢出检测
        const_142 = torch.cat([ones, zeros, zeros, zeros, ones, ones, ones, zeros], dim=-1)
        is_overflow, _ = self.overflow_cmp(fp32_exp, const_142)
        
        # 下溢检测
        const_113 = torch.cat([zeros, ones, ones, ones, zeros, zeros, zeros, ones], dim=-1)
        is_underflow, _ = self.underflow_cmp(const_113, fp32_exp)
        
        # RNE舍入
        fp16_mant_raw = fp32_mant[..., :10]
        L = fp32_mant[..., 9:10]
        R = fp32_mant[..., 10:11]
        S = fp32_mant[..., 11:12]
        for i in range(11):
            S = self.sticky_or[i](S, fp32_mant[..., 12+i:13+i] if 12+i < 23 else zeros)
        
        s_or_l = self.rne_or(S, L)
        round_up = self.rne_and(R, s_or_l)
        
        fp16_mant_lsb = fp16_mant_raw.flip(-1)
        mant_11bit_lsb = torch.cat([fp16_mant_lsb, zeros], dim=-1)
        round_inc_lsb = torch.cat([round_up] + [zeros]*10, dim=-1)
        mant_rounded_lsb, _ = self.mant_inc(mant_11bit_lsb, round_inc_lsb)
        
        mant_carry = mant_rounded_lsb[..., 10:11]
        fp16_mant = mant_rounded_lsb[..., :10].flip(-1)
        
        not_carry = self.not_carry(mant_carry)
        fp16_mant_final = []
        for i in range(10):
            fp16_mant_final.append(self.mant_clear_and[i](not_carry, fp16_mant[..., i:i+1]))
        fp16_mant_final = torch.cat(fp16_mant_final, dim=-1)
        
        exp_inc_lsb = torch.cat([mant_carry, zeros, zeros, zeros, zeros], dim=-1)
        exp_after_round_lsb, _ = self.exp_inc(fp16_exp_lsb, exp_inc_lsb)
        exp_after_round = exp_after_round_lsb.flip(-1)
        
        final_exp = []
        for i in range(5):
            e_sel = self.round_exp_mux[i](mant_carry, exp_after_round[..., i:i+1], fp16_exp[..., i:i+1])
            final_exp.append(e_sel)
        final_exp = torch.cat(final_exp, dim=-1)
        
        result = torch.cat([s, final_exp, fp16_mant_final], dim=-1)
        
        nan_val = torch.cat([s, ones, ones, ones, ones, ones, ones] + [zeros]*9, dim=-1)
        inf_val = torch.cat([s, ones, ones, ones, ones, ones] + [zeros]*10, dim=-1)
        zero_val = torch.cat([zeros] * 16, dim=-1)
        
        result_bits = []
        for i in range(16):
            bit = self.overflow_mux[i](is_overflow, inf_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        result_bits = []
        for i in range(16):
            bit = self.underflow_mux[i](is_underflow, zero_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        result_bits = []
        for i in range(16):
            bit = self.inf_mux[i](is_inf, inf_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        result_bits = []
        for i in range(16):
            bit = self.nan_mux[i](is_nan, nan_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        return result
    
    def reset(self):
        self.exp_sub.reset()
        self.overflow_cmp.reset()
        self.underflow_cmp.reset()
        for g in self.nan_exp_and: g.reset()
        for g in self.nan_mant_or: g.reset()
        self.is_nan_and.reset()
        self.mant_zero_not.reset()
        self.is_inf_and.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        for g in self.sticky_or: g.reset()
        self.mant_inc.reset()
        self.not_carry.reset()
        for g in self.mant_clear_and: g.reset()
        self.exp_inc.reset()
        for mux in self.overflow_mux: mux.reset()
        for mux in self.underflow_mux: mux.reset()
        for mux in self.nan_mux: mux.reset()
        for mux in self.inf_mux: mux.reset()
        for mux in self.round_exp_mux: mux.reset()
