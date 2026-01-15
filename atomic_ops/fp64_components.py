"""
FP64 相关组件 - 100%纯SNN门电路实现
用于高精度计算（支持Exp等超越函数的精确实现）

FP64 格式: [S | E10..E0 | M51..M0], bias=1023
内部精度: 57位尾数 (hidden + 52 mant + 4 guard)
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate, 
                          HalfAdder, FullAdder, RippleCarryAdder, ORTree, ANDTree)
from .vec_logic_gates import (VecAND, VecOR, VecXOR, VecNOT, VecMUX,
                               VecORTree, VecANDTree, VecAdder, VecSubtractor)


# ==============================================================================
# 参数化组件 (扩展到FP64位宽)
# ==============================================================================
class Comparator11Bit(nn.Module):
    """11位比较器 (FP64指数) - 向量化SNN实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 11
        nt = neuron_template
        
        # 向量化门电路
        self.vec_xor = VecXOR(neuron_template=nt)
        self.vec_not_xor = VecNOT(neuron_template=nt)
        self.vec_not_b = VecNOT(neuron_template=nt)
        self.vec_gt_and = VecAND(neuron_template=nt)
        self.eq_tree = VecANDTree(neuron_template=nt)
        self.eq_prefix_and = nn.ModuleList([VecAND(neuron_template=nt) for _ in range(10)])
        self.result_and = VecAND(neuron_template=nt)
        self.result_or = VecOR(neuron_template=nt)
        
    def forward(self, A, B):
        """A, B: [..., 11] (MSB first)"""
        self.reset()
        n = self.bits
        
        # 向量化计算
        xor_all = self.vec_xor(A, B)
        eq_all = self.vec_not_xor(xor_all)
        not_b_all = self.vec_not_b(B)
        gt_all = self.vec_gt_and(A, not_b_all)
        
        a_eq_b = self.eq_tree(eq_all)
        
        eq_prefix = eq_all[..., 0:1]
        a_gt_b = gt_all[..., 0:1]
        
        for i in range(1, n):
            term = self.result_and(eq_prefix, gt_all[..., i:i+1])
            a_gt_b = self.result_or(a_gt_b, term)
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


class Comparator53Bit(nn.Module):
    """53位比较器 (FP64尾数) - 向量化SNN实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 53
        nt = neuron_template
        
        # 向量化门电路
        self.vec_xor = VecXOR(neuron_template=nt)
        self.vec_not_xor = VecNOT(neuron_template=nt)
        self.vec_not_b = VecNOT(neuron_template=nt)
        self.vec_gt_and = VecAND(neuron_template=nt)
        self.eq_tree = VecANDTree(neuron_template=nt)
        self.eq_prefix_and = nn.ModuleList([VecAND(neuron_template=nt) for _ in range(52)])
        self.result_and = VecAND(neuron_template=nt)
        self.result_or = VecOR(neuron_template=nt)
        
    def forward(self, A, B):
        """A, B: [..., 53] (MSB first)"""
        self.reset()
        n = self.bits
        
        # 向量化计算
        xor_all = self.vec_xor(A, B)
        eq_all = self.vec_not_xor(xor_all)
        not_b_all = self.vec_not_b(B)
        gt_all = self.vec_gt_and(A, not_b_all)
        
        a_eq_b = self.eq_tree(eq_all)
        
        eq_prefix = eq_all[..., 0:1]
        a_gt_b = gt_all[..., 0:1]
        
        for i in range(1, n):
            term = self.result_and(eq_prefix, gt_all[..., i:i+1])
            a_gt_b = self.result_or(a_gt_b, term)
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


class Subtractor11Bit(nn.Module):
    """11位减法器 (FP64指数差) - 向量化SNN (LSB first)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 11
        nt = neuron_template
        
        # 向量化门电路复用
        self.vec_xor1 = VecXOR(neuron_template=nt)
        self.vec_xor2 = VecXOR(neuron_template=nt)
        self.vec_not_a = VecNOT(neuron_template=nt)
        self.vec_and1 = VecAND(neuron_template=nt)
        self.vec_and2 = VecAND(neuron_template=nt)
        self.vec_and3 = VecAND(neuron_template=nt)
        self.vec_or1 = VecOR(neuron_template=nt)
        self.vec_or2 = VecOR(neuron_template=nt)
        
    def forward(self, A, B, Bin=None):
        """A - B, LSB first"""
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
        
        return torch.cat(diffs, dim=-1), borrow
    
    def reset(self):
        self.vec_xor1.reset()
        self.vec_xor2.reset()
        self.vec_not_a.reset()
        self.vec_and1.reset()
        self.vec_and2.reset()
        self.vec_and3.reset()
        self.vec_or1.reset()
        self.vec_or2.reset()


class RippleCarryAdder57Bit(nn.Module):
    """57位加法器 (FP64内部精度) - 向量化SNN (LSB first)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 57
        self.vec_adder = VecAdder(57, neuron_template=neuron_template)
        
    def forward(self, A, B, Cin=None):
        """A + B, LSB first"""
        self.reset()
        result, cout = self.vec_adder(A, B)
        return result, cout
    
    def reset(self):
        self.vec_adder.reset()


class Subtractor57Bit(nn.Module):
    """57位减法器 (FP64内部精度) - 向量化SNN (LSB first)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 57
        nt = neuron_template
        
        # 向量化门电路复用
        self.vec_xor1 = VecXOR(neuron_template=nt)
        self.vec_xor2 = VecXOR(neuron_template=nt)
        self.vec_not_a = VecNOT(neuron_template=nt)
        self.vec_and1 = VecAND(neuron_template=nt)
        self.vec_and2 = VecAND(neuron_template=nt)
        self.vec_and3 = VecAND(neuron_template=nt)
        self.vec_or1 = VecOR(neuron_template=nt)
        self.vec_or2 = VecOR(neuron_template=nt)
        
    def forward(self, A, B, Bin=None):
        """A - B, LSB first"""
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
        
        return torch.cat(diffs, dim=-1), borrow
    
    def reset(self):
        self.vec_xor1.reset()
        self.vec_xor2.reset()
        self.vec_not_a.reset()
        self.vec_and1.reset()
        self.vec_and2.reset()
        self.vec_and3.reset()
        self.vec_or1.reset()
        self.vec_or2.reset()


# ==============================================================================
# 桶形移位器
# ==============================================================================
class BarrelShifterRight57(nn.Module):
    """57位桶形右移位器 - 向量化SNN实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.data_bits = 57
        self.shift_bits = 6  # 最多移63位
        nt = neuron_template
        
        # 每层使用一个 VecMUX 处理所有位
        self.vec_mux_layers = nn.ModuleList([VecMUX(neuron_template=nt) for _ in range(self.shift_bits)])
            
    def forward(self, X, shift):
        """X: [..., 57], shift: [..., 6] (MSB first)"""
        self.reset()
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 构建移位后的版本 (并行)
            if shift_amt < self.data_bits:
                shifted = torch.cat([
                    zeros.expand(*zeros.shape[:-1], shift_amt),
                    current[..., :-shift_amt]
                ], dim=-1)
            else:
                shifted = zeros.expand(*zeros.shape[:-1], self.data_bits)
            
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)
            current = self.vec_mux_layers[layer](s_bit_expanded, shifted, current)
        
        return current
    
    def reset(self):
        for mux in self.vec_mux_layers:
            mux.reset()


class BarrelShifterRight57WithSticky(nn.Module):
    """57位桶形右移位器 - 向量化SNN实现 - 输出sticky bit
    
    sticky = OR(所有被移出的位)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.data_bits = 57
        self.shift_bits = 6
        nt = neuron_template
        
        # 向量化门电路
        self.vec_mux_layers = nn.ModuleList([VecMUX(neuron_template=nt) for _ in range(self.shift_bits)])
        self.sticky_or_tree = VecORTree(neuron_template=nt)
        self.sticky_mux = nn.ModuleList([VecMUX(neuron_template=nt) for _ in range(self.shift_bits)])
        self.sticky_accum_or = VecOR(neuron_template=nt)
        
    def forward(self, X, shift):
        """X: [..., 57], shift: [..., 6] (MSB first)
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
                shifted_out_bits = current[..., start_idx:]
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
            
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)
            current = self.vec_mux_layers[layer](s_bit_expanded, shifted, current)
        
        return current, sticky_accum
    
    def reset(self):
        for mux in self.vec_mux_layers: mux.reset()
        self.sticky_or_tree.reset()
        for mux in self.sticky_mux: mux.reset()
        self.sticky_accum_or.reset()


class BarrelShifterLeft57(nn.Module):
    """57位桶形左移位器 - 向量化SNN实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.data_bits = 57
        self.shift_bits = 6
        nt = neuron_template
        
        # 向量化门电路
        self.vec_mux_layers = nn.ModuleList([VecMUX(neuron_template=nt) for _ in range(self.shift_bits)])
            
    def forward(self, X, shift):
        """X: [..., 57], shift: [..., 6] (MSB first)"""
        self.reset()
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]
            
            # 构建左移后的版本 (并行)
            if shift_amt < self.data_bits:
                shifted = torch.cat([
                    current[..., shift_amt:],
                    zeros.expand(*zeros.shape[:-1], shift_amt)
                ], dim=-1)
            else:
                shifted = zeros.expand(*zeros.shape[:-1], self.data_bits)
            
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)
            current = self.vec_mux_layers[layer](s_bit_expanded, shifted, current)
        
        return current
    
    def reset(self):
        for mux in self.vec_mux_layers:
            mux.reset()


# ==============================================================================
# 前导零检测器
# ==============================================================================
class LeadingZeroDetector57(nn.Module):
    """57位前导零检测器 - 向量化SNN实现
    
    使用向量化门电路复用
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化门电路复用
        self.vec_not_found = VecNOT(neuron_template=nt)
        self.vec_and_first = VecAND(neuron_template=nt)
        self.vec_or_found = VecOR(neuron_template=nt)
        self.vec_or_lzc = VecOR(neuron_template=nt)
        
        # 全零检测
        self.or_tree = VecORTree(neuron_template=nt)
        self.vec_not_all_zero = VecNOT(neuron_template=nt)
        self.vec_or_final = VecOR(neuron_template=nt)
        
    def forward(self, X):
        """X: [..., 57] MSB first, returns: [..., 6] LZC MSB first"""
        self.reset()
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        # 初始化lzc为全0
        lzc = [zeros.clone() for _ in range(6)]
        
        # found = 是否已找到第一个1
        found = zeros.clone()
        
        for i in range(57):
            bit = X[..., i:i+1]
            
            # 使用向量化门电路复用
            not_found = self.vec_not_found(found)
            is_first = self.vec_and_first(bit, not_found)
            
            # 如果is_first=1, 累积位置到lzc
            for j in range(6):
                pos_bit = (i >> (5 - j)) & 1
                if pos_bit:
                    lzc[j] = self.vec_or_lzc(lzc[j], is_first)
            
            # found = found OR is_first
            found = self.vec_or_found(found, is_first)
        
        # 检测是否全零 (使用 OR 树)
        any_one = self.or_tree(X)
        all_zero = self.vec_not_all_zero(any_one)
        
        # 如果全零, lzc = 57 = 0b111001
        lzc_57_bits = [1, 1, 1, 0, 0, 1]
        for j in range(6):
            if lzc_57_bits[j]:
                lzc[j] = self.vec_or_final(lzc[j], all_zero)
        
        return torch.cat(lzc, dim=-1)
    
    def reset(self):
        self.vec_not_found.reset()
        self.vec_and_first.reset()
        self.vec_or_found.reset()
        self.vec_or_lzc.reset()
        self.or_tree.reset()
        self.vec_not_all_zero.reset()
        self.vec_or_final.reset()


# ==============================================================================
# FP32 <-> FP64 转换器
# ==============================================================================
class FP32ToFP64Converter(nn.Module):
    """FP32 -> FP64 转换器 (100%纯SNN门电路)
    
    FP32: [S | E7..E0 | M22..M0], bias=127, 32位
    FP64: [S | E10..E0 | M51..M0], bias=1023, 64位
    
    转换规则 (Normal):
    - sign: 直接复制
    - exp: FP64_exp = FP32_exp + 896 (bias差 = 1023 - 127 = 896)
    - mant: 低位补零 (23位 -> 52位)
    
    转换规则 (Subnormal/Zero):
    - FP32 E=0 时保持 FP64 E=0
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 向量化检测门电路
        self.e_or_tree = VecORTree(neuron_template=nt)      # 检测 E≠0
        self.e_is_zero_not = VecNOT(neuron_template=nt)
        self.m_or_tree = VecORTree(neuron_template=nt)      # 检测 M≠0
        self.exp_all_one_and_tree = VecANDTree(neuron_template=nt)  # 检测 E=全1

        # 11位加法器: FP32_exp (8位扩展) + 896
        self.exp_adder = RippleCarryAdder(bits=11, neuron_template=nt)

        # 向量化MUX
        self.exp_mux = VecMUX(neuron_template=nt)           # E=0时选择0指数
        self.is_nan_and = VecAND(neuron_template=nt)
        self.mant_zero_not = VecNOT(neuron_template=nt)
        self.is_inf_and = VecAND(neuron_template=nt)
        self.nan_mux = VecMUX(neuron_template=nt)           # NaN输出选择
        self.inf_mux = VecMUX(neuron_template=nt)           # Inf输出选择

    def forward(self, fp32_pulse):
        """
        Args:
            fp32_pulse: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        Returns:
            fp64_pulse: [..., 64] FP64 脉冲 [S, E10..E0, M51..M0]
        """
        self.reset()
        device = fp32_pulse.device
        batch_shape = fp32_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        zeros_11 = torch.zeros(batch_shape + (11,), device=device)
        zeros_52 = torch.zeros(batch_shape + (52,), device=device)

        # 提取 FP32 各部分
        s = fp32_pulse[..., 0:1]
        e = fp32_pulse[..., 1:9]    # [E7..E0] MSB first
        m = fp32_pulse[..., 9:32]   # [M22..M0] MSB first

        # 向量化检测 E=0, M≠0, E=全1
        e_any = self.e_or_tree(e)                    # [..., 1]
        e_is_zero = self.e_is_zero_not(e_any)       # [..., 1]
        m_any = self.m_or_tree(m)                    # [..., 1]
        e_all_one = self.exp_all_one_and_tree(e)   # [..., 1]

        m_is_zero = self.mant_zero_not(m_any)
        is_nan = self.is_nan_and(e_all_one, m_any)
        is_inf = self.is_inf_and(e_all_one, m_is_zero)

        # FP32 exp 扩展到 11 位 (LSB first for adder)
        e_lsb = e.flip(-1)  # 转 LSB first
        fp32_exp_11bit_lsb = torch.cat([e_lsb, zeros, zeros, zeros], dim=-1)

        # +896 = 0b01110000000, LSB first: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        const_896_lsb = torch.cat([zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                                    ones, ones, ones, zeros], dim=-1)

        # 加法 (LSB first)
        fp64_exp_raw_lsb, _ = self.exp_adder(fp32_exp_11bit_lsb, const_896_lsb)

        # 转回 MSB first
        fp64_exp_raw = fp64_exp_raw_lsb.flip(-1)

        # E=0 时保持 0 (向量化MUX)
        e_is_zero_exp = e_is_zero.expand_as(fp64_exp_raw)
        fp64_exp = self.exp_mux(e_is_zero_exp, zeros_11, fp64_exp_raw)

        # 尾数: 高23位复制, 低29位补0
        fp64_mant = torch.cat([m, zeros.expand(batch_shape + (29,))], dim=-1)

        # 组装 FP64
        result = torch.cat([s, fp64_exp, fp64_mant], dim=-1)

        # NaN处理: FP64 NaN = S, E=全1, M=非零 (向量化MUX)
        nan_exp = ones.expand(batch_shape + (11,))
        nan_mant = torch.cat([ones, zeros.expand(batch_shape + (51,))], dim=-1)
        nan_result = torch.cat([s, nan_exp, nan_mant], dim=-1)
        is_nan_64 = is_nan.expand_as(result)
        result = self.nan_mux(is_nan_64, nan_result, result)

        # Inf处理: FP64 Inf = S, E=全1, M=0 (向量化MUX)
        inf_result = torch.cat([s, nan_exp, zeros_52], dim=-1)
        is_inf_64 = is_inf.expand_as(result)
        result = self.inf_mux(is_inf_64, inf_result, result)

        return result

    def reset(self):
        self.e_or_tree.reset()
        self.e_is_zero_not.reset()
        self.m_or_tree.reset()
        self.exp_all_one_and_tree.reset()
        self.exp_adder.reset()
        self.exp_mux.reset()
        self.is_nan_and.reset()
        self.mant_zero_not.reset()
        self.is_inf_and.reset()
        self.nan_mux.reset()
        self.inf_mux.reset()


class FP64ToFP32Converter(nn.Module):
    """FP64 -> FP32 转换器 (100%纯SNN门电路 - 向量化版本)

    FP64: [S | E10..E0 | M51..M0], bias=1023, 64位
    FP32: [S | E7..E0 | M22..M0], bias=127, 32位

    转换规则:
    - sign: 直接复制
    - exp: FP32_exp = FP64_exp - 896
    - mant: 截断 + RNE舍入 (52位 -> 23位)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 指数减法
        self.exp_sub = Subtractor11Bit(neuron_template=nt)

        # 溢出/下溢检测
        self.overflow_cmp = Comparator11Bit(neuron_template=nt)
        self.underflow_cmp = Comparator11Bit(neuron_template=nt)

        # 向量化 NaN/Inf 检测
        self.nan_exp_and_tree = VecANDTree(neuron_template=nt)
        self.nan_mant_or_tree = VecORTree(neuron_template=nt)
        self.is_nan_and = VecAND(neuron_template=nt)
        self.mant_zero_not = VecNOT(neuron_template=nt)
        self.is_inf_and = VecAND(neuron_template=nt)

        # 向量化 RNE 舍入
        self.sticky_or_tree = VecORTree(neuron_template=nt)
        self.rne_or = VecOR(neuron_template=nt)
        self.rne_and = VecAND(neuron_template=nt)

        # 尾数+1
        self.mant_inc = RippleCarryAdder(bits=24, neuron_template=nt)
        self.not_carry = VecNOT(neuron_template=nt)
        self.mant_clear_and = VecAND(neuron_template=nt)

        # 指数+1
        self.exp_inc = RippleCarryAdder(bits=8, neuron_template=nt)

        # 向量化结果选择MUX
        self.round_exp_mux = VecMUX(neuron_template=nt)
        self.overflow_mux = VecMUX(neuron_template=nt)
        self.underflow_mux = VecMUX(neuron_template=nt)
        self.nan_mux = VecMUX(neuron_template=nt)
        self.inf_mux = VecMUX(neuron_template=nt)

    def forward(self, fp64_pulse):
        """
        Args:
            fp64_pulse: [..., 64] FP64 脉冲 [S, E10..E0, M51..M0]
        Returns:
            fp32_pulse: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        self.reset()
        device = fp64_pulse.device
        batch_shape = fp64_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        s = fp64_pulse[..., 0:1]
        fp64_exp = fp64_pulse[..., 1:12]   # [E10..E0] MSB first
        fp64_mant = fp64_pulse[..., 12:64]  # [M51..M0] MSB first

        # 向量化 NaN/Inf 检测
        e_all_one = self.nan_exp_and_tree(fp64_exp)    # [..., 1]
        m_any_one = self.nan_mant_or_tree(fp64_mant)   # [..., 1]

        is_nan = self.is_nan_and(e_all_one, m_any_one)
        m_is_zero = self.mant_zero_not(m_any_one)
        is_inf = self.is_inf_and(e_all_one, m_is_zero)

        # 指数转换: FP32_exp = FP64_exp - 896 (LSB first)
        fp64_exp_lsb = fp64_exp.flip(-1)
        const_896_lsb = torch.cat([zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                                    ones, ones, ones, zeros], dim=-1)
        fp32_exp_raw_lsb, _ = self.exp_sub(fp64_exp_lsb, const_896_lsb)

        # 取低8位
        fp32_exp_lsb = fp32_exp_raw_lsb[..., :8]
        fp32_exp = fp32_exp_lsb.flip(-1)

        # 溢出检测: FP64_exp > 1150
        const_1150 = torch.cat([ones, zeros, zeros, zeros, ones, ones, ones, ones, ones, ones, zeros], dim=-1)
        is_overflow, _ = self.overflow_cmp(fp64_exp, const_1150)

        # 下溢检测: FP64_exp < 897
        const_897 = torch.cat([zeros, ones, ones, ones, zeros, zeros, zeros, zeros, zeros, zeros, ones], dim=-1)
        is_underflow, _ = self.underflow_cmp(const_897, fp64_exp)

        # 向量化 RNE 舍入
        fp32_mant_raw = fp64_mant[..., :23]
        L = fp64_mant[..., 22:23]   # LSB
        R = fp64_mant[..., 23:24]   # Round bit
        sticky_bits = fp64_mant[..., 24:52]  # 28 sticky bits
        S = self.sticky_or_tree(sticky_bits)  # 向量化 OR

        s_or_l = self.rne_or(S, L)
        round_up = self.rne_and(R, s_or_l)

        # 尾数+1 (LSB first)
        fp32_mant_lsb = fp32_mant_raw.flip(-1)
        mant_24bit_lsb = torch.cat([fp32_mant_lsb, zeros], dim=-1)
        round_inc_lsb = torch.cat([round_up, zeros.expand(batch_shape + (23,))], dim=-1)
        mant_rounded_lsb, _ = self.mant_inc(mant_24bit_lsb, round_inc_lsb)

        mant_carry = mant_rounded_lsb[..., 23:24]
        fp32_mant = mant_rounded_lsb[..., :23].flip(-1)

        # 向量化: 如果尾数进位, 尾数清零
        not_carry = self.not_carry(mant_carry)
        not_carry_23 = not_carry.expand_as(fp32_mant)
        fp32_mant_final = self.mant_clear_and(not_carry_23, fp32_mant)

        # 如果尾数进位, 指数+1
        exp_inc_lsb = torch.cat([mant_carry, zeros.expand(batch_shape + (7,))], dim=-1)
        exp_after_round_lsb, _ = self.exp_inc(fp32_exp_lsb, exp_inc_lsb)
        exp_after_round = exp_after_round_lsb.flip(-1)

        # 向量化: 选择最终指数
        mant_carry_8 = mant_carry.expand_as(fp32_exp)
        final_exp = self.round_exp_mux(mant_carry_8, exp_after_round, fp32_exp)

        # 组装结果
        result = torch.cat([s, final_exp, fp32_mant_final], dim=-1)

        # 特殊值处理 (向量化)
        ones_8 = ones.expand(batch_shape + (8,))
        zeros_22 = zeros.expand(batch_shape + (22,))
        zeros_23 = zeros.expand(batch_shape + (23,))
        zeros_31 = zeros.expand(batch_shape + (31,))

        nan_val = torch.cat([s, ones_8, ones, zeros_22], dim=-1)
        inf_val = torch.cat([s, ones_8, zeros_23], dim=-1)
        zero_val = torch.cat([s, zeros_31], dim=-1)

        # 向量化 MUX 选择
        is_overflow_32 = is_overflow.expand_as(result)
        result = self.overflow_mux(is_overflow_32, inf_val, result)

        is_underflow_32 = is_underflow.expand_as(result)
        result = self.underflow_mux(is_underflow_32, zero_val, result)

        is_inf_32 = is_inf.expand_as(result)
        result = self.inf_mux(is_inf_32, inf_val, result)

        is_nan_32 = is_nan.expand_as(result)
        result = self.nan_mux(is_nan_32, nan_val, result)

        return result

    def reset(self):
        self.exp_sub.reset()
        self.overflow_cmp.reset()
        self.underflow_cmp.reset()
        self.nan_exp_and_tree.reset()
        self.nan_mant_or_tree.reset()
        self.is_nan_and.reset()
        self.mant_zero_not.reset()
        self.is_inf_and.reset()
        self.sticky_or_tree.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.mant_inc.reset()
        self.not_carry.reset()
        self.mant_clear_and.reset()
        self.exp_inc.reset()
        self.round_exp_mux.reset()
        self.overflow_mux.reset()
        self.underflow_mux.reset()
        self.nan_mux.reset()
        self.inf_mux.reset()


