"""
FP16 相关组件 - 100%纯SNN门电路实现
用于多精度累加：FP8 -> FP16累加 -> FP8

FP16 格式: [S | E4 E3 E2 E1 E0 | M9..M0], bias=15
FP8 E4M3:  [S | E3 E2 E1 E0 | M2 M1 M0], bias=7
"""
import torch
import torch.nn as nn
from atomic_ops.core.logic_gates import (HalfAdder, FullAdder, ORTree, ANDTree)
# 注意：使用 VecAdder 代替旧的 RippleCarryAdder（支持 max_param_shape）
from atomic_ops.core.vec_logic_gates import (VecAND, VecOR, VecXOR, VecNOT, VecMUX,
                               VecORTree, VecANDTree, VecAdder, VecSubtractor)


# ==============================================================================
# 参数化比较器
# ==============================================================================
class ComparatorNBit(nn.Module):
    """N位比较器 - 向量化SNN实现

    使用向量化门电路：一次处理所有位的独立操作

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    MAX_BITS = 16  # FP16 最大位宽

    def __init__(self, bits, neuron_template=None):
        super().__init__()
        self.bits = bits
        nt = neuron_template

        # 预分配参数形状
        max_shape = (bits,)
        max_shape_1 = (1,)

        # 向量化门电路：单个实例处理所有位
        self.vec_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape)      # 计算 A XOR B (所有位并行)
        self.vec_not_xor = VecNOT(neuron_template=nt, max_param_shape=max_shape)  # 计算 NOT(A XOR B) = EQ (所有位并行)
        self.vec_not_b = VecNOT(neuron_template=nt, max_param_shape=max_shape)    # 计算 NOT(B) (所有位并行)
        self.vec_gt_and = VecAND(neuron_template=nt, max_param_shape=max_shape)   # 计算 A AND NOT(B) = GT (所有位并行)

        # 前缀逻辑有依赖，使用树归约
        self.eq_tree = VecANDTree(neuron_template=nt, max_param_shape=max_shape)  # 最终 a_eq_b

        # 用于 a_gt_b 的逐位计算（单实例，动态扩展支持）
        self.eq_prefix_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.result_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.result_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        
    def forward(self, A, B):
        """A, B: [..., bits] (MSB first)"""
        n = self.bits

        # 向量化计算：一次处理所有 n 位
        xor_all = self.vec_xor(A, B)                    # [..., n]
        eq_all = self.vec_not_xor(xor_all)             # [..., n]
        not_b_all = self.vec_not_b(B)                  # [..., n]
        gt_all = self.vec_gt_and(A, not_b_all)         # [..., n]
        
        if n == 1:
            return gt_all[..., 0:1], eq_all[..., 0:1]
        
        # a_eq_b = AND(所有 eq 位) - 使用 AND 树
        a_eq_b = self.eq_tree(eq_all)
        
        # a_gt_b 计算（有依赖，需要循环）
        eq_prefix = eq_all[..., 0:1]
        a_gt_b = gt_all[..., 0:1]
        
        for i in range(1, n):
            term = self.result_and(eq_prefix, gt_all[..., i:i+1])
            a_gt_b = self.result_or(a_gt_b, term)

            if i < n - 1:
                eq_prefix = self.eq_prefix_and(eq_prefix, eq_all[..., i:i+1])
        
        return a_gt_b, a_eq_b
    
    def reset(self):
        self.vec_xor.reset()
        self.vec_not_xor.reset()
        self.vec_not_b.reset()
        self.vec_gt_and.reset()
        self.eq_tree.reset()
        self.eq_prefix_and.reset()
        self.result_and.reset()
        self.result_or.reset()


class Comparator5Bit(ComparatorNBit):
    """5位比较器（FP16指数）"""
    def __init__(self, neuron_template=None):
        super().__init__(bits=5, neuron_template=neuron_template)


# ==============================================================================
# 参数化减法器
# ==============================================================================
class SubtractorNBit(nn.Module):
    """N位减法器 - 向量化SNN实现 (A - B with borrow)

    使用向量化门电路复用（借位链有依赖，需循环但复用门电路）

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    MAX_BITS = 16  # FP16 最大位宽

    def __init__(self, bits, neuron_template=None):
        super().__init__()
        self.bits = bits
        nt = neuron_template

        # 预分配参数形状 (单位处理，每次处理1位)
        max_shape_1 = (1,)

        # 向量化门电路复用
        self.vec_xor1 = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_xor2 = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_not_a = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and1 = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and2 = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and3 = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or1 = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or2 = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        
    def forward(self, A, B, Bin=None):
        """A - B, LSB first (index 0 = LSB)"""
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


class Subtractor5Bit(SubtractorNBit):
    """5位减法器（FP16指数差）"""
    def __init__(self, neuron_template=None):
        super().__init__(bits=5, neuron_template=neuron_template)


# ==============================================================================
# 参数化桶形移位器（右移）
# ==============================================================================
class BarrelShifterRightNBit(nn.Module):
    """N位桶形右移位器 - 向量化SNN实现

    每层 MUX 操作并行处理所有位

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    MAX_BITS = 16  # FP16 最大数据位宽

    def __init__(self, data_bits, shift_bits, neuron_template=None):
        super().__init__()
        self.data_bits = data_bits
        self.shift_bits = shift_bits
        nt = neuron_template

        # 预分配参数形状
        max_shape = (data_bits,)

        # 单个 VecMUX 处理所有层 (动态扩展机制支持)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape)
            
    def forward(self, X, shift):
        """X: [..., data_bits], shift: [..., shift_bits] (MSB first)"""
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
            
            # 扩展 s_bit 到所有位
            s_bit_expanded = s_bit.expand(*s_bit.shape[:-1], self.data_bits)

            # 并行 MUX
            current = self.vec_mux(s_bit_expanded, shifted, current)

        return current

    def reset(self):
        self.vec_mux.reset()


class BarrelShifterRight16(BarrelShifterRightNBit):
    """16位桶形右移位器（用于FP16尾数对齐）"""
    def __init__(self, neuron_template=None):
        super().__init__(data_bits=16, shift_bits=5, neuron_template=neuron_template)  # 移位0-31


# ==============================================================================
# FP8 -> FP16 转换器（纯SNN门电路）
# ==============================================================================
class FP8ToFP16Converter(nn.Module):
    """FP8 E4M3 -> FP16 转换器（100%纯SNN门电路）
    
    FP8 E4M3: [S | E3 E2 E1 E0 | M2 M1 M0], bias=7
    FP16:     [S | E4 E3 E2 E1 E0 | M9..M0], bias=15
    
    转换规则（Normal）：
    - sign: 直接复制
    - exp: FP16_exp = FP8_exp + 8 (bias差 = 15 - 7 = 8)
    - mant: 高位对齐，低位补0
    
    转换规则（Subnormal FP8, E=0, M≠0）：
    - 需要归一化：找到尾数中第一个1的位置，计算有效指数
    - FP8 subnormal value = 0.M2M1M0 * 2^(-6)
    - M=4..7 (1xx): lead at bit 2 -> value = 1.xx * 2^(-7) -> FP16 E = 8
    - M=2..3 (01x): lead at bit 1 -> value = 1.x * 2^(-8) -> FP16 E = 7
    - M=1    (001): lead at bit 0 -> value = 1 * 2^(-9) -> FP16 E = 6
    
    转换规则（Zero, E=0, M=0）：
    - FP16 E=0, M=0
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 预分配参数形状
        max_shape_10 = (10,)  # FP16 尾数
        max_shape_5 = (5,)    # FP16 指数

        # 检测 FP8 E=0
        self.e_or_01 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_or_23 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_or_all = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_is_zero_not = VecNOT(neuron_template=nt, max_param_shape=(1,))

        # 检测 M≠0
        self.m_or_01 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.m_or_all = VecOR(neuron_template=nt, max_param_shape=(1,))

        # 检测 subnormal (E=0 AND M≠0)
        self.is_subnorm_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 5位加法器: FP8_exp (4位扩展) + 8
        self.exp_adder = VecAdder(bits=5, neuron_template=nt, max_param_shape=(5,))

        # 前导1检测（用于subnormal）
        # m2=1 -> lead at bit 2
        # m2=0, m1=1 -> lead at bit 1
        # m2=0, m1=0, m0=1 -> lead at bit 0
        self.not_m2 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m1 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.lead_at_1_and = VecAND(neuron_template=nt, max_param_shape=(1,))  # NOT(m2) AND m1
        self.lead_at_0_and1 = VecAND(neuron_template=nt, max_param_shape=(1,))  # NOT(m2) AND NOT(m1)
        self.lead_at_0_and2 = VecAND(neuron_template=nt, max_param_shape=(1,))  # (NOT(m2) AND NOT(m1)) AND m0

        # subnormal 指数/尾数选择 - 统一向量化 MUX (预分配最大形状)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_10)

        # 纯SNN NOT门
        self.not_m_nonzero = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_is_zero = VecNOT(neuron_template=nt, max_param_shape=(1,))

        # 纯SNN AND门 - 用于 zero 检测和零化 - 向量化
        self.and_is_zero = VecAND(neuron_template=nt, max_param_shape=(1,))  # e_is_zero AND not_m_nonzero
        self.vec_and_zero_exp = VecAND(neuron_template=nt, max_param_shape=max_shape_5)    # e_sel AND not_is_zero (5位)
        self.vec_and_zero_mant = VecAND(neuron_template=nt, max_param_shape=max_shape_10)   # m_sel AND not_is_zero (10位)
        
    def forward(self, fp8_pulse):
        """
        Args:
            fp8_pulse: [..., 8] FP8 脉冲 [S, E3, E2, E1, E0, M2, M1, M0]
        Returns:
            fp16_pulse: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        """
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
        
        # 检测 M≠0
        m_or_01 = self.m_or_01(m0, m1)
        m_nonzero = self.m_or_all(m_or_01, m2)
        
        # 检测 subnormal (E=0 AND M≠0)
        is_subnormal = self.is_subnorm_and(e_is_zero, m_nonzero)
        
        # ===== Normal 路径 =====
        # FP8 exp 扩展到 5 位 (LSB first for RippleCarryAdder)
        fp8_exp_5bit_lsb = torch.cat([e0, e1, e2, e3, zeros], dim=-1)
        
        # +8 = 0b01000, LSB first: [0, 0, 0, 1, 0]
        const_8_lsb = torch.cat([zeros, zeros, zeros, ones, zeros], dim=-1)
        
        # 加法 (LSB first)
        fp16_exp_raw_lsb, _ = self.exp_adder(fp8_exp_5bit_lsb, const_8_lsb)
        
        # 转回 MSB first
        fp16_exp_normal = torch.cat([
            fp16_exp_raw_lsb[..., 4:5],
            fp16_exp_raw_lsb[..., 3:4],
            fp16_exp_raw_lsb[..., 2:3],
            fp16_exp_raw_lsb[..., 1:2],
            fp16_exp_raw_lsb[..., 0:1],
        ], dim=-1)
        
        # Normal 尾数: 3位 -> 10位 (高位对齐，低位补0)
        fp16_mant_normal = torch.cat([m2, m1, m0, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        
        # ===== Subnormal 路径 =====
        # 检测前导1位置
        not_m2 = self.not_m2(m2)
        not_m1 = self.not_m1(m1)
        
        lead_at_2 = m2  # m2=1
        lead_at_1 = self.lead_at_1_and(not_m2, m1)  # NOT(m2) AND m1
        lead_at_0_tmp = self.lead_at_0_and1(not_m2, not_m1)  # NOT(m2) AND NOT(m1)
        lead_at_0 = self.lead_at_0_and2(lead_at_0_tmp, m0)  # AND m0
        
        # Subnormal 指数 (MSB first)
        # lead at 2: E = 8 = 01000
        # lead at 1: E = 7 = 00111
        # lead at 0: E = 6 = 00110
        exp_8 = torch.cat([zeros, ones, zeros, zeros, zeros], dim=-1)
        exp_7 = torch.cat([zeros, zeros, ones, ones, ones], dim=-1)
        exp_6 = torch.cat([zeros, zeros, ones, ones, zeros], dim=-1)
        
        # 选择: 先从 lead_at_0 和 lead_at_1 中选，再与 lead_at_2 选 (向量化)
        lead_at_0_exp = lead_at_0.expand_as(exp_6)  # [..., 5]
        sub_exp_01 = self.vec_mux(lead_at_0_exp, exp_6, exp_7)  # [..., 5]

        lead_at_2_exp = lead_at_2.expand_as(exp_8)  # [..., 5]
        sub_exp = self.vec_mux(lead_at_2_exp, exp_8, sub_exp_01)  # [..., 5]
        
        # Subnormal 尾数
        # lead at 2: mant = [m1, m0, 0, 0, 0, 0, 0, 0, 0, 0]
        # lead at 1: mant = [m0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # lead at 0: mant = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mant_lead2 = torch.cat([m1, m0, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        mant_lead1 = torch.cat([m0, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        mant_lead0 = torch.cat([zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)

        # 选择: 先从 lead_at_0 和 lead_at_1 中选，再与 lead_at_2 选 (向量化)
        lead_at_0_mant = lead_at_0.expand_as(mant_lead0)  # [..., 10]
        sub_mant_01 = self.vec_mux(lead_at_0_mant, mant_lead0, mant_lead1)  # [..., 10]

        lead_at_2_mant = lead_at_2.expand_as(mant_lead2)  # [..., 10]
        sub_mant = self.vec_mux(lead_at_2_mant, mant_lead2, sub_mant_01)  # [..., 10]
        
        # ===== 最终选择 (Normal vs Subnormal) =====
        # 对于 zero (E=0, M=0): is_subnormal=0, 使用 normal 路径，但 fp16_exp_normal = 0+8 = 8
        # 这是不对的！zero 应该输出 E=0, M=0
        # 需要特殊处理 zero

        # Zero 检测 (纯SNN门电路)
        not_m_nonzero = self.not_m_nonzero(m_nonzero)
        is_zero = self.and_is_zero(e_is_zero, not_m_nonzero)  # E=0 AND M=0
        not_is_zero = self.not_is_zero(is_zero)

        # 向量化: 先选 normal vs subnormal，再用 AND 处理 zero 情况
        is_subnormal_exp = is_subnormal.expand_as(sub_exp)  # [..., 5]
        fp16_exp_sel = self.vec_mux(is_subnormal_exp, sub_exp, fp16_exp_normal)  # [..., 5]
        not_is_zero_exp = not_is_zero.expand_as(fp16_exp_sel)  # [..., 5]
        fp16_exp = self.vec_and_zero_exp(fp16_exp_sel, not_is_zero_exp)  # [..., 5]

        is_subnormal_mant = is_subnormal.expand_as(sub_mant)  # [..., 10]
        fp16_mant_sel = self.vec_mux(is_subnormal_mant, sub_mant, fp16_mant_normal)  # [..., 10]
        not_is_zero_mant = not_is_zero.expand_as(fp16_mant_sel)  # [..., 10]
        fp16_mant = self.vec_and_zero_mant(fp16_mant_sel, not_is_zero_mant)  # [..., 10]
        
        # 组装 FP16: [S, E4..E0, M9..M0]
        fp16_pulse = torch.cat([s, fp16_exp, fp16_mant], dim=-1)
        
        return fp16_pulse
    
    def reset(self):
        self.e_or_01.reset()
        self.e_or_23.reset()
        self.e_or_all.reset()
        self.e_is_zero_not.reset()
        self.m_or_01.reset()
        self.m_or_all.reset()
        self.is_subnorm_and.reset()
        self.exp_adder.reset()
        self.not_m2.reset()
        self.not_m1.reset()
        self.lead_at_1_and.reset()
        self.lead_at_0_and1.reset()
        self.lead_at_0_and2.reset()
        # 向量化门电路
        self.vec_mux.reset()
        self.not_m_nonzero.reset()
        self.not_is_zero.reset()
        self.vec_and_zero_exp.reset()
        self.vec_and_zero_mant.reset()


# ==============================================================================
# FP16 -> FP8 转换器（纯SNN门电路，带RNE舍入）
# ==============================================================================
class FP16ToFP8Converter(nn.Module):
    """FP16 -> FP8 E4M3 转换器（100%纯SNN门电路）
    
    需要处理：
    - 指数转换: FP8_exp = FP16_exp - 8
    - 尾数截断 + RNE舍入: 10位 -> 3位
    - 溢出/下溢/subnormal 处理
    
    FP8 subnormal 输出范围：
    - FP16 exp = 8: 边界，可能是 FP8 E=1 或 E=0
    - FP16 exp = 7: FP8 subnormal M = 4-7 (值 ≈ 2^(-7) 到 2^(-8))
    - FP16 exp = 6: FP8 subnormal M = 2-3 (值 ≈ 2^(-8) 到 2^(-9))
    - FP16 exp = 5: FP8 subnormal M = 1 (值 ≈ 2^(-9) 到 2^(-10))
    - FP16 exp < 5: 下溢到 0
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # 指数减法: FP16_exp - 8
        self.exp_sub = VecAdder(bits=5, neuron_template=nt, max_param_shape=(5,))
        
        # 指数范围检测
        self.overflow_cmp = Comparator5Bit(neuron_template=nt)
        self.underflow_cmp = Comparator5Bit(neuron_template=nt)
        
        # 检测 FP16 exp = 5, 6, 7 (subnormal 输出范围)
        self.exp_cmp_5 = Comparator5Bit(neuron_template=nt)
        self.exp_cmp_6 = Comparator5Bit(neuron_template=nt)
        self.exp_cmp_7 = Comparator5Bit(neuron_template=nt)
        self.exp_cmp_ge5 = Comparator5Bit(neuron_template=nt)
        
        # Sticky bit 计算
        self.sticky_or_01 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_or_23 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_or_45 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_or_0123 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_or_all = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # RNE 舍入 (normal path)
        self.rne_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.rne_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        
        # 尾数 +1 (4位)
        self.mant_inc = VecAdder(bits=4, neuron_template=nt, max_param_shape=(4,))
        
        # Subnormal 输出路径的 RNE 舍入
        self.sub_rne_or_7 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sub_rne_and_7 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.sub_rne_or_6 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sub_rne_and_6 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.sub_rne_or_5 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sub_rne_and_5 = VecAND(neuron_template=nt, max_param_shape=(1,))
        
        # Subnormal 结果选择
        self.is_subnorm_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.is_subnorm_or2 = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # 指数进位处理
        self.exp_inc = VecAdder(bits=4, neuron_template=nt, max_param_shape=(4,))
        
        # 尾数清零（当舍入进位时）
        self.not_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.mant_clear_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 纯SNN NOT门 (替换 ones - x) =====
        self.not_sub_m1_8_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_sub_m2_8_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_sub_m0_7_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_sub_m1_7_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_round_up_6 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_sub_exp_8_overflow = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_is_subnorm_output_with_8 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_is_true_underflow = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_is_overflow = VecNOT(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 纯SNN MUX门 (选择最终结果) - 向量化 =====
        # 统一 VecMUX (预分配最大形状: 4位)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=(4,))

        # ===== 纯SNN XOR/OR门 (替换 a+b-2*a*b 和 a+b-a*b) =====
        self.xor_m8_round = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.xor_m9_carry8 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.xor_m9_round7 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.or_s7 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_s6 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_s5 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_subnorm_exp8 = VecOR(neuron_template=nt, max_param_shape=(1,))

        # ===== 纯SNN AND门 (替换 a*b) =====
        self.and_m8_round = VecAND(neuron_template=nt, max_param_shape=(1,))      # m8 * round_up_8
        self.and_m9_carry8 = VecAND(neuron_template=nt, max_param_shape=(1,))     # m9 * sub_m0_8_carry
        self.and_m9_round7 = VecAND(neuron_template=nt, max_param_shape=(1,))     # m9 * round_up_7
        self.and_s5_round = VecAND(neuron_template=nt, max_param_shape=(1,))      # ones * S5 -> S5 (本质是直接使用 S5)

        # 用于清除尾数位的 AND 门
        self.and_m8_m2_clear = VecAND(neuron_template=nt, max_param_shape=(1,))   # not_sub_m2_8_carry * sub_m2_8_sum
        self.and_m8_m1_clear = VecAND(neuron_template=nt, max_param_shape=(1,))   # not_sub_m2_8_carry * sub_m1_8_sum
        self.and_m8_m0_clear = VecAND(neuron_template=nt, max_param_shape=(1,))   # not_sub_m2_8_carry * sub_m0_8_sum
        self.and_m7_m1_clear = VecAND(neuron_template=nt, max_param_shape=(1,))   # not_sub_m1_7_carry * sub_m1_7_sum
        self.and_m7_m0_clear = VecAND(neuron_template=nt, max_param_shape=(1,))   # not_sub_m1_7_carry * sub_m0_7_sum

        # Subnormal 尾数/指数选择 MUX（单实例）
        self.mux_subnorm_mant = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_subnorm_exp_overflow = VecAND(neuron_template=nt, max_param_shape=(1,))  # is_exp_8 AND sub_exp_8_overflow
        self.and_subnorm_exp_sel = VecAND(neuron_template=nt, max_param_shape=(1,))       # 用于选择 subnorm_exp_one
        
    def forward(self, fp16_pulse):
        """
        Args:
            fp16_pulse: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            fp8_pulse: [..., 8] FP8 脉冲 [S, E3..E0, M2..M0]
        """
        device = fp16_pulse.device
        batch_shape = fp16_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取 FP16 各部分 (MSB first)
        s = fp16_pulse[..., 0:1]
        e4 = fp16_pulse[..., 1:2]
        e3 = fp16_pulse[..., 2:3]
        e2 = fp16_pulse[..., 3:4]
        e1 = fp16_pulse[..., 4:5]
        e0 = fp16_pulse[..., 5:6]
        fp16_mant = fp16_pulse[..., 6:16]  # [M9..M0]
        
        fp16_exp_msb = torch.cat([e4, e3, e2, e1, e0], dim=-1)
        
        # ===== 指数范围检测 =====
        # 溢出: FP16_exp > 22
        const_22 = torch.cat([ones, zeros, ones, ones, zeros], dim=-1)
        is_overflow, _ = self.overflow_cmp(fp16_exp_msb, const_22)
        
        # 下溢: FP16_exp < 5 (真正的下溢，太小无法表示)
        const_5 = torch.cat([zeros, zeros, ones, zeros, ones], dim=-1)
        is_true_underflow, _ = self.underflow_cmp(const_5, fp16_exp_msb)  # 5 > exp
        
        # Subnormal 输出范围: FP16_exp ∈ [5, 7]
        const_7 = torch.cat([zeros, zeros, ones, ones, ones], dim=-1)
        const_6 = torch.cat([zeros, zeros, ones, ones, zeros], dim=-1)
        
        _, is_exp_7 = self.exp_cmp_7(fp16_exp_msb, const_7)  # exp == 7
        _, is_exp_6 = self.exp_cmp_6(fp16_exp_msb, const_6)  # exp == 6
        _, is_exp_5 = self.exp_cmp_5(fp16_exp_msb, const_5)  # exp == 5
        
        # FP16_exp < 8: 需要输出 FP8 subnormal 或 0
        const_8 = torch.cat([zeros, ones, zeros, zeros, zeros], dim=-1)
        is_lt_8, _ = self.underflow_cmp(const_8, fp16_exp_msb)
        
        # is_subnorm_output = (exp == 5) OR (exp == 6) OR (exp == 7)
        is_subnorm_56 = self.is_subnorm_or(is_exp_5, is_exp_6)
        is_subnorm_output = self.is_subnorm_or2(is_subnorm_56, is_exp_7)
        
        # ===== 指数转换（正常路径）=====
        fp16_exp_lsb = torch.cat([e0, e1, e2, e3, e4], dim=-1)
        const_neg8_lsb = torch.cat([zeros, zeros, zeros, ones, ones], dim=-1)
        fp8_exp_raw_lsb, _ = self.exp_sub(fp16_exp_lsb, const_neg8_lsb)
        
        fp8_exp = torch.cat([
            fp8_exp_raw_lsb[..., 3:4],
            fp8_exp_raw_lsb[..., 2:3],
            fp8_exp_raw_lsb[..., 1:2],
            fp8_exp_raw_lsb[..., 0:1],
        ], dim=-1)
        
        # ===== 尾数处理（正常路径）=====
        # 隐藏位 + FP16 尾数: 1.M9M8M7M6M5M4M3M2M1M0
        # FP8 尾数取 M9M8M7，RNE 舍入
        hidden = ones  # FP16 normal 有隐藏位
        m9 = fp16_mant[..., 0:1]
        m8 = fp16_mant[..., 1:2]
        m7 = fp16_mant[..., 2:3]
        m6 = fp16_mant[..., 3:4]
        m5 = fp16_mant[..., 4:5]
        m4 = fp16_mant[..., 5:6]
        m3 = fp16_mant[..., 6:7]
        m2_low = fp16_mant[..., 7:8]
        m1_low = fp16_mant[..., 8:9]
        m0_low = fp16_mant[..., 9:10]
        
        # Normal 路径的 RNE
        L_normal = m7
        R_normal = m6
        sticky_01 = self.sticky_or_01(m5, m4)
        sticky_23 = self.sticky_or_23(m3, m2_low)
        sticky_45 = self.sticky_or_45(m1_low, m0_low)
        sticky_0123 = self.sticky_or_0123(sticky_01, sticky_23)
        S_normal = self.sticky_or_all(sticky_0123, sticky_45)
        
        s_or_l = self.rne_or(S_normal, L_normal)
        round_up = self.rne_and(R_normal, s_or_l)
        
        mant_4bit_lsb = torch.cat([m7, m8, m9, zeros], dim=-1)
        round_inc_lsb = torch.cat([round_up, zeros, zeros, zeros], dim=-1)
        mant_rounded_lsb, _ = self.mant_inc(mant_4bit_lsb, round_inc_lsb)
        
        # 尾数溢出检测：使用 bit 3（第4位）而不是进位
        # 0111 + 0001 = 1000，bit 3 = 1 表示溢出
        mant_overflow = mant_rounded_lsb[..., 3:4]
        
        fp8_mant = torch.cat([
            mant_rounded_lsb[..., 2:3],
            mant_rounded_lsb[..., 1:2],
            mant_rounded_lsb[..., 0:1],
        ], dim=-1)
        
        fp8_exp_lsb = torch.cat([
            fp8_exp[..., 3:4], fp8_exp[..., 2:3], fp8_exp[..., 1:2], fp8_exp[..., 0:1]
        ], dim=-1)
        exp_inc_lsb = torch.cat([mant_overflow, zeros, zeros, zeros], dim=-1)
        exp_after_round_lsb, _ = self.exp_inc(fp8_exp_lsb, exp_inc_lsb)
        
        exp_after_round = torch.cat([
            exp_after_round_lsb[..., 3:4],
            exp_after_round_lsb[..., 2:3],
            exp_after_round_lsb[..., 1:2],
            exp_after_round_lsb[..., 0:1],
        ], dim=-1)
        
        not_overflow = self.not_carry(mant_overflow)
        fp8_mant_final = torch.cat([
            self.mant_clear_and(not_overflow, fp8_mant[..., 0:1]),
            self.mant_clear_and(not_overflow, fp8_mant[..., 1:2]),
            self.mant_clear_and(not_overflow, fp8_mant[..., 2:3]),
        ], dim=-1)
        
        # ===== Subnormal 输出路径 =====
        # FP16 exp=8: 1.M -> FP8 0.1M9M8 -> FP8 M = [1, M9, M8] + RNE(M7)
        # FP16 exp=7: 1.M -> FP8 0.01M9 -> FP8 M = [0, 1, M9] + RNE(M8)
        # FP16 exp=6: 1.M -> FP8 0.001 -> FP8 M = [0, 0, 1] + RNE(M9)
        # FP16 exp=5: 太小，可能舍入到 M=0 或 M=1
        
        # 检测 exp=8（FP8 exp=0，但有 FP16 隐藏位）
        _, is_exp_8 = self.exp_cmp_ge5(fp16_exp_msb, const_8)  # exp == 8
        
        # exp=8: M = [1, M9, M8], L=M8, R=M7, S=OR(M6..M0)
        # 隐藏位进入 FP8 尾数最高位
        # 使用纯SNN门电路：a + b 溢出 = a AND b
        L8 = m8
        R8 = m7
        S8 = S_normal
        s_or_l_8 = self.sub_rne_or_7(S8, L8)
        round_up_8 = self.sub_rne_and_7(R8, s_or_l_8)
        # 尾数进位链：使用纯SNN XOR 和 AND 实现二进制加法
        # m8 + round_up_8: sum = XOR(m8, round_up_8), carry = AND(m8, round_up_8)
        sub_m0_8_sum = self.xor_m8_round(m8, round_up_8)  # 纯SNN XOR
        sub_m0_8_carry = self.and_m8_round(m8, round_up_8)  # 纯SNN AND = carry
        # m9 + carry: sum = XOR(m9, carry), carry = AND(m9, carry)
        sub_m1_8_sum = self.xor_m9_carry8(m9, sub_m0_8_carry)  # 纯SNN XOR
        sub_m1_8_carry = self.and_m9_carry8(m9, sub_m0_8_carry)  # 纯SNN AND
        # 1 + carry: sum = XOR(1, carry) = NOT(carry), carry = AND(1, carry) = carry (纯SNN)
        sub_m2_8_sum = self.not_sub_m1_8_carry(sub_m1_8_carry)  # XOR(1, carry) = NOT(carry)
        sub_m2_8_carry = sub_m1_8_carry  # 溢出到 normal
        # 如果 exp=8 舍入溢出，结果应该是 FP8 E=1, M=0 (纯SNN)
        not_sub_m2_8_carry = self.not_sub_m2_8_carry(sub_m2_8_carry)
        sub_m_8_rounded = torch.cat([
            self.and_m8_m2_clear(not_sub_m2_8_carry, sub_m2_8_sum),  # 溢出时 M=0 (纯SNN AND)
            self.and_m8_m1_clear(not_sub_m2_8_carry, sub_m1_8_sum),
            self.and_m8_m0_clear(not_sub_m2_8_carry, sub_m0_8_sum),
        ], dim=-1)
        sub_exp_8_overflow = sub_m2_8_carry  # exp=8 溢出到 E=1
        
        # exp=7: M = [0, 1, M9], L=M9, R=M8, S=OR(M7..M0)
        L7 = m9
        R7 = m8
        S7 = self.or_s7(m7, S_normal)  # 纯SNN OR
        s_or_l_7 = self.sub_rne_or_6(S7, L7)
        round_up_7 = self.sub_rne_and_6(R7, s_or_l_7)
        # m9 + round_up_7 (纯SNN)
        sub_m0_7_sum = self.xor_m9_round7(m9, round_up_7)  # 纯SNN XOR
        sub_m0_7_carry = self.and_m9_round7(m9, round_up_7)  # 纯SNN AND
        # 1 + carry (纯SNN)
        sub_m1_7_sum = self.not_sub_m0_7_carry(sub_m0_7_carry)  # XOR(1, carry) = NOT(carry)
        sub_m1_7_carry = sub_m0_7_carry  # AND(1, carry) = carry
        not_sub_m1_7_carry = self.not_sub_m1_7_carry(sub_m1_7_carry)
        sub_m_7_rounded = torch.cat([
            sub_m1_7_carry,  # 如果进位，m2=1
            self.and_m7_m1_clear(not_sub_m1_7_carry, sub_m1_7_sum),  # 进位时 m1=0 (纯SNN AND)
            self.and_m7_m0_clear(not_sub_m1_7_carry, sub_m0_7_sum),
        ], dim=-1)
        
        # exp=6: M = [0, 0, 1], L=1, R=M9, S=OR(M8..M0)
        L6 = ones
        R6 = m9
        S6 = self.or_s6(m8, S7)  # 纯SNN OR
        s_or_l_6 = self.sub_rne_or_5(S6, L6)
        round_up_6 = self.sub_rne_and_5(R6, s_or_l_6)
        # 1 + round_up: sum = XOR(1, round_up) = NOT(round_up), carry = AND(1, round_up) = round_up
        # 如果 round_up=1: carry=1 -> m1=1, m0=0
        # 如果 round_up=0: carry=0 -> m1=0, m0=1
        not_round_up_6 = self.not_round_up_6(round_up_6)
        sub_m_6_rounded = torch.cat([
            zeros,
            round_up_6,  # carry -> m1
            not_round_up_6,  # NOT(carry) -> m0 (纯SNN)
        ], dim=-1)
        
        # exp=5: 值太小，可能舍入到 0 或 M=1
        # 值 = 1.M × 2^(-10)，最大 ≈ 2 × 2^(-10) = 2^(-9) = M=1 的值
        # R = 隐藏位 = 1, L = 0 (result M=0), S = OR(M9..M0)
        S5 = self.or_s5(m9, S6)  # 纯SNN OR
        round_up_5 = self.and_s5_round(ones, S5)  # 如果 sticky 有任何位，则舍入到 M=1 (纯SNN AND)
        sub_m_5_rounded = torch.cat([zeros, zeros, round_up_5], dim=-1)
        
        # 选择 subnormal 尾数（优先级: exp_8 > exp_7 > exp_6 > exp_5）(纯SNN MUX级联)
        not_sub_exp_8_overflow = self.not_sub_exp_8_overflow(sub_exp_8_overflow)
        # 使用 MUX 级联选择正确的尾数
        # 优先级从低到高: exp_5 -> exp_6 -> exp_7 -> (exp_8 AND not_overflow)
        is_exp_8_valid = self.mux_subnorm_exp_overflow(is_exp_8, not_sub_exp_8_overflow)  # AND
        subnorm_mant_t1 = self.mux_subnorm_mant(is_exp_5, sub_m_5_rounded, torch.cat([zeros]*3, dim=-1))
        subnorm_mant_t2 = self.mux_subnorm_mant(is_exp_6, sub_m_6_rounded, subnorm_mant_t1)
        subnorm_mant_t3 = self.mux_subnorm_mant(is_exp_7, sub_m_7_rounded, subnorm_mant_t2)
        subnorm_mant = self.mux_subnorm_mant(is_exp_8_valid, sub_m_8_rounded, subnorm_mant_t3)
        
        # Subnormal 指数 = 0（除非 exp=8 且溢出，则 E=1）(纯SNN AND)
        subnorm_exp_one = torch.cat([zeros, zeros, zeros, ones], dim=-1)
        subnorm_exp_sel = self.and_subnorm_exp_sel(is_exp_8, sub_exp_8_overflow)
        # 如果 subnorm_exp_sel=1 则用 subnorm_exp_one，否则用 zero_exp (向量化)
        subnorm_exp_zero = torch.cat([zeros, zeros, zeros, zeros], dim=-1)
        subnorm_exp_sel_4 = subnorm_exp_sel.expand_as(subnorm_exp_one)  # [..., 4]
        subnorm_exp = self.vec_mux(subnorm_exp_sel_4, subnorm_exp_one, subnorm_exp_zero)  # [..., 4]
        
        # 更新 is_subnorm_output 包含 exp=8 (纯SNN OR)
        is_subnorm_output_with_8 = self.or_subnorm_exp8(is_subnorm_output, is_exp_8)
        
        # ===== 最终选择 =====
        nan_exp = torch.cat([ones, ones, ones, ones], dim=-1)
        nan_mant = torch.cat([ones, ones, ones], dim=-1)
        zero_exp = torch.cat([zeros, zeros, zeros, zeros], dim=-1)
        zero_mant = torch.cat([zeros, zeros, zeros], dim=-1)
        
        # 构建最终结果 (纯SNN) - 向量化
        # 优先级: overflow > subnorm_output_with_8 > true_underflow > normal
        not_is_subnorm_output_with_8 = self.not_is_subnorm_output_with_8(is_subnorm_output_with_8)
        not_is_true_underflow = self.not_is_true_underflow(is_true_underflow)
        not_is_overflow = self.not_is_overflow(is_overflow)

        # 指数向量化 (4位)
        is_subnorm_out_4 = is_subnorm_output_with_8.expand_as(subnorm_exp)  # [..., 4]
        e_sub_or_norm = self.vec_mux(is_subnorm_out_4, subnorm_exp, exp_after_round)  # [..., 4]
        is_true_underflow_4 = is_true_underflow.expand_as(e_sub_or_norm)  # [..., 4]
        e_underflow = self.vec_mux(is_true_underflow_4, zero_exp, e_sub_or_norm)  # [..., 4]
        is_overflow_4 = is_overflow.expand_as(e_underflow)  # [..., 4]
        final_exp = self.vec_mux(is_overflow_4, nan_exp, e_underflow)  # [..., 4]

        # 尾数向量化 (3位)
        is_subnorm_out_3 = is_subnorm_output_with_8.expand_as(subnorm_mant)  # [..., 3]
        m_sub_or_norm = self.vec_mux(is_subnorm_out_3, subnorm_mant, fp8_mant_final)  # [..., 3]
        is_true_underflow_3 = is_true_underflow.expand_as(m_sub_or_norm)  # [..., 3]
        m_underflow = self.vec_mux(is_true_underflow_3, zero_mant, m_sub_or_norm)  # [..., 3]
        is_overflow_3 = is_overflow.expand_as(m_underflow)  # [..., 3]
        final_mant = self.vec_mux(is_overflow_3, nan_mant, m_underflow)  # [..., 3]
        
        # 组装 FP8
        fp8_pulse = torch.cat([s, final_exp, final_mant], dim=-1)
        
        return fp8_pulse
    
    def reset(self):
        self.exp_sub.reset()
        self.overflow_cmp.reset()
        self.underflow_cmp.reset()
        self.exp_cmp_5.reset()
        self.exp_cmp_6.reset()
        self.exp_cmp_7.reset()
        self.exp_cmp_ge5.reset()
        self.sticky_or_01.reset()
        self.sticky_or_23.reset()
        self.sticky_or_45.reset()
        self.sticky_or_0123.reset()
        self.sticky_or_all.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.mant_inc.reset()
        self.sub_rne_or_7.reset()
        self.sub_rne_and_7.reset()
        self.sub_rne_or_6.reset()
        self.sub_rne_and_6.reset()
        self.sub_rne_or_5.reset()
        self.sub_rne_and_5.reset()
        self.is_subnorm_or.reset()
        self.is_subnorm_or2.reset()
        self.exp_inc.reset()
        self.not_carry.reset()
        self.mant_clear_and.reset()
        # 纯SNN NOT门
        self.not_sub_m1_8_carry.reset()
        self.not_sub_m2_8_carry.reset()
        self.not_sub_m0_7_carry.reset()
        self.not_sub_m1_7_carry.reset()
        self.not_round_up_6.reset()
        self.not_sub_exp_8_overflow.reset()
        self.not_is_subnorm_output_with_8.reset()
        self.not_is_true_underflow.reset()
        self.not_is_overflow.reset()
        # 纯SNN XOR/OR门
        self.xor_m8_round.reset()
        self.xor_m9_carry8.reset()
        self.xor_m9_round7.reset()
        self.or_s7.reset()
        self.or_s6.reset()
        self.or_s5.reset()
        self.or_subnorm_exp8.reset()
        # 新增的纯SNN AND门
        self.and_m8_round.reset()
        self.and_m9_carry8.reset()
        self.and_m9_round7.reset()
        self.and_s5_round.reset()
        self.and_m8_m2_clear.reset()
        self.and_m8_m1_clear.reset()
        self.and_m8_m0_clear.reset()
        self.and_m7_m1_clear.reset()
        self.and_m7_m0_clear.reset()
        self.mux_subnorm_mant.reset()
        self.mux_subnorm_exp_overflow.reset()
        self.and_subnorm_exp_sel.reset()
        # 向量化门电路
        self.vec_mux.reset()

