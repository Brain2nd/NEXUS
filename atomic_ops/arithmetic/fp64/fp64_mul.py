"""
FP64 乘法器 - 100%纯SNN门电路实现 (向量化版本)
======================================

FP64 格式: [S | E10..E0 | M51..M0], bias=1023

核心算法:
1. 符号: Sr = Sa XOR Sb
2. 指数: Er = Ea + Eb - 1023
3. 尾数: 54x54 阵列乘法 (隐藏位 + 53位尾数)
4. 规格化: LZD + 桶形移位
5. RNE舍入: Guard/Round/Sticky位处理

向量化原则:
1. 使用 VecAND, VecOR, VecXOR, VecNOT, VecMUX 代替 ModuleList
2. 可并行操作一次处理所有位
3. 串行依赖仍保留循环
"""
import torch
import torch.nn as nn
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecORTree, VecANDTree, VecAdder, VecSubtractor
)
# 注意：使用 VecAdder 代替旧的 RippleCarryAdder（支持 max_param_shape）
from .fp64_components import Comparator11Bit


# ==============================================================================
# 向量化108位加法器 (用于尾数乘积累加)
# ==============================================================================
class VecRippleCarryAdder108Bit(nn.Module):
    """108位加法器 - 向量化版本 (LSB first)"""
    MAX_BITS = 108

    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 108
        nt = neuron_template
        max_shape = (self.MAX_BITS,)
        max_shape_1 = (1,)
        self.xor1 = VecXOR(neuron_template=nt, max_param_shape=max_shape)
        self.xor2 = VecXOR(neuron_template=nt, max_param_shape=max_shape)
        self.and1 = VecAND(neuron_template=nt, max_param_shape=max_shape)
        self.and2 = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.or1 = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        
    def forward(self, A, B, Cin=None):
        """A + B, LSB first"""
        device = A.device
        batch_shape = A.shape[:-1]

        if Cin is None:
            carry = torch.zeros(batch_shape + (1,), device=device)
        else:
            carry = Cin

        # 并行计算 P = A XOR B, G = A AND B
        P = self.xor1(A, B)
        G = self.and1(A, B)

        # 进位链
        carries = [carry]
        for i in range(self.bits):
            p_i = P[..., i:i+1]
            g_i = G[..., i:i+1]
            pc = self.and2(p_i, carry)
            carry = self.or1(g_i, pc)
            carries.append(carry)

        # 并行计算和
        all_carries = torch.cat(carries[:-1], dim=-1)
        S = self.xor2(P, all_carries)

        return S, carries[-1]
    
    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


# ==============================================================================
# 向量化54x54 阵列乘法器
# ==============================================================================
class VecArrayMultiplier54x54(nn.Module):
    """54x54位阵列乘法器 - 向量化版本 (解耦累加器)

    使用部分积累加方式:
    - 54个部分积, 每个54位
    - 使用 PartialProductAccumulator 累加生成108位结果

    输入: A, B: [..., 54] (LSB first)
    输出: P: [..., 108] (LSB first)

    Args:
        neuron_template: 神经元模板
        accumulator_mode: 累加模式 ('sequential' 或 'parallel')
    """
    MAX_BITS = 108

    def __init__(self, neuron_template=None, accumulator_mode='sequential'):
        super().__init__()
        nt = neuron_template
        self.accumulator_mode = accumulator_mode
        max_shape_54 = (54,)

        # 部分积生成: 使用单个VecAND处理所有位
        self.pp_and = VecAND(neuron_template=nt, max_param_shape=max_shape_54)

        # 解耦累加器 - 单个加法器 + 累加策略
        from atomic_ops.core.accumulator import create_partial_product_accumulator
        self.accum_adder = VecRippleCarryAdder108Bit(neuron_template=nt)
        self.accumulator = create_partial_product_accumulator(
            self.accum_adder, mode=accumulator_mode
        )

    def forward(self, A, B):
        """
        A, B: [..., 54] LSB first
        Returns: [..., 108] LSB first
        """
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)

        # 生成所有部分积
        partial_products = []
        for i in range(54):
            # 第i个部分积: A & B[i], 左移i位
            # 广播B[i]到所有A的位
            b_i = B[..., i:i+1].expand_as(A)
            pp_bits = self.pp_and(A, b_i)  # 向量化AND

            # 扩展到108位, 低位补零 (移位)
            low_zeros = zeros.expand(batch_shape + (i,)) if i > 0 else None
            high_zeros = zeros.expand(batch_shape + (108 - 54 - i,)) if (108 - 54 - i) > 0 else None

            if i == 0:
                pp_108 = torch.cat([pp_bits, high_zeros], dim=-1)
            elif (108 - 54 - i) == 0:
                pp_108 = torch.cat([low_zeros, pp_bits], dim=-1)
            else:
                pp_108 = torch.cat([low_zeros, pp_bits, high_zeros], dim=-1)
            partial_products.append(pp_108)

        # 使用解耦累加器累加所有部分积
        result = self.accumulator(partial_products)

        return result

    def reset(self):
        self.pp_and.reset()
        self.accum_adder.reset()
        self.accumulator.reset()


# ==============================================================================
# 向量化108位前导零检测器
# ==============================================================================
class VecLeadingZeroDetector108(nn.Module):
    """108位前导零检测器 - 向量化版本

    输入: X[107:0] MSB first
    输出: LZC[6:0] 前导零个数 (MSB first)
    """
    MAX_BITS = 108

    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        max_shape_108 = (self.MAX_BITS,)
        max_shape_7 = (7,)
        max_shape_1 = (1,)
        self.vec_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_7)
        self.vec_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_108)
        
    def forward(self, X):
        """X: [..., 108] MSB first, returns: [..., 7] LZC MSB first"""
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        lzc = [zeros.clone() for _ in range(7)]
        found = zeros.clone()
        
        for i in range(108):
            bit = X[..., i:i+1]
            not_found = self.vec_not(found)
            is_first = self.vec_and(bit, not_found)

            # 位置编码
            for j in range(7):
                pos_bit = ones if ((i >> (6-j)) & 1) else zeros
                lzc[j] = self.vec_mux(is_first, pos_bit, lzc[j])

            found = self.vec_or(found, is_first)

        any_one = self.vec_or_tree(X)
        all_zero = self.vec_not(any_one)

        # lzc = 108 = 0b1101100
        lzc_108 = [ones, ones, zeros, ones, ones, zeros, zeros]
        for j in range(7):
            lzc[j] = self.vec_mux(all_zero, lzc_108[j], lzc[j])
        
        return torch.cat(lzc, dim=-1)
    
    def reset(self):
        self.vec_not.reset()
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_mux.reset()
        self.vec_or_tree.reset()


# ==============================================================================
# 向量化108位桶形左移位器
# ==============================================================================
class VecBarrelShifterLeft108(nn.Module):
    """108位桶形左移位器 - 向量化版本"""
    MAX_BITS = 108

    def __init__(self, neuron_template=None):
        super().__init__()
        self.data_bits = 108
        self.shift_bits = 7
        self.vec_mux = VecMUX(neuron_template=neuron_template, max_param_shape=(self.MAX_BITS,))
            
    def forward(self, X, shift):
        """X: [..., 108], shift: [..., 7] (MSB first)"""
        device = X.device
        zeros = torch.zeros_like(X[..., 0:1])
        
        current = X
        for layer in range(self.shift_bits):
            shift_amt = 2 ** (self.shift_bits - 1 - layer)
            s_bit = shift[..., layer:layer+1]

            # 构建移位后的张量
            if shift_amt >= self.data_bits:
                shifted = torch.zeros_like(current)
            else:
                shifted = torch.cat([
                    current[..., shift_amt:],
                    zeros.expand_as(current[..., :shift_amt])
                ], dim=-1)

            current = self.vec_mux(s_bit.expand_as(current), shifted, current)
        
        return current
    
    def reset(self):
        self.vec_mux.reset()


# ==============================================================================
# 向量化12位加法器/减法器 (指数运算)
# ==============================================================================
class VecAdder12Bit(nn.Module):
    """12位加法器 - 向量化版本"""
    MAX_BITS = 12

    def __init__(self, neuron_template=None):
        super().__init__()
        self.adder = VecAdder(12, neuron_template=neuron_template, max_param_shape=(self.MAX_BITS,))
        
    def forward(self, A, B, Cin=None):
        return self.adder(A, B, Cin)
    
    def reset(self):
        self.adder.reset()


class VecSubtractor12Bit(nn.Module):
    """12位减法器 - 向量化版本"""
    MAX_BITS = 12

    def __init__(self, neuron_template=None):
        super().__init__()
        self.subtractor = VecSubtractor(12, neuron_template=neuron_template, max_param_shape=(self.MAX_BITS,))
        
    def forward(self, A, B, Bin=None):
        return self.subtractor(A, B)
    
    def reset(self):
        self.subtractor.reset()


# ==============================================================================
# FP64 乘法器主类 (向量化版本)
# ==============================================================================
class SpikeFP64Multiplier(nn.Module):
    """FP64 乘法器 - 100%纯SNN门电路实现 (向量化版本)

    输入: A, B: [..., 64] FP64脉冲 [S | E10..E0 | M51..M0]
    输出: [..., 64] FP64脉冲

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        accumulator_mode: 累加模式 ('sequential' 或 'parallel')
    """
    MAX_BITS = 64

    def __init__(self, neuron_template=None, accumulator_mode='sequential'):
        super().__init__()
        nt = neuron_template
        # 预分配参数形状
        max_shape_108 = (108,)
        max_shape_64 = (64,)
        max_shape_54 = (54,)
        max_shape_52 = (52,)
        max_shape_12 = (12,)
        max_shape_11 = (11,)
        max_shape_1 = (1,)

        # ===== 向量化基础门电路 =====
        self.vec_and = VecAND(neuron_template=nt, max_param_shape=max_shape_64)
        self.vec_or = VecOR(neuron_template=nt, max_param_shape=max_shape_64)
        self.vec_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_64)
        self.vec_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_64)
        # VecMUX 统一实例 (动态扩展机制支持不同位宽)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_64)

        # ===== 独立实例的 Tree (不同输入大小需要独立实例) =====
        # 指数相关 (11-bit): e_a_all_one, e_b_all_one, exp_ge_2047
        self.vec_and_tree_exp_a = VecANDTree(neuron_template=nt, max_param_shape=max_shape_11)
        self.vec_and_tree_exp_b = VecANDTree(neuron_template=nt, max_param_shape=max_shape_11)
        self.vec_and_tree_exp_overflow = VecANDTree(neuron_template=nt, max_param_shape=max_shape_11)

        # 指数非零检测 (11-bit): e_a_any_one, e_b_any_one, exp_any_one
        self.vec_or_tree_exp_a = VecORTree(neuron_template=nt, max_param_shape=max_shape_11)
        self.vec_or_tree_exp_b = VecORTree(neuron_template=nt, max_param_shape=max_shape_11)
        self.vec_or_tree_exp_final = VecORTree(neuron_template=nt, max_param_shape=max_shape_12)

        # 尾数非零检测 (52-bit): m_a_any_one, m_b_any_one
        self.vec_or_tree_mant_a = VecORTree(neuron_template=nt, max_param_shape=max_shape_52)
        self.vec_or_tree_mant_b = VecORTree(neuron_template=nt, max_param_shape=max_shape_52)

        # sticky位检测 (54-bit)
        self.vec_or_tree_sticky = VecORTree(neuron_template=nt, max_param_shape=max_shape_54)
        
        # ===== 指数运算 =====
        self.exp_adder = VecAdder12Bit(neuron_template=nt)
        self.bias_sub = VecSubtractor12Bit(neuron_template=nt)
        self.exp_inc = VecAdder12Bit(neuron_template=nt)
        self.exp_lzc_sub = VecSubtractor12Bit(neuron_template=nt)
        
        # ===== 尾数乘法 =====
        self.mantissa_mul = VecArrayMultiplier54x54(
            neuron_template=nt, accumulator_mode=accumulator_mode
        )
        
        # ===== 规格化 =====
        self.lzd = VecLeadingZeroDetector108(neuron_template=nt)
        self.norm_shifter = VecBarrelShifterLeft108(neuron_template=nt)
        
        # ===== 舍入 =====
        self.round_adder = VecAdder(bits=53, neuron_template=nt, max_param_shape=(53,))
        self.exp_round_inc = VecAdder(bits=11, neuron_template=nt, max_param_shape=(11,))
        
    def forward(self, A, B):
        """
        A, B: [..., 64] FP64脉冲 [S | E10..E0 | M51..M0] MSB first
        Returns: [..., 64] FP64脉冲
        """
        A, B = torch.broadcast_tensors(A, B)
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # ===== 1. 提取各部分 =====
        s_a = A[..., 0:1]
        e_a = A[..., 1:12]   # [E10..E0] MSB first
        m_a = A[..., 12:64]  # [M51..M0] MSB first
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:12]
        m_b = B[..., 12:64]
        
        # ===== 2. 符号 =====
        s_out = self.vec_xor(s_a, s_b)
        
        # ===== 3. 特殊值检测 (向量化) =====
        e_a_all_one = self.vec_and_tree_exp_a(e_a)
        e_b_all_one = self.vec_and_tree_exp_b(e_b)

        e_a_any_one = self.vec_or_tree_exp_a(e_a)
        e_b_any_one = self.vec_or_tree_exp_b(e_b)
        e_a_is_zero = self.vec_not(e_a_any_one)
        e_b_is_zero = self.vec_not(e_b_any_one)

        m_a_any_one = self.vec_or_tree_mant_a(m_a)
        m_b_any_one = self.vec_or_tree_mant_b(m_b)
        m_a_is_zero = self.vec_not(m_a_any_one)
        m_b_is_zero = self.vec_not(m_b_any_one)

        # 零检测
        a_is_zero = self.vec_and(e_a_is_zero, m_a_is_zero)
        b_is_zero = self.vec_and(e_b_is_zero, m_b_is_zero)
        either_zero = self.vec_or(a_is_zero, b_is_zero)

        # Inf检测
        a_is_inf = self.vec_and(e_a_all_one, m_a_is_zero)
        b_is_inf = self.vec_and(e_b_all_one, m_b_is_zero)
        either_inf = self.vec_or(a_is_inf, b_is_inf)

        # NaN检测
        a_is_nan = self.vec_and(e_a_all_one, m_a_any_one)
        b_is_nan = self.vec_and(e_b_all_one, m_b_any_one)
        either_nan = self.vec_or(a_is_nan, b_is_nan)

        # 0 × Inf = NaN
        zero_times_inf = self.vec_and(either_zero, either_inf)
        result_is_nan = self.vec_or(either_nan, zero_times_inf)

        # Subnormal检测
        a_is_subnormal = self.vec_and(e_a_is_zero, m_a_any_one)
        b_is_subnormal = self.vec_and(e_b_is_zero, m_b_any_one)
        
        # ===== 4. 指数处理 (向量化) =====
        e_a_le = e_a.flip(-1)
        e_b_le = e_b.flip(-1)
        
        # Subnormal时使用E=1
        e_one = torch.cat([ones] + [zeros]*10, dim=-1)  # LSB first
        e_a_corrected = self.vec_mux(a_is_subnormal.expand_as(e_a_le), e_one, e_a_le)
        e_b_corrected = self.vec_mux(b_is_subnormal.expand_as(e_b_le), e_one, e_b_le)
        
        # 扩展到12位
        e_a_12 = torch.cat([e_a_corrected, zeros], dim=-1)
        e_b_12 = torch.cat([e_b_corrected, zeros], dim=-1)
        
        # Ea + Eb (12位)
        sum_e_12, _ = self.exp_adder(e_a_12, e_b_12)
        
        # - 1023: bias减法
        const_1023 = torch.cat([ones]*10 + [zeros, zeros], dim=-1)
        raw_e_12, _ = self.bias_sub(sum_e_12, const_1023)
        
        # ===== 5. 尾数乘法 =====
        leading_a = self.vec_mux(a_is_subnormal, zeros, ones)
        leading_b = self.vec_mux(b_is_subnormal, zeros, ones)
        
        m_a_le = m_a.flip(-1)
        m_b_le = m_b.flip(-1)
        m_a_54 = torch.cat([m_a_le, leading_a, zeros], dim=-1)
        m_b_54 = torch.cat([m_b_le, leading_b, zeros], dim=-1)
        
        product_108 = self.mantissa_mul(m_a_54, m_b_54)
        
        # ===== 6. 规格化 =====
        product_108_be = product_108.flip(-1)
        lzc = self.lzd(product_108_be)
        product_norm = self.norm_shifter(product_108_be, lzc)
        
        # 调整指数
        const_3 = torch.cat([ones, ones] + [zeros]*10, dim=-1)
        exp_plus_3, _ = self.exp_inc(raw_e_12, const_3)
        
        lzc_le = lzc.flip(-1)
        lzc_12 = torch.cat([lzc_le, zeros, zeros, zeros, zeros, zeros], dim=-1)
        exp_final_pre, _ = self.exp_lzc_sub(exp_plus_3, lzc_12)
        
        # ===== 7. 提取尾数和舍入位 =====
        mant_norm = product_norm[..., 1:53]
        round_bit = product_norm[..., 53:54]
        
        sticky_bits = product_norm[..., 54:108]
        sticky = self.vec_or_tree_sticky(sticky_bits)
        
        # RNE舍入
        lsb = mant_norm[..., 51:52]
        s_or_l = self.vec_or(sticky, lsb)
        round_up = self.vec_and(round_bit, s_or_l)
        
        # 尾数+1
        mant_le = mant_norm.flip(-1)
        mant_53_le = torch.cat([mant_le, zeros], dim=-1)
        round_inc = torch.cat([round_up] + [zeros]*52, dim=-1)
        mant_rounded, _ = self.round_adder(mant_53_le, round_inc)
        
        mant_carry = mant_rounded[..., 52:53]

        not_carry = self.vec_not(mant_carry)
        mant_final_le = self.vec_and(not_carry.expand_as(mant_rounded[..., :52]), mant_rounded[..., :52])
        mant_final = mant_final_le.flip(-1)
        
        # 指数调整
        exp_11_le = exp_final_pre[..., :11]
        carry_inc = torch.cat([mant_carry] + [zeros]*10, dim=-1)
        exp_after_round_le, _ = self.exp_round_inc(exp_11_le, carry_inc)
        exp_final_le = self.vec_mux(mant_carry.expand_as(exp_11_le), exp_after_round_le, exp_11_le)
        exp_final = exp_final_le.flip(-1)
        
        # ===== 8. 溢出/下溢检测 =====
        exp_bit11 = exp_final_pre[..., 11:12]
        not_bit11 = self.vec_not(exp_bit11)

        exp_high_bits = exp_final_pre[..., :11]
        exp_ge_2047 = self.vec_and_tree_exp_overflow(exp_high_bits)
        is_overflow = self.vec_and(not_bit11, exp_ge_2047)

        # 下溢检测
        exp_any_one = self.vec_or_tree_exp_final(exp_final_pre[..., :11])
        exp_low_zero = self.vec_not(exp_any_one)
        is_underflow = self.vec_or(exp_bit11, exp_low_zero)
        
        # ===== 9. 特殊值输出 =====
        nan_exp = torch.cat([ones]*11, dim=-1)
        nan_mant = torch.cat([ones] + [zeros]*51, dim=-1)
        inf_exp = torch.cat([ones]*11, dim=-1)
        inf_mant = torch.cat([zeros]*52, dim=-1)
        zero_exp = torch.cat([zeros]*11, dim=-1)
        zero_mant = torch.cat([zeros]*52, dim=-1)
        
        # ===== 10. 选择最终结果 (向量化) =====
        # NaN
        e_out = self.vec_mux(result_is_nan.expand_as(exp_final), nan_exp, exp_final)
        m_out = self.vec_mux(result_is_nan.expand_as(mant_final), nan_mant, mant_final)

        # Inf
        not_nan = self.vec_not(result_is_nan)
        inf_and_not_nan = self.vec_and(either_inf, not_nan)
        e_out = self.vec_mux(inf_and_not_nan.expand_as(e_out), inf_exp, e_out)
        m_out = self.vec_mux(inf_and_not_nan.expand_as(m_out), inf_mant, m_out)

        # Zero
        not_either_inf = self.vec_not(either_inf)
        zero_and_not_nan = self.vec_and(either_zero, not_nan)
        zero_only = self.vec_and(zero_and_not_nan, not_either_inf)
        e_out = self.vec_mux(zero_only.expand_as(e_out), zero_exp, e_out)
        m_out = self.vec_mux(zero_only.expand_as(m_out), zero_mant, m_out)

        # 溢出 -> Inf
        overflow_and_valid = self.vec_and(is_overflow, not_nan)
        e_out = self.vec_mux(overflow_and_valid.expand_as(e_out), inf_exp, e_out)
        m_out = self.vec_mux(overflow_and_valid.expand_as(m_out), inf_mant, m_out)

        # 下溢 -> 0
        not_either_zero = self.vec_not(either_zero)
        underflow_temp = self.vec_and(is_underflow, not_nan)
        underflow_final = self.vec_and(underflow_temp, not_either_zero)
        e_out = self.vec_mux(underflow_final.expand_as(e_out), zero_exp, e_out)
        m_out = self.vec_mux(underflow_final.expand_as(m_out), zero_mant, m_out)
        
        # ===== 11. 组装输出 =====
        result = torch.cat([s_out, e_out, m_out], dim=-1)
        
        return result
    
    def reset(self):
        self.vec_and.reset()
        self.vec_or.reset()
        self.vec_xor.reset()
        self.vec_not.reset()
        self.vec_mux.reset()
        # Tree instances
        self.vec_and_tree_exp_a.reset()
        self.vec_and_tree_exp_b.reset()
        self.vec_and_tree_exp_overflow.reset()
        self.vec_or_tree_exp_a.reset()
        self.vec_or_tree_exp_b.reset()
        self.vec_or_tree_exp_final.reset()
        self.vec_or_tree_mant_a.reset()
        self.vec_or_tree_mant_b.reset()
        self.vec_or_tree_sticky.reset()
        self.exp_adder.reset()
        self.bias_sub.reset()
        self.exp_inc.reset()
        self.exp_lzc_sub.reset()
        self.mantissa_mul.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.round_adder.reset()
        self.exp_round_inc.reset()
