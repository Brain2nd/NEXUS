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

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
from atomic_ops.core.logic_gates import (HalfAdder, FullAdder, ORTree)
# 单比特门改用 Vec* 版本（支持 max_param_shape）
# 注意：使用 VecAdder 代替旧的 RippleCarryAdder（支持 max_param_shape）
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX, VecORTree, VecANDTree,
    VecFullAdder, VecAdder, VecSubtractor
)
from atomic_ops.core.accumulator import PartialProductAccumulator


# ==============================================================================
# 48位加法器 (用于尾数乘积) - 向量化
# ==============================================================================
class RippleCarryAdder48Bit(nn.Module):
    """48位加法器 - 向量化SNN (LSB first)"""
    MAX_BITS = 48

    def __init__(self, neuron_template=None):
        super().__init__()
        self.bits = 48
        max_shape = (self.MAX_BITS,)
        self.vec_adder = VecAdder(48, neuron_template=neuron_template, max_param_shape=max_shape)
        
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
    - 使用 PartialProductAccumulator 树形归约 O(log n)

    输入: A, B: [..., 24] (LSB first)
    输出: P: [..., 48] (LSB first)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        acc_mode: 累加模式，'parallel' (树形归约) 或 'sequential' (顺序)
    """
    MAX_BITS = 48

    def __init__(self, neuron_template=None, acc_mode='parallel'):
        super().__init__()
        nt = neuron_template
        max_shape = (self.MAX_BITS,)
        max_shape_24 = (24,)

        # 部分积生成: 单实例 VecAND（动态扩展支持）
        self.vec_ands = VecAND(neuron_template=nt, max_param_shape=max_shape_24)

        # 累加器: 使用 PartialProductAccumulator 支持树形归约
        self.vec_adder = VecAdder(48, neuron_template=nt, max_param_shape=max_shape)
        self.pp_acc = PartialProductAccumulator(self.vec_adder, mode=acc_mode)

    def forward(self, A, B):
        """
        A, B: [..., 24] LSB first
        Returns: [..., 48] LSB first
        """
        device = A.device
        batch_shape = A.shape[:-1]
        zeros_24 = torch.zeros(batch_shape + (24,), device=device)

        # 生成所有部分积（单实例，动态扩展支持）
        partial_products = []
        for i in range(24):
            # 第i个部分积: A & B[i] (广播)
            b_i = B[..., i:i+1].expand(*batch_shape, 24)  # [..., 24]
            pp = self.vec_ands(A, b_i)  # [..., 24]

            # 扩展到48位，低位补零（移位）
            if i > 0:
                low_zeros = torch.zeros(batch_shape + (i,), device=device)
                pp_48 = torch.cat([low_zeros, pp, zeros_24[..., :24-i]], dim=-1)
            else:
                pp_48 = torch.cat([pp, zeros_24], dim=-1)
            partial_products.append(pp_48)

        # 使用 PartialProductAccumulator 累加（树形归约 O(log n)）
        return self.pp_acc(partial_products)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 48位前导零检测器
# ==============================================================================
class LeadingZeroDetector48(nn.Module):
    """48位前导零检测器 - 输出6位LZC (100%纯SNN门电路)

    输入: X[47:0] MSB first
    输出: LZC[5:0] 前导零个数 (MSB first)

    实现: 使用纯门电路，禁止 * 和 + 操作脉冲值

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    MAX_BITS = 48

    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        max_shape_48 = (self.MAX_BITS,)
        max_shape_6 = (6,)
        max_shape_1 = (1,)
        # 单实例门电路 (动态扩展机制支持复用)
        self.vec_not_found = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_and_first = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux_lzc = VecMUX(neuron_template=nt, max_param_shape=max_shape_6)
        self.vec_or_found = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        # 全零检测
        self.vec_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_48)
        self.vec_not_allzero = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux_final = VecMUX(neuron_template=nt, max_param_shape=max_shape_6)

    def forward(self, X):
        """X: [..., 48] MSB first, returns: [..., 6] LZC MSB first"""
        device = X.device
        batch_shape = X.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)

        # 初始化lzc为全0 (6位)
        lzc = torch.zeros(batch_shape + (6,), device=device)

        # found = 是否已找到第一个1
        found = zeros.clone()

        for i in range(48):
            bit = X[..., i:i+1]

            # not_found = NOT(found)
            not_found = self.vec_not_found(found)

            # is_first = bit AND not_found
            is_first = self.vec_and_first(bit, not_found)

            # 如果is_first=1，设置lzc为当前位置i的二进制表示
            # i的6位二进制 (扩展到batch)
            pos_bits = torch.tensor([
                (i >> 5) & 1, (i >> 4) & 1, (i >> 3) & 1,
                (i >> 2) & 1, (i >> 1) & 1, i & 1
            ], device=device, dtype=torch.float32)
            pos_bits = pos_bits.expand(*batch_shape, 6)

            # lzc = MUX(is_first, pos_bits, lzc)
            is_first_exp = is_first.expand(*batch_shape, 6)
            lzc = self.vec_mux_lzc(is_first_exp, pos_bits, lzc)

            # found = found OR is_first
            found = self.vec_or_found(found, is_first)

        # 检测是否全零 (使用向量化 OR 树)
        any_one = self.vec_or_tree(X)
        all_zero = self.vec_not_allzero(any_one)

        # 如果全零，lzc = 48 = 0b110000
        lzc_48 = torch.tensor([1, 1, 0, 0, 0, 0], device=device, dtype=torch.float32)
        lzc_48 = lzc_48.expand(*batch_shape, 6)
        all_zero_exp = all_zero.expand(*batch_shape, 6)
        lzc = self.vec_mux_final(all_zero_exp, lzc_48, lzc)

        return lzc

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 48位桶形右移位器 (用于subnormal处理) - 向量化
# ==============================================================================
class BarrelShifterRight48(nn.Module):
    """48位桶形右移位器 (向量化SNN) - 输出sticky位"""
    MAX_BITS = 48

    def __init__(self, neuron_template=None):
        super().__init__()
        self.data_bits = 48
        self.shift_bits = 6  # 最多移63位
        nt = neuron_template
        max_shape_48 = (self.MAX_BITS,)
        max_shape_1 = (1,)

        # 单实例门电路 (动态扩展机制支持复用)
        self.sticky_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_48)
        self.sticky_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_1)
        self.sticky_accum_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.shift_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_48)

    def forward(self, X, shift):
        """
        X: [..., 48] MSB first
        shift: [..., 6] MSB first
        Returns: (shifted_data, sticky_bit)
        """
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
                layer_sticky = self.sticky_or_tree(shifted_out_bits)
            else:
                layer_sticky = zeros

            # 只有当s_bit=1时才累积sticky
            sticky_contrib = self.sticky_mux(s_bit, layer_sticky, zeros)
            sticky_accum = self.sticky_accum_or(sticky_accum, sticky_contrib)

            # 右移操作 (向量化)
            zeros_pad = torch.zeros(batch_shape + (shift_amt,), device=device)
            shifted = torch.cat([zeros_pad, current[..., :-shift_amt]], dim=-1)
            s_bit_exp = s_bit.expand(*batch_shape, self.data_bits)
            current = self.shift_mux(s_bit_exp, shifted, current)

        return current, sticky_accum

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 48位桶形左移位器 - 向量化
# ==============================================================================
class BarrelShifterLeft48(nn.Module):
    """48位桶形左移位器 (向量化SNN)"""
    MAX_BITS = 48

    def __init__(self, neuron_template=None):
        super().__init__()
        self.data_bits = 48
        self.shift_bits = 6  # 最多移63位
        nt = neuron_template
        max_shape = (self.MAX_BITS,)

        # 单实例门电路 (动态扩展机制支持复用)
        self.shift_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape)

    def forward(self, X, shift):
        """X: [..., 48], shift: [..., 6] (MSB first)"""
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
            current = self.shift_mux(s_bit_exp, shifted, current)

        return current

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# 9位加法器 (指数运算) - 向量化
# ==============================================================================
class RippleCarryAdder9Bit(nn.Module):
    """9位加法器 - 向量化SNN (LSB first)"""
    MAX_BITS = 9

    def __init__(self, neuron_template=None):
        super().__init__()
        max_shape = (self.MAX_BITS,)
        self.vec_adder = VecAdder(9, neuron_template=neuron_template, max_param_shape=max_shape)
        
    def forward(self, A, B, Cin=None):
        return self.vec_adder(A, B, Cin)
    
    def reset(self):
        self.vec_adder.reset()


# ==============================================================================
# 10位加法器 (指数运算 - 用于正确检测溢出/下溢) - 向量化
# ==============================================================================
class RippleCarryAdder10Bit(nn.Module):
    """10位加法器 - 向量化SNN (LSB first)"""
    MAX_BITS = 10

    def __init__(self, neuron_template=None):
        super().__init__()
        max_shape = (self.MAX_BITS,)
        self.vec_adder = VecAdder(10, neuron_template=neuron_template, max_param_shape=max_shape)
        
    def forward(self, A, B, Cin=None):
        return self.vec_adder(A, B, Cin)
    
    def reset(self):
        self.vec_adder.reset()


# ==============================================================================
# 10位减法器 - 向量化
# ==============================================================================
class Subtractor10Bit(nn.Module):
    """10位减法器 - 向量化SNN (LSB first)"""
    MAX_BITS = 10

    def __init__(self, neuron_template=None):
        super().__init__()
        max_shape = (self.MAX_BITS,)
        self.vec_subtractor = VecSubtractor(10, neuron_template=neuron_template, max_param_shape=max_shape)
        
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
    MAX_BITS = 9

    def __init__(self, neuron_template=None):
        super().__init__()
        max_shape = (self.MAX_BITS,)
        self.vec_subtractor = VecSubtractor(9, neuron_template=neuron_template, max_param_shape=max_shape)
        
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

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    MAX_BITS = 48  # 最大中间位宽 (24x24乘法产生48位)

    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 预分配参数形状
        max_shape_48 = (48,)
        max_shape_24 = (24,)
        max_shape_23 = (23,)
        max_shape_10 = (10,)
        max_shape_8 = (8,)
        max_shape_5 = (5,)
        max_shape_1 = (1,)

        # ===== 符号 =====
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))

        # ===== 指数运算 (使用10位确保正确检测溢出/下溢) =====
        # 指数加法: Ea + Eb (10位)
        self.exp_adder = RippleCarryAdder10Bit(neuron_template=nt)
        # 减bias: - 127
        self.bias_sub = Subtractor10Bit(neuron_template=nt)
        # 指数+1 (规格化调整) - 两个独立实例
        self.exp_inc_1 = RippleCarryAdder10Bit(neuron_template=nt)  # for exp_overflow
        self.exp_inc_2 = RippleCarryAdder10Bit(neuron_template=nt)  # for exp_final_pre
        # 指数减法 (LZC调整)
        self.exp_lzc_sub = Subtractor10Bit(neuron_template=nt)

        # ===== 尾数乘法 =====
        self.mantissa_mul = ArrayMultiplier24x24(neuron_template=nt)

        # ===== 规格化 =====
        self.lzd = LeadingZeroDetector48(neuron_template=nt)
        self.norm_shifter = BarrelShifterLeft48(neuron_template=nt)

        # ===== RNE舍入 =====
        self.rne_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.rne_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.round_adder = VecAdder(bits=24, neuron_template=nt, max_param_shape=(24,))

        # ===== Sticky bit OR（向量化树形归约）=====
        self.sticky_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_23)

        # ===== 特殊值检测（向量化树形归约）=====
        # 指数全1检测 (Inf/NaN) - 使用 VecANDTree
        self.exp_all_one_tree_a = VecANDTree(neuron_template=nt, max_param_shape=max_shape_8)
        self.exp_all_one_tree_b = VecANDTree(neuron_template=nt, max_param_shape=max_shape_8)

        # 指数全0检测 (Zero/Subnormal) - 使用 VecORTree + NOT
        self.exp_any_one_tree_a = VecORTree(neuron_template=nt, max_param_shape=max_shape_8)
        self.exp_zero_not_a = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.exp_any_one_tree_b = VecORTree(neuron_template=nt, max_param_shape=max_shape_8)
        self.exp_zero_not_b = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)

        # 尾数全0检测 - 使用 VecORTree + NOT
        self.mant_any_one_tree_a = VecORTree(neuron_template=nt, max_param_shape=max_shape_23)
        self.mant_zero_not_a = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.mant_any_one_tree_b = VecORTree(neuron_template=nt, max_param_shape=max_shape_23)
        self.mant_zero_not_b = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        
        # ===== 零检测 =====
        self.a_is_zero_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.b_is_zero_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.either_zero_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # ===== Inf检测 =====
        self.a_is_inf_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.b_is_inf_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.either_inf_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # ===== NaN检测 =====
        self.a_mant_nonzero_not = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.b_mant_nonzero_not = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.a_is_nan_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.b_is_nan_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.either_nan_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 0 × Inf = NaN =====
        self.zero_times_inf_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.result_is_nan_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # ===== Subnormal检测 =====
        self.a_is_subnormal_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.b_is_subnormal_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 尾数前导位选择 =====
        self.mux_a_leading = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_b_leading = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 指数修正 (subnormal有效指数=1)（向量化）=====
        self.mux_a_exp = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.mux_b_exp = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)

        # ===== 溢出/下溢处理 =====
        # 溢出检测: exp >= 255 (使用10位计算)
        # 需要检测9位指数的第9位(溢出位)和低8位>=255
        # 3个独立实例 (用于不同位置)
        self.overflow_and_1 = VecAND(neuron_template=nt, max_param_shape=(1,))  # is_overflow
        self.overflow_and_2 = VecAND(neuron_template=nt, max_param_shape=(1,))  # overflow_and_valid
        self.overflow_and_3 = VecAND(neuron_template=nt, max_param_shape=(1,))  # underflow_only
        self.overflow_255_tree = VecANDTree(neuron_template=nt, max_param_shape=max_shape_8)  # 检测低8位全1

        # 下溢检测: exp <= 0 (9位有符号)
        # bit8=1表示负数(下溢)，或低9位全0
        self.underflow_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        # 检测10位全0 - 向量化
        self.exp_zero_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_10)
        self.exp_zero_not_gate = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        # 独立实例 (禁止循环内复用)
        self.underflow_or_1 = VecOR(neuron_template=nt, max_param_shape=(1,))  # low9_ge_255
        self.underflow_or_2 = VecOR(neuron_template=nt, max_param_shape=(1,))  # is_underflow_or_subnormal
        self.is_underflow_gate = VecAND(neuron_template=nt, max_param_shape=(1,))  # is_underflow计算
        self.subnorm_valid_gate = VecAND(neuron_template=nt, max_param_shape=(1,))  # subnorm_and_valid计算
        self.underflow_final_gate = VecAND(neuron_template=nt, max_param_shape=(1,))  # underflow_final计算

        # 溢出结果选择MUX（向量化）
        self.overflow_mux_e = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.overflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)
        self.not_overflow = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)

        # 下溢结果选择MUX（向量化）
        self.underflow_mux_e = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.underflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)
        self.not_underflow = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.not_either_zero_gate = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)  # 独立实例

        # ===== Subnormal处理 =====
        # subnormal检测: exp <= 0 但 exp > -150 (大约)
        # 需要额外右移 (1 - exp) 位
        self.subnormal_shifter = BarrelShifterRight48(neuron_template=nt)  # 对尾数进行额外右移

        # 移位量计算: shift = 1 - exp = 1 + (-exp)
        # 需要将10位负指数转换为正的移位量
        self.shift_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_10)  # 取反（向量化）
        # 2个独立实例
        self.shift_add_one_1 = RippleCarryAdder10Bit(neuron_template=nt)  # -exp = ~exp + 1
        self.shift_add_one_2 = RippleCarryAdder10Bit(neuron_template=nt)  # shift = neg_exp + 1
        self.shift_add_one_const = VecAdder(bits=6, neuron_template=nt, max_param_shape=(6,))  # 加1得到移位量

        # subnormal舍入（向量化）
        self.subnorm_sticky_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_24)  # 合并sticky
        self.shift_high_bits_or_tree = VecORTree(neuron_template=nt, max_param_shape=max_shape_5)  # shift_bit5_or_higher
        self.subnorm_rne_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.subnorm_rne_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.subnorm_round_adder = VecAdder(bits=24, neuron_template=nt, max_param_shape=(24,))

        # subnormal结果选择（向量化）
        self.subnorm_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)
        # 2个独立实例
        self.is_subnormal_and_1 = VecAND(neuron_template=nt, max_param_shape=(1,))  # is_subnormal
        self.is_subnormal_and_2 = VecAND(neuron_template=nt, max_param_shape=(1,))  # subnorm_temp
        self.not_very_underflow = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)  # 非完全下溢

        # ===== 结果选择MUX（向量化）=====
        # NaN输出
        self.nan_mux_e = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.nan_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)
        # Inf输出
        self.inf_mux_e = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.inf_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)
        # 零输出
        self.zero_mux_e = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.zero_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)

        # ===== 舍入进位处理（向量化）=====
        self.round_carry_not = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.mant_clear_and = VecAND(neuron_template=nt, max_param_shape=max_shape_23)
        self.exp_round_inc = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))
        self.exp_round_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)

        # ===== 乘积溢出检测 (P[47]=1)（单实例）=====
        self.prod_overflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 特殊值选择 (纯SNN VecNOT/AND门) =====
        self.not_result_is_nan = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.not_either_inf = VecNOT(neuron_template=nt, max_param_shape=max_shape_1)
        self.inf_and_not_nan_gate = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.zero_and_not_nan_gate = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.zero_only_gate = VecAND(neuron_template=nt, max_param_shape=(1,))
        # 独立实例 (禁止循环内复用)
        self.underflow_temp_gate = VecAND(neuron_template=nt, max_param_shape=(1,))  # underflow_temp计算
        self.underflow_final_mux_e = VecMUX(neuron_template=nt, max_param_shape=max_shape_8)
        self.underflow_final_mux_m = VecMUX(neuron_template=nt, max_param_shape=max_shape_23)
        
    def forward(self, A, B):
        """
        A, B: [..., 32] FP32脉冲 [S | E7..E0 | M22..M0] MSB first
        Returns: [..., 32] FP32脉冲
        """
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
        
        # ===== 3. 特殊值检测（向量化树形归约）=====
        # 指数全1检测 (Inf/NaN) - VecANDTree 一次处理所有8位
        e_a_all_one = self.exp_all_one_tree_a(e_a)  # [..., 1]
        e_b_all_one = self.exp_all_one_tree_b(e_b)  # [..., 1]

        # 指数全0检测 (Zero/Subnormal) - VecORTree + NOT
        e_a_any_one = self.exp_any_one_tree_a(e_a)  # [..., 1]
        e_a_is_zero = self.exp_zero_not_a(e_a_any_one)
        e_b_any_one = self.exp_any_one_tree_b(e_b)  # [..., 1]
        e_b_is_zero = self.exp_zero_not_b(e_b_any_one)

        # 尾数全0检测 - VecORTree + NOT
        m_a_any_one = self.mant_any_one_tree_a(m_a)  # [..., 1]
        m_a_is_zero = self.mant_zero_not_a(m_a_any_one)
        m_b_any_one = self.mant_any_one_tree_b(m_b)  # [..., 1]
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
        # 向量化：构建常量并用 VecMUX 一次选择所有8位
        const_e_one = torch.cat([ones, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)  # [8]
        # 广播 sel 到 [batch..., 8]
        a_is_subnormal_8 = a_is_subnormal.expand(*batch_shape, 8)
        b_is_subnormal_8 = b_is_subnormal.expand(*batch_shape, 8)
        e_a_corrected = self.mux_a_exp(a_is_subnormal_8, const_e_one, e_a_le)
        e_b_corrected = self.mux_b_exp(b_is_subnormal_8, const_e_one, e_b_le)
        
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
        exp_overflow, _ = self.exp_inc_1(raw_e_10, one_10)

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
        exp_final_pre, _ = self.exp_inc_2(exp_adjusted, one_10)
        
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
        
        # Sticky = OR(bit[25:48]) - 向量化
        sticky_bits = product_norm[..., 25:48]  # [..., 23]
        sticky = self.sticky_or_tree(sticky_bits)  # [..., 1]
        
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
        
        # 如果进位，尾数清零 - 向量化
        not_carry = self.round_carry_not(mant_carry)
        not_carry_23 = not_carry.expand(*batch_shape, 23)  # 广播到23位
        mant_final_le = self.mant_clear_and(not_carry_23, mant_rounded[..., :23])
        mant_final = mant_final_le.flip(-1)  # 转MSB first
        
        # 如果进位，指数+1
        # exp_final_pre 是 LSB first，取低8位
        exp_8_le = exp_final_pre[..., :8]  # LSB first
        carry_inc = torch.cat([mant_carry, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        exp_after_round_le, _ = self.exp_round_inc(exp_8_le, carry_inc)  # LSB first
        
        # 选择最终指数 (LSB first) - 向量化
        mant_carry_8 = mant_carry.expand(*batch_shape, 8)  # 广播到8位
        exp_final_le = self.exp_round_mux(mant_carry_8, exp_after_round_le, exp_8_le)
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
        exp_low8 = exp_final_pre[..., 0:8]  # [..., 8]
        exp_all_255 = self.overflow_255_tree(exp_low8)  # VecANDTree: 所有位都为1
        
        # low9 >= 255: bit8=1 或 低8位全1
        low9_ge_255 = self.underflow_or_1(exp_bit8, exp_all_255)

        # 溢出条件: bit9=0 且 low9>=255
        not_bit9 = self.underflow_not(exp_bit9)
        is_overflow = self.overflow_and_1(not_bit9, low9_ge_255)
        
        # 下溢: 10位有符号值 <= 0
        # bit9=1 (负数) 或 全10位=0

        # 检测10位全0 - 向量化
        # 方法: OR所有位，如果结果为0则全0
        exp_any_one = self.exp_zero_or_tree(exp_final_pre)  # VecORTree
        exp_is_zero = self.exp_zero_not_gate(exp_any_one)  # NOT: any_one=0 → is_zero=1
        
        # 下溢条件: bit9=1 (负数) 或 全零
        is_underflow_or_subnormal = self.underflow_or_2(exp_bit9, exp_is_zero)
        
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
        
        # 计算 -exp (取反+1) - 向量化
        exp_neg = self.shift_not(exp_final_pre)  # VecNOT 处理所有10位
        
        one_10 = torch.cat([ones, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        neg_exp_10, _ = self.shift_add_one_1(exp_neg, one_10)  # -exp = ~exp + 1

        # 移位量 = 1 + (-exp) = 1 - exp (因为exp是负的或0)
        # 对于subnormal，我们需要移位 (1 - exp) 位
        # 如果exp=0, shift=1; 如果exp=-1, shift=2; 等等

        # 移位量 = neg_exp_10 + 1 (10位完整计算)
        shift_full_10, _ = self.shift_add_one_2(neg_exp_10, one_10)
        
        # 检测移位量是否过大 (>= 32，即bit5或更高位非零)
        # 如果bit5-9中任意一位为1，则移位量>=32，完全下溢 - 向量化
        high_bits = shift_full_10[..., 5:10]  # [..., 5]
        shift_bit5_or_higher = self.shift_high_bits_or_tree(high_bits)  # VecORTree
        
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
        
        # Sticky = OR(shifted_product[25:48]) OR shift_sticky - 向量化
        sticky_bits = shifted_product[..., 25:48]  # [..., 23]
        sticky_with_shift = torch.cat([sticky_bits, shift_sticky], dim=-1)  # [..., 24]
        subnorm_sticky = self.subnorm_sticky_or_tree(sticky_with_shift)  # [..., 1]
        
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
        is_subnormal = self.is_subnormal_and_1(is_underflow_or_subnormal, is_reasonable_shift)
        
        # 完全下溢: exp<=0 且 移位量不合理 (>=32)
        is_underflow = self.is_underflow_gate(is_underflow_or_subnormal, shift_bit5_or_higher)
        
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
        # 先应用NaN - 向量化
        result_is_nan_8 = result_is_nan.expand(*batch_shape, 8)
        result_is_nan_23 = result_is_nan.expand(*batch_shape, 23)
        e_out = self.nan_mux_e(result_is_nan_8, nan_exp, exp_final)
        m_out = self.nan_mux_m(result_is_nan_23, nan_mant, mant_final)

        # 应用Inf (非NaN的Inf情况) - 向量化
        not_nan = self.not_result_is_nan(result_is_nan)
        inf_and_not_nan = self.inf_and_not_nan_gate(either_inf, not_nan)
        inf_and_not_nan_8 = inf_and_not_nan.expand(*batch_shape, 8)
        inf_and_not_nan_23 = inf_and_not_nan.expand(*batch_shape, 23)
        e_out = self.inf_mux_e(inf_and_not_nan_8, inf_exp, e_out)
        m_out = self.inf_mux_m(inf_and_not_nan_23, inf_mant, m_out)

        # 应用Zero (非NaN非Inf的零情况) - 向量化
        not_either_inf = self.not_either_inf(either_inf)
        zero_and_not_nan = self.zero_and_not_nan_gate(either_zero, not_nan)
        zero_only = self.zero_only_gate(zero_and_not_nan, not_either_inf)
        zero_only_8 = zero_only.expand(*batch_shape, 8)
        zero_only_23 = zero_only.expand(*batch_shape, 23)
        e_out = self.zero_mux_e(zero_only_8, zero_exp, e_out)
        m_out = self.zero_mux_m(zero_only_23, zero_mant, m_out)
        
        # ===== 11. 应用计算溢出 (非特殊值情况下exp>=255) =====
        # 溢出 → Inf (E=FF, M=0) - 向量化
        not_special = self.not_overflow(result_is_nan)
        # 这里简化: 如果溢出且非NaN，输出Inf
        overflow_and_valid = self.overflow_and_2(is_overflow, not_nan)
        overflow_and_valid_8 = overflow_and_valid.expand(*batch_shape, 8)
        overflow_and_valid_23 = overflow_and_valid.expand(*batch_shape, 23)
        e_out = self.overflow_mux_e(overflow_and_valid_8, inf_exp, e_out)
        m_out = self.overflow_mux_m(overflow_and_valid_23, inf_mant, m_out)
        
        # ===== 12. 应用Subnormal (exp<=0 但 >-150，且非零输入) =====
        # subnormal: exp=0, mant=subnorm_mant_final - 向量化
        not_overflow_flag = self.not_underflow(is_overflow)
        not_either_zero = self.not_either_zero_gate(either_zero)
        subnorm_temp = self.is_subnormal_and_2(is_subnormal, not_nan)
        subnorm_and_valid = self.subnorm_valid_gate(subnorm_temp, not_either_zero)
        subnorm_and_valid_8 = subnorm_and_valid.expand(*batch_shape, 8)
        subnorm_and_valid_23 = subnorm_and_valid.expand(*batch_shape, 23)
        e_out = self.underflow_mux_e(subnorm_and_valid_8, zero_exp, e_out)
        m_out = self.subnorm_mux_m(subnorm_and_valid_23, subnorm_mant_final, m_out)
        
        # ===== 13. 应用完全下溢 (exp < -150左右 → 0, 且非零输入) =====
        # 向量化
        underflow_only = self.overflow_and_3(is_underflow, not_overflow_flag)
        underflow_temp = self.underflow_temp_gate(underflow_only, not_nan)  # 独立gate
        underflow_final = self.underflow_final_gate(underflow_temp, not_either_zero)
        underflow_final_8 = underflow_final.expand(*batch_shape, 8)
        underflow_final_23 = underflow_final.expand(*batch_shape, 23)
        e_out = self.underflow_final_mux_e(underflow_final_8, zero_exp, e_out)
        m_out = self.underflow_final_mux_m(underflow_final_23, zero_mant, m_out)
        
        # ===== 14. 组装输出 =====
        result = torch.cat([s_out, e_out, m_out], dim=-1)
        
        return result
    
    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)

