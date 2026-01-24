"""
FP64 三角函数 Sin/Cos - 100%纯SNN门电路实现
============================================

算法 (Taylor级数 + 范围缩减):
1. 范围缩减: q = round(x / (π/2)), r = x - q * (π/2)
2. 根据 q mod 4 选择计算方式和符号
3. Taylor级数逼近 (在 [-π/2, π/2] 范围内):
   - sin(r) ≈ r - r³/6 + r⁵/120 - r⁷/5040
   - cos(r) ≈ 1 - r²/2 + r⁴/24 - r⁶/720

核心原则:
- 禁止使用 ones - x, 必须用 NOTGate
- 禁止使用 a * b 逻辑AND, 必须用 ANDGate
- 所有操作通过纯SNN门电路完成

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
import struct
import math

from atomic_ops.core.vec_logic_gates import VecXOR, VecAND, VecOR, VecNOT, VecMUX
from atomic_ops.arithmetic.fp64.fp64_mul import SpikeFP64Multiplier
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_div import SpikeFP64Divider


# ==============================================================================
# FP64 辅助函数
# ==============================================================================

def float64_to_bits(f):
    """将Python float转换为64位整数表示"""
    return struct.unpack('>Q', struct.pack('>d', f))[0]


def make_fp64_constant(val, batch_shape, device):
    """创建FP64常量脉冲 (向量化)"""
    bits = float64_to_bits(val)
    # 向量化位提取: 使用位移和掩码
    bit_positions = torch.arange(63, -1, -1, device=device, dtype=torch.int64)
    pulse_1d = ((bits >> bit_positions) & 1).float()  # [64]
    # 广播到 batch_shape
    pulse = pulse_1d.expand(batch_shape + (64,)).clone()
    return pulse


# ==============================================================================
# FP64 Round (四舍五入到最近整数)
# ==============================================================================

class SpikeFP64Round(nn.Module):
    """FP64 四舍五入 - 100%纯SNN门电路

    round(x) = floor(x + 0.5) for x >= 0
    round(x) = ceil(x - 0.5) for x < 0

    简化实现: round(x) = floor(x + 0.5 * sign(x) + 0.5)
    即: round(x) = floor(x + 0.5) for positive
        round(x) = -floor(-x + 0.5) for negative
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 预分配参数形状
        max_shape_64 = (64,)
        max_shape_1 = (1,)

        # 需要导入 Floor
        from atomic_ops.activation.fp64.fp64_exp import SpikeFP64Floor

        self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
        self.fp64_floor = SpikeFP64Floor(neuron_template=nt)
        self.vec_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_64)

    def forward(self, x):
        """四舍五入

        输入: [..., 64] FP64脉冲
        输出: [..., 64] FP64脉冲 (整数值)
        """
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)

        # 常量
        const_half = make_fp64_constant(0.5, batch_shape, device)

        # 获取符号位
        sign_x = x[..., 0:1]

        # 取绝对值: 将符号位设为0 (使用 torch.cat 组合)
        x_abs = torch.cat([zeros, x[..., 1:]], dim=-1)

        # |x| + 0.5
        x_plus_half = self.fp64_adder(x_abs, const_half)

        # floor(|x| + 0.5)
        result_abs = self.fp64_floor(x_plus_half)

        # 恢复符号 (使用 torch.cat 组合)
        result = torch.cat([sign_x, result_abs[..., 1:]], dim=-1)

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# FP64 提取低2位 (用于 q mod 4)
# ==============================================================================

class SpikeFP64ExtractMod4(nn.Module):
    """从FP64整数中提取 mod 4 的结果 (低2位)

    输入: FP64 整数脉冲 (假设为小整数 0-7)
    输出: 2位脉冲 [b1, b0] 表示 mod 4 结果

    FP64 小整数的编码:
    - 0: E=0, M=0
    - 1: E=1023, M=0 (1.0 × 2^0, 隐含位=1)
    - 2: E=1024, M=0 (1.0 × 2^1)
    - 3: E=1024, M=0x8000... (1.5 × 2^1)
    - 4: E=1025, M=0 (1.0 × 2^2)
    - 5: E=1025, M=0x4000... (1.25 × 2^2)
    - 6: E=1025, M=0x8000... (1.5 × 2^2)
    - 7: E=1025, M=0xC000... (1.75 × 2^2)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 预分配参数形状
        max_shape_1 = (1,)

        self.vec_and = VecAND(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_or = VecOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_1)

    def forward(self, x):
        """提取 mod 4

        FP64格式: [S | E10..E0 | M51..M0]

        指数检测:
        - E = 0 (全0): 值 = 0
        - E = 1023 (01111111111): 值 = 1.xxx, 整数部分 = 1
        - E = 1024 (10000000000): 值 = 2~3
        - E = 1025 (10000000001): 值 = 4~7

        mod 4 逻辑:
        - E <= 1022: mod4 = 0
        - E = 1023: mod4 = 1 (隐含位)
        - E = 1024: mod4 = 2 + M[51]
        - E = 1025: mod4 = M[51]*2 + M[50]
        """
        device = x.device
        batch_shape = x.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        # 获取符号
        sign = x[..., 0:1]

        # 获取指数 E 和尾数 M
        e_bits = x[..., 1:12]  # E10..E0
        m_bits = x[..., 12:64]  # M51..M0

        # 指数关键位
        e10 = e_bits[..., 0:1]   # E 最高位 (bit 10)
        e0 = e_bits[..., 10:11]  # E 最低位 (bit 0)

        # 尾数高位
        m51 = m_bits[..., 0:1]   # M 最高位
        m50 = m_bits[..., 1:2]   # M 次高位

        # 指数值判断:
        # E = 1023 (0b01111111111): e10=0, e0=1
        # E = 1024 (0b10000000000): e10=1, e0=0
        # E = 1025 (0b10000000001): e10=1, e0=1

        # VecMUX 语义: MUX(sel, a, b) - sel=1 选 a, sel=0 选 b
        #
        # 当 e10=1 时 (E >= 1024):
        #   e0=0 (E=1024): b0 = m51, b1 = 1
        #   e0=1 (E >= 1025): b0 = m50, b1 = m51
        b0_e1024 = m51
        b0_e1025 = m50
        # e0=1 选 b0_e1025, e0=0 选 b0_e1024
        b0_high = self.vec_mux(e0.expand_as(m51), b0_e1025, b0_e1024)

        b1_e1024 = ones
        b1_e1025 = m51
        # e0=1 选 b1_e1025, e0=0 选 b1_e1024
        b1_high = self.vec_mux(e0.expand_as(m51), b1_e1025, b1_e1024)

        # 当 e10=0 时 (E <= 1023):
        #   e0=1 (E=1023): b0 = 1, b1 = 0 (值为1)
        #   e0=0 (E < 1023): b0 = 0, b1 = 0 (值 < 1)
        b0_low = e0  # 1 if E=1023, 0 otherwise
        b1_low = zeros

        # 最终选择: e10=1 选 high, e10=0 选 low
        b0 = self.vec_mux(e10.expand_as(b0_low), b0_high, b0_low)
        b1 = self.vec_mux(e10.expand_as(b1_low), b1_high, b1_low)

        result = torch.cat([b1, b0], dim=-1)
        return result, sign

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


# ==============================================================================
# FP64 Sin 核心实现
# ==============================================================================

class SpikeFP64Sin(nn.Module):
    """FP64 正弦函数 - 100%纯SNN门电路

    算法:
    1. 范围缩减: q = round(x / (π/2)), r = x - q * (π/2)
    2. 根据 q mod 4 选择:
       - 0: sin(r)
       - 1: cos(r)
       - 2: -sin(r)
       - 3: -cos(r)
    3. Taylor级数逼近

    输入: [..., 64] FP64脉冲
    输出: [..., 64] FP64脉冲
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 预分配参数形状
        max_shape_64 = (64,)
        max_shape_1 = (1,)

        # 基本运算
        self.mul = SpikeFP64Multiplier(neuron_template=nt)
        self.add = SpikeFP64Adder(neuron_template=nt)
        self.div = SpikeFP64Divider(neuron_template=nt)
        self.round = SpikeFP64Round(neuron_template=nt)
        self.extract_mod4 = SpikeFP64ExtractMod4(neuron_template=nt)

        # 符号翻转 (XOR)
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)

        # mod4 负数处理
        self.mod4_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)

        # 结果选择
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_64)

        # Taylor多项式需要的额外乘法器和加法器
        self.mul2 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul3 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul4 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul5 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul6 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul7 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul8 = SpikeFP64Multiplier(neuron_template=nt)  # 额外乘法器 (sin term7)

        self.add2 = SpikeFP64Adder(neuron_template=nt)
        self.add3 = SpikeFP64Adder(neuron_template=nt)
        self.add4 = SpikeFP64Adder(neuron_template=nt)

    def _compute_sin_taylor(self, r, batch_shape, device):
        """计算 sin(r) 的 Taylor 级数

        sin(r) = r - r³/6 + r⁵/120 - r⁷/5040
        """
        # 常量系数
        const_neg_1_6 = make_fp64_constant(-1.0/6.0, batch_shape, device)
        const_1_120 = make_fp64_constant(1.0/120.0, batch_shape, device)
        const_neg_1_5040 = make_fp64_constant(-1.0/5040.0, batch_shape, device)

        # r²
        r2 = self.mul2(r, r)

        # r³ = r² * r
        r3 = self.mul3(r2, r)

        # r⁵ = r³ * r²
        r5 = self.mul4(r3, r2)

        # r⁷ = r⁵ * r²
        r7 = self.mul5(r5, r2)

        # -r³/6
        term3 = self.mul6(r3, const_neg_1_6)

        # r⁵/120
        term5 = self.mul7(r5, const_1_120)

        # -r⁷/5040 (使用 __init__ 中创建的 mul8)
        term7 = self.mul8(r7, const_neg_1_5040)

        # r + term3
        sum1 = self.add2(r, term3)

        # sum1 + term5
        sum2 = self.add3(sum1, term5)

        # sum2 + term7
        result = self.add4(sum2, term7)

        return result

    def _compute_cos_taylor(self, r, batch_shape, device):
        """计算 cos(r) 的 Taylor 级数

        cos(r) = 1 - r²/2 + r⁴/24 - r⁶/720
        """
        # 常量
        const_1 = make_fp64_constant(1.0, batch_shape, device)
        const_neg_0_5 = make_fp64_constant(-0.5, batch_shape, device)
        const_1_24 = make_fp64_constant(1.0/24.0, batch_shape, device)
        const_neg_1_720 = make_fp64_constant(-1.0/720.0, batch_shape, device)

        # r²
        r2 = self.mul2(r, r)

        # r⁴ = r² * r²
        r4 = self.mul3(r2, r2)

        # r⁶ = r⁴ * r²
        r6 = self.mul4(r4, r2)

        # -r²/2
        term2 = self.mul5(r2, const_neg_0_5)

        # r⁴/24
        term4 = self.mul6(r4, const_1_24)

        # -r⁶/720
        term6 = self.mul7(r6, const_neg_1_720)

        # 1 + term2
        sum1 = self.add2(const_1, term2)

        # sum1 + term4
        sum2 = self.add3(sum1, term4)

        # sum2 + term6
        result = self.add4(sum2, term6)

        return result

    def forward(self, x):
        """计算 sin(x)

        输入: [..., 64] FP64脉冲
        输出: [..., 64] FP64脉冲
        """
        device = x.device
        batch_shape = x.shape[:-1]
        ones = torch.ones(batch_shape + (1,), device=device)

        # 常量
        const_pi_2 = make_fp64_constant(math.pi / 2, batch_shape, device)
        const_2_pi = make_fp64_constant(2.0 / math.pi, batch_shape, device)

        # 1. 范围缩减: q = round(x / (π/2))
        x_div_pi2 = self.mul(x, const_2_pi)  # x * (2/π) = x / (π/2)
        q = self.round(x_div_pi2)

        # r = x - q * (π/2)
        # 改用乘法: q * (π/2)
        q_times_pi2 = self.mul(q, const_pi_2)

        # r = x - q*(π/2) : 翻转符号位使用 VecXOR
        neg_sign = self.sign_xor(q_times_pi2[..., 0:1], ones)
        neg_q_times_pi2 = torch.cat([neg_sign, q_times_pi2[..., 1:]], dim=-1)
        r = self.add(x, neg_q_times_pi2)

        # 2. 提取 q mod 4
        q_mod4, q_sign = self.extract_mod4(q)

        # 处理负数 q: 当 q < 0 时，需要计算 (4 - |q| mod 4) mod 4
        # 2's complement 公式: new_b0 = b0 (不变), new_b1 = b1 XOR b0
        b0_raw = q_mod4[..., 1:2]  # 最低位
        b1_raw = q_mod4[..., 0:1]  # 次低位

        # 计算负数情况下的 new_b1 = b1 XOR b0
        neg_b1 = self.mod4_xor(b1_raw, b0_raw)

        # 根据符号选择: q_sign=1 (负数) 选 neg_b1, q_sign=0 (正数) 选 b1_raw
        b1 = self.vec_mux(q_sign.expand_as(b1_raw), neg_b1, b1_raw)
        b0 = b0_raw  # b0 保持不变

        # 3. 计算 sin(r) 和 cos(r)
        sin_r = self._compute_sin_taylor(r, batch_shape, device)
        cos_r = self._compute_cos_taylor(r, batch_shape, device)

        # 4. 根据 q mod 4 选择结果
        # q_mod4 = [b1, b0]
        # 00 (0): sin(r)
        # 01 (1): cos(r)
        # 10 (2): -sin(r)
        # 11 (3): -cos(r)

        # 选择 sin 或 cos (由 b0 决定)
        # VecMUX(sel, a, b): sel=1 选 a, sel=0 选 b
        # b0=0: sin_r, b0=1: cos_r
        base_result = self.vec_mux(b0.expand_as(sin_r), cos_r, sin_r)

        # 符号调整 (由 b1 决定): 使用 VecXOR 翻转符号位，然后用 torch.cat 组合
        # b1=1: 取负
        new_sign = self.sign_xor(base_result[..., 0:1], b1)
        result = torch.cat([new_sign, base_result[..., 1:]], dim=-1)

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


class SpikeFP64Cos(nn.Module):
    """FP64 余弦函数 - 100%纯SNN门电路

    cos(x) = sin(x + π/2)

    输入: [..., 64] FP64脉冲
    输出: [..., 64] FP64脉冲
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        self.add = SpikeFP64Adder(neuron_template=nt)
        self.sin = SpikeFP64Sin(neuron_template=nt)

    def forward(self, x):
        """计算 cos(x) = sin(x + π/2)

        输入: [..., 64] FP64脉冲
        输出: [..., 64] FP64脉冲
        """
        device = x.device
        batch_shape = x.shape[:-1]

        # x + π/2
        const_pi_2 = make_fp64_constant(math.pi / 2, batch_shape, device)
        x_plus_pi2 = self.add(x, const_pi_2)

        # sin(x + π/2) = cos(x)
        return self.sin(x_plus_pi2)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


class SpikeFP64SinCos(nn.Module):
    """FP64 同时计算 Sin 和 Cos - 优化版本

    同时返回 sin(x) 和 cos(x)，共享范围缩减计算。

    输入: [..., 64] FP64脉冲
    输出: (sin_result, cos_result) 各 [..., 64] FP64脉冲
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 预分配参数形状
        max_shape_64 = (64,)
        max_shape_1 = (1,)

        # 基本运算
        self.mul = SpikeFP64Multiplier(neuron_template=nt)
        self.add = SpikeFP64Adder(neuron_template=nt)
        self.round = SpikeFP64Round(neuron_template=nt)
        self.extract_mod4 = SpikeFP64ExtractMod4(neuron_template=nt)

        # 符号
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)
        self.vec_mux = VecMUX(neuron_template=nt, max_param_shape=max_shape_64)

        # mod4 负数处理
        self.mod4_xor = VecXOR(neuron_template=nt, max_param_shape=max_shape_1)

        # Taylor 多项式的乘法器/加法器 (共享)
        self.mul_r2 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_r3 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_r4 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_r5 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_r6 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_r7 = SpikeFP64Multiplier(neuron_template=nt)

        # sin 项乘法
        self.mul_sin_t3 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_sin_t5 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_sin_t7 = SpikeFP64Multiplier(neuron_template=nt)

        # cos 项乘法
        self.mul_cos_t2 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_cos_t4 = SpikeFP64Multiplier(neuron_template=nt)
        self.mul_cos_t6 = SpikeFP64Multiplier(neuron_template=nt)

        # sin 加法
        self.add_sin1 = SpikeFP64Adder(neuron_template=nt)
        self.add_sin2 = SpikeFP64Adder(neuron_template=nt)
        self.add_sin3 = SpikeFP64Adder(neuron_template=nt)

        # cos 加法
        self.add_cos1 = SpikeFP64Adder(neuron_template=nt)
        self.add_cos2 = SpikeFP64Adder(neuron_template=nt)
        self.add_cos3 = SpikeFP64Adder(neuron_template=nt)

        # 范围缩减
        self.add_range = SpikeFP64Adder(neuron_template=nt)

    def forward(self, x):
        """同时计算 sin(x) 和 cos(x)

        输入: [..., 64] FP64脉冲
        输出: (sin_result, cos_result)
        """
        device = x.device
        batch_shape = x.shape[:-1]
        ones = torch.ones(batch_shape + (1,), device=device)

        # 常量
        const_pi_2 = make_fp64_constant(math.pi / 2, batch_shape, device)
        const_2_pi = make_fp64_constant(2.0 / math.pi, batch_shape, device)
        const_1 = make_fp64_constant(1.0, batch_shape, device)

        # Taylor 系数
        const_neg_1_6 = make_fp64_constant(-1.0/6.0, batch_shape, device)
        const_1_120 = make_fp64_constant(1.0/120.0, batch_shape, device)
        const_neg_1_5040 = make_fp64_constant(-1.0/5040.0, batch_shape, device)
        const_neg_0_5 = make_fp64_constant(-0.5, batch_shape, device)
        const_1_24 = make_fp64_constant(1.0/24.0, batch_shape, device)
        const_neg_1_720 = make_fp64_constant(-1.0/720.0, batch_shape, device)

        # 1. 范围缩减: q = round(x * 2/π)
        x_scaled = self.mul(x, const_2_pi)
        q = self.round(x_scaled)

        # r = x - q * (π/2): 使用 VecXOR 翻转符号位，然后 torch.cat 组合
        q_times_pi2 = self.mul(q, const_pi_2)
        neg_sign = self.sign_xor(q_times_pi2[..., 0:1], ones)
        neg_q_times_pi2 = torch.cat([neg_sign, q_times_pi2[..., 1:]], dim=-1)
        r = self.add_range(x, neg_q_times_pi2)

        # 2. 计算 r 的幂次
        r2 = self.mul_r2(r, r)
        r3 = self.mul_r3(r2, r)
        r4 = self.mul_r4(r2, r2)
        r5 = self.mul_r5(r3, r2)
        r6 = self.mul_r6(r4, r2)
        r7 = self.mul_r7(r5, r2)

        # 3. 计算 sin(r)
        sin_t3 = self.mul_sin_t3(r3, const_neg_1_6)
        sin_t5 = self.mul_sin_t5(r5, const_1_120)
        sin_t7 = self.mul_sin_t7(r7, const_neg_1_5040)

        sin_sum1 = self.add_sin1(r, sin_t3)
        sin_sum2 = self.add_sin2(sin_sum1, sin_t5)
        sin_r = self.add_sin3(sin_sum2, sin_t7)

        # 4. 计算 cos(r)
        cos_t2 = self.mul_cos_t2(r2, const_neg_0_5)
        cos_t4 = self.mul_cos_t4(r4, const_1_24)
        cos_t6 = self.mul_cos_t6(r6, const_neg_1_720)

        cos_sum1 = self.add_cos1(const_1, cos_t2)
        cos_sum2 = self.add_cos2(cos_sum1, cos_t4)
        cos_r = self.add_cos3(cos_sum2, cos_t6)

        # 5. 提取 q mod 4 并选择结果
        q_mod4, q_sign = self.extract_mod4(q)

        # 处理负数 q: 当 q < 0 时，需要计算 (4 - |q| mod 4) mod 4
        # 2's complement 公式: new_b0 = b0 (不变), new_b1 = b1 XOR b0
        b0_raw = q_mod4[..., 1:2]
        b1_raw = q_mod4[..., 0:1]

        # 计算负数情况下的 new_b1 = b1 XOR b0
        neg_b1 = self.mod4_xor(b1_raw, b0_raw)

        # 根据符号选择: q_sign=1 (负数) 选 neg_b1, q_sign=0 (正数) 选 b1_raw
        b1 = self.vec_mux(q_sign.expand_as(b1_raw), neg_b1, b1_raw)
        b0 = b0_raw  # b0 保持不变

        # sin: q mod 4 = 0,2 用 sin_r; 1,3 用 cos_r
        # cos: q mod 4 = 0,2 用 cos_r; 1,3 用 -sin_r

        # sin 结果: VecMUX(sel, a, b): sel=1 选 a, sel=0 选 b
        # b0=0: sin_r, b0=1: cos_r
        sin_base = self.vec_mux(b0.expand_as(sin_r), cos_r, sin_r)
        sin_new_sign = self.sign_xor(sin_base[..., 0:1], b1)
        sin_result = torch.cat([sin_new_sign, sin_base[..., 1:]], dim=-1)

        # cos 结果: cos(x) = sin(x + π/2)
        # 更简单的方法: 直接使用 cos(x) 的象限关系
        # cos: q mod 4 = 0: cos_r, 1: -sin_r, 2: -cos_r, 3: sin_r
        # 构造 cos 的选择逻辑
        # b0=0: cos_r, b0=1: sin_r
        cos_base = self.vec_mux(b0.expand_as(cos_r), sin_r, cos_r)
        # cos 的符号: b1 XOR b0 决定是否取负
        cos_sign_flip = self.sign_xor(b0, b1)
        cos_new_sign = self.sign_xor(cos_base[..., 0:1], cos_sign_flip)
        cos_result = torch.cat([cos_new_sign, cos_base[..., 1:]], dim=-1)

        return sin_result, cos_result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)
