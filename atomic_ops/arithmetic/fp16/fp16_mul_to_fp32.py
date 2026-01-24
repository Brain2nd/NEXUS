"""
FP16 组件 - 100%纯SNN门电路实现
================================

包含:
- FP16ToFP32Converter: FP16 → FP32 转换器
- SpikeFP16MulToFP32: FP16 × FP16 → FP32 乘法器

FP16: [S | E4..E0 | M9..M0], bias=15
FP32: [S | E7..E0 | M22..M0], bias=127

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
from atomic_ops.core.logic_gates import (MUXGate)
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier


class FP16ToFP32Converter(nn.Module):
    """FP16 -> FP32 转换器（100%纯SNN门电路）

    FP16: [S | E4..E0 | M9..M0], bias=15
    FP32: [S | E7..E0 | M22..M0], bias=127

    转换规则 (Normal):
    - sign: 直接复制
    - exp: FP32_exp = FP16_exp + 112 (bias差 = 127 - 15 = 112)
    - mant: 10位扩展为23位（低位补0）

    特殊情况:
    - Zero (E=0, M=0): → Zero FP32
    - Subnormal (E=0, M≠0): 归一化后转换为 FP32 normal
    - Inf (E=31, M=0): → Inf FP32 (E=255)
    - NaN (E=31, M≠0): → NaN FP32 (E=255, M保留)

    Subnormal 转换:
    FP16 subnormal: 0.M × 2^(-14)
    找到尾数中第一个1的位置k，归一化后:
    1.xxx × 2^(-14-k-1) → FP32 exp = 127 - 15 - k = 112 - k
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 检测 FP16 E=0 - 单实例
        self.e_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_is_zero_not = VecNOT(neuron_template=nt, max_param_shape=(1,))

        # 检测 E=31 (Inf/NaN) - 单实例
        self.e_all_ones_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 检测 M≠0 - 单实例
        self.m_or = VecOR(neuron_template=nt, max_param_shape=(1,))

        # 检测 subnormal (E=0 AND M≠0)
        self.is_subnorm_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 检测 zero (E=0 AND M=0)
        self.not_m_nonzero = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.is_zero_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 检测 Inf (E=31 AND M=0)
        self.is_inf_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 检测 NaN (E=31 AND M≠0)
        self.is_nan_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 8位加法器: FP16_exp (5位扩展) + 112
        self.exp_adder = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))

        # 最终选择 MUX - 单实例
        self.final_exp_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.final_mant_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # 零/Inf/NaN 处理 MUX - 单实例
        self.zero_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.inf_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.nan_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== Subnormal 处理门电路 =====
        # 前导1检测 (优先级编码器): 找到 m[9..0] 中第一个1的位置
        # is_first[k] = m[k] AND NOT(m[9]) AND NOT(m[8]) AND ... AND NOT(m[k+1])
        self.subnorm_not = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.subnorm_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        # 用于选择 subnormal 指数和尾数
        self.subnorm_exp_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.subnorm_mant_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

    def forward(self, fp16_pulse):
        """
        Args:
            fp16_pulse: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            fp32_pulse: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        device = fp16_pulse.device
        batch_shape = fp16_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        # 提取 FP16 各部分
        s = fp16_pulse[..., 0:1]
        e4 = fp16_pulse[..., 1:2]
        e3 = fp16_pulse[..., 2:3]
        e2 = fp16_pulse[..., 3:4]
        e1 = fp16_pulse[..., 4:5]
        e0 = fp16_pulse[..., 5:6]
        m = fp16_pulse[..., 6:16]  # [M9..M0]

        # 检测 E=0 (tree reduction)
        e_or_01 = self.e_or(e0, e1)
        e_or_23 = self.e_or(e2, e3)
        e_or_0123 = self.e_or(e_or_01, e_or_23)
        e_nonzero = self.e_or(e_or_0123, e4)
        e_is_zero = self.e_is_zero_not(e_nonzero)

        # 检测 E=31 (11111) (tree reduction)
        e_and_01 = self.e_all_ones_and(e0, e1)
        e_and_23 = self.e_all_ones_and(e2, e3)
        e_and_0123 = self.e_all_ones_and(e_and_01, e_and_23)
        e_is_max = self.e_all_ones_and(e_and_0123, e4)

        # 检测 M≠0 (OR tree for 10 bits)
        m_or_0 = self.m_or(m[..., 0:1], m[..., 1:2])
        m_or_1 = self.m_or(m[..., 2:3], m[..., 3:4])
        m_or_2 = self.m_or(m[..., 4:5], m[..., 5:6])
        m_or_3 = self.m_or(m[..., 6:7], m[..., 7:8])
        m_or_4 = self.m_or(m[..., 8:9], m[..., 9:10])
        m_or_01 = self.m_or(m_or_0, m_or_1)
        m_or_23 = self.m_or(m_or_2, m_or_3)
        m_or_0123 = self.m_or(m_or_01, m_or_23)
        m_nonzero = self.m_or(m_or_0123, m_or_4)

        # 特殊情况检测
        not_m_nz = self.not_m_nonzero(m_nonzero)
        is_zero = self.is_zero_and(e_is_zero, not_m_nz)
        is_subnormal = self.is_subnorm_and(e_is_zero, m_nonzero)
        is_inf = self.is_inf_and(e_is_max, not_m_nz)
        is_nan = self.is_nan_and(e_is_max, m_nonzero)

        # ===== Normal 路径 =====
        # FP16 exp 扩展到 8 位 (LSB first)
        fp16_exp_8bit_lsb = torch.cat([e0, e1, e2, e3, e4, zeros, zeros, zeros], dim=-1)

        # +112 = 0b01110000, LSB first: [0, 0, 0, 0, 1, 1, 1, 0]
        const_112_lsb = torch.cat([zeros, zeros, zeros, zeros, ones, ones, ones, zeros], dim=-1)

        # 加法 (LSB first)
        fp32_exp_raw_lsb, _ = self.exp_adder(fp16_exp_8bit_lsb, const_112_lsb)

        # 转回 MSB first
        fp32_exp_normal = torch.cat([
            fp32_exp_raw_lsb[..., 7:8],
            fp32_exp_raw_lsb[..., 6:7],
            fp32_exp_raw_lsb[..., 5:6],
            fp32_exp_raw_lsb[..., 4:5],
            fp32_exp_raw_lsb[..., 3:4],
            fp32_exp_raw_lsb[..., 2:3],
            fp32_exp_raw_lsb[..., 1:2],
            fp32_exp_raw_lsb[..., 0:1],
        ], dim=-1)

        # 尾数扩展: 10位 → 23位 (低位补0)
        fp32_mant_normal = torch.cat([m] + [zeros] * 13, dim=-1)

        # ===== Subnormal 路径 =====
        # FP16 subnormal: E=0, M≠0, 值 = 0.M × 2^(-14)
        # 需要归一化: 找到第一个1的位置k，得到 1.xxx × 2^(-15-k)
        # FP32 exp = 127 + (-15-k) = 112 - k

        # 提取尾数各位
        m9, m8, m7, m6, m5 = m[..., 0:1], m[..., 1:2], m[..., 2:3], m[..., 3:4], m[..., 4:5]
        m4, m3, m2, m1, m0 = m[..., 5:6], m[..., 6:7], m[..., 7:8], m[..., 8:9], m[..., 9:10]

        # 优先级编码: 检测第一个1的位置
        # is_first[k] 表示 m[k] 是第一个1
        not_m9 = self.subnorm_not(m9)
        not_m8 = self.subnorm_not(m8)
        not_m7 = self.subnorm_not(m7)
        not_m6 = self.subnorm_not(m6)
        not_m5 = self.subnorm_not(m5)
        not_m4 = self.subnorm_not(m4)
        not_m3 = self.subnorm_not(m3)
        not_m2 = self.subnorm_not(m2)
        not_m1 = self.subnorm_not(m1)

        # is_first[9] = m9
        is_first_9 = m9

        # is_first[8] = m8 AND NOT(m9)
        is_first_8 = self.subnorm_and(m8, not_m9)

        # is_first[7] = m7 AND NOT(m9) AND NOT(m8)
        t87 = self.subnorm_and(not_m9, not_m8)
        is_first_7 = self.subnorm_and(m7, t87)

        # is_first[6] = m6 AND NOT(m9..m7)
        t876 = self.subnorm_and(t87, not_m7)
        is_first_6 = self.subnorm_and(m6, t876)

        # is_first[5] = m5 AND NOT(m9..m6)
        t8765 = self.subnorm_and(t876, not_m6)
        is_first_5 = self.subnorm_and(m5, t8765)

        # is_first[4] = m4 AND NOT(m9..m5)
        t87654 = self.subnorm_and(t8765, not_m5)
        is_first_4 = self.subnorm_and(m4, t87654)

        # is_first[3] = m3 AND NOT(m9..m4)
        t876543 = self.subnorm_and(t87654, not_m4)
        is_first_3 = self.subnorm_and(m3, t876543)

        # is_first[2] = m2 AND NOT(m9..m3)
        t8765432 = self.subnorm_and(t876543, not_m3)
        is_first_2 = self.subnorm_and(m2, t8765432)

        # is_first[1] = m1 AND NOT(m9..m2)
        t87654321 = self.subnorm_and(t8765432, not_m2)
        is_first_1 = self.subnorm_and(m1, t87654321)

        # is_first[0] = m0 AND NOT(m9..m1)
        t876543210 = self.subnorm_and(t87654321, not_m1)
        is_first_0 = self.subnorm_and(m0, t876543210)

        # FP32 指数值 (MSB first, 8位):
        # k=0 (m9=1): exp = 112 = 0b01110000
        # k=1 (m8=1): exp = 111 = 0b01101111
        # k=2 (m7=1): exp = 110 = 0b01101110
        # ...
        # k=9 (m0=1): exp = 103 = 0b01100111

        # 预计算各指数值 (MSB first)
        exp_112 = torch.cat([zeros, ones, ones, ones, zeros, zeros, zeros, zeros], dim=-1)  # 112
        exp_111 = torch.cat([zeros, ones, ones, zeros, ones, ones, ones, ones], dim=-1)  # 111
        exp_110 = torch.cat([zeros, ones, ones, zeros, ones, ones, ones, zeros], dim=-1)  # 110
        exp_109 = torch.cat([zeros, ones, ones, zeros, ones, ones, zeros, ones], dim=-1)  # 109
        exp_108 = torch.cat([zeros, ones, ones, zeros, ones, ones, zeros, zeros], dim=-1)  # 108
        exp_107 = torch.cat([zeros, ones, ones, zeros, ones, zeros, ones, ones], dim=-1)  # 107
        exp_106 = torch.cat([zeros, ones, ones, zeros, ones, zeros, ones, zeros], dim=-1)  # 106
        exp_105 = torch.cat([zeros, ones, ones, zeros, ones, zeros, zeros, ones], dim=-1)  # 105
        exp_104 = torch.cat([zeros, ones, ones, zeros, ones, zeros, zeros, zeros], dim=-1)  # 104
        exp_103 = torch.cat([zeros, ones, ones, zeros, zeros, ones, ones, ones], dim=-1)  # 103

        # 尾数左移后的值 (23位, 去掉隐含1后的部分)
        # k=0: 原尾数 m8..m0 补13个0 → [m8,m7,m6,m5,m4,m3,m2,m1,m0, 0x13]
        # k=1: 左移1位 m7..m0 补14个0 → [m7,m6,m5,m4,m3,m2,m1,m0, 0x14]
        # ...
        mant_k0 = torch.cat([m8, m7, m6, m5, m4, m3, m2, m1, m0] + [zeros] * 14, dim=-1)
        mant_k1 = torch.cat([m7, m6, m5, m4, m3, m2, m1, m0] + [zeros] * 15, dim=-1)
        mant_k2 = torch.cat([m6, m5, m4, m3, m2, m1, m0] + [zeros] * 16, dim=-1)
        mant_k3 = torch.cat([m5, m4, m3, m2, m1, m0] + [zeros] * 17, dim=-1)
        mant_k4 = torch.cat([m4, m3, m2, m1, m0] + [zeros] * 18, dim=-1)
        mant_k5 = torch.cat([m3, m2, m1, m0] + [zeros] * 19, dim=-1)
        mant_k6 = torch.cat([m2, m1, m0] + [zeros] * 20, dim=-1)
        mant_k7 = torch.cat([m1, m0] + [zeros] * 21, dim=-1)
        mant_k8 = torch.cat([m0] + [zeros] * 22, dim=-1)
        mant_k9 = torch.cat([zeros] * 23, dim=-1)

        # 使用 MUX 树选择正确的指数和尾数
        # is_first_9 (m9=1, k=0) → exp=112, mant=m8..m0,0x14
        # is_first_8 (m8=1, k=1) → exp=111, mant=m7..m0,0x15
        # ...
        # is_first_0 (m0=1, k=9) → exp=103, mant=0x23

        # 从低优先级开始，高优先级会覆盖
        subnorm_exp = exp_103  # 默认 is_first_0 (m0=1)
        subnorm_mant = mant_k9

        # is_first_1 (m1=1, k=8) → exp=104
        is_first_1_exp = is_first_1.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_1_exp, exp_104, subnorm_exp)
        is_first_1_mant = is_first_1.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_1_mant, mant_k8, subnorm_mant)

        # is_first_2 (m2=1, k=7) → exp=105
        is_first_2_exp = is_first_2.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_2_exp, exp_105, subnorm_exp)
        is_first_2_mant = is_first_2.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_2_mant, mant_k7, subnorm_mant)

        # is_first_3 (m3=1, k=6) → exp=106
        is_first_3_exp = is_first_3.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_3_exp, exp_106, subnorm_exp)
        is_first_3_mant = is_first_3.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_3_mant, mant_k6, subnorm_mant)

        # is_first_4 (m4=1, k=5) → exp=107
        is_first_4_exp = is_first_4.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_4_exp, exp_107, subnorm_exp)
        is_first_4_mant = is_first_4.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_4_mant, mant_k5, subnorm_mant)

        # is_first_5 (m5=1, k=4) → exp=108
        is_first_5_exp = is_first_5.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_5_exp, exp_108, subnorm_exp)
        is_first_5_mant = is_first_5.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_5_mant, mant_k4, subnorm_mant)

        # is_first_6 (m6=1, k=3) → exp=109
        is_first_6_exp = is_first_6.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_6_exp, exp_109, subnorm_exp)
        is_first_6_mant = is_first_6.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_6_mant, mant_k3, subnorm_mant)

        # is_first_7 (m7=1, k=2) → exp=110
        is_first_7_exp = is_first_7.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_7_exp, exp_110, subnorm_exp)
        is_first_7_mant = is_first_7.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_7_mant, mant_k2, subnorm_mant)

        # is_first_8 (m8=1, k=1) → exp=111
        is_first_8_exp = is_first_8.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_8_exp, exp_111, subnorm_exp)
        is_first_8_mant = is_first_8.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_8_mant, mant_k1, subnorm_mant)

        # is_first_9 (m9=1, k=0) → exp=112 (最高优先级)
        is_first_9_exp = is_first_9.expand_as(subnorm_exp)
        subnorm_exp = self.subnorm_exp_mux(is_first_9_exp, exp_112, subnorm_exp)
        is_first_9_mant = is_first_9.expand_as(subnorm_mant)
        subnorm_mant = self.subnorm_mant_mux(is_first_9_mant, mant_k0, subnorm_mant)

        # 选择 normal vs subnormal (vectorized)
        is_subnormal_8 = is_subnormal.expand_as(subnorm_exp)
        exp_sel = self.final_exp_mux(is_subnormal_8, subnorm_exp, fp32_exp_normal)

        is_subnormal_23 = is_subnormal.expand_as(subnorm_mant)
        mant_sel = self.final_mant_mux(is_subnormal_23, subnorm_mant, fp32_mant_normal)

        # ===== 零处理 ===== (vectorized)
        zero_result = torch.cat([zeros] * 31, dim=-1)
        em_sel = torch.cat([exp_sel, mant_sel], dim=-1)
        is_zero_31 = is_zero.expand_as(em_sel)
        after_zero = self.zero_mux(is_zero_31, zero_result, em_sel)

        # ===== Inf 处理 ===== (vectorized)
        inf_exp = torch.cat([ones] * 8, dim=-1)
        inf_mant = torch.cat([zeros] * 23, dim=-1)
        inf_result = torch.cat([inf_exp, inf_mant], dim=-1)
        is_inf_31 = is_inf.expand_as(after_zero)
        after_inf = self.inf_mux(is_inf_31, inf_result, after_zero)

        # ===== NaN 处理 ===== (vectorized)
        nan_exp = torch.cat([ones] * 8, dim=-1)
        nan_mant = torch.cat([m] + [zeros] * 13, dim=-1)  # 保留原尾数
        nan_result = torch.cat([nan_exp, nan_mant], dim=-1)
        is_nan_31 = is_nan.expand_as(after_inf)
        after_nan = self.nan_mux(is_nan_31, nan_result, after_inf)

        # 组装 FP32
        fp32_pulse = torch.cat([s, after_nan], dim=-1)

        return fp32_pulse

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


class SpikeFP16MulToFP32(nn.Module):
    """FP16 × FP16 → FP32 乘法器（纯SNN门电路）

    通过 FP16→FP32 转换 + FP32×FP32 乘法实现。
    输出完整精度的 FP32 结果。

    FP16: [S | E4..E0 | M9..M0], bias=15
    FP32: [S | E7..E0 | M22..M0], bias=127

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # FP16 → FP32 转换器
        self.conv_a = FP16ToFP32Converter(neuron_template=nt)
        self.conv_b = FP16ToFP32Converter(neuron_template=nt)

        # FP32 乘法器
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

    def forward(self, A, B):
        """
        Args:
            A, B: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        # 支持广播
        A, B = torch.broadcast_tensors(A, B)

        # FP16 → FP32
        A_fp32 = self.conv_a(A)
        B_fp32 = self.conv_b(B)

        # FP32 × FP32 → FP32
        result = self.mul(A_fp32, B_fp32)

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)
