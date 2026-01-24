"""
FP8 乘法器（输出 FP32）- 100%纯SNN门电路实现
用于对齐 PyTorch nn.Linear（FP32 精度乘法）

计算: FP8 × FP8 → FP32（保持完整精度，不舍入到 FP8）

FP8 E4M3: [S | E3..E0 | M2 M1 M0], bias=7
FP32:     [S | E7..E0 | M22..M0], bias=127

乘法结果：
- 符号: S_out = S_a XOR S_b
- 指数: E_out = E_a_eff + E_b_eff - 7 + 127 = E_a_eff + E_b_eff + 113 (其中 E_eff 是有效指数)
- 尾数: M_out = (1.M_a_eff) × (1.M_b_eff) = 8位结果 → 扩展到23位

对于 subnormal (E=0, M≠0):
- 值 = M/8 × 2^(-6)
- 需要归一化得到 1.xxx × 2^(exp)
- M=001: 1.000 × 2^(-9), FP32_E_eff = 118
- M=010: 1.000 × 2^(-8), FP32_E_eff = 119
- M=011: 1.500 × 2^(-8), FP32_E_eff = 119
- M=100: 1.000 × 2^(-7), FP32_E_eff = 120
- M=101: 1.250 × 2^(-7), FP32_E_eff = 120
- M=110: 1.500 × 2^(-7), FP32_E_eff = 120
- M=111: 1.750 × 2^(-7), FP32_E_eff = 120
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
# 使用向量化基础门以保持与 FP16/FP32 组件的一致性
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecHalfAdder as HalfAdder,
    VecFullAdder as FullAdder,
    VecORTree as ORTree,
    VecAdder
)
# 保留专用组件
from atomic_ops.core.logic_gates import ArrayMultiplier4x4_Strict
from atomic_ops.arithmetic.fp32.fp32_components import SubtractorNBit


class SpikeFP8MulToFP32(nn.Module):
    """FP8 × FP8 → FP32 乘法器（纯SNN门电路）
    
    输出完整精度的 FP32 结果，不进行中间舍入。
    完整支持 normal 和 subnormal 输入。
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # 符号
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))
        
        # 指数检测 (E=0 检测) - 单实例
        self.e_a_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_b_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_a_nonzero_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_b_nonzero_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # M≠0 检测 (纯 SNN OR 门链)
        self.m_a_or_01 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.m_a_or_all = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.m_b_or_01 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.m_b_or_all = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # 隐藏位 OR (替换 clamp)
        self.hidden_a_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.hidden_b_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # 零检测 OR (替换 clamp)
        self.is_zero_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # 4x4 尾数乘法器（输出8位）
        self.mant_mul = ArrayMultiplier4x4_Strict(neuron_template=nt, max_param_shape=(1,))
        
        # 8位指数加法 - 每次使用独立实例（避免复用）
        # 用于 subnormal 路径的 E+120 计算
        self.exp_add_normal_a = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))
        self.exp_add_normal_b = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))
        # 用于 subnormal 路径的 E_eff 求和
        self.exp_sum_subnorm = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))
        # 用于主计算路径
        self.exp_add_ab = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))
        self.exp_add_bias = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))

        # 指数调整（归一化时+1）
        self.exp_norm_inc = VecAdder(bits=8, neuron_template=nt, max_param_shape=(8,))
        
        # 纯 SNN 指数减法器 (替换 Python 算术)
        self.exp_adj_sub = SubtractorNBit(bits=8, neuron_template=nt)
        
        # 调整量编码门电路
        # adj 编码: adj_a ∈ {0, 1, 2} 对应 lead_at_2, lead_at_1, lead_at_0
        # adj_b 同理，total = adj_a + adj_b ∈ {0, 1, 2, 3, 4}
        # 需要用半加器/全加器来计算 adj_a + adj_b
        self.adj_ha0 = HalfAdder(neuron_template=nt, max_param_shape=(1,))  # 低位加法
        self.adj_ha1 = HalfAdder(neuron_template=nt, max_param_shape=(1,))  # 高位加法
        self.adj_or = VecOR(neuron_template=nt, max_param_shape=(1,))      # 进位处理
        
        # 零检测 - 单实例
        self.zero_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.zero_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 纯SNN NOT门 (替换 ones - x) =====
        self.not_e_a_nonzero = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_e_b_nonzero = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m2_a = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m1_a = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m2_b = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m1_b = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_is_a_subnormal = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_is_b_subnormal = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m_a_or = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_m_b_or = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_needs_norm = VecNOT(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 纯SNN XOR门 (替换 a+b-2*a*b) =====
        self.adj_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))
        
        # ===== 纯SNN AND门 (替换 a*b) =====
        self.and_a_subnormal = VecAND(neuron_template=nt, max_param_shape=(1,))  # not_e_a_nonzero AND m_a_or
        self.and_b_subnormal = VecAND(neuron_template=nt, max_param_shape=(1,))  # not_e_b_nonzero AND m_b_or
        self.and_a_true_zero = VecAND(neuron_template=nt, max_param_shape=(1,))  # not_e_a_nonzero AND not_m_a_or
        self.and_b_true_zero = VecAND(neuron_template=nt, max_param_shape=(1,))  # not_e_b_nonzero AND not_m_b_or
        self.and_adj_a_bit0 = VecAND(neuron_template=nt, max_param_shape=(1,))   # is_a_subnormal AND a_lead_at_1
        self.and_adj_a_bit1 = VecAND(neuron_template=nt, max_param_shape=(1,))   # is_a_subnormal AND a_lead_at_0
        self.and_adj_b_bit0 = VecAND(neuron_template=nt, max_param_shape=(1,))   # is_b_subnormal AND b_lead_at_1
        self.and_adj_b_bit1 = VecAND(neuron_template=nt, max_param_shape=(1,))   # is_b_subnormal AND b_lead_at_0
        
        # ===== 纯SNN MUX门 (替换 sel*a + not_sel*b) ===== - 单实例
        self.mux_eff_m_a = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 3位尾数
        self.mux_eff_m_b = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_e_eff_a = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_e_eff_b = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_final_exp = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_final_mant = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # A 端前导位检测门电路
        self.and_a_lead_1 = VecAND(neuron_template=nt, max_param_shape=(1,))    # not_m2_a AND m1_a
        self.and_a_lead_0_1 = VecAND(neuron_template=nt, max_param_shape=(1,))  # not_m2_a AND not_m1_a
        self.and_a_lead_0_2 = VecAND(neuron_template=nt, max_param_shape=(1,))  # (not_m2_a AND not_m1_a) AND m0_a
        
        # B 端前导位检测门电路
        self.and_b_lead_1 = VecAND(neuron_template=nt, max_param_shape=(1,))    # not_m2_b AND m1_b
        self.and_b_lead_0_1 = VecAND(neuron_template=nt, max_param_shape=(1,))  # not_m2_b AND not_m1_b
        self.and_b_lead_0_2 = VecAND(neuron_template=nt, max_param_shape=(1,))  # (not_m2_b AND not_m1_b) AND m0_b
        
        # A 端 subnormal 尾数/指数 MUX - 单实例
        self.mux_norm_m_a = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_e_sub_a = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # B 端 subnormal 尾数/指数 MUX - 单实例
        self.mux_norm_m_b = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_e_sub_b = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
        # 进位计算 AND 门
        self.and_carry1b = VecAND(neuron_template=nt, max_param_shape=(1,))  # temp1 AND carry0
        
    def forward(self, A, B):
        """
        Args:
            A, B: [..., 8] FP8 脉冲 [S, E3..E0, M2 M1 M0]
        Returns:
            [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        # 支持广播
        A, B = torch.broadcast_tensors(A, B)
        
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 提取各部分
        s_a = A[..., 0:1]
        e_a = A[..., 1:5]  # [E3..E0] MSB first
        m_a = A[..., 5:8]  # [M2 M1 M0]
        
        s_b = B[..., 0:1]
        e_b = B[..., 1:5]
        m_b = B[..., 5:8]
        
        # 符号
        s_out = self.sign_xor(s_a, s_b)
        
        # E=0 检测 (tree reduction)
        e_a_or_01 = self.e_a_or(e_a[..., 2:3], e_a[..., 3:4])
        e_a_or_23 = self.e_a_or(e_a[..., 0:1], e_a[..., 1:2])
        e_a_nonzero = self.e_a_nonzero_or(e_a_or_01, e_a_or_23)

        e_b_or_01 = self.e_b_or(e_b[..., 2:3], e_b[..., 3:4])
        e_b_or_23 = self.e_b_or(e_b[..., 0:1], e_b[..., 1:2])
        e_b_nonzero = self.e_b_nonzero_or(e_b_or_01, e_b_or_23)
        
        # M≠0 检测 (纯 SNN OR 门链)
        m_a_or_01 = self.m_a_or_01(m_a[..., 1:2], m_a[..., 2:3])
        m_a_or = self.m_a_or_all(m_a[..., 0:1], m_a_or_01)
        m_b_or_01 = self.m_b_or_01(m_b[..., 1:2], m_b[..., 2:3])
        m_b_or = self.m_b_or_all(m_b[..., 0:1], m_b_or_01)
        
        # Subnormal 检测: E=0 AND M≠0 (纯SNN AND门)
        not_e_a_nonzero = self.not_e_a_nonzero(e_a_nonzero)
        not_e_b_nonzero = self.not_e_b_nonzero(e_b_nonzero)
        is_a_subnormal = self.and_a_subnormal(not_e_a_nonzero, m_a_or)
        is_b_subnormal = self.and_b_subnormal(not_e_b_nonzero, m_b_or)
        
        # 提取尾数各位
        m2_a, m1_a, m0_a = m_a[..., 0:1], m_a[..., 1:2], m_a[..., 2:3]
        m2_b, m1_b, m0_b = m_b[..., 0:1], m_b[..., 1:2], m_b[..., 2:3]
        
        # ============================================================
        # Subnormal 归一化
        # ============================================================
        # 对于 subnormal，根据 leading one 位置计算归一化尾数和有效指数
        
        # A 的 leading one 检测 (纯SNN门电路)
        not_m2_a = self.not_m2_a(m2_a)
        not_m1_a = self.not_m1_a(m1_a)
        a_lead_at_2 = m2_a  # M >= 4
        a_lead_at_1 = self.and_a_lead_1(not_m2_a, m1_a)  # M in {2, 3}
        a_lead_0_tmp = self.and_a_lead_0_1(not_m2_a, not_m1_a)
        a_lead_at_0 = self.and_a_lead_0_2(a_lead_0_tmp, m0_a)  # M = 1
        
        # 归一化尾数 (去掉 leading one 后的部分，扩展到 3 位) (纯SNN MUX)
        # lead_at_2: [M1, M0, 0]
        # lead_at_1: [M0, 0, 0]
        # lead_at_0: [0, 0, 0]
        mant_at_2 = torch.cat([m1_a, m0_a, zeros], dim=-1)
        mant_at_1 = torch.cat([m0_a, zeros, zeros], dim=-1)
        mant_at_0 = torch.cat([zeros, zeros, zeros], dim=-1)
        # 先选 at_0 vs at_1，再选结果 vs at_2 (vectorized)
        a_lead_at_0_3 = a_lead_at_0.expand_as(mant_at_0)
        sel_01_a = self.mux_norm_m_a(a_lead_at_0_3, mant_at_0, mant_at_1)
        a_lead_at_2_3 = a_lead_at_2.expand_as(mant_at_2)
        norm_m_a = self.mux_norm_m_a(a_lead_at_2_3, mant_at_2, sel_01_a)
        
        # 有效 FP32 指数 (对于 subnormal)
        # lead_at_2 (M=4-7): 值 = 1.xxx × 2^(-7), FP32_E = 120
        # lead_at_1 (M=2-3): 值 = 1.x × 2^(-8), FP32_E = 119
        # lead_at_0 (M=1): 值 = 1 × 2^(-9), FP32_E = 118
        # 用 8 位表示 (MSB first)
        e_sub_a_120 = torch.cat([zeros, ones, ones, ones, ones, zeros, zeros, zeros], dim=-1)  # 120
        e_sub_a_119 = torch.cat([zeros, ones, ones, ones, zeros, ones, ones, ones], dim=-1)  # 119
        e_sub_a_118 = torch.cat([zeros, ones, ones, ones, zeros, ones, ones, zeros], dim=-1)  # 118
        
        # 使用 MUX 门选择指数 (vectorized)
        a_lead_at_0_8 = a_lead_at_0.expand_as(e_sub_a_118)
        sel_01_e_a = self.mux_e_sub_a(a_lead_at_0_8, e_sub_a_118, e_sub_a_119)
        a_lead_at_2_8 = a_lead_at_2.expand_as(e_sub_a_120)
        e_sub_a = self.mux_e_sub_a(a_lead_at_2_8, e_sub_a_120, sel_01_e_a)
        
        # B 同理 (纯SNN门电路)
        not_m2_b = self.not_m2_b(m2_b)
        not_m1_b = self.not_m1_b(m1_b)
        b_lead_at_2 = m2_b
        b_lead_at_1 = self.and_b_lead_1(not_m2_b, m1_b)
        b_lead_0_tmp = self.and_b_lead_0_1(not_m2_b, not_m1_b)
        b_lead_at_0 = self.and_b_lead_0_2(b_lead_0_tmp, m0_b)
        
        # B 端归一化尾数 (vectorized)
        mant_b_at_2 = torch.cat([m1_b, m0_b, zeros], dim=-1)
        mant_b_at_1 = torch.cat([m0_b, zeros, zeros], dim=-1)
        mant_b_at_0 = torch.cat([zeros, zeros, zeros], dim=-1)
        b_lead_at_0_3 = b_lead_at_0.expand_as(mant_b_at_0)
        sel_01_b = self.mux_norm_m_b(b_lead_at_0_3, mant_b_at_0, mant_b_at_1)
        b_lead_at_2_3 = b_lead_at_2.expand_as(mant_b_at_2)
        norm_m_b = self.mux_norm_m_b(b_lead_at_2_3, mant_b_at_2, sel_01_b)

        # B 端 subnormal 指数 (vectorized)
        b_lead_at_0_8 = b_lead_at_0.expand_as(e_sub_a_118)
        sel_01_e_b = self.mux_e_sub_b(b_lead_at_0_8, e_sub_a_118, e_sub_a_119)
        b_lead_at_2_8 = b_lead_at_2.expand_as(e_sub_a_120)
        e_sub_b = self.mux_e_sub_b(b_lead_at_2_8, e_sub_a_120, sel_01_e_b)
        
        # ============================================================
        # 构建有效尾数和指数
        # ============================================================
        # 有效尾数 (3位) (vectorized)
        not_is_a_subnormal = self.not_is_a_subnormal(is_a_subnormal)
        not_is_b_subnormal = self.not_is_b_subnormal(is_b_subnormal)
        is_a_subnormal_3 = is_a_subnormal.expand_as(norm_m_a)
        eff_m_a = self.mux_eff_m_a(is_a_subnormal_3, norm_m_a, m_a)
        is_b_subnormal_3 = is_b_subnormal.expand_as(norm_m_b)
        eff_m_b = self.mux_eff_m_b(is_b_subnormal_3, norm_m_b, m_b)
        
        # 隐藏位 (normal 或归一化后的 subnormal 都是 1)
        # 使用 OR 门: hidden = e_nonzero OR is_subnormal
        hidden_a = self.hidden_a_or(e_a_nonzero, is_a_subnormal)
        hidden_b = self.hidden_b_or(e_b_nonzero, is_b_subnormal)
        
        # 4位尾数（包含隐藏位）: [H, M2, M1, M0] MSB first → LSB first
        m_a_4bit_msb = torch.cat([hidden_a, eff_m_a], dim=-1)
        m_b_4bit_msb = torch.cat([hidden_b, eff_m_b], dim=-1)
        m_a_4bit = m_a_4bit_msb.flip(-1)
        m_b_4bit = m_b_4bit_msb.flip(-1)
        
        # 4x4 乘法 → 8位结果 (LSB first)
        product_lsb = self.mant_mul(m_a_4bit, m_b_4bit)
        product = product_lsb.flip(-1)  # 转回 MSB first
        
        # ============================================================
        # 指数计算
        # ============================================================
        # 对于 normal: FP32_E = E + 120 (因为 E_fp32 = E_fp8 - 7 + 127 = E + 120)
        # 对于 subnormal: FP32_E = e_sub_a/b (直接使用归一化后的值)
        
        # Normal 的 FP32 有效指数
        e_a_8bit_lsb = torch.cat([e_a[..., 3:4], e_a[..., 2:3], e_a[..., 1:2], e_a[..., 0:1], 
                                   zeros, zeros, zeros, zeros], dim=-1)
        const_120_lsb = torch.cat([zeros, zeros, zeros, ones, ones, ones, ones, zeros], dim=-1)
        e_normal_a_lsb, _ = self.exp_add_normal_a(e_a_8bit_lsb, const_120_lsb)
        e_normal_a = e_normal_a_lsb.flip(-1)
        
        e_b_8bit_lsb = torch.cat([e_b[..., 3:4], e_b[..., 2:3], e_b[..., 1:2], e_b[..., 0:1],
                                   zeros, zeros, zeros, zeros], dim=-1)
        e_normal_b_lsb, _ = self.exp_add_normal_b(e_b_8bit_lsb, const_120_lsb)
        e_normal_b = e_normal_b_lsb.flip(-1)
        
        # 选择有效 FP32 指数 (vectorized)
        is_a_subnormal_8 = is_a_subnormal.expand_as(e_sub_a)
        e_eff_a = self.mux_e_eff_a(is_a_subnormal_8, e_sub_a, e_normal_a)
        is_b_subnormal_8 = is_b_subnormal.expand_as(e_sub_b)
        e_eff_b = self.mux_e_eff_b(is_b_subnormal_8, e_sub_b, e_normal_b)
        
        # 乘积指数: E_out = E_a_eff + E_b_eff - 127 (因为两个都已经 biased 了)
        # 实际上: E_out_unbiased = E_a_eff_unbiased + E_b_eff_unbiased
        # E_out_biased = E_a_eff + E_b_eff - 127
        
        # 使用加法器计算 E_a_eff + E_b_eff
        e_eff_a_lsb = e_eff_a.flip(-1)
        e_eff_b_lsb = e_eff_b.flip(-1)
        exp_sum_raw_lsb, _ = self.exp_sum_subnorm(e_eff_a_lsb, e_eff_b_lsb)
        
        # - 127 = + (-127) = + 129 (8位二进制补码, 但我们用 9 位所以用加法)
        # 更简单: 使用减法
        # E_a_eff + E_b_eff - 127
        # = (E_a_eff + E_b_eff) + 129 (mod 256) - 256
        # 不对，让我直接用减法
        # 127 = 0b01111111, LSB first = [1,1,1,1,1,1,1,0]
        # 减法: A - B = A + (~B) + 1 (注: const_127_comp_lsb 未使用，已移除)
        # A + NOT(127) + 1 = A - 127
        
        # 两步减法太复杂，让我用不同方法
        # E_out = E_a_eff + E_b_eff - 127
        #       = E_a_eff + (E_b_eff - 127)
        # E_b_eff - 127: 如果 E_b_eff >= 127 (大多数情况)，结果是正的小数
        
        # 更简单的方法：直接计算
        # 对于 normal × normal: E_out = (E_a + 120) + (E_b + 120) - 127 = E_a + E_b + 113
        # 这就是原来的公式！
        
        # 对于 normal × subnormal:
        # E_out = (E_a + 120) + e_sub_b - 127
        #       = E_a + 120 + e_sub_b - 127
        #       = E_a + e_sub_b - 7
        
        # 让我重新用原来的思路，但正确处理 subnormal
        # 基准公式: E_out = E_a_eff + E_b_eff - 127
        # 其中 E_a_eff = E_a + 120 (normal) 或 e_sub_a (subnormal)
        
        # 计算 E_a_eff + E_b_eff (可能溢出 8 位，但我们只取低 8 位)
        # 然后减 127
        
        # 简化实现：使用原始公式 E_a + E_b + 113，但对 subnormal 进行调整
        # E_out = E_a + E_b + 113 + subnormal_adjustment
        # 对于 normal a: adjustment = 0
        # 对于 subnormal a (e_sub = 118, 119, 120):
        #   正常公式会算 0 + E_b + 113 = E_b + 113
        #   实际需要 e_sub + E_b + 120 - 127 = e_sub + E_b - 7
        #   差异 = (e_sub + E_b - 7) - (0 + E_b + 113) = e_sub - 120
        #   e_sub=118: adj = -2
        #   e_sub=119: adj = -1
        #   e_sub=120: adj = 0
        
        # ============================================================
        # 调整量编码 (纯 SNN 门电路)
        # ============================================================
        # adj_a 编码为 2 位:
        #   a_lead_at_0 (M=1): adj = 2 = 10b
        #   a_lead_at_1 (M=2,3): adj = 1 = 01b
        #   a_lead_at_2 (M=4-7): adj = 0 = 00b
        # 只对 subnormal 应用
        adj_a_bit0 = self.and_adj_a_bit0(is_a_subnormal, a_lead_at_1)  # adj_a[0] = 1 when adj_a=1
        adj_a_bit1 = self.and_adj_a_bit1(is_a_subnormal, a_lead_at_0)  # adj_a[1] = 1 when adj_a=2
        
        adj_b_bit0 = self.and_adj_b_bit0(is_b_subnormal, b_lead_at_1)
        adj_b_bit1 = self.and_adj_b_bit1(is_b_subnormal, b_lead_at_0)
        
        # total_adj = adj_a + adj_b (2位 + 2位 = 最多3位)
        # 使用半加器
        # bit0: adj_a_bit0 + adj_b_bit0
        sum0, carry0 = self.adj_ha0(adj_a_bit0, adj_b_bit0)
        # bit1: adj_a_bit1 + adj_b_bit1 + carry0
        temp1, carry1a = self.adj_ha1(adj_a_bit1, adj_b_bit1)
        # 还需要加 carry0，用 OR 处理进位
        # sum1 = temp1 XOR carry0, carry1 = (temp1 AND carry0) OR carry1a
        sum1_xor = self.adj_xor(temp1, carry0)  # 纯SNN XOR
        carry1b = self.and_carry1b(temp1, carry0)  # 纯SNN AND
        # bit2: carry1a OR carry1b
        sum2 = self.adj_or(carry1a, carry1b)
        
        # total_adj 的 3 位表示 (LSB first): [sum0, sum1_xor, sum2]
        # 扩展到 8 位用于减法
        adj_8bit_lsb = torch.cat([sum0, sum1_xor, sum2, zeros, zeros, zeros, zeros, zeros], dim=-1)
        
        # ============================================================
        # 原始指数公式计算
        # ============================================================
        sum_ab_lsb, _ = self.exp_add_ab(e_a_8bit_lsb, e_b_8bit_lsb)
        const_113_lsb = torch.cat([ones, zeros, zeros, zeros, ones, ones, ones, zeros], dim=-1)
        exp_base_lsb, _ = self.exp_add_bias(sum_ab_lsb, const_113_lsb)
        
        one_8bit_lsb = torch.cat([ones] + [zeros]*7, dim=-1)
        
        # 归一化检测
        needs_norm = product[..., 0:1]
        
        # 归一化时指数+1
        exp_norm_lsb, _ = self.exp_norm_inc(exp_base_lsb, one_8bit_lsb)
        
        # 选择指数 (归一化调整)
        exp_sum = exp_base_lsb.flip(-1)  # 原始公式结果
        exp_norm = exp_norm_lsb.flip(-1)
        not_needs_norm = self.not_needs_norm(needs_norm)
        # 选择指数 (vectorized)
        needs_norm_8 = needs_norm.expand_as(exp_norm)
        final_exp_pre = self.mux_final_exp(needs_norm_8, exp_norm, exp_sum)
        
        # ============================================================
        # Subnormal 指数修正 (纯 SNN 减法器)
        # ============================================================
        # 计算 final_exp_pre - total_adj
        # 使用 8 位减法器 (LSB first)
        final_exp_pre_lsb = final_exp_pre.flip(-1)
        final_exp_lsb, _ = self.exp_adj_sub(final_exp_pre_lsb, adj_8bit_lsb)
        final_exp = final_exp_lsb.flip(-1)
        
        # ============================================================
        # 尾数处理
        # ============================================================
        # 归一化: product[7]=1 时取 P6..P0，否则取 P5..P0
        m_norm = torch.cat([product[..., 1:8], 
                           zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                           zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        
        m_no_norm = torch.cat([product[..., 2:8], 
                               zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                               zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        
        # 选择尾数 (vectorized)
        needs_norm_23 = needs_norm.expand_as(m_norm)
        final_mant = self.mux_final_mant(needs_norm_23, m_norm, m_no_norm)
        
        # ============================================================
        # 零检测 (纯 SNN)
        # ============================================================
        not_m_a_or = self.not_m_a_or(m_a_or)
        not_m_b_or = self.not_m_b_or(m_b_or)
        is_a_true_zero = self.and_a_true_zero(not_e_a_nonzero, not_m_a_or)  # 纯SNN AND
        is_b_true_zero = self.and_b_true_zero(not_e_b_nonzero, not_m_b_or)  # 纯SNN AND
        is_zero = self.is_zero_or(is_a_true_zero, is_b_true_zero)
        
        # 零时清零指数和尾数 (vectorized)
        is_zero_8 = is_zero.expand_as(final_exp)
        zero_exp = torch.zeros_like(final_exp)
        result_exp = self.zero_mux_e(is_zero_8, zero_exp, final_exp)

        is_zero_23 = is_zero.expand_as(final_mant)
        zero_mant = torch.zeros_like(final_mant)
        result_mant = self.zero_mux_m(is_zero_23, zero_mant, final_mant)
        
        # 组装 FP32
        fp32_pulse = torch.cat([s_out, result_exp, result_mant], dim=-1)
        
        return fp32_pulse
    
    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)

