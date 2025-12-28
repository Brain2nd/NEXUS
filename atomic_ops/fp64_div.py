"""
FP64 除法器 - 100%纯SNN门电路实现 (并行化版本)
=================================================

核心原则：
1. 所有逻辑操作使用IFNode神经元，禁止直接算术运算
2. 批量+位并行：使用向量化门电路，一次处理所有样本的所有位
3. 仅在必要的数据依赖处使用最小循环

FP64 格式: [S | E10..E0 | M51..M0], bias=1023

算法: 恢复余数除法
- 符号: Sr = Sa XOR Sb
- 指数: Er = Ea - Eb + 1023
- 尾数: 109次迭代产生54位商 + 舍入信息
- RNE舍入

作者: HumanBrain Project
"""
import torch
import torch.nn as nn

# 从统一的向量化门电路模块导入，避免代码重复
from .vec_logic_gates import (
    VecAND, VecOR, VecNOT, VecXOR, VecMUX,
    VecORTree, VecANDTree,
    VecAdder, VecSubtractor
)


# ==============================================================================
# 除法迭代单元
# ==============================================================================

class VecDivIteration(nn.Module):
    """向量化除法迭代 - 一次处理所有样本
    
    恢复余数算法：
    1. 尝试 R - D
    2. 如果 R >= D (无借位)，Q=1，R=R-D
    3. 如果 R < D (有借位)，Q=0，R保持不变
    """
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.sub = VecSubtractor(bits)
        self.not_borrow = VecNOT()
        self.mux = VecMUX()
        
    def forward(self, R, D):
        """
        R: [..., bits] 当前余数 (LSB first)
        D: [..., bits] 除数 (LSB first)
        返回: Q_bit [..., 1], R_next [..., bits]
        """
        # 尝试减法
        R_trial, borrow = self.sub(R, D)
        
        # Q = NOT(borrow): 无借位时Q=1
        Q_bit = self.not_borrow(borrow)
        
        # 选择余数: borrow=0时选R_trial, borrow=1时选R
        # MUX(borrow, R, R_trial): borrow=1选R, borrow=0选R_trial
        R_next = self.mux(borrow, R, R_trial)
        
        return Q_bit, R_next
    
    def reset(self):
        self.sub.reset()
        self.not_borrow.reset()
        self.mux.reset()


# ==============================================================================
# FP64 除法器主类
# ==============================================================================

class SpikeFP64Divider(nn.Module):
    """FP64 除法器 - 100%纯SNN并行化实现
    
    输入: A, B: [..., 64] FP64脉冲 [S | E10..E0 | M51..M0] MSB first
    输出: [..., 64] FP64脉冲 (A / B)
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        
        # ===== 符号 =====
        self.sign_xor = VecXOR(neuron_template=nt)
        
        # ===== 指数运算 =====
        self.exp_sub = VecSubtractor(13, neuron_template=nt)  # 13位防溢出
        self.exp_add = VecAdder(13, neuron_template=nt)
        
        # ===== 特殊值检测 =====
        self.exp_and_tree = VecANDTree(neuron_template=nt)  # 检测E全1
        self.exp_or_tree = VecORTree(neuron_template=nt)    # 检测E非零
        self.mant_or_tree = VecORTree(neuron_template=nt)   # 检测M非零
        self.not_gate = VecNOT(neuron_template=nt)
        self.and_gate = VecAND(neuron_template=nt)
        self.or_gate = VecOR(neuron_template=nt)
        
        # ===== 尾数除法 (55位用于54位尾数+1位扩展) =====
        # 109次迭代产生足够精度
        self.div_iterations = nn.ModuleList([VecDivIteration(55, neuron_template=nt) for _ in range(109)])
        
        # ===== 尾数比较 =====
        self.mant_cmp = VecSubtractor(55, neuron_template=nt)
        
        # ===== RNE舍入 =====
        self.rne_or = VecOR(neuron_template=nt)
        self.rne_and = VecAND(neuron_template=nt)
        self.round_adder = VecAdder(53, neuron_template=nt)
        
        # ===== 归一化选择 =====
        self.normalize_mux = VecMUX(neuron_template=nt)
        self.exp_mux = VecMUX(neuron_template=nt)
        
        # ===== 特殊值输出选择 =====
        self.nan_mux = VecMUX(neuron_template=nt)
        self.inf_mux = VecMUX(neuron_template=nt)
        self.zero_mux = VecMUX(neuron_template=nt)
        self.overflow_mux = VecMUX(neuron_template=nt)
        self.underflow_mux = VecMUX(neuron_template=nt)
        
    def forward(self, A, B):
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
        s_out = self.sign_xor(s_a, s_b)
        
        # ===== 3. 特殊值检测 (并行树结构) =====
        # 指数全1检测 (Inf/NaN)
        e_a_all_one = self.exp_and_tree(e_a)
        e_b_all_one = self.exp_and_tree(e_b)
        
        # 指数非零检测
        e_a_any = self.exp_or_tree(e_a)
        e_b_any = self.exp_or_tree(e_b)
        e_a_is_zero = self.not_gate(e_a_any)
        e_b_is_zero = self.not_gate(e_b_any)
        
        # 尾数非零检测
        m_a_any = self.mant_or_tree(m_a)
        m_b_any = self.mant_or_tree(m_b)
        m_a_is_zero = self.not_gate(m_a_any)
        m_b_is_zero = self.not_gate(m_b_any)
        
        # 零检测: E=0 AND M=0
        a_is_zero = self.and_gate(e_a_is_zero, m_a_is_zero)
        b_is_zero = self.and_gate(e_b_is_zero, m_b_is_zero)
        
        # Inf检测: E=全1 AND M=0
        a_is_inf = self.and_gate(e_a_all_one, m_a_is_zero)
        b_is_inf = self.and_gate(e_b_all_one, m_b_is_zero)
        
        # NaN检测: E=全1 AND M!=0
        a_is_nan = self.and_gate(e_a_all_one, m_a_any)
        b_is_nan = self.and_gate(e_b_all_one, m_b_any)
        
        # 组合特殊情况
        either_nan = self.or_gate(a_is_nan, b_is_nan)
        not_a_zero = self.not_gate(a_is_zero)
        div_by_zero = self.and_gate(not_a_zero, b_is_zero)  # x/0 (x!=0)
        zero_div_zero = self.and_gate(a_is_zero, b_is_zero)  # 0/0
        inf_div_inf = self.and_gate(a_is_inf, b_is_inf)      # Inf/Inf
        
        result_is_nan = self.or_gate(either_nan, self.or_gate(zero_div_zero, inf_div_inf))
        not_b_inf = self.not_gate(b_is_inf)
        inf_div_y = self.and_gate(a_is_inf, not_b_inf)
        result_is_inf = self.or_gate(div_by_zero, inf_div_y)
        
        not_b_zero = self.not_gate(b_is_zero)
        zero_div_x = self.and_gate(a_is_zero, not_b_zero)  # 0/x (x!=0)
        not_a_inf = self.not_gate(a_is_inf)
        x_div_inf = self.and_gate(not_a_inf, b_is_inf)     # x/Inf (x!=Inf)
        result_is_zero = self.or_gate(zero_div_x, x_div_inf)
        
        # ===== 4. 指数处理 (13位防溢出) =====
        # 转换为LSB first
        e_a_lsb = e_a.flip(-1)
        e_b_lsb = e_b.flip(-1)
        
        # 扩展到13位
        e_a_13 = torch.cat([e_a_lsb, zeros, zeros], dim=-1)
        e_b_13 = torch.cat([e_b_lsb, zeros, zeros], dim=-1)
        
        # Ea - Eb
        exp_diff, _ = self.exp_sub(e_a_13, e_b_13)
        
        # + 1023
        const_1023 = torch.cat([ones]*10 + [zeros]*3, dim=-1)  # 1023 = 0b01111111111
        exp_result, _ = self.exp_add(exp_diff, const_1023)
        
        # ===== 5. 尾数除法 =====
        # 恢复余数除法 (Non-Restoring/Restoring Division)
        # 
        # 被除数 A = 1.m_a (54位: 隐藏位 + 52尾数 + 1保护位)
        # 除数 D = 1.m_b (54位)
        # 
        # 标准方法:
        # 1. 初始化 R = A (被除数), D = 除数
        # 2. 循环 N 次 (N = 需要的精度位数):
        #    a. 如果 R >= D: Q[i] = 1, R = (R - D) << 1
        #    b. 否则: Q[i] = 0, R = R << 1
        # 3. 最后一步不左移
        
        # 构建54位尾数: [1, M51..M0, 0] MSB first
        A_54 = torch.cat([ones, m_a, zeros], dim=-1)  # 被除数
        D_54 = torch.cat([ones, m_b, zeros], dim=-1)  # 除数
        
        # 扩展到55位用于防止溢出
        A_55 = torch.cat([zeros, A_54], dim=-1)  # [0, 1, M51..M0, 0] = 55位
        D_55 = torch.cat([zeros, D_54], dim=-1)  # [0, 1, M51..M0, 0] = 55位
        
        # 初始余数 R = A (被除数)
        R = A_55.clone()
        
        # 57次迭代产生54位商 + 3位舍入信息
        Q_bits = []
        for i in range(57):
            # 比较 R 与 D
            R_lsb = R.flip(-1)
            D_lsb = D_55.flip(-1)
            q_bit, R_sub_lsb = self.div_iterations[i](R_lsb, D_lsb)
            R_sub = R_sub_lsb.flip(-1)
            
            Q_bits.append(q_bit)
            
            # 选择新余数: 如果 Q=1，用 R-D；否则用 R
            # 由于 VecDivIteration 已经处理了这个，R_sub 就是正确的余数
            R = R_sub
            
            # 左移余数 (除了最后一次)
            if i < 56:
                R = torch.cat([R[..., 1:], zeros], dim=-1)
        
        # 商 Q: 57位 MSB first
        Q_raw = torch.cat(Q_bits, dim=-1)
        
        # 扩展到109位以保持兼容性（后面补0）
        Q = torch.cat([Q_raw, torch.zeros(batch_shape + (52,), device=device)], dim=-1)
        
        # ===== 6. 商位提取和归一化 =====
        # Q_raw: 57位 MSB first
        # 
        # 对于 1.m_a / 1.m_b:
        # - 如果 m_a >= m_b: 商 >= 1.0, Q[0] = 1
        # - 如果 m_a < m_b: 商 < 1.0, Q[0] = 0, Q[1] = 1
        
        # 检测 Q[0] 是否为1 (决定是否需要归一化)
        q0 = Q_raw[..., 0:1]
        need_normalize = self.not_gate(q0)  # Q[0]=0 时需要归一化
        
        # 检测余数非零 (sticky)
        remainder_nonzero = self.mant_or_tree(R)
        
        # 商Q_raw结构 (57位 MSB first):
        # 正常路径 (Q[0]=1, 商 >= 1):
        #   Q[0] = 1 (隐藏位)
        #   Q[1:53] = 52位尾数
        #   Q[53] = round bit
        #   Q[54:57] + 余数 = sticky bits
        #
        # 归一化路径 (Q[0]=0, 商 < 1):
        #   Q[0] = 0
        #   Q[1] = 1 (归一化后的隐藏位)
        #   Q[2:54] = 52位尾数
        #   Q[54] = round bit
        #   Q[55:57] + 余数 = sticky bits
        
        # 正常路径 (Q[0]=1)
        mant_normal = Q_raw[..., 1:53]      # 52位尾数 MSB first
        round_normal = Q_raw[..., 53:54]    # round bit
        sticky_normal_bits = torch.cat([Q_raw[..., 54:57], remainder_nonzero], dim=-1)
        sticky_normal = self.mant_or_tree(sticky_normal_bits)
        
        # 归一化路径 (Q[0]=0)
        mant_shifted = Q_raw[..., 2:54]     # 52位尾数 MSB first
        round_shifted = Q_raw[..., 54:55]   # round bit
        sticky_shifted_bits = torch.cat([Q_raw[..., 55:57], remainder_nonzero], dim=-1)
        sticky_shifted = self.mant_or_tree(sticky_shifted_bits)
        
        # 选择尾数 (Q[0]=1 选 normal, Q[0]=0 选 shifted)
        sel_52 = q0.expand(batch_shape + (52,))
        mant_52 = self.normalize_mux(sel_52, mant_normal, mant_shifted)
        
        # 选择round和sticky
        round_bit = self.normalize_mux(q0, round_normal, round_shifted)
        sticky_bit = self.normalize_mux(q0, sticky_normal, sticky_shifted)
        
        # 指数调整: Q[0]=0 时指数-1
        const_1_13 = torch.cat([ones] + [zeros]*12, dim=-1)
        exp_minus1, _ = self.exp_sub(exp_result, const_1_13)
        
        sel_13 = need_normalize.expand(batch_shape + (13,))
        exp_adjusted = self.exp_mux(sel_13, exp_minus1, exp_result)
        
        # ===== 7. RNE舍入 =====
        # 尾数是MSB first格式，LSB在最后一位
        lsb = mant_52[..., -1:]  # LSB (MSB first的最后一位)
        s_or_l = self.rne_or(sticky_bit, lsb)
        round_up = self.rne_and(round_bit, s_or_l)
        
        # 构建53位用于舍入 (52位尾数 + 1位保护位)
        # 转换为LSB first进行加法
        mant_52_lsb = mant_52.flip(-1)
        mant_53_lsb = torch.cat([mant_52_lsb, zeros], dim=-1)
        round_inc_lsb = torch.cat([round_up] + [zeros]*52, dim=-1)
        mant_rounded_lsb, carry = self.round_adder(mant_53_lsb, round_inc_lsb)
        
        # 取52位尾数，转回MSB first
        mant_final = mant_rounded_lsb[..., :52].flip(-1)
        
        # 如果进位，指数+1
        exp_plus1, _ = self.exp_add(exp_adjusted, const_1_13)
        sel_13_carry = carry.expand(batch_shape + (13,))
        exp_final_13 = self.exp_mux(sel_13_carry, exp_plus1, exp_adjusted)
        
        # 如果进位，尾数变为0
        zero_52 = torch.zeros(batch_shape + (52,), device=device)
        sel_52_carry = carry.expand(batch_shape + (52,))
        mant_final = self.normalize_mux(sel_52_carry, zero_52, mant_final)
        
        # 取11位指数 (LSB first的低11位)，转回MSB first
        exp_final = exp_final_13[..., :11].flip(-1)
        
        # ===== 8. 溢出/下溢检测 =====
        # 检测exp >= 2047 (溢出)
        exp_high_bits = exp_final_13[..., 11:13]
        is_overflow = self.exp_or_tree(exp_high_bits)
        
        # 检测exp <= 0 (下溢)
        exp_is_zero = self.not_gate(self.exp_or_tree(exp_final_13))
        exp_sign = exp_final_13[..., 12:13]  # 符号位
        is_underflow = self.or_gate(exp_sign, exp_is_zero)
        
        # ===== 9. 组装结果 =====
        # 正常结果
        result = torch.cat([s_out, exp_final, mant_final], dim=-1)
        
        # NaN: E=全1, M=非零
        nan_exp = torch.ones(batch_shape + (11,), device=device)
        nan_mant = torch.cat([ones] + [zeros]*51, dim=-1).expand(batch_shape + (52,))
        nan_val = torch.cat([s_out, nan_exp, nan_mant], dim=-1)
        
        # Inf: E=全1, M=0
        inf_exp = torch.ones(batch_shape + (11,), device=device)
        inf_mant = torch.zeros(batch_shape + (52,), device=device)
        inf_val = torch.cat([s_out, inf_exp, inf_mant], dim=-1)
        
        # Zero: E=0, M=0
        zero_exp = torch.zeros(batch_shape + (11,), device=device)
        zero_mant = torch.zeros(batch_shape + (52,), device=device)
        zero_val = torch.cat([s_out, zero_exp, zero_mant], dim=-1)
        
        # 应用特殊值 (优先级: NaN > Inf > Zero > Overflow > Underflow > 正常)
        is_nan_64 = result_is_nan.expand(batch_shape + (64,))
        result = self.nan_mux(is_nan_64, nan_val, result)
        
        not_nan = self.not_gate(result_is_nan)
        is_inf_valid = self.and_gate(result_is_inf, not_nan)
        is_inf_64 = is_inf_valid.expand(batch_shape + (64,))
        result = self.inf_mux(is_inf_64, inf_val, result)
        
        not_inf = self.not_gate(result_is_inf)
        is_zero_valid = self.and_gate(result_is_zero, self.and_gate(not_nan, not_inf))
        is_zero_64 = is_zero_valid.expand(batch_shape + (64,))
        result = self.zero_mux(is_zero_64, zero_val, result)
        
        # 应用计算溢出
        overflow_valid = self.and_gate(is_overflow, self.and_gate(not_nan, self.and_gate(not_inf, self.not_gate(result_is_zero))))
        overflow_64 = overflow_valid.expand(batch_shape + (64,))
        result = self.overflow_mux(overflow_64, inf_val, result)
        
        # 应用计算下溢
        underflow_valid = self.and_gate(is_underflow, self.and_gate(not_nan, self.and_gate(not_inf, self.and_gate(self.not_gate(is_overflow), self.not_gate(result_is_zero)))))
        underflow_64 = underflow_valid.expand(batch_shape + (64,))
        result = self.underflow_mux(underflow_64, zero_val, result)
        
        return result
    
    def reset(self):
        self.sign_xor.reset()
        self.exp_sub.reset()
        self.exp_add.reset()
        self.exp_and_tree.reset()
        self.exp_or_tree.reset()
        self.mant_or_tree.reset()
        self.not_gate.reset()
        self.and_gate.reset()
        self.or_gate.reset()
        for div in self.div_iterations:
            div.reset()
        self.mant_cmp.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.round_adder.reset()
        self.normalize_mux.reset()
        self.exp_mux.reset()
        self.nan_mux.reset()
        self.inf_mux.reset()
        self.zero_mux.reset()
        self.overflow_mux.reset()
        self.underflow_mux.reset()

