"""
FP32 平方根函数 - 100%纯SNN门电路实现
======================================

使用SNN原生逐位算法，类似于恢复余数除法。

25次迭代获得24位精度 + 1位舍入。

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate,
                          FullAdder, RippleCarryAdder)


class Subtractor50Bit(nn.Module):
    """50位减法器 - 纯SNN (LSB first)"""
    def __init__(self):
        super().__init__()
        self.bits = 50
        self.not_gates = nn.ModuleList([NOTGate() for _ in range(50)])
        self.adders = nn.ModuleList([FullAdder() for _ in range(50)])
        self.borrow_not = NOTGate()
        
    def forward(self, A, B):
        """A - B, LSB first. Returns (result, borrow)"""
        not_b = []
        for i in range(self.bits):
            not_b.append(self.not_gates[i](B[..., i:i+1]))
        not_b = torch.cat(not_b, dim=-1)
        
        c = torch.ones_like(A[..., 0:1])
        sum_bits = []
        for i in range(self.bits):
            s, c = self.adders[i](A[..., i:i+1], not_b[..., i:i+1], c)
            sum_bits.append(s)
        
        result = torch.cat(sum_bits, dim=-1)
        borrow = self.borrow_not(c)  # c=1 → no borrow (A>=B)
        return result, borrow
    
    def reset(self):
        for g in self.not_gates: g.reset()
        for g in self.adders: g.reset()
        self.borrow_not.reset()


class SpikeFP32Sqrt(nn.Module):
    """FP32 平方根 - 100%纯SNN门电路实现
    
    使用SNN原生逐位算法，25次迭代。
    
    输入: x [..., 32] FP32脉冲
    输出: sqrt(x) [..., 32] FP32脉冲
    """
    def __init__(self):
        super().__init__()
        
        # 50位减法器 (用于 R - T)
        self.subtractor = Subtractor50Bit()
        
        # 余数选择MUX (50位)
        self.mux_r = nn.ModuleList([MUXGate() for _ in range(50)])
        
        # Q bit NOT (用于判断 R >= T)
        self.q_not = NOTGate()
        
        # 指数处理
        self.exp_add = RippleCarryAdder(bits=9)
        self.exp_mux = nn.ModuleList([MUXGate() for _ in range(9)])
        
        # 舍入
        self.rne_or = ORGate()
        self.rne_and = ANDGate()
        self.round_adder = RippleCarryAdder(bits=23)
        
        # 特殊值检测
        self.exp_all_one_and = nn.ModuleList([ANDGate() for _ in range(7)])
        self.exp_zero_or = nn.ModuleList([ORGate() for _ in range(7)])
        self.exp_zero_not = NOTGate()
        self.exp_odd_not = NOTGate()  # 用于判断 (e-127) 奇偶
        self.mant_zero_or = nn.ModuleList([ORGate() for _ in range(22)])
        self.mant_zero_not = NOTGate()
        
        self.is_zero_and = ANDGate()
        self.is_inf_and = ANDGate()
        self.is_nan_and = ANDGate()
        self.is_neg_and = ANDGate()
        self.not_is_zero = NOTGate()
        
        # sticky检测
        self.sticky_or = nn.ModuleList([ORGate() for _ in range(49)])
        
        # 输出选择MUX
        self.nan_mux = nn.ModuleList([MUXGate() for _ in range(32)])
        self.inf_mux = nn.ModuleList([MUXGate() for _ in range(32)])
        self.zero_mux = nn.ModuleList([MUXGate() for _ in range(32)])
        self.radicand_mux = nn.ModuleList([MUXGate() for _ in range(25)])
        
        self.result_nan_or = ORGate()
        
    def forward(self, x):
        device = x.device
        batch_shape = x.shape[:-1]
        ones = torch.ones(batch_shape + (1,), device=device)
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        # 提取各部分
        s_x = x[..., 0:1]
        e_x = x[..., 1:9]   # MSB first
        m_x = x[..., 9:32]  # MSB first
        
        # ===== 特殊值检测 =====
        e_all_one = e_x[..., 0:1]
        for i in range(1, 8):
            e_all_one = self.exp_all_one_and[i-1](e_all_one, e_x[..., i:i+1])
        
        e_any_one = e_x[..., 0:1]
        for i in range(1, 8):
            e_any_one = self.exp_zero_or[i-1](e_any_one, e_x[..., i:i+1])
        e_is_zero = self.exp_zero_not(e_any_one)
        
        m_any_one = m_x[..., 0:1]
        for i in range(1, 23):
            m_any_one = self.mant_zero_or[i-1](m_any_one, m_x[..., i:i+1])
        m_is_zero = self.mant_zero_not(m_any_one)
        
        is_zero = self.is_zero_and(e_is_zero, m_is_zero)
        is_inf = self.is_inf_and(e_all_one, m_is_zero)
        is_nan = self.is_nan_and(e_all_one, m_any_one)
        
        not_zero = self.not_is_zero(is_zero)
        is_neg = self.is_neg_and(s_x, not_zero)
        
        # ===== 指数处理 =====
        e_x_le = e_x.flip(-1)  # LSB first
        # (e - 127) 是奇数 ⟺ e 是偶数 (因为127是奇数)
        # 所以 exp_is_odd = NOT(e & 1)
        e_lsb = e_x_le[..., 0:1]
        exp_is_odd = self.exp_odd_not(e_lsb)
        
        # (e + 127) / 2 for even, (e + 126) / 2 for odd
        e_9 = torch.cat([e_x_le, zeros], dim=-1)
        const_127_9 = torch.cat([ones, ones, ones, ones, ones, ones, ones, zeros, zeros], dim=-1)
        const_126_9 = torch.cat([zeros, ones, ones, ones, ones, ones, ones, zeros, zeros], dim=-1)
        
        # 选择加数
        add_const = []
        for i in range(9):
            bit = self.exp_mux[i](exp_is_odd, const_126_9[..., i:i+1], const_127_9[..., i:i+1])
            add_const.append(bit)
        add_const = torch.cat(add_const, dim=-1)
        
        exp_sum_9, _ = self.exp_add(e_9, add_const)
        exp_result_le = exp_sum_9[..., 1:9]  # 右移1位 (/2)
        exp_result = exp_result_le.flip(-1)  # MSB first
        
        # ===== 构建被开方数 =====
        m_x_le = m_x.flip(-1)  # LSB first
        mant_24_le = torch.cat([m_x_le, ones], dim=-1)  # 24位含隐藏位
        
        # 奇指数时左移1位
        mant_shifted_le = torch.cat([zeros, mant_24_le[..., :-1]], dim=-1)
        
        # 选择被开方数 (25位)
        radicand_25_le = []
        for i in range(24):
            bit = self.radicand_mux[i](exp_is_odd, mant_shifted_le[..., i:i+1], mant_24_le[..., i:i+1])
            radicand_25_le.append(bit)
        # 第25位：奇数时是mant_24_le的最高位，偶数时是0
        bit_24 = self.radicand_mux[24](exp_is_odd, mant_24_le[..., 23:24], zeros)
        radicand_25_le.append(bit_24)
        radicand_25_le = torch.cat(radicand_25_le, dim=-1)
        
        # 扩展到48位，放在高位 (LSB first: 高位在右边)
        radicand_48_le = torch.cat([zeros.expand(batch_shape + (23,)), radicand_25_le], dim=-1)
        
        # 转为MSB first用于逐对提取
        radicand_48_msb = radicand_48_le.flip(-1)
        
        # ===== 逐位平方根 (25次迭代) =====
        R = torch.zeros(batch_shape + (50,), device=device)  # 余数 LSB first
        Q = torch.zeros(batch_shape + (25,), device=device)  # 结果 LSB first
        
        for i in range(25):
            # 取radicand的两位 (MSB first: index 2i, 2i+1)
            if 2*i < 48:
                bit1 = radicand_48_msb[..., 2*i:2*i+1]
            else:
                bit1 = zeros
            if 2*i+1 < 48:
                bit0 = radicand_48_msb[..., 2*i+1:2*i+2]
            else:
                bit0 = zeros
            
            # R左移2位，加入这两位
            R = torch.cat([bit0, bit1, R[..., :-2]], dim=-1)
            
            # 计算 T = (Q << 2) | 1 (扩展到50位)
            # Q << 2 在LSB first中是在低位补两个0
            # T = [1, 0, Q0, Q1, ..., Q24, 0, ..., 0] 共50位 LSB first
            T = torch.cat([ones, zeros, Q, zeros.expand(batch_shape + (23,))], dim=-1)  # 50位
            
            # 比较 R 和 T
            self.subtractor.reset()
            R_minus_T, borrow = self.subtractor(R, T)
            
            # q = NOT(borrow) = 1 当 R >= T
            self.q_not.reset()
            q_bit = self.q_not(borrow)
            
            # 更新R: q=1时R=R-T, q=0时R不变
            R_next = []
            for j in range(50):
                self.mux_r[j].reset()
                r_bit = self.mux_r[j](q_bit, R_minus_T[..., j:j+1], R[..., j:j+1])
                R_next.append(r_bit)
            R = torch.cat(R_next, dim=-1)
            
            # 更新Q: Q = (Q << 1) | q
            # LSB first中左移是高位补0，然后最低位设为q
            Q = torch.cat([q_bit, Q[..., :-1]], dim=-1)
        
        # Q现在是25位 LSB first
        # 转为MSB first
        Q_msb = Q.flip(-1)
        
        # 取24位 (索引0-23) 作为尾数，索引24作为round bit
        mant_24_msb = Q_msb[..., :24]
        round_bit = Q_msb[..., 24:25]
        
        # sticky = R非零
        sticky = R[..., 0:1]
        for j in range(1, 50):
            if j-1 < len(self.sticky_or):
                self.sticky_or[j-1].reset()
                sticky = self.sticky_or[j-1](sticky, R[..., j:j+1])
        
        # ===== RNE舍入 =====
        mant_23_msb = mant_24_msb[..., 1:24]  # 去隐藏位
        lsb = mant_23_msb[..., 22:23]
        
        self.rne_or.reset()
        self.rne_and.reset()
        s_or_l = self.rne_or(sticky, lsb)
        round_up = self.rne_and(round_bit, s_or_l)
        
        mant_23_le = mant_23_msb.flip(-1)
        round_inc = torch.cat([round_up] + [zeros]*22, dim=-1)
        self.round_adder.reset()
        mant_rounded_le, carry = self.round_adder(mant_23_le, round_inc)
        mant_final = mant_rounded_le.flip(-1)
        
        # ===== 输出组装 =====
        result = torch.cat([zeros, exp_result, mant_final], dim=-1)
        
        # ===== 特殊值处理 =====
        self.result_nan_or.reset()
        result_is_nan = self.result_nan_or(is_nan, is_neg)
        nan_val = torch.cat([zeros] + [ones]*8 + [ones] + [zeros]*22, dim=-1)
        
        result_bits = []
        for i in range(32):
            self.nan_mux[i].reset()
            bit = self.nan_mux[i](result_is_nan, nan_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        # Zero
        zero_val = torch.cat([zeros]*32, dim=-1)
        result_bits = []
        for i in range(32):
            self.zero_mux[i].reset()
            bit = self.zero_mux[i](is_zero, zero_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        # Inf
        inf_val = torch.cat([zeros] + [ones]*8 + [zeros]*23, dim=-1)
        result_bits = []
        for i in range(32):
            self.inf_mux[i].reset()
            bit = self.inf_mux[i](is_inf, inf_val[..., i:i+1], result[..., i:i+1])
            result_bits.append(bit)
        result = torch.cat(result_bits, dim=-1)
        
        return result
    
    def reset(self):
        self.subtractor.reset()
        for mux in self.mux_r: mux.reset()
        self.q_not.reset()
        self.exp_add.reset()
        for mux in self.exp_mux: mux.reset()
        self.rne_or.reset()
        self.rne_and.reset()
        self.round_adder.reset()
        for g in self.exp_all_one_and: g.reset()
        for g in self.exp_zero_or: g.reset()
        self.exp_zero_not.reset()
        self.exp_odd_not.reset()
        for g in self.mant_zero_or: g.reset()
        self.mant_zero_not.reset()
        self.is_zero_and.reset()
        self.is_inf_and.reset()
        self.is_nan_and.reset()
        self.is_neg_and.reset()
        self.not_is_zero.reset()
        for g in self.sticky_or: g.reset()
        for mux in self.nan_mux: mux.reset()
        for mux in self.inf_mux: mux.reset()
        for mux in self.zero_mux: mux.reset()
        for mux in self.radicand_mux: mux.reset()
        self.result_nan_or.reset()

