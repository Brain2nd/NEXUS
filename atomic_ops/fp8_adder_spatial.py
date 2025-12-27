"""
空间编码FP8加法器 - 1个时间步完成
100%纯SNN：所有计算通过IF神经元门电路实现，无实数运算
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate, 
                          HalfAdder, FullAdder, RippleCarryAdder)


class Comparator4Bit(nn.Module):
    """4位比较器 - 纯SNN实现"""
    def __init__(self):
        super().__init__()
        # XOR用于检测不相等，NOT(XOR)=XNOR用于检测相等
        self.xor = nn.ModuleList([XORGate() for _ in range(4)])
        self.not_xor = nn.ModuleList([NOTGate() for _ in range(4)])  # XNOR = NOT(XOR)
        
        # A[i] > B[i] 需要 A[i]=1, B[i]=0，即 A[i] AND NOT(B[i])
        self.not_b = nn.ModuleList([NOTGate() for _ in range(4)])
        self.a_gt_b_and = nn.ModuleList([ANDGate() for _ in range(4)])
        
        # 累积相等条件
        self.eq_and = nn.ModuleList([ANDGate() for _ in range(3)])
        
        # 最终结果组合
        self.result_and = nn.ModuleList([ANDGate() for _ in range(3)])
        self.result_or = nn.ModuleList([ORGate() for _ in range(3)])
        
        # 检测全部相等
        self.final_eq_and = nn.ModuleList([ANDGate() for _ in range(3)])
        
    def forward(self, A, B):
        """A, B: [..., 4] (MSB first)"""
        self.reset()
        
        a3, a2, a1, a0 = A[..., 0:1], A[..., 1:2], A[..., 2:3], A[..., 3:4]
        b3, b2, b1, b0 = B[..., 0:1], B[..., 1:2], B[..., 2:3], B[..., 3:4]
        
        # 检测每位是否相等: eq = NOT(XOR(a, b))
        xor3 = self.xor[3](a3, b3)
        xor2 = self.xor[2](a2, b2)
        xor1 = self.xor[1](a1, b1)
        xor0 = self.xor[0](a0, b0)
        
        eq3 = self.not_xor[3](xor3)
        eq2 = self.not_xor[2](xor2)
        eq1 = self.not_xor[1](xor1)
        eq0 = self.not_xor[0](xor0)
        
        # A[i] > B[i]: A[i] AND NOT(B[i])
        not_b3 = self.not_b[3](b3)
        not_b2 = self.not_b[2](b2)
        not_b1 = self.not_b[1](b1)
        not_b0 = self.not_b[0](b0)
        
        gt3 = self.a_gt_b_and[3](a3, not_b3)
        gt2 = self.a_gt_b_and[2](a2, not_b2)
        gt1 = self.a_gt_b_and[1](a1, not_b1)
        gt0 = self.a_gt_b_and[0](a0, not_b0)
        
        # A > B: gt3 OR (eq3 AND gt2) OR (eq3 AND eq2 AND gt1) OR (eq3 AND eq2 AND eq1 AND gt0)
        eq32 = self.eq_and[0](eq3, eq2)
        eq321 = self.eq_and[1](eq32, eq1)
        
        term2 = self.result_and[0](eq3, gt2)
        term3 = self.result_and[1](eq32, gt1)
        term4 = self.result_and[2](eq321, gt0)
        
        t12 = self.result_or[0](gt3, term2)
        t123 = self.result_or[1](t12, term3)
        a_gt_b = self.result_or[2](t123, term4)
        
        # A == B: 所有位都相等
        eq_all_1 = self.final_eq_and[0](eq3, eq2)
        eq_all_2 = self.final_eq_and[1](eq_all_1, eq1)
        a_eq_b = self.final_eq_and[2](eq_all_2, eq0)
        
        return a_gt_b, a_eq_b
    
    def reset(self):
        for g in self.xor: g.reset()
        for g in self.not_xor: g.reset()
        for g in self.not_b: g.reset()
        for g in self.a_gt_b_and: g.reset()
        for g in self.eq_and: g.reset()
        for g in self.result_and: g.reset()
        for g in self.result_or: g.reset()
        for g in self.final_eq_and: g.reset()


class Comparator3Bit(nn.Module):
    """3位比较器 - 纯SNN实现"""
    def __init__(self):
        super().__init__()
        self.xor = nn.ModuleList([XORGate() for _ in range(3)])
        self.not_xor = nn.ModuleList([NOTGate() for _ in range(3)])
        self.not_b = nn.ModuleList([NOTGate() for _ in range(3)])
        self.a_gt_b_and = nn.ModuleList([ANDGate() for _ in range(3)])
        self.eq_and = nn.ModuleList([ANDGate() for _ in range(2)])
        self.result_and = nn.ModuleList([ANDGate() for _ in range(2)])
        self.result_or = nn.ModuleList([ORGate() for _ in range(2)])
        self.final_eq_and = nn.ModuleList([ANDGate() for _ in range(2)])
        
    def forward(self, A, B):
        """A, B: [..., 3] (MSB first)"""
        self.reset()
        
        a2, a1, a0 = A[..., 0:1], A[..., 1:2], A[..., 2:3]
        b2, b1, b0 = B[..., 0:1], B[..., 1:2], B[..., 2:3]
        
        xor2 = self.xor[2](a2, b2)
        xor1 = self.xor[1](a1, b1)
        xor0 = self.xor[0](a0, b0)
        
        eq2 = self.not_xor[2](xor2)
        eq1 = self.not_xor[1](xor1)
        eq0 = self.not_xor[0](xor0)
        
        not_b2 = self.not_b[2](b2)
        not_b1 = self.not_b[1](b1)
        not_b0 = self.not_b[0](b0)
        
        gt2 = self.a_gt_b_and[2](a2, not_b2)
        gt1 = self.a_gt_b_and[1](a1, not_b1)
        gt0 = self.a_gt_b_and[0](a0, not_b0)
        
        eq21 = self.eq_and[0](eq2, eq1)
        
        term2 = self.result_and[0](eq2, gt1)
        term3 = self.result_and[1](eq21, gt0)
        
        t12 = self.result_or[0](gt2, term2)
        a_gt_b = self.result_or[1](t12, term3)
        
        a_eq_b_t = self.final_eq_and[0](eq2, eq1)
        a_eq_b = self.final_eq_and[1](a_eq_b_t, eq0)
        
        return a_gt_b, a_eq_b
    
    def reset(self):
        for g in self.xor: g.reset()
        for g in self.not_xor: g.reset()
        for g in self.not_b: g.reset()
        for g in self.a_gt_b_and: g.reset()
        for g in self.eq_and: g.reset()
        for g in self.result_and: g.reset()
        for g in self.result_or: g.reset()
        for g in self.final_eq_and: g.reset()


class Subtractor4Bit(nn.Module):
    """4位减法器 - 纯SNN实现"""
    def __init__(self):
        super().__init__()
        self.xor1 = nn.ModuleList([XORGate() for _ in range(4)])
        self.xor2 = nn.ModuleList([XORGate() for _ in range(4)])
        self.not_a = nn.ModuleList([NOTGate() for _ in range(4)])
        self.and1 = nn.ModuleList([ANDGate() for _ in range(4)])
        self.and2 = nn.ModuleList([ANDGate() for _ in range(4)])
        self.and3 = nn.ModuleList([ANDGate() for _ in range(4)])
        self.or1 = nn.ModuleList([ORGate() for _ in range(4)])
        self.or2 = nn.ModuleList([ORGate() for _ in range(4)])
        
    def forward(self, A, B, Bin=None):
        """A - B"""
        self.reset()
        
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin
            
        diffs = []
        for i in [3, 2, 1, 0]:
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            t1 = self.xor1[i](a_i, b_i)
            diff = self.xor2[i](t1, borrow)
            
            not_a_i = self.not_a[i](a_i)
            term1 = self.and1[i](not_a_i, b_i)
            term2 = self.and2[i](not_a_i, borrow)
            term3 = self.and3[i](b_i, borrow)
            t12 = self.or1[i](term1, term2)
            new_borrow = self.or2[i](t12, term3)
            
            diffs.append((i, diff))
            borrow = new_borrow
        
        diffs.sort(key=lambda x: x[0])
        result = torch.cat([d[1] for d in diffs], dim=-1)
        
        return result, borrow
    
    def reset(self):
        for g in self.xor1: g.reset()
        for g in self.xor2: g.reset()
        for g in self.not_a: g.reset()
        for g in self.and1: g.reset()
        for g in self.and2: g.reset()
        for g in self.and3: g.reset()
        for g in self.or1: g.reset()
        for g in self.or2: g.reset()


class Subtractor8Bit(nn.Module):
    """8位减法器 - 纯SNN实现"""
    def __init__(self):
        super().__init__()
        self.xor1 = nn.ModuleList([XORGate() for _ in range(8)])
        self.xor2 = nn.ModuleList([XORGate() for _ in range(8)])
        self.not_a = nn.ModuleList([NOTGate() for _ in range(8)])
        self.and1 = nn.ModuleList([ANDGate() for _ in range(8)])
        self.and2 = nn.ModuleList([ANDGate() for _ in range(8)])
        self.and3 = nn.ModuleList([ANDGate() for _ in range(8)])
        self.or1 = nn.ModuleList([ORGate() for _ in range(8)])
        self.or2 = nn.ModuleList([ORGate() for _ in range(8)])
        
    def forward(self, A, B, Bin=None):
        """A - B"""
        self.reset()
        
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin
            
        diffs = []
        for i in [7, 6, 5, 4, 3, 2, 1, 0]:
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            t1 = self.xor1[i](a_i, b_i)
            diff = self.xor2[i](t1, borrow)
            
            not_a_i = self.not_a[i](a_i)
            term1 = self.and1[i](not_a_i, b_i)
            term2 = self.and2[i](not_a_i, borrow)
            term3 = self.and3[i](b_i, borrow)
            t12 = self.or1[i](term1, term2)
            new_borrow = self.or2[i](t12, term3)
            
            diffs.append((i, diff))
            borrow = new_borrow
        
        diffs.sort(key=lambda x: x[0])
        return torch.cat([d[1] for d in diffs], dim=-1), borrow
    
    def reset(self):
        for g in self.xor1: g.reset()
        for g in self.xor2: g.reset()
        for g in self.not_a: g.reset()
        for g in self.and1: g.reset()
        for g in self.and2: g.reset()
        for g in self.and3: g.reset()
        for g in self.or1: g.reset()
        for g in self.or2: g.reset()


class BarrelShifterRight8(nn.Module):
    """8位桶形右移位器 - 纯SNN"""
    def __init__(self):
        super().__init__()
        self.mux_layer0 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_layer1 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_layer2 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_layer3 = nn.ModuleList([MUXGate() for _ in range(8)])
        
    def forward(self, X, shift):
        """X: [..., 8], shift: [..., 4]"""
        self.reset()
        
        s0 = shift[..., 3:4]
        s1 = shift[..., 2:3]
        s2 = shift[..., 1:2]
        s3 = shift[..., 0:1]
        
        zeros = torch.zeros_like(X[..., 0:1])
        
        out0 = []
        for i in range(8):
            shifted = zeros if i == 0 else X[..., i-1:i]
            original = X[..., i:i+1]
            out0.append(self.mux_layer0[i](s0, shifted, original))
        x0 = torch.cat(out0, dim=-1)
        
        out1 = []
        for i in range(8):
            shifted = zeros if i < 2 else x0[..., i-2:i-1]
            original = x0[..., i:i+1]
            out1.append(self.mux_layer1[i](s1, shifted, original))
        x1 = torch.cat(out1, dim=-1)
        
        out2 = []
        for i in range(8):
            shifted = zeros if i < 4 else x1[..., i-4:i-3]
            original = x1[..., i:i+1]
            out2.append(self.mux_layer2[i](s2, shifted, original))
        x2 = torch.cat(out2, dim=-1)
        
        out3 = []
        for i in range(8):
            out3.append(self.mux_layer3[i](s3, zeros, x2[..., i:i+1]))
        
        return torch.cat(out3, dim=-1)
    
    def reset(self):
        for m in self.mux_layer0: m.reset()
        for m in self.mux_layer1: m.reset()
        for m in self.mux_layer2: m.reset()
        for m in self.mux_layer3: m.reset()


class BarrelShifterRight12(nn.Module):
    """12位桶形右移位器 - 纯SNN (用于尾数对齐)"""
    def __init__(self):
        super().__init__()
        # 移位0-15位
        self.mux_layer0 = nn.ModuleList([MUXGate() for _ in range(12)])  # shift by 1
        self.mux_layer1 = nn.ModuleList([MUXGate() for _ in range(12)])  # shift by 2
        self.mux_layer2 = nn.ModuleList([MUXGate() for _ in range(12)])  # shift by 4
        self.mux_layer3 = nn.ModuleList([MUXGate() for _ in range(12)])  # shift by 8
        
    def forward(self, X, shift):
        """X: [..., 12], shift: [..., 4]"""
        self.reset()
        
        s0 = shift[..., 3:4]  # shift by 1
        s1 = shift[..., 2:3]  # shift by 2
        s2 = shift[..., 1:2]  # shift by 4
        s3 = shift[..., 0:1]  # shift by 8
        
        zeros = torch.zeros_like(X[..., 0:1])
        
        # Shift by 1
        out0 = []
        for i in range(12):
            shifted = zeros if i == 0 else X[..., i-1:i]
            original = X[..., i:i+1]
            out0.append(self.mux_layer0[i](s0, shifted, original))
        x0 = torch.cat(out0, dim=-1)
        
        # Shift by 2
        out1 = []
        for i in range(12):
            shifted = zeros if i < 2 else x0[..., i-2:i-1]
            original = x0[..., i:i+1]
            out1.append(self.mux_layer1[i](s1, shifted, original))
        x1 = torch.cat(out1, dim=-1)
        
        # Shift by 4
        out2 = []
        for i in range(12):
            shifted = zeros if i < 4 else x1[..., i-4:i-3]
            original = x1[..., i:i+1]
            out2.append(self.mux_layer2[i](s2, shifted, original))
        x2 = torch.cat(out2, dim=-1)
        
        # Shift by 8
        out3 = []
        for i in range(12):
            shifted = zeros if i < 8 else x2[..., i-8:i-7]
            original = x2[..., i:i+1]
            out3.append(self.mux_layer3[i](s3, shifted, original))
        
        return torch.cat(out3, dim=-1)
    
    def reset(self):
        for m in self.mux_layer0: m.reset()
        for m in self.mux_layer1: m.reset()
        for m in self.mux_layer2: m.reset()
        for m in self.mux_layer3: m.reset()


class Subtractor12Bit(nn.Module):
    """12位减法器 - 纯SNN"""
    def __init__(self):
        super().__init__()
        self.xor1 = nn.ModuleList([XORGate() for _ in range(12)])
        self.xor2 = nn.ModuleList([XORGate() for _ in range(12)])
        self.not_a = nn.ModuleList([NOTGate() for _ in range(12)])
        self.and1 = nn.ModuleList([ANDGate() for _ in range(12)])
        self.and2 = nn.ModuleList([ANDGate() for _ in range(12)])
        self.and3 = nn.ModuleList([ANDGate() for _ in range(12)])
        self.or1 = nn.ModuleList([ORGate() for _ in range(12)])
        self.or2 = nn.ModuleList([ORGate() for _ in range(12)])
        
    def forward(self, A, B, Bin=None):
        """A - B"""
        self.reset()
        
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin
            
        diffs = []
        for i in range(11, -1, -1):  # LSB first
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            t1 = self.xor1[i](a_i, b_i)
            diff = self.xor2[i](t1, borrow)
            
            not_a_i = self.not_a[i](a_i)
            term1 = self.and1[i](not_a_i, b_i)
            term2 = self.and2[i](not_a_i, borrow)
            term3 = self.and3[i](b_i, borrow)
            t12 = self.or1[i](term1, term2)
            new_borrow = self.or2[i](t12, term3)
            
            diffs.append((i, diff))
            borrow = new_borrow
        
        diffs.sort(key=lambda x: x[0])
        return torch.cat([d[1] for d in diffs], dim=-1), borrow
    
    def reset(self):
        for g in self.xor1: g.reset()
        for g in self.xor2: g.reset()
        for g in self.not_a: g.reset()
        for g in self.and1: g.reset()
        for g in self.and2: g.reset()
        for g in self.and3: g.reset()
        for g in self.or1: g.reset()
        for g in self.or2: g.reset()


class Adder12Bit(nn.Module):
    """12位加法器 - 纯SNN"""
    def __init__(self):
        super().__init__()
        self.xor1 = nn.ModuleList([XORGate() for _ in range(12)])
        self.xor2 = nn.ModuleList([XORGate() for _ in range(12)])
        self.and1 = nn.ModuleList([ANDGate() for _ in range(12)])
        self.and2 = nn.ModuleList([ANDGate() for _ in range(12)])
        self.and3 = nn.ModuleList([ANDGate() for _ in range(12)])
        self.or1 = nn.ModuleList([ORGate() for _ in range(12)])
        self.or2 = nn.ModuleList([ORGate() for _ in range(12)])
        
    def forward(self, A, B, Cin=None):
        """A + B"""
        self.reset()
        
        if Cin is None:
            carry = torch.zeros_like(A[..., 0:1])
        else:
            carry = Cin
            
        sums = []
        for i in range(11, -1, -1):  # LSB first
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            t1 = self.xor1[i](a_i, b_i)
            s = self.xor2[i](t1, carry)
            
            term1 = self.and1[i](a_i, b_i)
            term2 = self.and2[i](a_i, carry)
            term3 = self.and3[i](b_i, carry)
            t12 = self.or1[i](term1, term2)
            new_carry = self.or2[i](t12, term3)
            
            sums.append((i, s))
            carry = new_carry
        
        sums.sort(key=lambda x: x[0])
        return torch.cat([s[1] for s in sums], dim=-1), carry
    
    def reset(self):
        for g in self.xor1: g.reset()
        for g in self.xor2: g.reset()
        for g in self.and1: g.reset()
        for g in self.and2: g.reset()
        for g in self.and3: g.reset()
        for g in self.or1: g.reset()
        for g in self.or2: g.reset()


class BarrelShifterLeft8(nn.Module):
    """8位桶形左移位器 - 纯SNN"""
    def __init__(self):
        super().__init__()
        self.mux_layer0 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_layer1 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_layer2 = nn.ModuleList([MUXGate() for _ in range(8)])
        
    def forward(self, X, shift):
        """X: [..., 8], shift: [..., 3]"""
        self.reset()
        
        s0 = shift[..., 2:3]
        s1 = shift[..., 1:2]
        s2 = shift[..., 0:1]
        
        zeros = torch.zeros_like(X[..., 0:1])
        
        out0 = []
        for i in range(8):
            shifted = zeros if i == 7 else X[..., i+1:i+2]
            original = X[..., i:i+1]
            out0.append(self.mux_layer0[i](s0, shifted, original))
        x0 = torch.cat(out0, dim=-1)
        
        out1 = []
        for i in range(8):
            shifted = zeros if i >= 6 else x0[..., i+2:i+3]
            original = x0[..., i:i+1]
            out1.append(self.mux_layer1[i](s1, shifted, original))
        x1 = torch.cat(out1, dim=-1)
        
        out2 = []
        for i in range(8):
            shifted = zeros if i >= 4 else x1[..., i+4:i+5]
            original = x1[..., i:i+1]
            out2.append(self.mux_layer2[i](s2, shifted, original))
        
        return torch.cat(out2, dim=-1)
    
    def reset(self):
        for m in self.mux_layer0: m.reset()
        for m in self.mux_layer1: m.reset()
        for m in self.mux_layer2: m.reset()


class LeadingZeroDetector8(nn.Module):
    """8位前导零检测器 - 纯SNN"""
    def __init__(self):
        super().__init__()
        self.or_67 = ORGate()
        self.or_45 = ORGate()
        self.or_23 = ORGate()
        self.or_01 = ORGate()
        self.or_7654 = ORGate()
        self.or_3210 = ORGate()
        
        # NOT门用于计算LZC
        self.not_7654 = NOTGate()
        self.not_76 = NOTGate()
        self.not_32 = NOTGate()
        self.not_b7 = NOTGate()
        self.not_b5 = NOTGate()
        self.not_b3 = NOTGate()
        self.not_b1 = NOTGate()
        
        # MUX门
        self.mux_lzc1 = MUXGate()
        self.mux_lzc0_high = MUXGate()
        self.mux_lzc0_low = MUXGate()
        self.mux_lzc0 = MUXGate()
        
    def forward(self, X):
        """X: [..., 8] (MSB at 0), 返回: [..., 3] 前导零计数"""
        self.reset()
        
        b7, b6, b5, b4 = X[..., 0:1], X[..., 1:2], X[..., 2:3], X[..., 3:4]
        b3, b2, b1, b0 = X[..., 4:5], X[..., 5:6], X[..., 6:7], X[..., 7:8]
        
        has_76 = self.or_67(b7, b6)
        has_54 = self.or_45(b5, b4)
        has_32 = self.or_23(b3, b2)
        has_10 = self.or_01(b1, b0)
        
        has_7654 = self.or_7654(has_76, has_54)
        has_3210 = self.or_3210(has_32, has_10)
        
        # LZC[2]: 高4位没有1
        lzc2 = self.not_7654(has_7654)
        
        # LZC[1]
        lzc1_high = self.not_76(has_76)
        lzc1_low = self.not_32(has_32)
        lzc1 = self.mux_lzc1(has_7654, lzc1_high, lzc1_low)
        
        # LZC[0]
        not_b7 = self.not_b7(b7)
        not_b5 = self.not_b5(b5)
        not_b3 = self.not_b3(b3)
        not_b1 = self.not_b1(b1)
        
        lzc0_high = self.mux_lzc0_high(has_76, not_b7, not_b5)
        lzc0_low = self.mux_lzc0_low(has_32, not_b3, not_b1)
        lzc0 = self.mux_lzc0(has_7654, lzc0_high, lzc0_low)
        
        return torch.cat([lzc2, lzc1, lzc0], dim=-1)
    
    def reset(self):
        self.or_67.reset()
        self.or_45.reset()
        self.or_23.reset()
        self.or_01.reset()
        self.or_7654.reset()
        self.or_3210.reset()
        self.not_7654.reset()
        self.not_76.reset()
        self.not_32.reset()
        self.not_b7.reset()
        self.not_b5.reset()
        self.not_b3.reset()
        self.not_b1.reset()
        self.mux_lzc1.reset()
        self.mux_lzc0_high.reset()
        self.mux_lzc0_low.reset()
        self.mux_lzc0.reset()


class Adder8Bit(nn.Module):
    """8位加法器 - 纯SNN"""
    def __init__(self):
        super().__init__()
        self.adders = nn.ModuleList([FullAdder() for _ in range(8)])
        
    def forward(self, A, B, Cin=None):
        """A, B: [..., 8] (MSB first)"""
        self.reset()
        
        if Cin is None:
            carry = torch.zeros_like(A[..., 0:1])
        else:
            carry = Cin
            
        sums = []
        for i in [7, 6, 5, 4, 3, 2, 1, 0]:
            s, carry = self.adders[i](A[..., i:i+1], B[..., i:i+1], carry)
            sums.append((i, s))
        
        sums.sort(key=lambda x: x[0])
        return torch.cat([s[1] for s in sums], dim=-1), carry
    
    def reset(self):
        for a in self.adders: a.reset()


class SpikeFP8Adder_Spatial(nn.Module):
    """空间编码FP8加法器 - 100%纯SNN实现
    
    所有计算通过IF神经元门电路实现，无任何实数域运算
    """
    def __init__(self):
        super().__init__()
        
        # ===== 指数比较与差值 =====
        self.exp_cmp = Comparator4Bit()
        self.exp_sub_ab = Subtractor4Bit()
        self.exp_sub_ba = Subtractor4Bit()
        self.exp_diff_mux = nn.ModuleList([MUXGate() for _ in range(4)])
        
        # ===== 尾数比较（含隐藏位，4位）=====
        self.mantissa_cmp = Comparator4Bit()  # 比较 [hidden, M2, M1, M0]
        
        # ===== 绝对值比较组合逻辑 =====
        self.abs_eq_and = ANDGate()  # E相等 AND M相等
        self.mant_ge_or = ORGate()   # M_gt OR M_eq
        self.abs_ge_and = ANDGate()  # E_eq AND M_ge
        self.abs_ge_or = ORGate()    # E_gt OR (E_eq AND M_ge)
        
        # ===== 尾数对齐（12位以支持大移位）=====
        self.align_shifter = BarrelShifterRight12()
        
        # ===== 零检测（指数）=====
        self.e_zero_or = nn.ModuleList([ORGate() for _ in range(6)])  # 检测E是否非零
        self.e_zero_not = nn.ModuleList([NOTGate() for _ in range(2)])  # NOT得到E==0
        
        # ===== Subnormal指数修正 (E=0 -> E_eff=1) =====
        self.subnorm_exp_mux_a = nn.ModuleList([MUXGate() for _ in range(4)])
        self.subnorm_exp_mux_b = nn.ModuleList([MUXGate() for _ in range(4)])
        
        # ===== 尾数加法/减法（12位精度）=====
        self.mantissa_adder = Adder12Bit()
        self.mantissa_sub = Subtractor12Bit()
        
        # ===== 符号处理 =====
        self.sign_xor = XORGate()
        self.not_diff_sign = NOTGate()
        self.exact_cancel_and = ANDGate()  # is_diff_sign AND abs_eq
        
        # ===== 操作数选择MUX（12位尾数）=====
        self.swap_mux_s = MUXGate()
        self.swap_mux_e = nn.ModuleList([MUXGate() for _ in range(4)])
        self.swap_mux_m = nn.ModuleList([MUXGate() for _ in range(24)])  # 12个用于m_large, 12个用于m_small
        
        # ===== 结果选择（加法 vs 减法，12位+carry）=====
        self.result_mux = nn.ModuleList([MUXGate() for _ in range(13)])
        
        # ===== 归一化 =====
        self.lzd = LeadingZeroDetector8()
        self.norm_shifter = BarrelShifterLeft8()
        self.exp_adj_sub = Subtractor4Bit()
        
        # ===== 溢出处理 =====
        self.exp_overflow_mux = nn.ModuleList([MUXGate() for _ in range(4)])
        self.post_round_exp_inc = RippleCarryAdder(bits=4)
        
        # ===== 指数下溢检测（subnormal）=====
        self.underflow_cmp = Comparator4Bit()  # 比较 e_max 和 lzc
        self.underflow_or = ORGate()  # lzc >= e_max (gt OR eq)
        self.underflow_mux_e = nn.ModuleList([MUXGate() for _ in range(4)])
        self.underflow_mux_m = nn.ModuleList([MUXGate() for _ in range(3)])
        self.subnorm_shifter = BarrelShifterLeft8()  # 部分左移
        
        # ===== 指数溢出检测（NaN）=====
        self.exp_overflow_and = ANDGate()  # result_carry AND exp_inc_carry
        self.nan_mux_s = MUXGate()
        self.nan_mux_e = nn.ModuleList([MUXGate() for _ in range(4)])
        self.nan_mux_m = nn.ModuleList([MUXGate() for _ in range(3)])
        
        # ===== 尾数选择（溢出 vs 正常）=====
        self.m_overflow_mux = nn.ModuleList([MUXGate() for _ in range(5)])  # m2, m1, m0, round, sticky
        
        # ===== Sticky OR =====
        self.sticky_or_overflow = nn.ModuleList([ORGate() for _ in range(3)])
        self.sticky_or_normal = nn.ModuleList([ORGate() for _ in range(2)])
        self.sticky_extra_or = nn.ModuleList([ORGate() for _ in range(4)])  # 额外位的sticky(overflow)
        self.sticky_extra_or_norm = ORGate()  # 额外位的sticky(normal)
        
        # ===== 舍入 =====
        self.round_or = ORGate()
        self.round_and = ANDGate()
        self.round_ha0 = HalfAdder()
        self.round_ha1 = HalfAdder()
        self.round_ha2 = HalfAdder()
        
        # ===== 舍入溢出处理 =====
        self.not_m_carry = NOTGate()
        self.m_final_and = nn.ModuleList([ANDGate() for _ in range(3)])
        self.round_exp_mux = nn.ModuleList([MUXGate() for _ in range(4)])
        
        # ===== 符号结果选择 =====
        self.final_sign_mux = MUXGate()
        
        # ===== 完全抵消路径选择 =====
        self.cancel_mux_s = MUXGate()
        self.cancel_mux_e = nn.ModuleList([MUXGate() for _ in range(4)])
        self.cancel_mux_m = nn.ModuleList([MUXGate() for _ in range(3)])
        
    def forward(self, A, B):
        """纯SNN FP8加法"""
        self.reset()
        
        s_a, e_a, m_a_raw = A[..., 0:1], A[..., 1:5], A[..., 5:8]
        s_b, e_b, m_b_raw = B[..., 0:1], B[..., 1:5], B[..., 5:8]
        
        zeros = torch.zeros_like(s_a)
        ones = torch.ones_like(s_a)
        
        # ===== 隐藏位检测与指数修正 =====
        # E是否为零（subnormal）: 检测E的任一位是否为1
        e_a_or_01 = self.e_zero_or[0](e_a[..., 0:1], e_a[..., 1:2])
        e_a_or_23 = self.e_zero_or[1](e_a[..., 2:3], e_a[..., 3:4])
        e_a_nonzero = self.e_zero_or[2](e_a_or_01, e_a_or_23)
        e_a_is_zero = self.e_zero_not[0](e_a_nonzero)
        hidden_a = e_a_nonzero  # E非零则隐藏位为1
        
        e_b_or_01 = self.e_zero_or[3](e_b[..., 0:1], e_b[..., 1:2])
        e_b_or_23 = self.e_zero_or[4](e_b[..., 2:3], e_b[..., 3:4])
        e_b_nonzero = self.e_zero_or[5](e_b_or_01, e_b_or_23)
        e_b_is_zero = self.e_zero_not[1](e_b_nonzero)
        hidden_b = e_b_nonzero
        
        # Subnormal的有效指数是1，不是0
        # 当E=0时，实际使用E=1进行计算
        e_a_eff = []
        for i in range(4):
            if i == 3:  # LSB: 如果E=0，则设为1
                e_a_eff.append(self.subnorm_exp_mux_a[i](e_a_is_zero, ones, e_a[..., i:i+1]))
            else:
                e_a_eff.append(self.subnorm_exp_mux_a[i](e_a_is_zero, zeros, e_a[..., i:i+1]))
        e_a_eff = torch.cat(e_a_eff, dim=-1)
        
        e_b_eff = []
        for i in range(4):
            if i == 3:
                e_b_eff.append(self.subnorm_exp_mux_b[i](e_b_is_zero, ones, e_b[..., i:i+1]))
            else:
                e_b_eff.append(self.subnorm_exp_mux_b[i](e_b_is_zero, zeros, e_b[..., i:i+1]))
        e_b_eff = torch.cat(e_b_eff, dim=-1)
        
        # 12位尾数: hidden(1) + M(3) + guard(8)
        m_a = torch.cat([hidden_a, m_a_raw, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        m_b = torch.cat([hidden_b, m_b_raw, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], dim=-1)
        
        # ===== Step 1: 比较绝对值（使用有效指数）=====
        a_exp_gt, a_exp_eq = self.exp_cmp(e_a_eff, e_b_eff)
        # 尾数比较需要包含隐藏位：[hidden, M2, M1, M0]
        m_a_with_hidden = torch.cat([hidden_a, m_a_raw], dim=-1)
        m_b_with_hidden = torch.cat([hidden_b, m_b_raw], dim=-1)
        a_mant_gt, a_mant_eq = self.mantissa_cmp(m_a_with_hidden, m_b_with_hidden)
        
        # |A| == |B|: E_eq AND M_eq
        a_abs_eq_b = self.abs_eq_and(a_exp_eq, a_mant_eq)
        
        # |A| >= |B|: E_gt OR (E_eq AND (M_gt OR M_eq))
        a_mant_ge = self.mant_ge_or(a_mant_gt, a_mant_eq)
        exp_eq_and_mant_ge = self.abs_ge_and(a_exp_eq, a_mant_ge)
        a_ge_b = self.abs_ge_or(a_exp_gt, exp_eq_and_mant_ge)
        
        # ===== Step 2: 指数差（使用有效指数）=====
        diff_ab, _ = self.exp_sub_ab(e_a_eff, e_b_eff)
        diff_ba, _ = self.exp_sub_ba(e_b_eff, e_a_eff)
        
        exp_diff = []
        for i in range(4):
            d = self.exp_diff_mux[i](a_ge_b, diff_ab[..., i:i+1], diff_ba[..., i:i+1])
            exp_diff.append(d)
        exp_diff = torch.cat(exp_diff, dim=-1)
        
        # e_max使用有效指数
        e_max = []
        for i in range(4):
            e = self.swap_mux_e[i](a_ge_b, e_a_eff[..., i:i+1], e_b_eff[..., i:i+1])
            e_max.append(e)
        e_max = torch.cat(e_max, dim=-1)
        
        # ===== Step 3: 尾数对齐（12位）=====
        m_large = []
        m_small_unshifted = []
        for i in range(12):
            ml = self.swap_mux_m[i](a_ge_b, m_a[..., i:i+1], m_b[..., i:i+1])
            ms = self.swap_mux_m[i+12](a_ge_b, m_b[..., i:i+1], m_a[..., i:i+1])
            m_large.append(ml)
            m_small_unshifted.append(ms)
        m_large = torch.cat(m_large, dim=-1)
        m_small_unshifted = torch.cat(m_small_unshifted, dim=-1)
        
        m_small = self.align_shifter(m_small_unshifted, exp_diff)
        
        # ===== Step 4: 符号处理 =====
        is_diff_sign = self.sign_xor(s_a, s_b)
        exact_cancel = self.exact_cancel_and(is_diff_sign, a_abs_eq_b)
        s_large = self.swap_mux_s(a_ge_b, s_a, s_b)
        
        # ===== Step 5: 尾数运算（12位）=====
        sum_result, sum_carry = self.mantissa_adder(m_large, m_small)
        diff_result, _ = self.mantissa_sub(m_large, m_small)
        
        mantissa_result_12 = []
        for i in range(12):
            r = self.result_mux[i](is_diff_sign, diff_result[..., i:i+1], sum_result[..., i:i+1])
            mantissa_result_12.append(r)
        mantissa_result_12 = torch.cat(mantissa_result_12, dim=-1)
        result_carry = self.result_mux[12](is_diff_sign, zeros, sum_carry)
        
        # 取高8位用于归一化，低4位用于sticky计算
        mantissa_result = mantissa_result_12[..., :8]
        extra_bits = mantissa_result_12[..., 8:12]
        
        # ===== Step 6: 归一化 =====
        lzc = self.lzd(mantissa_result)
        lzc_4bit = torch.cat([zeros, lzc], dim=-1)
        
        # 检测指数下溢: lzc >= e_max (结果是subnormal)
        lzc_gt_emax, lzc_eq_emax = self.underflow_cmp(lzc_4bit, e_max)
        # 下溢条件: lzc >= e_max (包括lzc==e_max时e_after_norm=0的情况)
        is_underflow = self.underflow_or(lzc_gt_emax, lzc_eq_emax)
        
        # 正常归一化（左移lzc位）
        norm_mantissa = self.norm_shifter(mantissa_result, lzc)
        e_after_norm, _ = self.exp_adj_sub(e_max, lzc_4bit)
        
        # 下溢时: 左移e_max位（不是lzc位），指数=0
        # 取e_max的低3位作为移位量
        e_max_3bit = e_max[..., 1:4]  # 最多移7位
        subnorm_mantissa = self.subnorm_shifter(mantissa_result, e_max_3bit)
        
        # 溢出路径：E + 1
        one_4bit = torch.cat([zeros, zeros, zeros, ones], dim=-1)
        e_inc_le, exp_inc_carry = self.post_round_exp_inc(e_max.flip(-1), one_4bit.flip(-1))
        e_plus_one = e_inc_le.flip(-1)
        # 如果exp_inc_carry=1且result_carry=1，说明E>15，应该输出NaN
        # is_exp_overflow = result_carry AND exp_inc_carry
        
        # 选择指数：下溢时为0，否则为e_after_norm
        e_normal = []
        for i in range(4):
            e_sel = self.underflow_mux_e[i](is_underflow, zeros, e_after_norm[..., i:i+1])
            e_normal.append(e_sel)
        e_normal = torch.cat(e_normal, dim=-1)
        
        # 最终指数选择：溢出 > 下溢 > 正常
        final_e_pre = []
        for i in range(4):
            e_sel = self.exp_overflow_mux[i](result_carry, e_plus_one[..., i:i+1], e_normal[..., i:i+1])
            final_e_pre.append(e_sel)
        final_e_pre = torch.cat(final_e_pre, dim=-1)
        
        # ===== Step 7: 提取尾数并舍入 =====
        # 溢出情况（12位中的位置0-7 + extra_bits用于sticky）
        m2_overflow = mantissa_result[..., 0:1]
        m1_overflow = mantissa_result[..., 1:2]
        m0_overflow = mantissa_result[..., 2:3]
        round_overflow = mantissa_result[..., 3:4]
        # sticky = OR(bits 4-7) OR OR(extra_bits)
        sticky_ov_t1 = self.sticky_or_overflow[0](mantissa_result[..., 4:5], mantissa_result[..., 5:6])
        sticky_ov_t2 = self.sticky_or_overflow[1](sticky_ov_t1, mantissa_result[..., 6:7])
        sticky_ov_t3 = self.sticky_or_overflow[2](sticky_ov_t2, mantissa_result[..., 7:8])
        # 合并extra_bits的sticky
        extra_or_01 = self.sticky_extra_or[0](extra_bits[..., 0:1], extra_bits[..., 1:2])
        extra_or_23 = self.sticky_extra_or[1](extra_bits[..., 2:3], extra_bits[..., 3:4])
        extra_or_all = self.sticky_extra_or[2](extra_or_01, extra_or_23)
        sticky_overflow = self.sticky_extra_or[3](sticky_ov_t3, extra_or_all)
        
        # 正常归一化情况（extra_bits也参与sticky）
        m2_norm = norm_mantissa[..., 1:2]
        m1_norm = norm_mantissa[..., 2:3]
        m0_norm = norm_mantissa[..., 3:4]
        round_norm = norm_mantissa[..., 4:5]
        sticky_nm_t = self.sticky_or_normal[0](norm_mantissa[..., 5:6], norm_mantissa[..., 6:7])
        sticky_nm_t2 = self.sticky_or_normal[1](sticky_nm_t, norm_mantissa[..., 7:8])
        # extra_bits已经计算过: extra_or_all
        sticky_norm = self.sticky_extra_or_norm(sticky_nm_t2, extra_or_all)
        
        # subnormal情况（下溢）: 使用部分归一化的结果，取位置0-2作为尾数（无隐藏位）
        m2_subnorm = subnorm_mantissa[..., 0:1]
        m1_subnorm = subnorm_mantissa[..., 1:2]
        m0_subnorm = subnorm_mantissa[..., 2:3]
        
        # 选择尾数：先选择是否下溢，再选择是否溢出
        m2_normal = self.underflow_mux_m[0](is_underflow, m2_subnorm, m2_norm)
        m1_normal = self.underflow_mux_m[1](is_underflow, m1_subnorm, m1_norm)
        m0_normal = self.underflow_mux_m[2](is_underflow, m0_subnorm, m0_norm)
        
        m2 = self.m_overflow_mux[0](result_carry, m2_overflow, m2_normal)
        m1 = self.m_overflow_mux[1](result_carry, m1_overflow, m1_normal)
        m0 = self.m_overflow_mux[2](result_carry, m0_overflow, m0_normal)
        round_bit = self.m_overflow_mux[3](result_carry, round_overflow, round_norm)
        sticky = self.m_overflow_mux[4](result_carry, sticky_overflow, sticky_norm)
        
        # RNE舍入
        sticky_or_m0 = self.round_or(sticky, m0)
        do_round = self.round_and(round_bit, sticky_or_m0)
        
        m0_r, c0 = self.round_ha0(m0, do_round)
        m1_r, c1 = self.round_ha1(m1, c0)
        m2_r, m_carry = self.round_ha2(m2, c1)
        
        # 舍入溢出
        not_mc = self.not_m_carry(m_carry)
        m2_final = self.m_final_and[0](m2_r, not_mc)
        m1_final = self.m_final_and[1](m1_r, not_mc)
        m0_final = self.m_final_and[2](m0_r, not_mc)
        
        e_round_inc, _ = self.post_round_exp_inc(final_e_pre.flip(-1), 
                                                  torch.cat([zeros, zeros, zeros, m_carry], dim=-1).flip(-1))
        computed_e = []
        for i in range(4):
            e_sel = self.round_exp_mux[i](m_carry, e_round_inc.flip(-1)[..., i:i+1], final_e_pre[..., i:i+1])
            computed_e.append(e_sel)
        computed_e = torch.cat(computed_e, dim=-1)
        
        # ===== Step 8: 符号 =====
        computed_s = self.final_sign_mux(is_diff_sign, s_large, s_a)
        
        # ===== 指数溢出检测 (NaN) =====
        # 当尾数溢出(result_carry=1)且指数加1后也溢出(exp_inc_carry=1)时，输出NaN
        is_exp_overflow = self.exp_overflow_and(result_carry, exp_inc_carry)
        
        # ===== 路径选择 =====
        # 先处理exact_cancel
        cancel_s = self.cancel_mux_s(exact_cancel, zeros, computed_s)
        
        cancel_e = []
        for i in range(4):
            e_sel = self.cancel_mux_e[i](exact_cancel, zeros, computed_e[..., i:i+1])
            cancel_e.append(e_sel)
        cancel_e = torch.cat(cancel_e, dim=-1)
        
        cancel_m2 = self.cancel_mux_m[0](exact_cancel, zeros, m2_final)
        cancel_m1 = self.cancel_mux_m[1](exact_cancel, zeros, m1_final)
        cancel_m0 = self.cancel_mux_m[2](exact_cancel, zeros, m0_final)
        
        # NaN路径选择: 溢出时输出NaN (E=1111, M=111)
        # NaN的符号保持原始符号
        final_s = self.nan_mux_s(is_exp_overflow, computed_s, cancel_s)
        
        final_e = []
        for i in range(4):
            # NaN的指数是1111
            e_sel = self.nan_mux_e[i](is_exp_overflow, ones, cancel_e[..., i:i+1])
            final_e.append(e_sel)
        final_e = torch.cat(final_e, dim=-1)
        
        # NaN的尾数是111
        final_m2 = self.nan_mux_m[0](is_exp_overflow, ones, cancel_m2)
        final_m1 = self.nan_mux_m[1](is_exp_overflow, ones, cancel_m1)
        final_m0 = self.nan_mux_m[2](is_exp_overflow, ones, cancel_m0)
        
        m_out = torch.cat([final_m2, final_m1, final_m0], dim=-1)
        
        return torch.cat([final_s, final_e, m_out], dim=-1)
    
    def reset(self):
        self.exp_cmp.reset()
        self.mantissa_cmp.reset()
        self.exp_sub_ab.reset()
        self.exp_sub_ba.reset()
        for m in self.exp_diff_mux: m.reset()
        self.abs_eq_and.reset()
        self.mant_ge_or.reset()
        self.abs_ge_and.reset()
        self.abs_ge_or.reset()
        self.align_shifter.reset()
        for g in self.e_zero_or: g.reset()
        for g in self.e_zero_not: g.reset()
        for m in self.subnorm_exp_mux_a: m.reset()
        for m in self.subnorm_exp_mux_b: m.reset()
        self.mantissa_adder.reset()
        self.mantissa_sub.reset()
        self.sign_xor.reset()
        self.not_diff_sign.reset()
        self.exact_cancel_and.reset()
        self.swap_mux_s.reset()
        for m in self.swap_mux_e: m.reset()
        for m in self.swap_mux_m: m.reset()
        for m in self.result_mux: m.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.exp_adj_sub.reset()
        for m in self.exp_overflow_mux: m.reset()
        self.post_round_exp_inc.reset()
        self.underflow_cmp.reset()
        self.underflow_or.reset()
        for m in self.underflow_mux_e: m.reset()
        for m in self.underflow_mux_m: m.reset()
        self.subnorm_shifter.reset()
        self.exp_overflow_and.reset()
        self.nan_mux_s.reset()
        for m in self.nan_mux_e: m.reset()
        for m in self.nan_mux_m: m.reset()
        for m in self.m_overflow_mux: m.reset()
        for g in self.sticky_or_overflow: g.reset()
        for g in self.sticky_or_normal: g.reset()
        for g in self.sticky_extra_or: g.reset()
        self.sticky_extra_or_norm.reset()
        self.round_or.reset()
        self.round_and.reset()
        self.round_ha0.reset()
        self.round_ha1.reset()
        self.round_ha2.reset()
        self.not_m_carry.reset()
        for g in self.m_final_and: g.reset()
        for m in self.round_exp_mux: m.reset()
        self.final_sign_mux.reset()
        self.cancel_mux_s.reset()
        for m in self.cancel_mux_e: m.reset()
        for m in self.cancel_mux_m: m.reset()
