"""
空间编码FP8加法器 - 1个时间步完成
100%纯SNN：所有计算通过IF神经元门电路实现，无实数运算

支持统一的神经元模板机制，可在 IF/LIF 之间切换用于物理仿真。

注意：使用向量化组件以保持与 FP16/FP32 组件的一致性。
"""
import torch
import torch.nn as nn
from atomic_ops.core.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecHalfAdder as HalfAdder,
    VecFullAdder as FullAdder,
    VecAdder
)


class Comparator4Bit(nn.Module):
    """4位比较器 - 纯SNN向量化实现

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化门电路 - 一次处理所有4位
        self.vec_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))      # XOR所有位
        self.vec_not_xor = VecNOT(neuron_template=nt, max_param_shape=(1,))  # XNOR = NOT(XOR)
        self.vec_not_b = VecNOT(neuron_template=nt, max_param_shape=(1,))    # NOT(B) 所有位
        self.vec_a_gt_b_and = VecAND(neuron_template=nt, max_param_shape=(1,))  # A AND NOT(B) 所有位

        # 累积相等条件 - 单实例 (动态扩展机制支持复用)
        self.eq_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # 最终结果组合 - 单实例
        self.result_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.result_or = VecOR(neuron_template=nt, max_param_shape=(1,))

        # 检测全部相等 - 单实例
        self.final_eq_and = VecAND(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B):
        """A, B: [..., 4] (MSB first)"""

        xor_all = self.vec_xor(A, B)  # [..., 4]
        eq_all = self.vec_not_xor(xor_all)  # [..., 4] eq[i] = NOT(XOR(A[i], B[i]))
        not_b_all = self.vec_not_b(B)  # [..., 4]
        gt_all = self.vec_a_gt_b_and(A, not_b_all)  # [..., 4] gt[i] = A[i] AND NOT(B[i])

        # 提取各位
        eq3, eq2, eq1, eq0 = eq_all[..., 0:1], eq_all[..., 1:2], eq_all[..., 2:3], eq_all[..., 3:4]
        gt3, gt2, gt1, gt0 = gt_all[..., 0:1], gt_all[..., 1:2], gt_all[..., 2:3], gt_all[..., 3:4]

        # A > B: gt3 OR (eq3 AND gt2) OR (eq3 AND eq2 AND gt1) OR (eq3 AND eq2 AND eq1 AND gt0)
        # 树形结构 - 有依赖关系
        eq32 = self.eq_and(eq3, eq2)
        eq321 = self.eq_and(eq32, eq1)

        term2 = self.result_and(eq3, gt2)
        term3 = self.result_and(eq32, gt1)
        term4 = self.result_and(eq321, gt0)

        t12 = self.result_or(gt3, term2)
        t123 = self.result_or(t12, term3)
        a_gt_b = self.result_or(t123, term4)

        # A == B: 所有位都相等 - 树形结构
        eq_all_1 = self.final_eq_and(eq3, eq2)
        eq_all_2 = self.final_eq_and(eq_all_1, eq1)
        a_eq_b = self.final_eq_and(eq_all_2, eq0)

        return a_gt_b, a_eq_b

    def reset(self):
        self.vec_xor.reset()
        self.vec_not_xor.reset()
        self.vec_not_b.reset()
        self.vec_a_gt_b_and.reset()
        self.eq_and.reset()
        self.result_and.reset()
        self.result_or.reset()
        self.final_eq_and.reset()


class Comparator3Bit(nn.Module):
    """3位比较器 - 纯SNN向量化实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化门电路 - 一次处理所有3位
        self.vec_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.vec_not_xor = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.vec_not_b = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.vec_a_gt_b_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        # 树形结构 - 单实例 (动态扩展机制支持复用)
        self.eq_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.result_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.result_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.final_eq_and = VecAND(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B):
        """A, B: [..., 3] (MSB first)"""

        xor_all = self.vec_xor(A, B)  # [..., 3]
        eq_all = self.vec_not_xor(xor_all)  # [..., 3]
        not_b_all = self.vec_not_b(B)  # [..., 3]
        gt_all = self.vec_a_gt_b_and(A, not_b_all)  # [..., 3]

        # 提取各位
        eq2, eq1, eq0 = eq_all[..., 0:1], eq_all[..., 1:2], eq_all[..., 2:3]
        gt2, gt1, gt0 = gt_all[..., 0:1], gt_all[..., 1:2], gt_all[..., 2:3]

        # 树形结构 - 有依赖
        eq21 = self.eq_and(eq2, eq1)

        term2 = self.result_and(eq2, gt1)
        term3 = self.result_and(eq21, gt0)

        t12 = self.result_or(gt2, term2)
        a_gt_b = self.result_or(t12, term3)

        a_eq_b_t = self.final_eq_and(eq2, eq1)
        a_eq_b = self.final_eq_and(a_eq_b_t, eq0)

        return a_gt_b, a_eq_b

    def reset(self):
        self.vec_xor.reset()
        self.vec_not_xor.reset()
        self.vec_not_b.reset()
        self.vec_a_gt_b_and.reset()
        self.eq_and.reset()
        self.result_and.reset()
        self.result_or.reset()
        self.final_eq_and.reset()


class Subtractor4Bit(nn.Module):
    """4位减法器 - 纯SNN实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 (动态扩展机制支持复用)
        self.xor1 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.xor2 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.not_a = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.and1 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and2 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and3 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.or1 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or2 = VecOR(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B, Bin=None):
        """A - B"""
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin

        diffs = []
        for i in [3, 2, 1, 0]:
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]

            t1 = self.xor1(a_i, b_i)
            diff = self.xor2(t1, borrow)

            not_a_i = self.not_a(a_i)
            term1 = self.and1(not_a_i, b_i)
            term2 = self.and2(not_a_i, borrow)
            term3 = self.and3(b_i, borrow)
            t12 = self.or1(term1, term2)
            new_borrow = self.or2(t12, term3)

            diffs.append((i, diff))
            borrow = new_borrow

        diffs.sort(key=lambda x: x[0])
        result = torch.cat([d[1] for d in diffs], dim=-1)

        return result, borrow

    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.not_a.reset()
        self.and1.reset()
        self.and2.reset()
        self.and3.reset()
        self.or1.reset()
        self.or2.reset()


class Subtractor8Bit(nn.Module):
    """8位减法器 - 纯SNN实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 (动态扩展机制支持复用)
        self.xor1 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.xor2 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.not_a = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.and1 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and2 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and3 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.or1 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or2 = VecOR(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B, Bin=None):
        """A - B"""
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin

        diffs = []
        for i in [7, 6, 5, 4, 3, 2, 1, 0]:
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]

            t1 = self.xor1(a_i, b_i)
            diff = self.xor2(t1, borrow)

            not_a_i = self.not_a(a_i)
            term1 = self.and1(not_a_i, b_i)
            term2 = self.and2(not_a_i, borrow)
            term3 = self.and3(b_i, borrow)
            t12 = self.or1(term1, term2)
            new_borrow = self.or2(t12, term3)

            diffs.append((i, diff))
            borrow = new_borrow

        diffs.sort(key=lambda x: x[0])
        return torch.cat([d[1] for d in diffs], dim=-1), borrow

    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.not_a.reset()
        self.and1.reset()
        self.and2.reset()
        self.and3.reset()
        self.or1.reset()
        self.or2.reset()


class BarrelShifterRight8(nn.Module):
    """8位桶形右移位器 - 纯SNN向量化实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化MUX - 每层一个MUX处理所有8位
        self.vec_mux_layer0 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 1
        self.vec_mux_layer1 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 2
        self.vec_mux_layer2 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 4
        self.vec_mux_layer3 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 8

    def forward(self, X, shift):
        """X: [..., 8], shift: [..., 4]"""

        s0 = shift[..., 3:4]  # shift by 1
        s1 = shift[..., 2:3]  # shift by 2
        s2 = shift[..., 1:2]  # shift by 4
        s3 = shift[..., 0:1]  # shift by 8

        zeros = torch.zeros_like(X[..., 0:1])

        # Layer 0: shift by 1 - 向量化
        # shifted = [0, X0, X1, X2, X3, X4, X5, X6]
        # original = [X0, X1, X2, X3, X4, X5, X6, X7]
        shifted0 = torch.cat([zeros, X[..., :7]], dim=-1)
        s0_expanded = s0.expand_as(X)
        x0 = self.vec_mux_layer0(s0_expanded, shifted0, X)

        # Layer 1: shift by 2 - 向量化
        # shifted = [0, 0, x0_0, x0_1, x0_2, x0_3, x0_4, x0_5]
        shifted1 = torch.cat([zeros, zeros, x0[..., :6]], dim=-1)
        s1_expanded = s1.expand_as(x0)
        x1 = self.vec_mux_layer1(s1_expanded, shifted1, x0)

        # Layer 2: shift by 4 - 向量化
        # shifted = [0, 0, 0, 0, x1_0, x1_1, x1_2, x1_3]
        shifted2 = torch.cat([zeros, zeros, zeros, zeros, x1[..., :4]], dim=-1)
        s2_expanded = s2.expand_as(x1)
        x2 = self.vec_mux_layer2(s2_expanded, shifted2, x1)

        # Layer 3: shift by 8 - 向量化
        # shifted = [0, 0, 0, 0, 0, 0, 0, 0] (全零)
        zeros_8 = torch.zeros_like(X)
        s3_expanded = s3.expand_as(x2)
        result = self.vec_mux_layer3(s3_expanded, zeros_8, x2)

        return result

    def reset(self):
        self.vec_mux_layer0.reset()
        self.vec_mux_layer1.reset()
        self.vec_mux_layer2.reset()
        self.vec_mux_layer3.reset()


class BarrelShifterRight12(nn.Module):
    """12位桶形右移位器 - 纯SNN向量化实现 (用于尾数对齐)"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化MUX - 每层一个MUX处理所有12位
        self.vec_mux_layer0 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 1
        self.vec_mux_layer1 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 2
        self.vec_mux_layer2 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 4
        self.vec_mux_layer3 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 8

    def forward(self, X, shift):
        """X: [..., 12], shift: [..., 4]"""

        s0 = shift[..., 3:4]  # shift by 1
        s1 = shift[..., 2:3]  # shift by 2
        s2 = shift[..., 1:2]  # shift by 4
        s3 = shift[..., 0:1]  # shift by 8

        zeros = torch.zeros_like(X[..., 0:1])

        # Layer 0: shift by 1 - 向量化
        shifted0 = torch.cat([zeros, X[..., :11]], dim=-1)
        s0_expanded = s0.expand_as(X)
        x0 = self.vec_mux_layer0(s0_expanded, shifted0, X)

        # Layer 1: shift by 2 - 向量化
        shifted1 = torch.cat([zeros, zeros, x0[..., :10]], dim=-1)
        s1_expanded = s1.expand_as(x0)
        x1 = self.vec_mux_layer1(s1_expanded, shifted1, x0)

        # Layer 2: shift by 4 - 向量化
        shifted2 = torch.cat([zeros, zeros, zeros, zeros, x1[..., :8]], dim=-1)
        s2_expanded = s2.expand_as(x1)
        x2 = self.vec_mux_layer2(s2_expanded, shifted2, x1)

        # Layer 3: shift by 8 - 向量化
        shifted3 = torch.cat([zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, x2[..., :4]], dim=-1)
        s3_expanded = s3.expand_as(x2)
        result = self.vec_mux_layer3(s3_expanded, shifted3, x2)

        return result

    def reset(self):
        self.vec_mux_layer0.reset()
        self.vec_mux_layer1.reset()
        self.vec_mux_layer2.reset()
        self.vec_mux_layer3.reset()


class Subtractor12Bit(nn.Module):
    """12位减法器 - 纯SNN"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 (动态扩展机制支持不同位宽)
        self.xor1 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.xor2 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.not_a = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.and1 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and2 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and3 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.or1 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or2 = VecOR(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B, Bin=None):
        """A - B"""
        if Bin is None:
            borrow = torch.zeros_like(A[..., 0:1])
        else:
            borrow = Bin
            
        diffs = []
        for i in range(11, -1, -1):  # LSB first
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            t1 = self.xor1(a_i, b_i)
            diff = self.xor2(t1, borrow)

            not_a_i = self.not_a(a_i)
            term1 = self.and1(not_a_i, b_i)
            term2 = self.and2(not_a_i, borrow)
            term3 = self.and3(b_i, borrow)
            t12 = self.or1(term1, term2)
            new_borrow = self.or2(t12, term3)
            
            diffs.append((i, diff))
            borrow = new_borrow
        
        diffs.sort(key=lambda x: x[0])
        return torch.cat([d[1] for d in diffs], dim=-1), borrow
    
    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.not_a.reset()
        self.and1.reset()
        self.and2.reset()
        self.and3.reset()
        self.or1.reset()
        self.or2.reset()


class Adder12Bit(nn.Module):
    """12位加法器 - 纯SNN"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 (动态扩展机制支持不同位宽)
        self.xor1 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.xor2 = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.and1 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and2 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.and3 = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.or1 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or2 = VecOR(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B, Cin=None):
        """A + B"""
        if Cin is None:
            carry = torch.zeros_like(A[..., 0:1])
        else:
            carry = Cin
            
        sums = []
        for i in range(11, -1, -1):  # LSB first
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            
            t1 = self.xor1(a_i, b_i)
            s = self.xor2(t1, carry)

            term1 = self.and1(a_i, b_i)
            term2 = self.and2(a_i, carry)
            term3 = self.and3(b_i, carry)
            t12 = self.or1(term1, term2)
            new_carry = self.or2(t12, term3)
            
            sums.append((i, s))
            carry = new_carry
        
        sums.sort(key=lambda x: x[0])
        return torch.cat([s[1] for s in sums], dim=-1), carry
    
    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.and3.reset()
        self.or1.reset()
        self.or2.reset()


class BarrelShifterLeft8(nn.Module):
    """8位桶形左移位器 - 纯SNN向量化实现"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 向量化MUX - 每层一个MUX处理所有8位
        self.vec_mux_layer0 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 1
        self.vec_mux_layer1 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 2
        self.vec_mux_layer2 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # shift by 4

    def forward(self, X, shift):
        """X: [..., 8], shift: [..., 3]"""

        s0 = shift[..., 2:3]  # shift by 1
        s1 = shift[..., 1:2]  # shift by 2
        s2 = shift[..., 0:1]  # shift by 4

        zeros = torch.zeros_like(X[..., 0:1])

        # Layer 0: shift by 1 (left) - 向量化
        # shifted = [X1, X2, X3, X4, X5, X6, X7, 0]
        shifted0 = torch.cat([X[..., 1:], zeros], dim=-1)
        s0_expanded = s0.expand_as(X)
        x0 = self.vec_mux_layer0(s0_expanded, shifted0, X)

        # Layer 1: shift by 2 (left) - 向量化
        # shifted = [x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, 0, 0]
        shifted1 = torch.cat([x0[..., 2:], zeros, zeros], dim=-1)
        s1_expanded = s1.expand_as(x0)
        x1 = self.vec_mux_layer1(s1_expanded, shifted1, x0)

        # Layer 2: shift by 4 (left) - 向量化
        # shifted = [x1_4, x1_5, x1_6, x1_7, 0, 0, 0, 0]
        shifted2 = torch.cat([x1[..., 4:], zeros, zeros, zeros, zeros], dim=-1)
        s2_expanded = s2.expand_as(x1)
        result = self.vec_mux_layer2(s2_expanded, shifted2, x1)

        return result

    def reset(self):
        self.vec_mux_layer0.reset()
        self.vec_mux_layer1.reset()
        self.vec_mux_layer2.reset()


class LeadingZeroDetector8(nn.Module):
    """8位前导零检测器 - 纯SNN"""
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        self.or_67 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_45 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_23 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_01 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_7654 = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.or_3210 = VecOR(neuron_template=nt, max_param_shape=(1,))
        
        # NOT门用于计算LZC
        self.not_7654 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_76 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_32 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_b7 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_b5 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_b3 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.not_b1 = VecNOT(neuron_template=nt, max_param_shape=(1,))
        
        # MUX门
        self.mux_lzc1 = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_lzc0_high = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_lzc0_low = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.mux_lzc0 = VecMUX(neuron_template=nt, max_param_shape=(1,))
        
    def forward(self, X):
        """X: [..., 8] (MSB at 0), 返回: [..., 3] 前导零计数"""
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
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 (动态扩展机制支持不同位宽)
        self.adder = FullAdder(neuron_template=nt, max_param_shape=(1,))

    def forward(self, A, B, Cin=None):
        """A, B: [..., 8] (MSB first)"""
        if Cin is None:
            carry = torch.zeros_like(A[..., 0:1])
        else:
            carry = Cin

        sums = []
        for i in [7, 6, 5, 4, 3, 2, 1, 0]:
            s, carry = self.adder(A[..., i:i+1], B[..., i:i+1], carry)
            sums.append((i, s))

        sums.sort(key=lambda x: x[0])
        return torch.cat([s[1] for s in sums], dim=-1), carry

    def reset(self):
        self.adder.reset()


class SpikeFP8Adder_Spatial(nn.Module):
    """空间编码FP8加法器 - 100%纯SNN实现
    
    所有计算通过IF神经元门电路实现，无任何实数域运算
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # ===== 指数比较与差值 =====
        self.exp_cmp = Comparator4Bit(neuron_template=nt)
        self.exp_sub_ab = Subtractor4Bit(neuron_template=nt)
        self.exp_sub_ba = Subtractor4Bit(neuron_template=nt)
        self.vec_exp_diff_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 向量化: 4位

        # ===== 尾数比较（含隐藏位，4位）=====
        self.mantissa_cmp = Comparator4Bit(neuron_template=nt)

        # ===== 绝对值比较组合逻辑 =====
        self.abs_eq_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.mant_ge_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.abs_ge_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.abs_ge_or = VecOR(neuron_template=nt, max_param_shape=(1,))

        # ===== 尾数对齐 =====
        self.align_shifter = BarrelShifterRight12(neuron_template=nt)

        # ===== 零检测（树形结构 - 有依赖）=====
        # 单实例 (动态扩展机制支持不同位宽)
        self.e_zero_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.e_zero_not = VecNOT(neuron_template=nt, max_param_shape=(1,))

        # ===== Subnormal指数修正 - 向量化 =====
        self.vec_subnorm_exp_mux_a = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位
        self.vec_subnorm_exp_mux_b = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位

        # ===== 尾数加法/减法 =====
        self.mantissa_adder = Adder12Bit(neuron_template=nt)
        self.mantissa_sub = Subtractor12Bit(neuron_template=nt)

        # ===== 符号处理 =====
        self.sign_xor = VecXOR(neuron_template=nt, max_param_shape=(1,))
        self.not_diff_sign = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.exact_cancel_and = VecAND(neuron_template=nt, max_param_shape=(1,))

        # ===== 操作数选择MUX - 向量化 =====
        self.swap_mux_s = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.vec_swap_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位
        self.vec_swap_mux_m_large = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 12位
        self.vec_swap_mux_m_small = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 12位

        # ===== 结果选择 - 向量化 =====
        self.vec_result_mux_12 = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 12位尾数
        self.result_mux_carry = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 1位carry

        # ===== 归一化 =====
        self.lzd = LeadingZeroDetector8(neuron_template=nt)
        self.norm_shifter = BarrelShifterLeft8(neuron_template=nt)
        self.exp_adj_sub = Subtractor4Bit(neuron_template=nt)

        # ===== 溢出处理 - 向量化 =====
        self.vec_exp_overflow_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位
        self.post_round_exp_inc = VecAdder(bits=4, neuron_template=nt, max_param_shape=(4,))

        # ===== 指数下溢检测 =====
        self.underflow_cmp = Comparator4Bit(neuron_template=nt)
        self.underflow_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.vec_underflow_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位
        self.vec_underflow_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 3位
        self.subnorm_shifter = BarrelShifterLeft8(neuron_template=nt)

        # ===== 指数溢出检测（NaN）=====
        self.exp_overflow_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.nan_mux_s = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.vec_nan_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位
        self.vec_nan_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 3位

        # ===== 尾数选择 - 向量化 =====
        self.vec_m_overflow_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 5位: m2, m1, m0, round, sticky

        # ===== Sticky OR (树形结构 - 有依赖) =====
        # 单实例 (动态扩展机制支持不同位宽)
        self.sticky_or_overflow = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_or_normal = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_extra_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.sticky_extra_or_norm = VecOR(neuron_template=nt, max_param_shape=(1,))

        # ===== 舍入 =====
        self.round_or = VecOR(neuron_template=nt, max_param_shape=(1,))
        self.round_and = VecAND(neuron_template=nt, max_param_shape=(1,))
        self.round_ha0 = HalfAdder(neuron_template=nt, max_param_shape=(1,))
        self.round_ha1 = HalfAdder(neuron_template=nt, max_param_shape=(1,))
        self.round_ha2 = HalfAdder(neuron_template=nt, max_param_shape=(1,))

        # ===== 舍入溢出处理 - 向量化 =====
        self.not_m_carry = VecNOT(neuron_template=nt, max_param_shape=(1,))
        self.vec_m_final_and = VecAND(neuron_template=nt, max_param_shape=(1,))  # 3位
        self.vec_round_exp_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位

        # ===== 符号结果选择 =====
        self.final_sign_mux = VecMUX(neuron_template=nt, max_param_shape=(1,))

        # ===== 完全抵消路径选择 - 向量化 =====
        self.cancel_mux_s = VecMUX(neuron_template=nt, max_param_shape=(1,))
        self.vec_cancel_mux_e = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 4位
        self.vec_cancel_mux_m = VecMUX(neuron_template=nt, max_param_shape=(1,))  # 3位
        
    def forward(self, A, B):
        """纯SNN FP8加法 - 向量化实现"""

        s_a, e_a, m_a_raw = A[..., 0:1], A[..., 1:5], A[..., 5:8]
        s_b, e_b, m_b_raw = B[..., 0:1], B[..., 1:5], B[..., 5:8]

        zeros = torch.zeros_like(s_a)
        ones = torch.ones_like(s_a)
        zeros_4 = zeros.expand_as(e_a)
        ones_4 = ones.expand_as(e_a)

        # ===== 隐藏位检测与指数修正 (树形结构 - 有依赖) =====
        e_a_or_01 = self.e_zero_or(e_a[..., 0:1], e_a[..., 1:2])
        e_a_or_23 = self.e_zero_or(e_a[..., 2:3], e_a[..., 3:4])
        e_a_nonzero = self.e_zero_or(e_a_or_01, e_a_or_23)
        e_a_is_zero = self.e_zero_not(e_a_nonzero)
        hidden_a = e_a_nonzero

        e_b_or_01 = self.e_zero_or(e_b[..., 0:1], e_b[..., 1:2])
        e_b_or_23 = self.e_zero_or(e_b[..., 2:3], e_b[..., 3:4])
        e_b_nonzero = self.e_zero_or(e_b_or_01, e_b_or_23)
        e_b_is_zero = self.e_zero_not(e_b_nonzero)
        hidden_b = e_b_nonzero

        # Subnormal指数修正 - 向量化
        # 当E=0时，使用E=1: [0,0,0,1]
        subnorm_val_a = torch.cat([zeros, zeros, zeros, ones], dim=-1)
        e_a_is_zero_4 = e_a_is_zero.expand_as(e_a)
        e_a_eff = self.vec_subnorm_exp_mux_a(e_a_is_zero_4, subnorm_val_a, e_a)

        subnorm_val_b = torch.cat([zeros, zeros, zeros, ones], dim=-1)
        e_b_is_zero_4 = e_b_is_zero.expand_as(e_b)
        e_b_eff = self.vec_subnorm_exp_mux_b(e_b_is_zero_4, subnorm_val_b, e_b)

        # 12位尾数
        zeros_8 = torch.cat([zeros]*8, dim=-1)
        m_a = torch.cat([hidden_a, m_a_raw, zeros_8], dim=-1)
        m_b = torch.cat([hidden_b, m_b_raw, zeros_8], dim=-1)

        # ===== Step 1: 比较绝对值 =====
        a_exp_gt, a_exp_eq = self.exp_cmp(e_a_eff, e_b_eff)
        m_a_with_hidden = torch.cat([hidden_a, m_a_raw], dim=-1)
        m_b_with_hidden = torch.cat([hidden_b, m_b_raw], dim=-1)
        a_mant_gt, a_mant_eq = self.mantissa_cmp(m_a_with_hidden, m_b_with_hidden)

        a_abs_eq_b = self.abs_eq_and(a_exp_eq, a_mant_eq)
        a_mant_ge = self.mant_ge_or(a_mant_gt, a_mant_eq)
        exp_eq_and_mant_ge = self.abs_ge_and(a_exp_eq, a_mant_ge)
        a_ge_b = self.abs_ge_or(a_exp_gt, exp_eq_and_mant_ge)

        # ===== Step 2: 指数差 - 向量化 =====
        diff_ab, _ = self.exp_sub_ab(e_a_eff, e_b_eff)
        diff_ba, _ = self.exp_sub_ba(e_b_eff, e_a_eff)

        a_ge_b_4 = a_ge_b.expand_as(diff_ab)
        exp_diff = self.vec_exp_diff_mux(a_ge_b_4, diff_ab, diff_ba)

        # e_max - 向量化
        e_max = self.vec_swap_mux_e(a_ge_b_4, e_a_eff, e_b_eff)

        # ===== Step 3: 尾数对齐 - 向量化 =====
        a_ge_b_12 = a_ge_b.expand_as(m_a)
        m_large = self.vec_swap_mux_m_large(a_ge_b_12, m_a, m_b)
        m_small_unshifted = self.vec_swap_mux_m_small(a_ge_b_12, m_b, m_a)

        m_small = self.align_shifter(m_small_unshifted, exp_diff)

        # ===== Step 4: 符号处理 =====
        is_diff_sign = self.sign_xor(s_a, s_b)
        exact_cancel = self.exact_cancel_and(is_diff_sign, a_abs_eq_b)
        s_large = self.swap_mux_s(a_ge_b, s_a, s_b)

        # ===== Step 5: 尾数运算 - 向量化 =====
        sum_result, sum_carry = self.mantissa_adder(m_large, m_small)
        diff_result, _ = self.mantissa_sub(m_large, m_small)

        is_diff_sign_12 = is_diff_sign.expand_as(sum_result)
        mantissa_result_12 = self.vec_result_mux_12(is_diff_sign_12, diff_result, sum_result)

        result_carry = self.result_mux_carry(is_diff_sign, zeros, sum_carry)

        mantissa_result = mantissa_result_12[..., :8]
        extra_bits = mantissa_result_12[..., 8:12]

        # ===== Step 6: 归一化 =====
        lzc = self.lzd(mantissa_result)
        lzc_4bit = torch.cat([zeros, lzc], dim=-1)

        lzc_gt_emax, lzc_eq_emax = self.underflow_cmp(lzc_4bit, e_max)
        is_underflow = self.underflow_or(lzc_gt_emax, lzc_eq_emax)

        norm_mantissa = self.norm_shifter(mantissa_result, lzc)
        e_after_norm, _ = self.exp_adj_sub(e_max, lzc_4bit)

        e_max_3bit = e_max[..., 1:4]
        subnorm_mantissa = self.subnorm_shifter(mantissa_result, e_max_3bit)

        one_4bit = torch.cat([zeros, zeros, zeros, ones], dim=-1)
        e_inc_le, exp_inc_carry = self.post_round_exp_inc(e_max.flip(-1), one_4bit.flip(-1))
        e_plus_one = e_inc_le.flip(-1)

        # 选择指数 - 向量化
        is_underflow_4 = is_underflow.expand_as(e_after_norm)
        e_normal = self.vec_underflow_mux_e(is_underflow_4, zeros_4, e_after_norm)

        result_carry_4 = result_carry.expand_as(e_normal)
        final_e_pre = self.vec_exp_overflow_mux(result_carry_4, e_plus_one, e_normal)

        # ===== Step 7: 提取尾数并舍入 =====
        m2_overflow = mantissa_result[..., 0:1]
        m1_overflow = mantissa_result[..., 1:2]
        m0_overflow = mantissa_result[..., 2:3]
        round_overflow = mantissa_result[..., 3:4]
        # sticky - 树形结构
        sticky_ov_t1 = self.sticky_or_overflow(mantissa_result[..., 4:5], mantissa_result[..., 5:6])
        sticky_ov_t2 = self.sticky_or_overflow(sticky_ov_t1, mantissa_result[..., 6:7])
        sticky_ov_t3 = self.sticky_or_overflow(sticky_ov_t2, mantissa_result[..., 7:8])
        extra_or_01 = self.sticky_extra_or(extra_bits[..., 0:1], extra_bits[..., 1:2])
        extra_or_23 = self.sticky_extra_or(extra_bits[..., 2:3], extra_bits[..., 3:4])
        extra_or_all = self.sticky_extra_or(extra_or_01, extra_or_23)
        sticky_overflow = self.sticky_extra_or(sticky_ov_t3, extra_or_all)

        m2_norm = norm_mantissa[..., 1:2]
        m1_norm = norm_mantissa[..., 2:3]
        m0_norm = norm_mantissa[..., 3:4]
        round_norm = norm_mantissa[..., 4:5]
        sticky_nm_t = self.sticky_or_normal(norm_mantissa[..., 5:6], norm_mantissa[..., 6:7])
        sticky_nm_t2 = self.sticky_or_normal(sticky_nm_t, norm_mantissa[..., 7:8])
        sticky_norm = self.sticky_extra_or_norm(sticky_nm_t2, extra_or_all)

        m2_subnorm = subnorm_mantissa[..., 0:1]
        m1_subnorm = subnorm_mantissa[..., 1:2]
        m0_subnorm = subnorm_mantissa[..., 2:3]

        # 选择尾数 - 向量化
        m_norm_3 = torch.cat([m2_norm, m1_norm, m0_norm], dim=-1)
        m_subnorm_3 = torch.cat([m2_subnorm, m1_subnorm, m0_subnorm], dim=-1)
        is_underflow_3 = is_underflow.expand_as(m_norm_3)
        m_normal_3 = self.vec_underflow_mux_m(is_underflow_3, m_subnorm_3, m_norm_3)

        m_overflow_5 = torch.cat([m2_overflow, m1_overflow, m0_overflow, round_overflow, sticky_overflow], dim=-1)
        m_normal_5 = torch.cat([m_normal_3, round_norm, sticky_norm], dim=-1)
        result_carry_5 = result_carry.expand_as(m_overflow_5)
        m_selected = self.vec_m_overflow_mux(result_carry_5, m_overflow_5, m_normal_5)

        m2, m1, m0 = m_selected[..., 0:1], m_selected[..., 1:2], m_selected[..., 2:3]
        round_bit, sticky = m_selected[..., 3:4], m_selected[..., 4:5]

        # RNE舍入
        sticky_or_m0 = self.round_or(sticky, m0)
        do_round = self.round_and(round_bit, sticky_or_m0)

        m0_r, c0 = self.round_ha0(m0, do_round)
        m1_r, c1 = self.round_ha1(m1, c0)
        m2_r, m_carry = self.round_ha2(m2, c1)

        # 舍入溢出 - 向量化
        not_mc = self.not_m_carry(m_carry)
        m_r_3 = torch.cat([m2_r, m1_r, m0_r], dim=-1)
        not_mc_3 = not_mc.expand_as(m_r_3)
        m_final_3 = self.vec_m_final_and(m_r_3, not_mc_3)
        m2_final, m1_final, m0_final = m_final_3[..., 0:1], m_final_3[..., 1:2], m_final_3[..., 2:3]

        e_round_inc, _ = self.post_round_exp_inc(final_e_pre.flip(-1),
                                                  torch.cat([zeros, zeros, zeros, m_carry], dim=-1).flip(-1))
        m_carry_4 = m_carry.expand_as(final_e_pre)
        computed_e = self.vec_round_exp_mux(m_carry_4, e_round_inc.flip(-1), final_e_pre)

        # ===== Step 8: 符号 =====
        computed_s = self.final_sign_mux(is_diff_sign, s_large, s_a)

        # 指数溢出检测
        is_exp_overflow = self.exp_overflow_and(result_carry, exp_inc_carry)

        # 路径选择 - 向量化
        cancel_s = self.cancel_mux_s(exact_cancel, zeros, computed_s)

        exact_cancel_4 = exact_cancel.expand_as(computed_e)
        cancel_e = self.vec_cancel_mux_e(exact_cancel_4, zeros_4, computed_e)

        exact_cancel_3 = exact_cancel.expand_as(m_final_3)
        zeros_3 = torch.cat([zeros, zeros, zeros], dim=-1)
        cancel_m = self.vec_cancel_mux_m(exact_cancel_3, zeros_3, m_final_3)

        # NaN路径选择 - 向量化
        final_s = self.nan_mux_s(is_exp_overflow, computed_s, cancel_s)

        is_exp_overflow_4 = is_exp_overflow.expand_as(cancel_e)
        final_e = self.vec_nan_mux_e(is_exp_overflow_4, ones_4, cancel_e)

        ones_3 = torch.cat([ones, ones, ones], dim=-1)
        is_exp_overflow_3 = is_exp_overflow.expand_as(cancel_m)
        final_m = self.vec_nan_mux_m(is_exp_overflow_3, ones_3, cancel_m)

        return torch.cat([final_s, final_e, final_m], dim=-1)

    def reset(self):
        self.exp_cmp.reset()
        self.mantissa_cmp.reset()
        self.exp_sub_ab.reset()
        self.exp_sub_ba.reset()
        self.vec_exp_diff_mux.reset()
        self.abs_eq_and.reset()
        self.mant_ge_or.reset()
        self.abs_ge_and.reset()
        self.abs_ge_or.reset()
        self.align_shifter.reset()
        self.e_zero_or.reset()
        self.e_zero_not.reset()
        self.vec_subnorm_exp_mux_a.reset()
        self.vec_subnorm_exp_mux_b.reset()
        self.mantissa_adder.reset()
        self.mantissa_sub.reset()
        self.sign_xor.reset()
        self.not_diff_sign.reset()
        self.exact_cancel_and.reset()
        self.swap_mux_s.reset()
        self.vec_swap_mux_e.reset()
        self.vec_swap_mux_m_large.reset()
        self.vec_swap_mux_m_small.reset()
        self.vec_result_mux_12.reset()
        self.result_mux_carry.reset()
        self.lzd.reset()
        self.norm_shifter.reset()
        self.exp_adj_sub.reset()
        self.vec_exp_overflow_mux.reset()
        self.post_round_exp_inc.reset()
        self.underflow_cmp.reset()
        self.underflow_or.reset()
        self.vec_underflow_mux_e.reset()
        self.vec_underflow_mux_m.reset()
        self.subnorm_shifter.reset()
        self.exp_overflow_and.reset()
        self.nan_mux_s.reset()
        self.vec_nan_mux_e.reset()
        self.vec_nan_mux_m.reset()
        self.vec_m_overflow_mux.reset()
        self.sticky_or_overflow.reset()
        self.sticky_or_normal.reset()
        self.sticky_extra_or.reset()
        self.sticky_extra_or_norm.reset()
        self.round_or.reset()
        self.round_and.reset()
        self.round_ha0.reset()
        self.round_ha1.reset()
        self.round_ha2.reset()
        self.not_m_carry.reset()
        self.vec_m_final_and.reset()
        self.vec_round_exp_mux.reset()
        self.final_sign_mux.reset()
        self.cancel_mux_s.reset()
        self.vec_cancel_mux_e.reset()
        self.vec_cancel_mux_m.reset()
