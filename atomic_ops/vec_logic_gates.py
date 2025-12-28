"""
向量化SNN逻辑门电路库 (Vectorized SNN Logic Gates Library)
==============================================================

所有门电路使用单个神经元实例处理任意形状的张量输入，实现真正的向量化。
支持统一的神经元模板机制，可在 IF/LIF 之间切换用于物理仿真。

核心原则：
1. 批次向量化：所有batch样本同时处理
2. 位级向量化：可并行的操作（如 P = A XOR B）一次处理所有位
3. 门电路复用：只创建少量门电路实例，通过复用减少内存

组件列表：
- VecAND, VecOR, VecXOR, VecNOT, VecMUX: 基础向量化门
- VecORTree, VecANDTree: 并行归约树
- VecFullAdder, VecHalfAdder: 向量化加法器单元
- VecAdder, VecSubtractor: 向量化多位加减法器

使用示例:
```python
from SNNTorch.atomic_ops.vec_logic_gates import VecAdder
from SNNTorch.atomic_ops.logic_gates import SimpleLIFNode

# 默认 IF 神经元 (理想数字逻辑)
adder = VecAdder(bits=8)

# LIF 神经元 (物理仿真)
lif_template = SimpleLIFNode(beta=0.9)
adder_lif = VecAdder(bits=8, neuron_template=lif_template)
```

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import neuron, surrogate

# 从 logic_gates 导入 SimpleLIFNode 和辅助函数
from .logic_gates import SimpleLIFNode, _create_neuron


# ==============================================================================
# 基础向量化逻辑门
# ==============================================================================

class VecAND(nn.Module):
    """向量化AND门 - 一个神经元处理任意形状输入
    
    数学原理: 阈值1.5，只有当两个输入都为1时 (V=2>1.5) 才发放
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.node = _create_neuron(neuron_template, threshold=1.5)
    
    def forward(self, a, b):
        self.node.reset()
        return self.node(a + b)
    
    def reset(self):
        self.node.reset()


class VecOR(nn.Module):
    """向量化OR门 - 一个神经元处理任意形状输入
    
    数学原理: 阈值0.5，任一输入为1时 (V>=1>0.5) 就发放
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.node = _create_neuron(neuron_template, threshold=0.5)
    
    def forward(self, a, b):
        self.node.reset()
        return self.node(a + b)
    
    def reset(self):
        self.node.reset()


class VecNOT(nn.Module):
    """向量化NOT门 - 一个神经元处理任意形状输入
    
    数学原理: 偏置1.0 + 抑制性权重-1.0，输入0时V=1>0.5发放，输入1时V=0不发放
    注意: 1.0-x 是突触电流计算，不是逻辑运算
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.node = _create_neuron(neuron_template, threshold=0.5)
    
    def forward(self, x):
        self.node.reset()
        return self.node(1.0 - x)  # 突触电流 = bias(1) + weight(-1)*x
    
    def reset(self):
        self.node.reset()


class VecXOR(nn.Module):
    """向量化XOR门 - 两层神经元
    
    数学原理: XOR = (A + B) - 2*AND(A,B)
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.hidden = _create_neuron(neuron_template, threshold=1.5)
        self.output = _create_neuron(neuron_template, threshold=0.5)
    
    def forward(self, a, b):
        self.hidden.reset()
        self.output.reset()
        h = self.hidden(a + b)  # AND(a, b)
        return self.output(a + b - 2.0 * h)  # XOR = a+b-2*AND
    
    def reset(self):
        self.hidden.reset()
        self.output.reset()


class VecMUX(nn.Module):
    """向量化MUX门 - sel=1选a, sel=0选b
    
    MUX(s, a, b) = (s AND a) OR (NOT(s) AND b)
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.not_s = VecNOT(neuron_template)
        self.and1 = VecAND(neuron_template)  # s AND a
        self.and2 = VecAND(neuron_template)  # NOT(s) AND b
        self.or_gate = VecOR(neuron_template)
    
    def forward(self, sel, a, b):
        ns = self.not_s(sel)
        sa = self.and1(sel, a)
        nsb = self.and2(ns, b)
        return self.or_gate(sa, nsb)
    
    def reset(self):
        self.not_s.reset()
        self.and1.reset()
        self.and2.reset()
        self.or_gate.reset()


# ==============================================================================
# 并行归约树
# ==============================================================================

class VecORTree(nn.Module):
    """并行OR树 - 检测任意位是否为1
    
    使用log2(n)层OR门并行归约，最多支持1024位输入
    
    Args:
        max_layers: 最大层数
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, max_layers=10, neuron_template=None):
        super().__init__()
        self.or_gates = nn.ModuleList([VecOR(neuron_template) for _ in range(max_layers)])
    
    def forward(self, x):
        """
        x: [..., N] 输入位
        返回: [..., 1] 任意位为1则输出1
        """
        current = x
        layer = 0
        while current.shape[-1] > 1:
            n = current.shape[-1]
            if n % 2 == 1:
                pairs = current[..., :-1]
                last = current[..., -1:]
                left = pairs[..., 0::2]
                right = pairs[..., 1::2]
                paired = self.or_gates[layer](left, right)
                current = torch.cat([paired, last], dim=-1)
            else:
                left = current[..., 0::2]
                right = current[..., 1::2]
                current = self.or_gates[layer](left, right)
            layer += 1
        return current
    
    def reset(self):
        for g in self.or_gates:
            g.reset()


class VecANDTree(nn.Module):
    """并行AND树 - 检测所有位是否为1
    
    使用log2(n)层AND门并行归约，最多支持1024位输入
    
    Args:
        max_layers: 最大层数
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, max_layers=10, neuron_template=None):
        super().__init__()
        self.and_gates = nn.ModuleList([VecAND(neuron_template) for _ in range(max_layers)])
    
    def forward(self, x):
        """
        x: [..., N] 输入位
        返回: [..., 1] 所有位为1则输出1
        """
        current = x
        layer = 0
        while current.shape[-1] > 1:
            n = current.shape[-1]
            if n % 2 == 1:
                pairs = current[..., :-1]
                last = current[..., -1:]
                left = pairs[..., 0::2]
                right = pairs[..., 1::2]
                paired = self.and_gates[layer](left, right)
                current = torch.cat([paired, last], dim=-1)
            else:
                left = current[..., 0::2]
                right = current[..., 1::2]
                current = self.and_gates[layer](left, right)
            layer += 1
        return current
    
    def reset(self):
        for g in self.and_gates:
            g.reset()


# ==============================================================================
# 向量化算术单元
# ==============================================================================

class VecHalfAdder(nn.Module):
    """向量化半加器 - 处理任意形状输入
    
    S = A XOR B
    C = A AND B
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.xor_gate = VecXOR(neuron_template)
        self.and_gate = VecAND(neuron_template)
    
    def forward(self, a, b):
        s = self.xor_gate(a, b)
        c = self.and_gate(a, b)
        return s, c
    
    def reset(self):
        self.xor_gate.reset()
        self.and_gate.reset()


class VecFullAdder(nn.Module):
    """向量化全加器 - 处理任意形状输入
    
    S = A XOR B XOR Cin
    Cout = (A AND B) OR ((A XOR B) AND Cin)
    
    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.xor1 = VecXOR(neuron_template)
        self.xor2 = VecXOR(neuron_template)
        self.and1 = VecAND(neuron_template)
        self.and2 = VecAND(neuron_template)
        self.or1 = VecOR(neuron_template)
    
    def forward(self, a, b, cin):
        p = self.xor1(a, b)         # P = A XOR B
        s = self.xor2(p, cin)       # S = P XOR Cin
        g = self.and1(a, b)         # G = A AND B
        pc = self.and2(p, cin)      # P AND Cin
        cout = self.or1(g, pc)      # Cout = G OR (P AND Cin)
        return s, cout
    
    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


# ==============================================================================
# 向量化多位加法器/减法器
# ==============================================================================

class VecAdder(nn.Module):
    """向量化N位加法器 - A + B (LSB first)
    
    特点：
    1. P = A XOR B, G = A AND B 一次处理所有位
    2. 进位链仍是串行的（数学依赖不可避免）
    3. S = P XOR carries 一次处理所有位
    
    Args:
        bits: 加法器位宽
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, bits, neuron_template=None):
        super().__init__()
        self.bits = bits
        
        self.xor1 = VecXOR(neuron_template)  # 计算 P
        self.xor2 = VecXOR(neuron_template)  # 计算 S
        self.and1 = VecAND(neuron_template)  # 计算 G
        self.and2 = VecAND(neuron_template)  # P AND carry
        self.or1 = VecOR(neuron_template)    # G OR (P AND carry)
        
    def forward(self, A, B, Cin=None):
        """
        A, B: [..., bits] LSB first
        Cin: [..., 1] 可选进位输入
        返回: (Sum [..., bits], Cout [..., 1])
        """
        device = A.device
        batch_shape = A.shape[:-1]
        
        if Cin is None:
            carry = torch.zeros(batch_shape + (1,), device=device)
        else:
            carry = Cin
        
        # 1. 并行计算 P = A XOR B, G = A AND B
        P = self.xor1(A, B)  # [..., bits]
        G = self.and1(A, B)  # [..., bits]
        
        # 2. 进位链 (串行依赖，不可避免)
        carries = [carry]
        for i in range(self.bits):
            p_i = P[..., i:i+1]
            g_i = G[..., i:i+1]
            pc = self.and2(p_i, carry)
            carry = self.or1(g_i, pc)
            carries.append(carry)
        
        # 3. 并行计算和
        all_carries = torch.cat(carries[:-1], dim=-1)
        S = self.xor2(P, all_carries)
        
        return S, carries[-1]
    
    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


class VecSubtractor(nn.Module):
    """向量化N位减法器 - A - B (LSB first)
    
    使用补码减法: A - B = A + NOT(B) + 1
    
    Args:
        bits: 减法器位宽
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, bits, neuron_template=None):
        super().__init__()
        self.bits = bits
        
        self.not_b = VecNOT(neuron_template)
        self.xor1 = VecXOR(neuron_template)   # A XOR NOT(B)
        self.xor2 = VecXOR(neuron_template)   # P XOR carry
        self.and1 = VecAND(neuron_template)   # A AND NOT(B) (generate)
        self.and2 = VecAND(neuron_template)   # P AND carry
        self.or1 = VecOR(neuron_template)     # G OR (P AND carry)
        
    def forward(self, A, B):
        """
        A, B: [..., bits] LSB first
        返回: (差 [..., bits], 借位 [..., 1])
        借位=1 表示 A < B
        """
        device = A.device
        batch_shape = A.shape[:-1]
        ones = torch.ones(batch_shape + (1,), device=device)
        
        # 1. 并行计算 NOT(B)
        not_b = self.not_b(B)
        
        # 2. 并行计算 P = A XOR NOT(B), G = A AND NOT(B)
        P = self.xor1(A, not_b)
        G = self.and1(A, not_b)
        
        # 3. 进位链 (初始进位为1，补码+1)
        carry = ones
        carries = [carry]
        
        for i in range(self.bits):
            p_i = P[..., i:i+1]
            g_i = G[..., i:i+1]
            pc = self.and2(p_i, carry)
            carry = self.or1(g_i, pc)
            carries.append(carry)
        
        # 4. 并行计算差
        all_carries = torch.cat(carries[:-1], dim=-1)
        D = self.xor2(P, all_carries)
        
        # 借位 = NOT(最终进位)
        borrow = self.not_b(carries[-1])
        
        return D, borrow
    
    def reset(self):
        self.not_b.reset()
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


# ==============================================================================
# 向量化比较器
# ==============================================================================

class VecComparator(nn.Module):
    """向量化N位比较器 - 比较 A 和 B (MSB first)
    
    返回: (A > B, A == B)
    
    Args:
        bits: 比较器位宽
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, bits, neuron_template=None):
        super().__init__()
        self.bits = bits
        
        self.xor_eq = VecXOR(neuron_template)      # 检测位相等
        self.not_b = VecNOT(neuron_template)       # NOT(B)
        self.and_gt = VecAND(neuron_template)      # A AND NOT(B) - A位为1且B位为0
        self.and_tree = VecANDTree(neuron_template=neuron_template)
        self.or_tree = VecORTree(neuron_template=neuron_template)
        
        # 用于逐位比较的门
        self.and_prefix = VecAND(neuron_template)  # 前缀相等 AND 当前位大于
        self.or_result = VecOR(neuron_template)    # 累积结果
        
    def forward(self, A, B):
        """
        A, B: [..., bits] MSB first
        返回: (gt [..., 1], eq [..., 1])
        """
        # 1. 并行计算每位是否相等: eq[i] = NOT(A[i] XOR B[i])
        xor_bits = self.xor_eq(A, B)
        eq_bits = self.not_b(xor_bits)  # eq_bits[i] = 1 if A[i] == B[i]
        
        # 2. 全等: 所有位都相等
        all_eq = self.and_tree(eq_bits)
        
        # 3. 大于: 找到第一个不相等的位，且该位 A > B
        # gt_bits[i] = A[i] AND NOT(B[i])
        not_b_bits = self.not_b(B)
        gt_bits = self.and_gt(A, not_b_bits)
        
        # 4. 从 MSB 开始扫描，找到第一个不等位
        # 使用前缀积累: prefix_eq[i] = AND(eq_bits[0:i])
        device = A.device
        batch_shape = A.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        gt_result = zeros
        prefix_eq = torch.ones(batch_shape + (1,), device=device)
        
        for i in range(self.bits):
            # 当前位大于且前缀相等
            curr_gt = self.and_prefix(prefix_eq, gt_bits[..., i:i+1])
            gt_result = self.or_result(gt_result, curr_gt)
            # 更新前缀相等
            prefix_eq = self.and_prefix(prefix_eq, eq_bits[..., i:i+1])
        
        return gt_result, all_eq
    
    def reset(self):
        self.xor_eq.reset()
        self.not_b.reset()
        self.and_gt.reset()
        self.and_tree.reset()
        self.or_tree.reset()
        self.and_prefix.reset()
        self.or_result.reset()
