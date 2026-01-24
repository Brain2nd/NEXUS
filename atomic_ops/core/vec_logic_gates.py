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
from atomic_ops.vec_logic_gates import VecAdder
from atomic_ops.logic_gates import SimpleLIFNode

# 默认 IF 神经元 (理想数字逻辑)
adder = VecAdder(bits=8)

# LIF 神经元 (物理仿真)
lif_template = SimpleLIFNode(beta=0.9)
adder_lif = VecAdder(bits=8, neuron_template=lif_template)
```

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from copy import deepcopy

# 从 neurons 和 logic_gates 导入
from .neurons import SimpleLIFNode
from .logic_gates import _create_neuron
from .spike_mode import SpikeMode


# ==============================================================================
# 基础向量化逻辑门
# ==============================================================================

class VecAND(nn.Module):
    """向量化AND门 - 一个神经元处理任意形状输入

    数学原理: 阈值1.5，只有当两个输入都为1时 (V=2>1.5) 才发放

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self.node = _create_neuron(neuron_template, threshold=1.5,
                                   max_param_shape=max_param_shape)
        self._instance_mode = mode  # None = 跟随全局/上下文

    def forward(self, a, b):
        result = self.node(a + b)
        # BIT_EXACT 模式：forward 结束后清理膜电位，避免显存累积
        if SpikeMode.should_reset(self._instance_mode):
            self.reset_state()
        return result

    def reset_state(self):
        """只重置膜电位（高效版本，用于 BIT_EXACT 模式）"""
        if hasattr(self.node, 'reset_state'):
            self.node.reset_state()
        else:
            self.node.reset()

    def reset(self):
        """完全重置"""
        self.node.reset()

    def _reset(self):
        if hasattr(self.node, '_reset'):
            self.node._reset()
        else:
            self.node.reset()


class VecOR(nn.Module):
    """向量化OR门 - 一个神经元处理任意形状输入

    数学原理: 阈值0.5，任一输入为1时 (V>=1>0.5) 就发放

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self.node = _create_neuron(neuron_template, threshold=0.5,
                                   max_param_shape=max_param_shape)
        self._instance_mode = mode

    def forward(self, a, b):
        result = self.node(a + b)
        # BIT_EXACT 模式：forward 结束后清理膜电位，避免显存累积
        if SpikeMode.should_reset(self._instance_mode):
            self.reset_state()
        return result

    def reset_state(self):
        """只重置膜电位（高效版本，用于 BIT_EXACT 模式）"""
        if hasattr(self.node, 'reset_state'):
            self.node.reset_state()
        else:
            self.node.reset()

    def reset(self):
        """完全重置"""
        self.node.reset()

    def _reset(self):
        if hasattr(self.node, '_reset'):
            self.node._reset()
        else:
            self.node.reset()


class VecNOT(nn.Module):
    """向量化NOT门 - 神经形态抑制实现 (Neuromorphic Inhibition)

    **纯 SNN 原理**:
    使用抑制性神经元模型代替 `1-x`：
    - Bias = 1.5
    - Inhibitory Input = -1.0 * x
    - Threshold = 1.0

    Logic:
    - x=0 -> Net=1.5 -> Spike
    - x=1 -> Net=0.5 -> No Spike

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self.node = _create_neuron(neuron_template, threshold=1.0,
                                   max_param_shape=max_param_shape)
        self._instance_mode = mode

    def forward(self, x):
        # 物理模拟: Bias(1.5) + Inhibitory(-x)
        result = self.node(1.5 - x)
        # BIT_EXACT 模式：forward 结束后清理膜电位，避免显存累积
        if SpikeMode.should_reset(self._instance_mode):
            self.reset_state()
        return result

    def reset_state(self):
        """只重置膜电位（高效版本，用于 BIT_EXACT 模式）"""
        if hasattr(self.node, 'reset_state'):
            self.node.reset_state()
        else:
            self.node.reset()

    def reset(self):
        """完全重置"""
        self.node.reset()

    def _reset(self):
        if hasattr(self.node, '_reset'):
            self.node._reset()
        else:
            self.node.reset()


class VecXOR(nn.Module):
    """向量化XOR门 - 纯 SNN 实现

    使用内部 VecNOT 生成反相信号，消除 `1-x` 数学计算。

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self._instance_mode = mode
        # 使用两个独立NOT门实例避免状态累积
        self.not_a_gate = VecNOT(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.not_b_gate = VecNOT(neuron_template, mode=mode, max_param_shape=max_param_shape)
        # XOR = (A AND NOT_B) OR (NOT_A AND B)
        self.and1 = _create_neuron(neuron_template, threshold=1.5,
                                   max_param_shape=max_param_shape)  # A AND NOT_B
        self.and2 = _create_neuron(neuron_template, threshold=1.5,
                                   max_param_shape=max_param_shape)  # NOT_A AND B
        self.or_out = _create_neuron(neuron_template, threshold=0.5,
                                     max_param_shape=max_param_shape)  # 输出 OR

    def forward(self, a, b):
        # 内部生成反相信号 - 每个输入使用独立的NOT门
        not_a = self.not_a_gate(a)
        not_b = self.not_b_gate(b)

        # XOR逻辑
        term1 = self.and1(a + not_b)  # A AND NOT_B
        term2 = self.and2(not_a + b)  # NOT_A AND B
        result = self.or_out(term1 + term2)  # OR

        # BIT_EXACT 模式：forward 结束后清理内部神经元的膜电位
        if SpikeMode.should_reset(self._instance_mode):
            self._reset_internal_nodes()
        return result

    def _reset_internal_nodes(self):
        """重置内部神经元节点（不包括已经有模式控制的子门）"""
        for node in [self.and1, self.and2, self.or_out]:
            if hasattr(node, 'reset_state'):
                node.reset_state()
            else:
                node.reset()

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.not_a_gate.reset_state()
        self.not_b_gate.reset_state()
        self._reset_internal_nodes()

    def reset(self):
        """完全重置"""
        self.not_a_gate.reset()
        self.not_b_gate.reset()
        self.and1.reset()
        self.and2.reset()
        self.or_out.reset()

    def _reset(self):
        for gate in [self.not_a_gate, self.not_b_gate]:
            gate._reset() if hasattr(gate, '_reset') else gate.reset()
        for node in [self.and1, self.and2, self.or_out]:
            node._reset() if hasattr(node, '_reset') else node.reset()


class VecMUX(nn.Module):
    """向量化MUX门 - sel=1选a, sel=0选b

    MUX(s, a, b) = (s AND a) OR (NOT(s) AND b)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self._instance_mode = mode
        # 子组件继承模式
        self.not_s = VecNOT(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.and1 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)  # s AND a
        self.and2 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)  # NOT(s) AND b
        self.or_gate = VecOR(neuron_template, mode=mode, max_param_shape=max_param_shape)

    def forward(self, sel, a, b):
        # 子组件已经有模式检查，无需额外处理
        ns = self.not_s(sel)
        sa = self.and1(sel, a)
        nsb = self.and2(ns, b)
        return self.or_gate(sa, nsb)

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.not_s.reset_state()
        self.and1.reset_state()
        self.and2.reset_state()
        self.or_gate.reset_state()

    def reset(self):
        """完全重置"""
        self.not_s.reset()
        self.and1.reset()
        self.and2.reset()
        self.or_gate.reset()

    def _reset(self):
        for comp in [self.not_s, self.and1, self.and2, self.or_gate]:
            comp._reset() if hasattr(comp, '_reset') else comp.reset()


# ==============================================================================
# 并行归约树
# ==============================================================================

class VecORTree(nn.Module):
    """并行OR树 - 检测任意位是否为1

    使用log2(n)层OR门并行归约，最多支持1024位输入

    Args:
        max_layers: 最大层数 (保留用于接口兼容，实际使用单实例)
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, max_layers=10, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self._instance_mode = mode
        # 单实例 (动态扩展机制支持不同位宽)
        self.or_gate = VecOR(neuron_template, mode=mode, max_param_shape=max_param_shape)

    def forward(self, x):
        """
        x: [..., N] 输入位
        返回: [..., 1] 任意位为1则输出1
        """
        current = x
        while current.shape[-1] > 1:
            n = current.shape[-1]
            if n % 2 == 1:
                pairs = current[..., :-1]
                last = current[..., -1:]
                left = pairs[..., 0::2]
                right = pairs[..., 1::2]
                paired = self.or_gate(left, right)
                current = torch.cat([paired, last], dim=-1)
            else:
                left = current[..., 0::2]
                right = current[..., 1::2]
                current = self.or_gate(left, right)
        return current

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.or_gate.reset_state()

    def reset(self):
        """完全重置"""
        self.or_gate.reset()

    def _reset(self):
        self.or_gate._reset() if hasattr(self.or_gate, '_reset') else self.or_gate.reset()


class VecANDTree(nn.Module):
    """并行AND树 - 检测所有位是否为1

    使用log2(n)层AND门并行归约，最多支持1024位输入

    Args:
        max_layers: 最大层数 (保留用于接口兼容，实际使用单实例)
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, max_layers=10, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self._instance_mode = mode
        # 单实例 (动态扩展机制支持不同位宽)
        self.and_gate = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)

    def forward(self, x):
        """
        x: [..., N] 输入位
        返回: [..., 1] 所有位为1则输出1
        """
        current = x
        while current.shape[-1] > 1:
            n = current.shape[-1]
            if n % 2 == 1:
                pairs = current[..., :-1]
                last = current[..., -1:]
                left = pairs[..., 0::2]
                right = pairs[..., 1::2]
                paired = self.and_gate(left, right)
                current = torch.cat([paired, last], dim=-1)
            else:
                left = current[..., 0::2]
                right = current[..., 1::2]
                current = self.and_gate(left, right)
        return current

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.and_gate.reset_state()

    def reset(self):
        """完全重置"""
        self.and_gate.reset()

    def _reset(self):
        self.and_gate._reset() if hasattr(self.and_gate, '_reset') else self.and_gate.reset()


# ==============================================================================
# 向量化算术单元
# ==============================================================================

class VecHalfAdder(nn.Module):
    """向量化半加器 - 处理任意形状输入

    S = A XOR B
    C = A AND B

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self._instance_mode = mode
        self.xor_gate = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.and_gate = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)

    def forward(self, a, b):
        s = self.xor_gate(a, b)
        c = self.and_gate(a, b)
        return s, c

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.xor_gate.reset_state()
        self.and_gate.reset_state()

    def reset(self):
        """完全重置"""
        self.xor_gate.reset()
        self.and_gate.reset()

    def _reset(self):
        for comp in [self.xor_gate, self.and_gate]:
            comp._reset() if hasattr(comp, '_reset') else comp.reset()


class VecFullAdder(nn.Module):
    """向量化全加器 - 处理任意形状输入

    S = A XOR B XOR Cin
    Cout = (A AND B) OR ((A XOR B) AND Cin)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self._instance_mode = mode
        self.xor1 = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.xor2 = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.and1 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.and2 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.or1 = VecOR(neuron_template, mode=mode, max_param_shape=max_param_shape)

    def forward(self, a, b, cin):
        p = self.xor1(a, b)         # P = A XOR B
        s = self.xor2(p, cin)       # S = P XOR Cin
        g = self.and1(a, b)         # G = A AND B
        pc = self.and2(p, cin)      # P AND Cin
        cout = self.or1(g, pc)      # Cout = G OR (P AND Cin)
        return s, cout

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.xor1.reset_state()
        self.xor2.reset_state()
        self.and1.reset_state()
        self.and2.reset_state()
        self.or1.reset_state()

    def reset(self):
        """完全重置"""
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()

    def _reset(self):
        for comp in [self.xor1, self.xor2, self.and1, self.and2, self.or1]:
            comp._reset() if hasattr(comp, '_reset') else comp.reset()


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
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, bits, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self.bits = bits
        self._instance_mode = mode

        self.xor1 = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)  # 计算 P
        self.xor2 = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)  # 计算 S
        self.and1 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)  # 计算 G
        # 进位链：单实例 (动态扩展机制支持复用)
        self.and2 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)  # P AND carry
        self.or1 = VecOR(neuron_template, mode=mode, max_param_shape=max_param_shape)    # G OR (P AND carry)

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

        # 2. 进位链 (串行依赖)
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

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.xor1.reset_state()
        self.xor2.reset_state()
        self.and1.reset_state()
        self.and2.reset_state()
        self.or1.reset_state()

    def reset(self):
        """完全重置"""
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()

    def _reset(self):
        for comp in [self.xor1, self.xor2, self.and1, self.and2, self.or1]:
            comp._reset() if hasattr(comp, '_reset') else comp.reset()


class VecSubtractor(nn.Module):
    """向量化N位减法器 - A - B (LSB first)

    使用补码减法: A - B = A + NOT(B) + 1

    Args:
        bits: 减法器位宽
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, bits, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self.bits = bits
        self._instance_mode = mode

        self.not_b = VecNOT(neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.not_borrow = VecNOT(neuron_template, mode=mode, max_param_shape=max_param_shape)  # 用于计算最终借位
        self.xor1 = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)   # A XOR NOT(B)
        self.xor2 = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)   # P XOR carry
        self.and1 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)   # A AND NOT(B) (generate)
        # 进位链：单实例 (动态扩展机制支持复用)
        self.and2 = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)  # P AND carry
        self.or1 = VecOR(neuron_template, mode=mode, max_param_shape=max_param_shape)    # G OR (P AND carry)

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

        # 借位 = NOT(最终进位)，使用独立的NOT门
        borrow = self.not_borrow(carries[-1])

        return D, borrow

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.not_b.reset_state()
        self.not_borrow.reset_state()
        self.xor1.reset_state()
        self.xor2.reset_state()
        self.and1.reset_state()
        self.and2.reset_state()
        self.or1.reset_state()

    def reset(self):
        """完全重置"""
        self.not_b.reset()
        self.not_borrow.reset()
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()

    def _reset(self):
        for comp in [self.not_b, self.not_borrow, self.xor1, self.xor2, self.and1, self.and2, self.or1]:
            comp._reset() if hasattr(comp, '_reset') else comp.reset()


# ==============================================================================
# 向量化比较器
# ==============================================================================

class VecComparator(nn.Module):
    """向量化N位比较器 - 比较 A 和 B (MSB first)

    返回: (A > B, A == B)

    Args:
        bits: 比较器位宽
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        mode: SpikeMode 模式覆盖，None 表示跟随全局/上下文
        max_param_shape: 预分配最大参数形状，例如 (32,) 表示32位
    """
    def __init__(self, bits, neuron_template=None, mode=None, max_param_shape=None):
        super().__init__()
        self.bits = bits
        self._instance_mode = mode

        self.xor_eq = VecXOR(neuron_template, mode=mode, max_param_shape=max_param_shape)      # 检测位相等
        self.not_b = VecNOT(neuron_template, mode=mode, max_param_shape=max_param_shape)       # NOT(B)
        self.and_gt = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)      # A AND NOT(B) - A位为1且B位为0
        self.and_tree = VecANDTree(neuron_template=neuron_template, mode=mode, max_param_shape=max_param_shape)
        self.or_tree = VecORTree(neuron_template=neuron_template, mode=mode, max_param_shape=max_param_shape)

        # 用于逐位比较的门
        self.and_prefix = VecAND(neuron_template, mode=mode, max_param_shape=max_param_shape)  # 前缀相等 AND 当前位大于
        self.or_result = VecOR(neuron_template, mode=mode, max_param_shape=max_param_shape)    # 累积结果

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

    def reset_state(self):
        """只重置膜电位（高效版本）"""
        self.xor_eq.reset_state()
        self.not_b.reset_state()
        self.and_gt.reset_state()
        self.and_tree.reset_state()
        self.or_tree.reset_state()
        self.and_prefix.reset_state()
        self.or_result.reset_state()

    def reset(self):
        """完全重置"""
        self.xor_eq.reset()
        self.not_b.reset()
        self.and_gt.reset()
        self.and_tree.reset()
        self.or_tree.reset()
        self.and_prefix.reset()
        self.or_result.reset()

    def _reset(self):
        for comp in [self.xor_eq, self.not_b, self.and_gt, self.and_tree,
                     self.or_tree, self.and_prefix, self.or_result]:
            comp._reset() if hasattr(comp, '_reset') else comp.reset()
