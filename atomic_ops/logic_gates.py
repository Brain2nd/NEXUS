"""
SNN 逻辑门电路库 (SNN Logic Gates Library)
==========================================

基于 Integrate-and-Fire (IF) 神经元实现的纯 SNN 逻辑门电路。

这些门电路是构建复杂 SNN 浮点运算的基础原子操作。

核心原理
--------

**IF 神经元动力学**:
```
V[t] = V[t-1] + I[t]           # 膜电位积累
S[t] = H(V[t] - V_th)          # 阈值判断 (Heaviside)
V[t] = V[t] × (1 - S[t])       # 硬复位 (Hard Reset)
```

**逻辑门实现**:

| 门类型 | 阈值 | 输入电流 | 逻辑公式 |
|--------|------|----------|----------|
| AND    | 1.5  | A + B    | A ∧ B: 仅当 A=B=1 时 V=2 > 1.5 |
| OR     | 0.5  | A + B    | A ∨ B: 任一为1时 V≥1 > 0.5 |
| XOR    | -    | 两层级联  | A ⊕ B = (A+B) - 2×AND(A,B) |
| NOT    | 0.5  | 1 - A    | ¬A: 偏置1.0 + 抑制性输入 |

**算术单元**:

半加器 (Half Adder):
```
S = A ⊕ B        (XOR)
C = A ∧ B        (AND)
```

全加器 (Full Adder):
```
S = A ⊕ B ⊕ Cin
C = (A ∧ B) ∨ ((A ⊕ B) ∧ Cin)
```

组件列表
--------
- 基础门: ANDGate, ORGate, XORGate, NOTGate, MUXGate
- 算术: HalfAdder, FullAdder, RippleCarryAdder
- 乘法: ArrayMultiplier4x4_Strict
- 树结构: ORTree, ANDTree
- 移位器: BarrelShifter8, Denormalizer
- 检测器: FirstSpikeDetector, PriorityEncoder8

使用示例
--------
```python
from SNNTorch.atomic_ops.logic_gates import ANDGate, FullAdder

# 基础 AND 门
and_gate = ANDGate()
result = and_gate(a, b)  # a ∧ b

# 全加器
fa = FullAdder()
s, c = fa(a, b, cin)  # s = a⊕b⊕cin, c = carry
```

作者: HumanBrain Project
许可: MIT License
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


# ==============================================================================
# 基础类
# ==============================================================================

class BaseLogicGate(nn.Module):
    """所有 SNN 逻辑门的基类"""
    def __init__(self):
        super().__init__()

# ==============================================================================
# 基础逻辑门
# ==============================================================================

class ANDGate(BaseLogicGate):
    """AND 门 - 使用 IF 神经元实现
    
    **数学原理**:
    ```
    V = A + B
    输出 = H(V - 1.5)  # Heaviside 阶跃函数
    ```
    
    **真值表**:
    | A | B | V | 输出 |
    |---|---|---|------|
    | 0 | 0 | 0 | 0    |
    | 0 | 1 | 1 | 0    |
    | 1 | 0 | 1 | 0    |
    | 1 | 1 | 2 | 1    | ← 仅此情况 V > 1.5
    
    门电路计数: 1 个 IF 神经元
    """
    def __init__(self, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.node = neuron.IFNode(v_threshold=1.5, v_reset=0.0, surrogate_function=surrogate_function)
        
    def forward(self, x_a, x_b):
        self.reset()
        return self.node(x_a + x_b)
        
    def reset(self):
        self.node.reset()


class ORGate(BaseLogicGate):
    """OR 门 - 使用 IF 神经元实现
    
    **数学原理**:
    ```
    V = A + B
    输出 = H(V - 0.5)
    ```
    
    **真值表**:
    | A | B | V | 输出 |
    |---|---|---|------|
    | 0 | 0 | 0 | 0    | ← 仅此情况 V < 0.5
    | 0 | 1 | 1 | 1    |
    | 1 | 0 | 1 | 1    |
    | 1 | 1 | 2 | 1    |
    
    门电路计数: 1 个 IF 神经元
    """
    def __init__(self, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.node = neuron.IFNode(v_threshold=0.5, v_reset=0.0, surrogate_function=surrogate_function)
        
    def forward(self, x_a, x_b):
        self.reset()
        return self.node(x_a + x_b)

    def reset(self):
        self.node.reset()


class XORGate(BaseLogicGate):
    """XOR 门 - 使用两层 IF 神经元实现
    
    **数学原理**:
    ```
    层1: H = AND(A, B) = H(A + B - 1.5)
    层2: 输出 = H(A + B - 2H - 0.5)
    
    简化: XOR = (A + B) - 2×AND(A, B)
    ```
    
    **真值表**:
    | A | B | H | A+B-2H | 输出 |
    |---|---|---|--------|------|
    | 0 | 0 | 0 |   0    | 0    |
    | 0 | 1 | 0 |   1    | 1    |
    | 1 | 0 | 0 |   1    | 1    |
    | 1 | 1 | 1 |   0    | 0    |
    
    门电路计数: 2 个 IF 神经元
    """
    def __init__(self, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.hidden_node = neuron.IFNode(v_threshold=1.5, v_reset=0.0, surrogate_function=surrogate_function)
        self.out_node = neuron.IFNode(v_threshold=0.5, v_reset=0.0, surrogate_function=surrogate_function)
        
    def forward(self, x_a, x_b):
        self.reset()
        hidden_spike = self.hidden_node(x_a + x_b)  # AND(A, B)
        out_spike = self.out_node(x_a + x_b - 2.0 * hidden_spike)  # XOR
        return out_spike

    def reset(self):
        self.hidden_node.reset()
        self.out_node.reset()


class NOTGate(BaseLogicGate):
    """NOT 门 - 使用 IF 神经元 + 抑制性突触实现
    
    **数学原理**:
    ```
    V = bias + weight × A = 1.0 + (-1.0) × A = 1 - A
    输出 = H(V - 0.5)
    ```
    
    **真值表**:
    | A | V = 1-A | 输出 |
    |---|---------|------|
    | 0 |    1    | 1    | ← V=1 > 0.5
    | 1 |    0    | 0    | ← V=0 < 0.5
    
    **实现细节**:
    - 偏置电流 (bias): 1.0 (恒定兴奋性)
    - 突触权重 (weight): -1.0 (抑制性)
    
    门电路计数: 1 个 IF 神经元
    """
    def __init__(self, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.node = neuron.IFNode(v_threshold=0.5, v_reset=0.0, surrogate_function=surrogate_function)
        self.bias = 1.0    # 恒定偏置电流（兴奋性）
        self.weight = -1.0  # 抑制性突触权重
        
    def forward(self, x):
        self.reset()
        current = self.bias + self.weight * x  # V = 1 - x
        return self.node(current)
    
    def reset(self):
        self.node.reset()

class XNORGate(nn.Module):
    """XNOR门：当两个输入相同时输出1
    XNOR = NOT(XOR) = (a AND b) OR (NOT_a AND NOT_b)
    纯SNN实现：使用NOTGate代替直接的1-x操作
    """
    def __init__(self):
        super().__init__()
        self.not_a = NOTGate()  # NOT(a)
        self.not_b = NOTGate()  # NOT(b)
        self.and1 = ANDGate()   # a AND b
        self.and2 = ANDGate()   # NOT_a AND NOT_b
        self.or1 = ORGate()
        
    def forward(self, a, b):
        self.reset()
        ab = self.and1(a, b)
        na = self.not_a(a)
        nb = self.not_b(b)
        na_nb = self.and2(na, nb)
        return self.or1(ab, na_nb)
    
    def reset(self):
        self.not_a.reset()
        self.not_b.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()

class AND4Gate(nn.Module):
    """四输入AND门"""
    def __init__(self):
        super().__init__()
        self.and1 = ANDGate()
        self.and2 = ANDGate()
        self.and3 = ANDGate()
        
    def forward(self, a, b, c, d):
        self.reset()
        ab = self.and1(a, b)
        cd = self.and2(c, d)
        return self.and3(ab, cd)
    
    def reset(self):
        self.and1.reset()
        self.and2.reset()
        self.and3.reset()

class SpikeDetector(nn.Module):
    """脉冲检测器：当输入>=1时输出1，否则输出0
    使用阈值为0.5的IF神经元
    """
    def __init__(self):
        super().__init__()
        self.node = neuron.IFNode(v_threshold=0.5, v_reset=0.0, surrogate_function=surrogate.ATan())
        
    def forward(self, x):
        self.reset()
        return self.node(x)
    
    def reset(self):
        self.node.reset()

class MUXGate(nn.Module):
    """脉冲MUX选择器：MUX(S, A, B) = OR(AND(S, A), AND(NOT_S, B))
    当S=1时选择A，S=0时选择B
    纯SNN实现：使用NOTGate代替直接的1-x操作
    """
    def __init__(self):
        super().__init__()
        self.not_s = NOTGate()  # NOT(S)
        self.and1 = ANDGate()   # S AND A
        self.and2 = ANDGate()   # NOT_S AND B
        self.or1 = ORGate()
        
    def forward(self, s, a, b):
        self.reset()
        ns = self.not_s(s)
        sa = self.and1(s, a)
        nsb = self.and2(ns, b)
        return self.or1(sa, nsb)
    
    def reset(self):
        self.not_s.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()

class OR3Gate(nn.Module):
    """三输入OR门"""
    def __init__(self):
        super().__init__()
        self.or1 = ORGate()
        self.or2 = ORGate()
        
    def forward(self, a, b, c):
        self.reset()
        ab = self.or1(a, b)
        return self.or2(ab, c)
    
    def reset(self):
        self.or1.reset()
        self.or2.reset()

# ==============================================================================
# 算术单元
# ==============================================================================

class HalfAdder(nn.Module):
    """半加器 - 两个 1 位输入的加法
    
    **数学公式**:
    ```
    S = A ⊕ B        (和，XOR)
    C = A ∧ B        (进位，AND)
    ```
    
    **真值表**:
    | A | B | S | C |
    |---|---|---|---|
    | 0 | 0 | 0 | 0 |
    | 0 | 1 | 1 | 0 |
    | 1 | 0 | 1 | 0 |
    | 1 | 1 | 0 | 1 |
    
    门电路计数: 1 XOR + 1 AND = 3 IF 神经元
    
    Returns:
        (S, C): 和位与进位位
    """
    def __init__(self):
        super().__init__()
        self.xor1 = XORGate()
        self.and1 = ANDGate()
        
    def forward(self, a, b):
        self.reset()
        s = self.xor1(a, b)   # S = A ⊕ B
        c = self.and1(a, b)   # C = A ∧ B
        return s, c
        
    def reset(self):
        self.xor1.reset()
        self.and1.reset()


class FullAdder(nn.Module):
    """全加器 - 三个 1 位输入的加法 (含进位输入)
    
    **数学公式**:
    ```
    S = A ⊕ B ⊕ Cin
    Cout = (A ∧ B) ∨ ((A ⊕ B) ∧ Cin)
    ```
    
    **电路结构**:
    ```
    A ──┬──[XOR1]──┬──[XOR2]── S
        │          │
    B ──┘          └──[AND2]──┐
                              ├──[OR]── Cout
    A ──[AND1]────────────────┘
    B ──┘
    Cin ──────────────┘
    ```
    
    门电路计数: 2 XOR + 2 AND + 1 OR = 7 IF 神经元
    
    Returns:
        (S, Cout): 和位与进位输出
    """
    def __init__(self):
        super().__init__()
        self.xor1 = XORGate()
        self.xor2 = XORGate()
        self.and1 = ANDGate()
        self.and2 = ANDGate()
        self.or1 = ORGate()
        
    def forward(self, a, b, cin):
        self.reset()
        s1 = self.xor1(a, b)       # A ⊕ B
        sum_out = self.xor2(s1, cin)  # S = (A ⊕ B) ⊕ Cin
        c1 = self.and1(a, b)       # A ∧ B
        c2 = self.and2(s1, cin)    # (A ⊕ B) ∧ Cin
        cout = self.or1(c1, c2)    # Cout = c1 ∨ c2
        return sum_out, cout

    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


class RippleCarryAdder(nn.Module):
    """行波进位加法器 - N 位二进制加法
    
    **数学公式**:
    ```
    对于 i = 0 to N-1:
        S[i], C[i+1] = FullAdder(A[i], B[i], C[i])
    ```
    
    **电路结构** (4位示例):
    ```
    A[0] B[0]   A[1] B[1]   A[2] B[2]   A[3] B[3]
      │   │       │   │       │   │       │   │
      └─[FA]──C──[FA]──C──[FA]──C──[FA]── Cout
         │          │          │          │
        S[0]       S[1]       S[2]       S[3]
    ```
    
    **注意**: 输入格式为 LSB first (最低位在索引 0)
    
    Args:
        bits: 加法器位宽
        
    门电路计数: bits × 7 = 7N IF 神经元
    
    Returns:
        (Sum, Cout): N位和 与 最终进位
    """
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.adders = nn.ModuleList([FullAdder() for _ in range(bits)])
        
    def forward(self, A, B, Cin=None):
        """
        Args:
            A: [..., bits] 加数 (LSB first)
            B: [..., bits] 被加数 (LSB first)
            Cin: [..., 1] 可选进位输入
            
        Returns:
            Sum: [..., bits] 和 (LSB first)
            Cout: [..., 1] 进位输出
        """
        sum_bits = []
        if Cin is None:
            c = torch.zeros_like(A[..., 0:1])
        else:
            c = Cin
            
        for i in range(self.bits):
            a_i = A[..., i:i+1]
            b_i = B[..., i:i+1]
            s, c = self.adders[i](a_i, b_i, c)
            sum_bits.append(s)
            
        return torch.cat(sum_bits, dim=-1), c

    def reset(self):
        for adder in self.adders:
            adder.reset()

class ORTree(nn.Module):
    """N输入OR树，纯SNN实现"""
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs
        # 构建二叉OR树
        n_gates = n_inputs - 1
        self.or_gates = nn.ModuleList([ORGate() for _ in range(n_gates)])
        
    def forward(self, inputs):
        """
        inputs: [batch, n_inputs] 或 list of [batch, 1]
        输出: [batch, 1]
        """
        if isinstance(inputs, list):
            current = inputs
        else:
            current = [inputs[..., i:i+1] for i in range(self.n_inputs)]
        
        gate_idx = 0
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    result = self.or_gates[gate_idx](current[i], current[i+1])
                    gate_idx += 1
                    next_level.append(result)
                else:
                    next_level.append(current[i])
            current = next_level
        
        return current[0]
    
    def reset(self):
        for gate in self.or_gates:
            gate.reset()


class ANDTree(nn.Module):
    """N输入AND树，纯SNN实现"""
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs
        n_gates = n_inputs - 1
        self.and_gates = nn.ModuleList([ANDGate() for _ in range(n_gates)])
        
    def forward(self, inputs):
        if isinstance(inputs, list):
            current = inputs
        else:
            current = [inputs[..., i:i+1] for i in range(self.n_inputs)]
        
        gate_idx = 0
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    result = self.and_gates[gate_idx](current[i], current[i+1])
                    gate_idx += 1
                    next_level.append(result)
                else:
                    next_level.append(current[i])
            current = next_level
        
        return current[0]
    
    def reset(self):
        for gate in self.and_gates:
            gate.reset()


class FirstSpikeDetector(nn.Module):
    """纯SNN首脉冲检测器
    
    输入: spike_train [batch, T] - 脉冲序列
    输出: one_hot [batch, T] - 只有首脉冲位置为1
    
    原理: first_spike[t] = spike[t] AND NOT(any_previous_spike[t])
    any_previous_spike[t] = OR(spike[0], spike[1], ..., spike[t-1])
    """
    def __init__(self, max_steps):
        super().__init__()
        self.max_steps = max_steps
        # OR门用于累积"之前是否有脉冲"
        self.or_gates = nn.ModuleList([ORGate() for _ in range(max_steps - 1)])
        # AND门用于输出首脉冲
        self.and_gates = nn.ModuleList([ANDGate() for _ in range(max_steps)])
        # NOT门
        self.not_gates = nn.ModuleList([NOTGate() for _ in range(max_steps)])
        
    def forward(self, spike_train):
        """
        spike_train: [batch, max_steps]
        返回: [batch, max_steps] one-hot向量
        """
        batch_shape = spike_train.shape[:-1]
        device = spike_train.device
        
        # 初始化"之前是否有脉冲"为0
        prev_any = torch.zeros(batch_shape + (1,), device=device)
        first_spikes = []
        
        for t in range(self.max_steps):
            current = spike_train[..., t:t+1]
            # NOT(prev_any)
            not_prev = self.not_gates[t](prev_any)
            # first_spike = current AND NOT(prev_any)
            first_spike = self.and_gates[t](current, not_prev)
            first_spikes.append(first_spike)
            
            # 更新 prev_any = prev_any OR current
            if t < self.max_steps - 1:
                prev_any = self.or_gates[t](prev_any, current)
        
        return torch.cat(first_spikes, dim=-1)
    
    def reset(self):
        for gate in self.or_gates:
            gate.reset()
        for gate in self.and_gates:
            gate.reset()


class OneHotToExponent(nn.Module):
    """将首脉冲位置(one-hot)转换为4位二进制指数
    
    纯SNN实现：
    - 预计算每个位置对应的指数值
    - 对于每个指数位：OR所有该位为1的位置的AND结果
    
    E[b] = OR over all k where exponent[k] has bit b=1
    """
    def __init__(self, max_steps, n_integer_bits, bias=7, e_bits=4):
        super().__init__()
        self.max_steps = max_steps
        self.e_bits = e_bits
        self.bias = bias
        
        # 预计算每个位置的指数
        # 位置k → E = clamp(n_integer_bits - 1 - k + bias, 0, 2^e_bits - 1)
        max_e = (1 << e_bits) - 1
        self.exponents = []
        for k in range(max_steps):
            e = (n_integer_bits - 1) - k + bias
            e = max(0, min(e, max_e))
            self.exponents.append(e)
        
        # 为每个指数位创建AND门（选择该位置的贡献）
        # 以及OR树合并所有贡献
        self.and_gates = nn.ModuleList()
        self.or_trees = nn.ModuleList()
        
        for b in range(e_bits):
            # 找出哪些位置的该指数位为1
            positions_with_bit = []
            for k in range(max_steps):
                if (self.exponents[k] >> (e_bits - 1 - b)) & 1:
                    positions_with_bit.append(k)
            
            if len(positions_with_bit) > 0:
                self.and_gates.append(nn.ModuleList([ANDGate() for _ in positions_with_bit]))
                if len(positions_with_bit) > 1:
                    self.or_trees.append(ORTree(len(positions_with_bit)))
                else:
                    self.or_trees.append(None)
            else:
                self.and_gates.append(None)
                self.or_trees.append(None)
        
        # 记录哪些位置对应哪个bit
        self.positions_per_bit = []
        for b in range(e_bits):
            positions = [k for k in range(max_steps) 
                        if (self.exponents[k] >> (e_bits - 1 - b)) & 1]
            self.positions_per_bit.append(positions)
    
    def forward(self, position_onehot):
        """
        position_onehot: [batch, max_steps]
        返回: [batch, e_bits] 二进制指数
        """
        device = position_onehot.device
        batch_shape = position_onehot.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)
        
        e_bits_out = []
        
        for b in range(self.e_bits):
            positions = self.positions_per_bit[b]
            
            if len(positions) == 0:
                # 该位始终为0
                e_bits_out.append(zeros)
            elif len(positions) == 1:
                # 只有一个位置贡献
                k = positions[0]
                # 该位 = position_onehot[k]（直接取，相当于 AND(pos[k], 1)）
                contrib = self.and_gates[b][0](position_onehot[..., k:k+1], ones)
                e_bits_out.append(contrib)
            else:
                # 多个位置贡献，需要OR合并
                contribs = []
                for i, k in enumerate(positions):
                    contrib = self.and_gates[b][i](position_onehot[..., k:k+1], ones)
                    contribs.append(contrib)
                bit_val = self.or_trees[b](contribs)
                e_bits_out.append(bit_val)
        
        return torch.cat(e_bits_out, dim=-1)
    
    def reset(self):
        for gates in self.and_gates:
            if gates is not None:
                for g in gates:
                    g.reset()
        for tree in self.or_trees:
            if tree is not None:
                tree.reset()


class NormalMantissaExtractor(nn.Module):
    """从脉冲序列中提取Normal数的尾数
    
    原理：首脉冲在位置k，尾数是spike[k+1], spike[k+2], spike[k+3]
    
    纯SNN实现：
    M[i] = OR over all k: (first_spike[k] AND spike[k+1+i])
    """
    def __init__(self, max_steps, m_bits=3):
        super().__init__()
        self.max_steps = max_steps
        self.m_bits = m_bits
        
        # 对于每个尾数位，为每个可能的首脉冲位置创建AND门
        # M[i] = OR over k: (first_spike[k] AND spike[k+1+i])
        self.and_gates = nn.ModuleList()
        self.or_trees = nn.ModuleList()
        
        for m_idx in range(m_bits):
            # 有效位置：首脉冲位置k使得k+1+m_idx < max_steps
            valid_positions = [k for k in range(max_steps) 
                              if k + 1 + m_idx < max_steps]
            
            if len(valid_positions) > 0:
                self.and_gates.append(nn.ModuleList([ANDGate() for _ in valid_positions]))
                if len(valid_positions) > 1:
                    self.or_trees.append(ORTree(len(valid_positions)))
                else:
                    self.or_trees.append(None)
            else:
                self.and_gates.append(None)
                self.or_trees.append(None)
        
        # 记录每个尾数位的有效位置
        self.valid_positions = []
        for m_idx in range(m_bits):
            self.valid_positions.append([k for k in range(max_steps) 
                                        if k + 1 + m_idx < max_steps])
    
    def forward(self, first_spike_onehot, spike_train):
        """
        first_spike_onehot: [batch, max_steps]
        spike_train: [batch, max_steps]
        返回: [batch, m_bits]
        """
        device = spike_train.device
        batch_shape = spike_train.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        
        m_bits_out = []
        
        for m_idx in range(self.m_bits):
            positions = self.valid_positions[m_idx]
            
            if len(positions) == 0:
                m_bits_out.append(zeros)
            elif len(positions) == 1:
                k = positions[0]
                spike_idx = k + 1 + m_idx
                contrib = self.and_gates[m_idx][0](
                    first_spike_onehot[..., k:k+1],
                    spike_train[..., spike_idx:spike_idx+1]
                )
                m_bits_out.append(contrib)
            else:
                contribs = []
                for i, k in enumerate(positions):
                    spike_idx = k + 1 + m_idx
                    contrib = self.and_gates[m_idx][i](
                        first_spike_onehot[..., k:k+1],
                        spike_train[..., spike_idx:spike_idx+1]
                    )
                    contribs.append(contrib)
                bit_val = self.or_trees[m_idx](contribs)
                m_bits_out.append(bit_val)
        
        return torch.cat(m_bits_out, dim=-1)
    
    def reset(self):
        for gates in self.and_gates:
            if gates is not None:
                for g in gates:
                    g.reset()
        for tree in self.or_trees:
            if tree is not None:
                tree.reset()


class PriorityEncoder8(nn.Module):
    """8-bit Priority Encoder (Leading One Detector) - Pure SNN
    
    Input: P [batch, 8] (Little Endian: P0...P7)
    Output: Valid [batch, 8] (One-Hot, only the MSB '1' is active)
    """
    def __init__(self):
        super().__init__()
        self.or_chain = nn.ModuleList([ORGate() for _ in range(7)])
        self.not_gates = nn.ModuleList([NOTGate() for _ in range(8)])
        self.and_gates = nn.ModuleList([ANDGate() for _ in range(8)])
        
    def forward(self, P):
        p_bits = [P[..., i:i+1] for i in range(8)]
        
        inhibits = [None] * 8
        
        curr_inhibit = p_bits[7]
        inhibits[6] = curr_inhibit
        
        for i in range(5, -1, -1):
            curr_inhibit = self.or_chain[i](p_bits[i+1], curr_inhibit)
            inhibits[i] = curr_inhibit
            
        valid = []
        for i in range(8):
            if i == 7:
                valid.append(p_bits[7])
            else:
                not_inh = self.not_gates[i](inhibits[i])
                v = self.and_gates[i](p_bits[i], not_inh)
                valid.append(v)
                
        return torch.cat(valid, dim=-1)

    def reset(self):
        for g in self.or_chain: g.reset()
        for g in self.not_gates: g.reset()
        for g in self.and_gates: g.reset()


class ShiftAmountEncoder8to3(nn.Module):
    """Encode One-Hot 8-bit to 3-bit Binary - Pure SNN"""
    def __init__(self):
        super().__init__()
        self.or_tree_s0 = ORTree(4)
        self.or_tree_s1 = ORTree(4)
        self.or_tree_s2 = ORTree(4)
        
    def forward(self, V):
        v = [V[..., i:i+1] for i in range(8)]
        
        # S0: 6, 4, 2, 0
        s0 = self.or_tree_s0([v[6], v[4], v[2], v[0]])
        
        # S1: 5, 4, 1, 0
        s1 = self.or_tree_s1([v[5], v[4], v[1], v[0]])
        
        # S2: 3, 2, 1, 0
        s2 = self.or_tree_s2([v[3], v[2], v[1], v[0]])
        
        return torch.cat([s0, s1, s2], dim=-1)

    def reset(self):
        self.or_tree_s0.reset()
        self.or_tree_s1.reset()
        self.or_tree_s2.reset()


class BarrelShifter8(nn.Module):
    """8-bit Left Barrel Shifter - Pure SNN"""
    def __init__(self):
        super().__init__()
        self.mux_s4 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_s2 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_s1 = nn.ModuleList([MUXGate() for _ in range(8)])
        
    def forward(self, P, S):
        zeros = torch.zeros_like(P[..., 0:1])
        p_bits = [P[..., i:i+1] for i in range(8)]
        
        s0 = S[..., 0:1]
        s1 = S[..., 1:2]
        s2 = S[..., 2:3]
        
        stage1 = []
        for i in range(8):
            if i >= 4:
                val = self.mux_s4[i](s2, p_bits[i-4], p_bits[i])
            else:
                val = self.mux_s4[i](s2, zeros, p_bits[i])
            stage1.append(val)
            
        stage2 = []
        for i in range(8):
            if i >= 2:
                val = self.mux_s2[i](s1, stage1[i-2], stage1[i])
            else:
                val = self.mux_s2[i](s1, zeros, stage1[i])
            stage2.append(val)
            
        stage3 = []
        for i in range(8):
            if i >= 1:
                val = self.mux_s1[i](s0, stage2[i-1], stage2[i])
            else:
                val = self.mux_s1[i](s0, zeros, stage2[i])
            stage3.append(val)
            
        return torch.cat(stage3, dim=-1)

    def reset(self):
        for m in self.mux_s4: m.reset()
        for m in self.mux_s2: m.reset()
        for m in self.mux_s1: m.reset()


class ExponentAdjuster(nn.Module):
    """Convert One-Hot Position to 5-bit Exponent Adjustment - Pure SNN"""
    def __init__(self):
        super().__init__()
        self.or_tree_e0 = ORTree(8)
        self.or_tree_e1 = ORTree(8)
        self.or_tree_e2 = ORTree(8)
        self.or_tree_e3 = ORTree(8)
        self.or_tree_e4 = ORTree(8)
        self.nor_all_valid = NOTGate()
        self.or_tree_valid = ORTree(8)
        
    def forward(self, V):
        v = [V[..., i:i+1] for i in range(8)]
        
        any_valid = self.or_tree_valid(v)
        is_zero = self.nor_all_valid(any_valid)
        
        # V[7] -> +1 (00001)
        # V[6] -> 0 (00000)
        # V[5] -> -1 (11111) ...
        
        # E0: +1, -1, -3, -5, -7 -> V[7], V[5], V[3], V[1], Zero
        e0 = self.or_tree_e0([v[7], v[5], v[3], v[1], is_zero, 
                              torch.zeros_like(is_zero), torch.zeros_like(is_zero), torch.zeros_like(is_zero)])
        
        # E1: -1, -2, -5, -6 -> V[5], V[4], V[1], V[0]
        e1 = self.or_tree_e1([v[5], v[4], v[1], v[0], 
                              torch.zeros_like(is_zero), torch.zeros_like(is_zero), torch.zeros_like(is_zero), torch.zeros_like(is_zero)])
                              
        # E2: -1..-4 -> V[5]..V[2]
        e2 = self.or_tree_e2([v[5], v[4], v[3], v[2],
                              torch.zeros_like(is_zero), torch.zeros_like(is_zero), torch.zeros_like(is_zero), torch.zeros_like(is_zero)])
        
        # E3, E4 (Sign extension): -1..-7 are negative (1)
        # V[5]..V[0], Zero.
        e3 = self.or_tree_e3([v[5], v[4], v[3], v[2], v[1], v[0], is_zero, torch.zeros_like(is_zero)])
        e4 = self.or_tree_e4([v[5], v[4], v[3], v[2], v[1], v[0], is_zero, torch.zeros_like(is_zero)])
        
        return torch.cat([e0, e1, e2, e3, e4], dim=-1)

    def reset(self):
        self.or_tree_e0.reset()
        self.or_tree_e1.reset()
        self.or_tree_e2.reset()
        self.or_tree_e3.reset()
        self.or_tree_e4.reset()
        self.or_tree_valid.reset()


class NewNormalizationUnit(nn.Module):
    """Integrated Pure SNN Normalization Unit"""
    def __init__(self):
        super().__init__()
        self.priority_encoder = PriorityEncoder8()
        self.shift_encoder = ShiftAmountEncoder8to3()
        self.barrel_shifter = BarrelShifter8()
        self.exponent_adjuster = ExponentAdjuster()
        
    def forward(self, P):
        # P is Little Endian [P0...P7]
        
        # 1. Find leading one
        valid = self.priority_encoder(P)
        
        # 2. Determine left shift amount (0..7)
        # Map V[7] -> 0, V[6] -> 1, ... V[0] -> 7
        shift_amt = self.shift_encoder(valid)
        
        # 3. Determine Exponent Adjustment (+1 to -6, or -7)
        exp_adj = self.exponent_adjuster(valid)
        
        # 4. Pre-shift Right 1 (Hardwired)
        # P_pre = P >> 1.  P_pre[6] = P[7], P_pre[5] = P[6]...
        # We can just rewire input to Barrel Shifter
        # BS Input: [P1, P2, P3, P4, P5, P6, P7, 0]
        zeros = torch.zeros_like(P[..., 0:1])
        p_bits = [P[..., i:i+1] for i in range(8)]
        
        # 保存 P[0] 作为 sticky_extra (被丢弃的最低位)
        sticky_extra = p_bits[0]
        
        # 保存 P[7] 作为 overflow 标志 (乘积是否溢出到 2.xxx)
        overflow = p_bits[7]
        
        p_pre_list = p_bits[1:] + [zeros] # [P1...P7, 0] -> Right Shift 1
        p_pre = torch.cat(p_pre_list, dim=-1)
        
        # 5. Shift Left
        p_norm = self.barrel_shifter(p_pre, shift_amt)
        
        # 返回 shift_amt 用于 Subnormal 路径修正
        return p_norm, exp_adj, sticky_extra, overflow, shift_amt

    def reset(self):
        self.priority_encoder.reset()
        self.shift_encoder.reset()
        self.barrel_shifter.reset()
        self.exponent_adjuster.reset()


class TemporalExponentGenerator(nn.Module):
    """时序指数生成器 (纯SNN)
    
    原理：
    - 内部维护一个4-bit递减计数器，初始值为最大指数 (E_max)。
    - 每个时间步，如果还没有检测到首脉冲，计数器减1。
    - 当检测到首脉冲时，当前的计数器值即为指数 E。
    - 如果减到0还没有首脉冲，保持为0 (Subnormal/Zero)。
    """
    def __init__(self, start_value=15, bits=4):
        super().__init__()
        self.bits = bits
        self.start_value = start_value
        
        # 4-bit 减法器 (加 -1)
        self.adder = RippleCarryAdder(bits=bits)
        
        # 状态寄存器 (4位)
        self.register_buffer('state', None)
        
        # 辅助门
        self.mux_update = nn.ModuleList([MUXGate() for _ in range(bits)])
        self.not_gate = NOTGate()
        
    def forward(self, has_fired, time_step_pulse):
        batch_size = has_fired.shape[0]
        device = has_fired.device
        
        if self.state is None:
            init_val = torch.tensor([self.start_value], device=device, dtype=torch.float32)
            state_bits = []
            for i in range(self.bits):
                bit = (init_val.int() >> i) & 1
                state_bits.append(bit.float().expand(batch_size, 1))
            self.state = torch.cat(state_bits, dim=-1)
            
        ones = torch.ones(batch_size, 1, device=device)
        minus_one = torch.cat([ones] * self.bits, dim=-1)
        
        state_minus_1, _ = self.adder(self.state, minus_one)
        do_update = self.not_gate(has_fired)
        
        new_state_list = []
        for i in range(self.bits):
            s_bit = self.mux_update[i](do_update, state_minus_1[..., i:i+1], self.state[..., i:i+1])
            new_state_list.append(s_bit)
            
        self.state = torch.cat(new_state_list, dim=-1)
        return self.state
    
    def reset(self):
        self.state = None
        self.adder.reset()
        for m in self.mux_update: m.reset()


class DelayNode(nn.Module):
    """纯SNN延迟单元 (D触发器)
    out[t] = in[t-1]
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('state', None)
        
    def forward(self, x):
        if self.state is None:
            self.state = torch.zeros_like(x)
        
        out = self.state
        self.state = x
        return out
    
    def reset(self):
        self.state = None


class ArrayMultiplier4x4_Strict(nn.Module):
    def __init__(self):
        super().__init__()
        self.pp_gates = nn.ModuleList([ANDGate() for _ in range(16)])
        
        # Row 1: 1 HA, 3 FA
        self.row1_ha = HalfAdder()
        self.row1_fa1 = FullAdder()
        self.row1_fa2 = FullAdder()
        self.row1_fa3 = FullAdder() 
        
        # Row 2: 1 HA, 3 FA
        self.row2_ha = HalfAdder()
        self.row2_fa1 = FullAdder()
        self.row2_fa2 = FullAdder()
        self.row2_fa3 = FullAdder()
        
        # Row 3: 1 HA, 3 FA
        self.row3_ha = HalfAdder()
        self.row3_fa1 = FullAdder()
        self.row3_fa2 = FullAdder()
        self.row3_fa3 = FullAdder()

    def forward(self, A, B):
        # Generate Partial Products
        p = [[None for _ in range(4)] for _ in range(4)]
        k = 0
        for i in range(4):
            for j in range(4):
                p[i][j] = self.pp_gates[k](A[..., j:j+1], B[..., i:i+1])
                k += 1
                
        # --- Layer 1 ---
        out0 = p[0][0] # P0
        s1_1, c1_1 = self.row1_ha(p[0][1], p[1][0])
        s1_2, c1_2 = self.row1_fa1(p[0][2], p[1][1], c1_1)
        s1_3, c1_3 = self.row1_fa2(p[0][3], p[1][2], c1_2)
        zeros = torch.zeros_like(out0)
        s1_4, c1_4 = self.row1_fa3(zeros, p[1][3], c1_3)
        
        # --- Layer 2 ---
        out1 = s1_1 # P1
        s2_2, c2_2 = self.row2_ha(s1_2, p[2][0])
        s2_3, c2_3 = self.row2_fa1(s1_3, p[2][1], c2_2)
        s2_4, c2_4 = self.row2_fa2(s1_4, p[2][2], c2_3)
        s2_5, c2_5 = self.row2_fa3(c1_4, p[2][3], c2_4)
        
        # --- Layer 3 ---
        out2 = s2_2 # P2
        s3_3, c3_3 = self.row3_ha(s2_3, p[3][0])
        out3 = s3_3 # P3
        s3_4, c3_4 = self.row3_fa1(s2_4, p[3][1], c3_3)
        out4 = s3_4 # P4
        s3_5, c3_5 = self.row3_fa2(s2_5, p[3][2], c3_4)
        out5 = s3_5 # P5
        s3_6, c3_6 = self.row3_fa3(c2_5, p[3][3], c3_5)
        out6 = s3_6 # P6
        out7 = c3_6 # P7
        
        return torch.cat([out0, out1, out2, out3, out4, out5, out6, out7], dim=-1)

    def reset(self):
        for g in self.pp_gates: g.reset()
        self.row1_ha.reset()
        self.row1_fa1.reset()
        self.row1_fa2.reset()
        self.row1_fa3.reset()
        self.row2_ha.reset()
        self.row2_fa1.reset()
        self.row2_fa2.reset()
        self.row2_fa3.reset()
        self.row3_ha.reset()
        self.row3_fa1.reset()
        self.row3_fa2.reset()
        self.row3_fa3.reset()


class Denormalizer(nn.Module):
    """Right Barrel Shifter for Denormalization
    
    Input: P [batch, 8], Shift [batch, 3]
    Output: P >> Shift
    """
    def __init__(self):
        super().__init__()
        self.mux_s4 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_s2 = nn.ModuleList([MUXGate() for _ in range(8)])
        self.mux_s1 = nn.ModuleList([MUXGate() for _ in range(8)])
        
    def forward(self, P, S):
        # P: Little Endian
        zeros = torch.zeros_like(P[..., 0:1])
        p_bits = [P[..., i:i+1] for i in range(8)]
        
        s0 = S[..., 0:1]
        s1 = S[..., 1:2]
        s2 = S[..., 2:3]
        
        # Stage 1: Shift 4
        stage1 = []
        for i in range(8):
            if i + 4 < 8:
                val = self.mux_s4[i](s2, p_bits[i+4], p_bits[i])
            else:
                val = self.mux_s4[i](s2, zeros, p_bits[i])
            stage1.append(val)
            
        # Stage 2: Shift 2
        stage2 = []
        for i in range(8):
            if i + 2 < 8:
                val = self.mux_s2[i](s1, stage1[i+2], stage1[i])
            else:
                val = self.mux_s2[i](s1, zeros, stage1[i])
            stage2.append(val)
            
        # Stage 3: Shift 1
        stage3 = []
        for i in range(8):
            if i + 1 < 8:
                val = self.mux_s1[i](s0, stage2[i+1], stage2[i])
            else:
                val = self.mux_s1[i](s0, zeros, stage2[i])
            stage3.append(val)
            
        return torch.cat(stage3, dim=-1)

    def reset(self):
        for m in self.mux_s4: m.reset()
        for m in self.mux_s2: m.reset()
        for m in self.mux_s1: m.reset()