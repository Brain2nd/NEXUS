"""
⚠️ 已弃用 (DEPRECATED) ⚠️
========================

本模块已被 neuron_template 统一架构取代。

请使用 logic_gates.py 中的组件，并通过 neuron_template 参数传递 SimpleLIFNode：

新用法示例
----------
```python
from SNNTorch.atomic_ops.logic_gates import ANDGate, SimpleLIFNode

# 使用 neuron_template 动态切换神经元类型
lif_template = SimpleLIFNode(beta=0.9)
and_gate = ANDGate(neuron_template=lif_template)
result = and_gate(a, b)

# 不同 beta 值测试
for beta in [1.0, 0.9, 0.5, 0.1]:
    gate = ANDGate(neuron_template=SimpleLIFNode(beta=beta))
    result = gate(a, b)
```

旧用法 (已弃用)
--------------
```python
# ❌ 不推荐
from SNNTorch.atomic_ops.logic_gates_lif import ANDGate_LIF
and_gate = ANDGate_LIF(beta=0.9)  # 硬编码 LIF
```

迁移指南
--------
| 旧类 (弃用) | 新用法 |
|-------------|--------|
| ANDGate_LIF(beta) | ANDGate(neuron_template=SimpleLIFNode(beta)) |
| ORGate_LIF(beta) | ORGate(neuron_template=SimpleLIFNode(beta)) |
| XORGate_LIF(beta) | XORGate(neuron_template=SimpleLIFNode(beta)) |
| RippleCarryAdder_LIF(bits, beta) | RippleCarryAdder(bits, neuron_template=SimpleLIFNode(beta)) |
| ArrayMultiplier4x4_LIF(beta) | ArrayMultiplier4x4_Strict(neuron_template=SimpleLIFNode(beta)) |

---

LIF 神经元逻辑门 (LIF Neuron Logic Gates) - 已弃用
=================================================

用于物理硬件模拟和鲁棒性测试的 LIF 版本逻辑门。

神经元动力学
-----------

**Leaky Integrate-and-Fire (LIF) 模型**:

```
膜电位更新: V(t+1) = β × V(t) + I(t)
脉冲发放:   S(t) = H(V(t) - V_th)
软重置:     V(t) = V(t) - S(t) × V_th
```

作者: HumanBrain Project
许可: MIT License
"""
import warnings
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

# 发出弃用警告
warnings.warn(
    "logic_gates_lif 模块已弃用。请使用 logic_gates 中的组件并传递 neuron_template=SimpleLIFNode(beta)。"
    "详见模块文档中的迁移指南。",
    DeprecationWarning,
    stacklevel=2
)


class SimpleLIFNode(nn.Module):
    """简化的LIF神经元，直接使用beta参数控制泄漏"""
    def __init__(self, beta=1.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.beta = beta
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.register_buffer('v', None)
        
    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)
        
        # LIF动力学: V = beta * V + I
        self.v = self.beta * self.v + x
        
        # 发放判断
        spike = (self.v >= self.v_threshold).float()
        
        # 软重置
        self.v = self.v - spike * self.v_threshold
        
        return spike
    
    def reset(self):
        self.v = None


class ANDGate_LIF(nn.Module):
    """LIF版本AND门"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.node = SimpleLIFNode(beta=beta, v_threshold=1.5, v_reset=0.0)
        
    def forward(self, x_a, x_b):
        self.reset()
        return self.node(x_a + x_b)
        
    def reset(self):
        self.node.reset()


class ORGate_LIF(nn.Module):
    """LIF版本OR门"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.node = SimpleLIFNode(beta=beta, v_threshold=0.5, v_reset=0.0)
        
    def forward(self, x_a, x_b):
        self.reset()
        return self.node(x_a + x_b)

    def reset(self):
        self.node.reset()


class XORGate_LIF(nn.Module):
    """LIF版本XOR门"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.hidden_node = SimpleLIFNode(beta=beta, v_threshold=1.5, v_reset=0.0)
        self.out_node = SimpleLIFNode(beta=beta, v_threshold=0.5, v_reset=0.0)
        
    def forward(self, x_a, x_b):
        self.reset()
        hidden_spike = self.hidden_node(x_a + x_b)
        out_spike = self.out_node(x_a + x_b - 2.0 * hidden_spike)
        return out_spike

    def reset(self):
        self.hidden_node.reset()
        self.out_node.reset()


class NOTGate_LIF(nn.Module):
    """LIF版本NOT门 - 使用LIF神经元 + 抑制性突触实现
    
    数学原理:
    - 偏置电流 (bias): 1.0 (恒定兴奋性)
    - 突触权重 (weight): -1.0 (抑制性)
    - 突触电流: I = bias + weight * x = 1.0 - x
    - 通过LIF神经元发放脉冲
    
    真值表:
    | A | I = 1-A | 输出 |
    |---|---------|------|
    | 0 |    1    | 1    |
    | 1 |    0    | 0    |
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.node = SimpleLIFNode(beta=beta, v_threshold=0.5, v_reset=0.0)
        self.bias = 1.0
        self.weight = -1.0
        
    def forward(self, x):
        self.reset()
        # 计算突触电流 (这是神经元输入，不是逻辑操作)
        current = self.bias + self.weight * x
        return self.node(current)
    
    def reset(self):
        self.node.reset()


class HalfAdder_LIF(nn.Module):
    """LIF版本半加器"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.xor1 = XORGate_LIF(beta)
        self.and1 = ANDGate_LIF(beta)
        
    def forward(self, a, b):
        self.reset()
        s = self.xor1(a, b)
        c = self.and1(a, b)
        return s, c
        
    def reset(self):
        self.xor1.reset()
        self.and1.reset()


class FullAdder_LIF(nn.Module):
    """LIF版本全加器"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.xor1 = XORGate_LIF(beta)
        self.xor2 = XORGate_LIF(beta)
        self.and1 = ANDGate_LIF(beta)
        self.and2 = ANDGate_LIF(beta)
        self.or1 = ORGate_LIF(beta)
        
    def forward(self, a, b, cin):
        self.reset()
        s1 = self.xor1(a, b)
        sum_out = self.xor2(s1, cin)
        c1 = self.and1(a, b)
        c2 = self.and2(s1, cin)
        cout = self.or1(c1, c2)
        return sum_out, cout

    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


class MUXGate_LIF(nn.Module):
    """LIF版本MUX选择器
    
    MUX(s, a, b) = (s AND a) OR (NOT(s) AND b)
    s=1 选择 a, s=0 选择 b
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.not_s = NOTGate_LIF(beta)  # 使用 LIF NOT 门
        self.and1 = ANDGate_LIF(beta)
        self.and2 = ANDGate_LIF(beta)
        self.or1 = ORGate_LIF(beta)
        
    def forward(self, s, a, b):
        self.reset()
        not_s = self.not_s(s)  # 通过 LIF 神经元取反
        sa = self.and1(s, a)
        nsb = self.and2(not_s, b)
        return self.or1(sa, nsb)
    
    def reset(self):
        self.not_s.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


class RippleCarryAdder_LIF(nn.Module):
    """LIF版本波纹进位加法器"""
    def __init__(self, bits=4, beta=1.0):
        super().__init__()
        self.bits = bits
        self.beta = beta
        self.adders = nn.ModuleList([FullAdder_LIF(beta) for _ in range(bits)])
        
    def forward(self, A, B, Cin=None):
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


# ==============================================================================
# 扩展 LIF 组件 - 用于 FP8/FP16/FP32 物理模拟
# ==============================================================================

class OR3Gate_LIF(nn.Module):
    """LIF版本三输入OR门"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.or1 = ORGate_LIF(beta)
        self.or2 = ORGate_LIF(beta)
        
    def forward(self, a, b, c):
        self.reset()
        ab = self.or1(a, b)
        return self.or2(ab, c)
    
    def reset(self):
        self.or1.reset()
        self.or2.reset()


class ORTree_LIF(nn.Module):
    """LIF版本 N输入OR树"""
    def __init__(self, n_inputs, beta=1.0):
        super().__init__()
        self.n_inputs = n_inputs
        self.beta = beta
        n_gates = n_inputs - 1 if n_inputs > 1 else 0
        self.or_gates = nn.ModuleList([ORGate_LIF(beta) for _ in range(n_gates)])
        
    def forward(self, inputs):
        if isinstance(inputs, list):
            current = inputs
        else:
            current = [inputs[..., i:i+1] for i in range(self.n_inputs)]
        
        if len(current) == 0:
            return torch.zeros_like(current[0]) if current else None
        if len(current) == 1:
            return current[0]
            
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


class ANDTree_LIF(nn.Module):
    """LIF版本 N输入AND树"""
    def __init__(self, n_inputs, beta=1.0):
        super().__init__()
        self.n_inputs = n_inputs
        self.beta = beta
        n_gates = n_inputs - 1 if n_inputs > 1 else 0
        self.and_gates = nn.ModuleList([ANDGate_LIF(beta) for _ in range(n_gates)])
        
    def forward(self, inputs):
        if isinstance(inputs, list):
            current = inputs
        else:
            current = [inputs[..., i:i+1] for i in range(self.n_inputs)]
        
        if len(current) == 0:
            return torch.ones_like(current[0]) if current else None
        if len(current) == 1:
            return current[0]
            
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


class PriorityEncoder8_LIF(nn.Module):
    """LIF版本 8-bit Priority Encoder (Leading One Detector)
    
    Input: P [batch, 8] (Little Endian: P0...P7)
    Output: Valid [batch, 8] (One-Hot, only the MSB '1' is active)
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.or_chain = nn.ModuleList([ORGate_LIF(beta) for _ in range(7)])
        self.not_gates = nn.ModuleList([NOTGate_LIF(beta) for _ in range(8)])
        self.and_gates = nn.ModuleList([ANDGate_LIF(beta) for _ in range(8)])
        
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


class ShiftAmountEncoder8to3_LIF(nn.Module):
    """LIF版本 Encode One-Hot 8-bit to 3-bit Binary"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.or_tree_s0 = ORTree_LIF(4, beta)
        self.or_tree_s1 = ORTree_LIF(4, beta)
        self.or_tree_s2 = ORTree_LIF(4, beta)
        
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


class BarrelShifter8_LIF(nn.Module):
    """LIF版本 8-bit Left Barrel Shifter"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.mux_s4 = nn.ModuleList([MUXGate_LIF(beta) for _ in range(8)])
        self.mux_s2 = nn.ModuleList([MUXGate_LIF(beta) for _ in range(8)])
        self.mux_s1 = nn.ModuleList([MUXGate_LIF(beta) for _ in range(8)])
        
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


class Denormalizer_LIF(nn.Module):
    """LIF版本 Right Barrel Shifter for Denormalization
    
    Input: P [batch, 8], Shift [batch, 3]
    Output: P >> Shift
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.mux_s4 = nn.ModuleList([MUXGate_LIF(beta) for _ in range(8)])
        self.mux_s2 = nn.ModuleList([MUXGate_LIF(beta) for _ in range(8)])
        self.mux_s1 = nn.ModuleList([MUXGate_LIF(beta) for _ in range(8)])
        
    def forward(self, P, S):
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


class ArrayMultiplier4x4_LIF(nn.Module):
    """LIF版本 4x4位阵列乘法器"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.pp_gates = nn.ModuleList([ANDGate_LIF(beta) for _ in range(16)])
        
        # Row 1: 1 HA, 3 FA
        self.row1_ha = HalfAdder_LIF(beta)
        self.row1_fa1 = FullAdder_LIF(beta)
        self.row1_fa2 = FullAdder_LIF(beta)
        self.row1_fa3 = FullAdder_LIF(beta)
        
        # Row 2: 1 HA, 3 FA
        self.row2_ha = HalfAdder_LIF(beta)
        self.row2_fa1 = FullAdder_LIF(beta)
        self.row2_fa2 = FullAdder_LIF(beta)
        self.row2_fa3 = FullAdder_LIF(beta)
        
        # Row 3: 1 HA, 3 FA
        self.row3_ha = HalfAdder_LIF(beta)
        self.row3_fa1 = FullAdder_LIF(beta)
        self.row3_fa2 = FullAdder_LIF(beta)
        self.row3_fa3 = FullAdder_LIF(beta)

    def forward(self, A, B):
        # Generate Partial Products
        p = [[None for _ in range(4)] for _ in range(4)]
        k = 0
        for i in range(4):
            for j in range(4):
                p[i][j] = self.pp_gates[k](A[..., j:j+1], B[..., i:i+1])
                k += 1
                
        # --- Layer 1 ---
        out0 = p[0][0]
        s1_1, c1_1 = self.row1_ha(p[0][1], p[1][0])
        s1_2, c1_2 = self.row1_fa1(p[0][2], p[1][1], c1_1)
        s1_3, c1_3 = self.row1_fa2(p[0][3], p[1][2], c1_2)
        zeros = torch.zeros_like(out0)
        s1_4, c1_4 = self.row1_fa3(zeros, p[1][3], c1_3)
        
        # --- Layer 2 ---
        out1 = s1_1
        s2_2, c2_2 = self.row2_ha(s1_2, p[2][0])
        s2_3, c2_3 = self.row2_fa1(s1_3, p[2][1], c2_2)
        s2_4, c2_4 = self.row2_fa2(s1_4, p[2][2], c2_3)
        s2_5, c2_5 = self.row2_fa3(c1_4, p[2][3], c2_4)
        
        # --- Layer 3 ---
        out2 = s2_2
        s3_3, c3_3 = self.row3_ha(s2_3, p[3][0])
        out3 = s3_3
        s3_4, c3_4 = self.row3_fa1(s2_4, p[3][1], c3_3)
        out4 = s3_4
        s3_5, c3_5 = self.row3_fa2(s2_5, p[3][2], c3_4)
        out5 = s3_5
        s3_6, c3_6 = self.row3_fa3(c2_5, p[3][3], c3_5)
        out6 = s3_6
        out7 = c3_6
        
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


class SubtractorNBit_LIF(nn.Module):
    """LIF版本 N-bit 减法器 (使用补码加法)
    
    A - B = A + (~B) + 1
    """
    def __init__(self, bits=4, beta=1.0):
        super().__init__()
        self.bits = bits
        self.beta = beta
        self.not_gates = nn.ModuleList([NOTGate_LIF(beta) for _ in range(bits)])
        self.adder = RippleCarryAdder_LIF(bits, beta)
        
    def forward(self, A, B):
        # 计算 ~B
        not_b_list = []
        for i in range(self.bits):
            not_b_list.append(self.not_gates[i](B[..., i:i+1]))
        not_b = torch.cat(not_b_list, dim=-1)
        
        # A + (~B) + 1
        ones = torch.ones_like(A[..., 0:1])
        result, cout = self.adder(A, not_b, Cin=ones)
        
        return result, cout
    
    def reset(self):
        for g in self.not_gates: g.reset()
        self.adder.reset()


class Comparator4Bit_LIF(nn.Module):
    """LIF版本 4-bit 比较器
    
    比较 A 和 B，输出 A > B, A == B, A < B
    使用减法实现: A - B 的符号位
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.sub = SubtractorNBit_LIF(bits=4, beta=beta)
        self.not_borrow = NOTGate_LIF(beta)
        self.or_tree = ORTree_LIF(4, beta)
        self.not_any_diff = NOTGate_LIF(beta)
        self.and_gt = ANDGate_LIF(beta)
        
    def forward(self, A, B):
        # A - B
        diff, borrow = self.sub(A, B)
        
        # A > B: 没有借位且结果不为0
        no_borrow = self.not_borrow(borrow)
        diff_bits = [diff[..., i:i+1] for i in range(4)]
        any_diff = self.or_tree(diff_bits)
        a_gt_b = self.and_gt(no_borrow, any_diff)
        
        # A == B: 差为0
        a_eq_b = self.not_any_diff(any_diff)
        
        # A < B: 有借位
        a_lt_b = borrow
        
        return a_gt_b, a_eq_b, a_lt_b
    
    def reset(self):
        self.sub.reset()
        self.not_borrow.reset()
        self.or_tree.reset()
        self.not_any_diff.reset()
        self.and_gt.reset()


# ==============================================================================
# 测试函数
# ==============================================================================

def test_lif_gates():
    """测试LIF逻辑门的正确性"""
    print("=" * 60)
    print("测试LIF版本逻辑门 (SimpleLIF)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 测试不同beta值
    betas = [1.0, 0.99, 0.95, 0.9]
    
    for beta in betas:
        print(f"\n--- β = {beta} ---")
        
        # 创建门
        and_gate = ANDGate_LIF(beta).to(device)
        or_gate = ORGate_LIF(beta).to(device)
        xor_gate = XORGate_LIF(beta).to(device)
        
        # 测试所有输入组合
        inputs = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0)
        ]
        
        # 期望输出
        expected_and = [0, 0, 0, 1]
        expected_or = [0, 1, 1, 1]
        expected_xor = [0, 1, 1, 0]
        
        and_correct = 0
        or_correct = 0
        xor_correct = 0
        
        for i, (a, b) in enumerate(inputs):
            a_t = torch.tensor([a], device=device)
            b_t = torch.tensor([b], device=device)
            
            and_gate.reset()
            or_gate.reset()
            xor_gate.reset()
            
            and_out = and_gate(a_t, b_t).item()
            or_out = or_gate(a_t, b_t).item()
            xor_out = xor_gate(a_t, b_t).item()
            
            if round(and_out) == expected_and[i]:
                and_correct += 1
            if round(or_out) == expected_or[i]:
                or_correct += 1
            if round(xor_out) == expected_xor[i]:
                xor_correct += 1
        
        print(f"  AND: {and_correct}/4")
        print(f"  OR:  {or_correct}/4")
        print(f"  XOR: {xor_correct}/4")
        
    # 测试全加器
    print("\n--- 全加器测试 (β=1.0) ---")
    fa = FullAdder_LIF(beta=1.0).to(device)
    
    fa_cases = [
        (0, 0, 0, 0, 0),  # a, b, cin, sum, cout
        (1, 0, 0, 1, 0),
        (0, 1, 0, 1, 0),
        (1, 1, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (1, 0, 1, 0, 1),
        (0, 1, 1, 0, 1),
        (1, 1, 1, 1, 1),
    ]
    
    fa_correct = 0
    for a, b, cin, exp_s, exp_c in fa_cases:
        fa.reset()
        a_t = torch.tensor([float(a)], device=device)
        b_t = torch.tensor([float(b)], device=device)
        cin_t = torch.tensor([float(cin)], device=device)
        s, c = fa(a_t, b_t, cin_t)
        if round(s.item()) == exp_s and round(c.item()) == exp_c:
            fa_correct += 1
    
    print(f"  FullAdder: {fa_correct}/8")
    
    # 测试扩展组件
    print("\n--- 扩展组件测试 (β=1.0) ---")
    
    # 4x4 乘法器
    mul = ArrayMultiplier4x4_LIF(beta=1.0).to(device)
    mul_correct = 0
    for a_val in range(4):  # 测试部分
        for b_val in range(4):
            a_bits = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
            b_bits = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
            mul.reset()
            result = mul(a_bits, b_bits)
            result_val = sum(int(round(result[0, i].item())) << i for i in range(8))
            if result_val == a_val * b_val:
                mul_correct += 1
    print(f"  4x4 Multiplier: {mul_correct}/16")
    
    # Barrel Shifter
    bs = BarrelShifter8_LIF(beta=1.0).to(device)
    bs_correct = 0
    for shift in range(4):  # 测试 shift 0-3
        p = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]], device=device)  # 值=1
        s = torch.tensor([[float((shift >> 0) & 1), float((shift >> 1) & 1), float((shift >> 2) & 1)]], device=device)
        bs.reset()
        result = bs(p, s)
        result_val = sum(int(round(result[0, i].item())) << i for i in range(8))
        expected = 1 << shift
        if result_val == expected:
            bs_correct += 1
    print(f"  BarrelShifter8: {bs_correct}/4")


if __name__ == "__main__":
    test_lif_gates()

