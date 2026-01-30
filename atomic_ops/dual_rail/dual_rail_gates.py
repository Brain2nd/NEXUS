"""
双轨编码 SNN 逻辑门电路库 (Dual-Rail SNN Logic Gates Library)
=============================================================

基于双轨编码的纯 SNN 逻辑门电路实现。

核心原则
--------

**双轨编码**:
- 每个逻辑信号 A 表示为 (A_pos, A_neg) = (A, NOT_A)
- A=1 → (1, 0)
- A=0 → (0, 1)

**纯 SNN 约束**:
- 只使用 +1 权重的脉冲汇聚
- 只使用阈值判断（IF神经元）
- NOT 门 = 线路交换（零计算）
- 无负权重，无浮点乘法

**逻辑门实现**:

| 门类型 | 正极性输出 | 负极性输出 | 说明 |
|--------|------------|------------|------|
| NOT    | A_neg      | A_pos      | 线路交换 |
| AND    | A_pos AND B_pos | A_neg OR B_neg | 德摩根定律 |
| OR     | A_pos OR B_pos | A_neg AND B_neg | 德摩根定律 |
| XOR    | (A_pos AND B_neg) OR (A_neg AND B_pos) | (A_pos AND B_pos) OR (A_neg AND B_neg) | XNOR |

**学术价值**:
- 100%位精确完全来自空间组合逻辑
- 不依赖权重精度
- NOT 是纯拓扑操作，零计算

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn
from copy import deepcopy
from atomic_ops.core.neurons import SimpleIFNode, SimpleLIFNode


# ==============================================================================
# 神经元模板 (已移至 neurons.py)
# ==============================================================================

def _create_neuron(template, threshold, v_reset=None,
                   beta=None, max_param_shape=None,
                   # 兼容旧调用，忽略这些参数
                   trainable_threshold=True, trainable_beta=True):
    """从模板创建指定阈值的神经元

    所有参数始终为 nn.Parameter (可训练)，无需 trainable 开关。

    Args:
        template: 神经元模板，None 则创建默认 SimpleLIFNode
        threshold: 目标阈值 (float 或 Tensor)
        v_reset: 复位电压 (None=软复位, 数值=硬复位)
        beta: 泄漏因子 (None=DEFAULT_BETA)
        max_param_shape: 预分配参数形状，None 使用全局默认
    """
    if template is None:
        return SimpleLIFNode(
            beta=beta,
            v_threshold=threshold,
            v_reset=v_reset,
            max_param_shape=max_param_shape
        )
    else:
        node = deepcopy(template)
        node.v_threshold = threshold
        if hasattr(node, 'v_reset'):
            node.v_reset = v_reset
        if max_param_shape is not None and hasattr(node, 'max_param_shape'):
            node.max_param_shape = max_param_shape
            if hasattr(node, '_preallocate_params'):
                node._preallocate_params(max_param_shape)
        return node


# ==============================================================================
# 双轨编码工具
# ==============================================================================

def to_dual_rail(x):
    """单轨转双轨（边界组件，允许 1-x）
    
    Args:
        x: [...] 单轨脉冲
    Returns:
        [..., 2] 双轨脉冲 (pos, neg)
    """
    pos = x.unsqueeze(-1)
    neg = (1.0 - x).unsqueeze(-1)  # 边界转换，允许
    return torch.cat([pos, neg], dim=-1)


def from_dual_rail(x):
    """双轨转单轨（只取正极性）
    
    Args:
        x: [..., 2] 双轨脉冲
    Returns:
        [...] 单轨脉冲（正极性）
    """
    return x[..., 0]


# ==============================================================================
# 基础双轨门电路
# ==============================================================================

class DualRailNOT(nn.Module):
    """双轨 NOT 门 - 纯拓扑操作，零计算！
    
    **原理**: NOT(A_pos, A_neg) = (A_neg, A_pos)
    只需交换正负极性线路，不需要任何神经元。
    
    **这是双轨编码的核心优势**：NOT 完全不需要计算。
    """
    def forward(self, x):
        # x: [..., 2] 双轨输入
        # 交换正负极性
        return x.flip(-1)
    
    def reset(self):
        pass  # 无状态


class DualRailAND(nn.Module):
    """双轨 AND 门 - 只使用 +1 权重
    
    **原理**:
    - Y_pos = A_pos AND B_pos (脉冲汇聚，阈值 1.5)
    - Y_neg = NOT(A AND B) = A_neg OR B_neg (脉冲汇聚，阈值 0.5)
    
    **无负权重**：所有操作都是 +1 权重的脉冲汇聚。
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.and_node = _create_neuron(neuron_template, threshold=1.5)
        self.or_node = _create_neuron(neuron_template, threshold=0.5)
    
    def forward(self, a, b):
        # a, b: [..., 2] 双轨
        a_pos, a_neg = a[..., 0:1], a[..., 1:2]
        b_pos, b_neg = b[..., 0:1], b[..., 1:2]

        # Y_pos = A AND B (脉冲汇聚，阈值1.5)
        y_pos = self.and_node(a_pos + b_pos)
        
        # Y_neg = NOT(A AND B) = A_neg OR B_neg (脉冲汇聚，阈值0.5)
        y_neg = self.or_node(a_neg + b_neg)
        
        return torch.cat([y_pos, y_neg], dim=-1)
    
    def reset(self):
        self.and_node.reset()
        self.or_node.reset()


class DualRailOR(nn.Module):
    """双轨 OR 门 - 只使用 +1 权重
    
    **原理**:
    - Y_pos = A_pos OR B_pos (脉冲汇聚，阈值 0.5)
    - Y_neg = NOT(A OR B) = A_neg AND B_neg (脉冲汇聚，阈值 1.5)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.or_node = _create_neuron(neuron_template, threshold=0.5)
        self.and_node = _create_neuron(neuron_template, threshold=1.5)
    
    def forward(self, a, b):
        a_pos, a_neg = a[..., 0:1], a[..., 1:2]
        b_pos, b_neg = b[..., 0:1], b[..., 1:2]

        # Y_pos = A OR B
        y_pos = self.or_node(a_pos + b_pos)
        
        # Y_neg = NOT(A OR B) = A_neg AND B_neg
        y_neg = self.and_node(a_neg + b_neg)
        
        return torch.cat([y_pos, y_neg], dim=-1)
    
    def reset(self):
        self.or_node.reset()
        self.and_node.reset()


class DualRailXOR(nn.Module):
    """双轨 XOR 门 - 只使用 +1 权重，无 -2.0*h！
    
    **原理**:
    - XOR = (A AND NOT_B) OR (NOT_A AND B)
    - Y_pos = (A_pos AND B_neg) OR (A_neg AND B_pos)
    - Y_neg = XNOR = (A_pos AND B_pos) OR (A_neg AND B_neg)
    
    **关键优势**: 完全不需要 -2.0 权重！
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        # XOR: (A AND NOT_B) OR (NOT_A AND B)
        self.and1 = _create_neuron(neuron_template, threshold=1.5)
        self.and2 = _create_neuron(neuron_template, threshold=1.5)
        self.or_xor = _create_neuron(neuron_template, threshold=0.5)
        
        # XNOR: (A AND B) OR (NOT_A AND NOT_B)
        self.and3 = _create_neuron(neuron_template, threshold=1.5)
        self.and4 = _create_neuron(neuron_template, threshold=1.5)
        self.or_xnor = _create_neuron(neuron_template, threshold=0.5)
    
    def forward(self, a, b):
        a_pos, a_neg = a[..., 0:1], a[..., 1:2]
        b_pos, b_neg = b[..., 0:1], b[..., 1:2]

        # XOR = (A_pos AND B_neg) OR (A_neg AND B_pos)
        term1 = self.and1(a_pos + b_neg)  # A AND NOT_B
        term2 = self.and2(a_neg + b_pos)  # NOT_A AND B
        y_pos = self.or_xor(term1 + term2)
        
        # XNOR = (A_pos AND B_pos) OR (A_neg AND B_neg)
        term3 = self.and3(a_pos + b_pos)  # A AND B
        term4 = self.and4(a_neg + b_neg)  # NOT_A AND NOT_B
        y_neg = self.or_xnor(term3 + term4)
        
        return torch.cat([y_pos, y_neg], dim=-1)
    
    def reset(self):
        self.and1.reset()
        self.and2.reset()
        self.or_xor.reset()
        self.and3.reset()
        self.and4.reset()
        self.or_xnor.reset()


class DualRailMUX(nn.Module):
    """双轨 MUX 门 - sel=1选a, sel=0选b
    
    MUX(s, a, b) = (s AND a) OR (NOT_s AND b)
    
    使用双轨编码，NOT_s 直接从 s_neg 获取，无需计算。
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.and1 = DualRailAND(neuron_template)  # s AND a
        self.and2 = DualRailAND(neuron_template)  # NOT_s AND b
        self.or_gate = DualRailOR(neuron_template)
    
    def forward(self, sel, a, b):
        # sel, a, b: [..., 2] 双轨
        # NOT_sel 直接从 sel 的负极性获取
        not_sel = sel.flip(-1)  # 交换线路 = NOT
        
        sa = self.and1(sel, a)       # s AND a
        nsb = self.and2(not_sel, b)  # NOT_s AND b
        return self.or_gate(sa, nsb)
    
    def reset(self):
        self.and1.reset()
        self.and2.reset()
        self.or_gate.reset()


# ==============================================================================
# 双轨算术单元
# ==============================================================================

class DualRailHalfAdder(nn.Module):
    """双轨半加器
    
    S = A XOR B
    C = A AND B
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.xor_gate = DualRailXOR(neuron_template)
        self.and_gate = DualRailAND(neuron_template)
    
    def forward(self, a, b):
        s = self.xor_gate(a, b)  # S = A XOR B
        c = self.and_gate(a, b)  # C = A AND B
        return s, c
    
    def reset(self):
        self.xor_gate.reset()
        self.and_gate.reset()


class DualRailFullAdder(nn.Module):
    """双轨全加器
    
    S = A XOR B XOR Cin
    Cout = (A AND B) OR ((A XOR B) AND Cin)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.xor1 = DualRailXOR(neuron_template)
        self.xor2 = DualRailXOR(neuron_template)
        self.and1 = DualRailAND(neuron_template)
        self.and2 = DualRailAND(neuron_template)
        self.or1 = DualRailOR(neuron_template)
    
    def forward(self, a, b, cin):
        s1 = self.xor1(a, b)          # A XOR B
        sum_out = self.xor2(s1, cin)  # S = (A XOR B) XOR Cin
        
        c1 = self.and1(a, b)          # A AND B
        c2 = self.and2(s1, cin)       # (A XOR B) AND Cin
        cout = self.or1(c1, c2)       # Cout = c1 OR c2
        
        return sum_out, cout
    
    def reset(self):
        self.xor1.reset()
        self.xor2.reset()
        self.and1.reset()
        self.and2.reset()
        self.or1.reset()


# ==============================================================================
# 单轨接口包装器（对上层透明）
# ==============================================================================

class PureSNN_NOT(nn.Module):
    """纯 SNN NOT 门 - 对上层透明的单轨接口
    
    内部使用双轨编码，外部看到的是单轨接口。
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_not = DualRailNOT()
    
    def forward(self, x):
        # 单轨 -> 双轨
        x_dual = to_dual_rail(x)
        # 双轨 NOT
        y_dual = self.dual_not(x_dual)
        # 双轨 -> 单轨
        return from_dual_rail(y_dual)
    
    def reset(self):
        pass


class PureSNN_AND(nn.Module):
    """纯 SNN AND 门 - 单轨接口"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_and = DualRailAND(neuron_template)
    
    def forward(self, a, b):
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        y_dual = self.dual_and(a_dual, b_dual)
        return from_dual_rail(y_dual)
    
    def reset(self):
        self.dual_and.reset()


class PureSNN_OR(nn.Module):
    """纯 SNN OR 门 - 单轨接口"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_or = DualRailOR(neuron_template)
    
    def forward(self, a, b):
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        y_dual = self.dual_or(a_dual, b_dual)
        return from_dual_rail(y_dual)
    
    def reset(self):
        self.dual_or.reset()


class PureSNN_XOR(nn.Module):
    """纯 SNN XOR 门 - 单轨接口"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_xor = DualRailXOR(neuron_template)
    
    def forward(self, a, b):
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        y_dual = self.dual_xor(a_dual, b_dual)
        return from_dual_rail(y_dual)
    
    def reset(self):
        self.dual_xor.reset()


class PureSNN_MUX(nn.Module):
    """纯 SNN MUX 门 - 单轨接口"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_mux = DualRailMUX(neuron_template)
    
    def forward(self, sel, a, b):
        sel_dual = to_dual_rail(sel)
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        y_dual = self.dual_mux(sel_dual, a_dual, b_dual)
        return from_dual_rail(y_dual)
    
    def reset(self):
        self.dual_mux.reset()


class PureSNN_HalfAdder(nn.Module):
    """纯 SNN 半加器 - 单轨接口"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_ha = DualRailHalfAdder(neuron_template)
    
    def forward(self, a, b):
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        s_dual, c_dual = self.dual_ha(a_dual, b_dual)
        return from_dual_rail(s_dual), from_dual_rail(c_dual)
    
    def reset(self):
        self.dual_ha.reset()


class PureSNN_FullAdder(nn.Module):
    """纯 SNN 全加器 - 单轨接口"""
    def __init__(self, neuron_template=None):
        super().__init__()
        self.dual_fa = DualRailFullAdder(neuron_template)
    
    def forward(self, a, b, cin):
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        cin_dual = to_dual_rail(cin)
        s_dual, cout_dual = self.dual_fa(a_dual, b_dual, cin_dual)
        return from_dual_rail(s_dual), from_dual_rail(cout_dual)
    
    def reset(self):
        self.dual_fa.reset()

