"""
统一神经元架构测试 - 验证 neuron_template 参数功能
===================================================

测试 IF/LIF 神经元切换功能，确保：
1. 默认 IF 神经元正常工作
2. LIF 神经元模板正常工作
3. 参数向下传递正确
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from atomic_ops.logic_gates import (
    ANDGate, ORGate, XORGate, NOTGate, MUXGate,
    HalfAdder, FullAdder, RippleCarryAdder,
    ORTree, ANDTree, ArrayMultiplier4x4_Strict,
    SimpleLIFNode
)
from atomic_ops.vec_logic_gates import (
    VecAND, VecOR, VecXOR, VecNOT, VecMUX,
    VecORTree, VecANDTree,
    VecAdder, VecSubtractor, VecComparator
)


def test_basic_gates_if():
    """测试基础门电路 - 默认 IF 神经元"""
    print("\n--- 基础门电路 (IF) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 AND 门
    and_gate = ANDGate().to(device)
    test_cases = [
        ((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)
    ]
    
    and_correct = 0
    for (a, b), expected in test_cases:
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        and_gate.reset()
        result = and_gate(a_t, b_t)
        if round(result.item()) == expected:
            and_correct += 1
    
    print(f"  AND 门: {and_correct}/4")
    
    # 测试 OR 门
    or_gate = ORGate().to(device)
    or_expected = [0, 1, 1, 1]
    
    or_correct = 0
    for i, ((a, b), _) in enumerate(test_cases):
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        or_gate.reset()
        result = or_gate(a_t, b_t)
        if round(result.item()) == or_expected[i]:
            or_correct += 1
    
    print(f"  OR 门: {or_correct}/4")
    
    # 测试 XOR 门
    xor_gate = XORGate().to(device)
    xor_expected = [0, 1, 1, 0]
    
    xor_correct = 0
    for i, ((a, b), _) in enumerate(test_cases):
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        xor_gate.reset()
        result = xor_gate(a_t, b_t)
        if round(result.item()) == xor_expected[i]:
            xor_correct += 1
    
    print(f"  XOR 门: {xor_correct}/4")
    
    # 测试 NOT 门
    not_gate = NOTGate().to(device)
    not_correct = 0
    for inp, exp in [(0, 1), (1, 0)]:
        not_gate.reset()
        result = not_gate(torch.tensor([[float(inp)]], device=device))
        if round(result.item()) == exp:
            not_correct += 1
    
    print(f"  NOT 门: {not_correct}/2")
    
    all_correct = and_correct == 4 and or_correct == 4 and xor_correct == 4 and not_correct == 2
    return all_correct


def test_basic_gates_lif():
    """测试基础门电路 - LIF 神经元"""
    print("\n--- 基础门电路 (LIF, β=0.9) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lif_template = SimpleLIFNode(beta=0.9)
    
    # 测试 AND 门
    and_gate = ANDGate(neuron_template=lif_template).to(device)
    test_cases = [
        ((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)
    ]
    
    and_correct = 0
    for (a, b), expected in test_cases:
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        and_gate.reset()
        result = and_gate(a_t, b_t)
        if round(result.item()) == expected:
            and_correct += 1
    
    print(f"  AND 门: {and_correct}/4")
    
    # 测试 OR 门
    or_gate = ORGate(neuron_template=lif_template).to(device)
    or_expected = [0, 1, 1, 1]
    
    or_correct = 0
    for i, ((a, b), _) in enumerate(test_cases):
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        or_gate.reset()
        result = or_gate(a_t, b_t)
        if round(result.item()) == or_expected[i]:
            or_correct += 1
    
    print(f"  OR 门: {or_correct}/4")
    
    # 测试 XOR 门
    xor_gate = XORGate(neuron_template=lif_template).to(device)
    xor_expected = [0, 1, 1, 0]
    
    xor_correct = 0
    for i, ((a, b), _) in enumerate(test_cases):
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        xor_gate.reset()
        result = xor_gate(a_t, b_t)
        if round(result.item()) == xor_expected[i]:
            xor_correct += 1
    
    print(f"  XOR 门: {xor_correct}/4")
    
    all_correct = and_correct == 4 and or_correct == 4 and xor_correct == 4
    return all_correct


def test_mux_gate():
    """测试 MUX 选择器"""
    print("\n--- MUX 选择器 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # IF 版本
    mux_if = MUXGate().to(device)
    
    # MUX(s, a, b): s=1 选 a, s=0 选 b
    test_cases = [
        # (s, a, b, expected)
        (0, 0, 0, 0),  # s=0, 选 b=0
        (0, 0, 1, 1),  # s=0, 选 b=1
        (0, 1, 0, 0),  # s=0, 选 b=0
        (0, 1, 1, 1),  # s=0, 选 b=1
        (1, 0, 0, 0),  # s=1, 选 a=0
        (1, 0, 1, 0),  # s=1, 选 a=0
        (1, 1, 0, 1),  # s=1, 选 a=1
        (1, 1, 1, 1),  # s=1, 选 a=1
    ]
    
    mux_if_correct = 0
    for s, a, b, expected in test_cases:
        mux_if.reset()
        s_t = torch.tensor([[float(s)]], device=device)
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        result = mux_if(s_t, a_t, b_t)
        if round(result.item()) == expected:
            mux_if_correct += 1
    
    print(f"  MUX (IF): {mux_if_correct}/8")
    
    # LIF 版本
    lif = SimpleLIFNode(beta=0.9)
    mux_lif = MUXGate(neuron_template=lif).to(device)
    
    mux_lif_correct = 0
    for s, a, b, expected in test_cases:
        mux_lif.reset()
        s_t = torch.tensor([[float(s)]], device=device)
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        result = mux_lif(s_t, a_t, b_t)
        if round(result.item()) == expected:
            mux_lif_correct += 1
    
    print(f"  MUX (LIF): {mux_lif_correct}/8")
    
    return mux_if_correct == 8 and mux_lif_correct == 8


def test_ripple_carry_adder():
    """测试多位加法器"""
    print("\n--- 多位加法器 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4-bit 加法器 (IF)
    rca_if = RippleCarryAdder(bits=4).to(device)
    
    test_cases = [
        (0, 0, 0),    # 0 + 0 = 0
        (1, 1, 2),    # 1 + 1 = 2
        (3, 5, 8),    # 3 + 5 = 8
        (7, 7, 14),   # 7 + 7 = 14
        (15, 1, 16),  # 15 + 1 = 16 (溢出)
    ]
    
    rca_if_correct = 0
    for a_val, b_val, expected in test_cases:
        rca_if.reset()
        # 转为 LSB first 位表示
        a = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
        b = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
        
        s, cout = rca_if(a, b)
        result_val = sum(int(round(s[0, i].item())) << i for i in range(4))
        result_val += int(round(cout[0, 0].item())) << 4
        
        if result_val == expected:
            rca_if_correct += 1
    
    print(f"  RippleCarryAdder (IF): {rca_if_correct}/5")
    
    # LIF 版本
    lif = SimpleLIFNode(beta=0.95)
    rca_lif = RippleCarryAdder(bits=4, neuron_template=lif).to(device)
    
    rca_lif_correct = 0
    for a_val, b_val, expected in test_cases:
        rca_lif.reset()
        a = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
        b = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
        
        s, cout = rca_lif(a, b)
        result_val = sum(int(round(s[0, i].item())) << i for i in range(4))
        result_val += int(round(cout[0, 0].item())) << 4
        
        if result_val == expected:
            rca_lif_correct += 1
    
    print(f"  RippleCarryAdder (LIF): {rca_lif_correct}/5")
    
    return rca_if_correct == 5 and rca_lif_correct == 5


def test_trees():
    """测试 OR/AND 树"""
    print("\n--- OR/AND 归约树 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # OR 树 (IF)
    or_tree = ORTree(n_inputs=4).to(device)
    
    or_cases = [
        ([0, 0, 0, 0], 0),  # 全0 -> 0
        ([1, 0, 0, 0], 1),  # 有1 -> 1
        ([0, 1, 0, 0], 1),
        ([0, 0, 1, 0], 1),
        ([0, 0, 0, 1], 1),
        ([1, 1, 1, 1], 1),  # 全1 -> 1
    ]
    
    or_correct = 0
    for inputs, expected in or_cases:
        or_tree.reset()
        x = torch.tensor([[float(v) for v in inputs]], device=device)
        result = or_tree(x)
        if round(result.item()) == expected:
            or_correct += 1
    
    print(f"  ORTree (IF): {or_correct}/6")
    
    # AND 树 (IF)
    and_tree = ANDTree(n_inputs=4).to(device)
    
    and_cases = [
        ([0, 0, 0, 0], 0),  # 全0 -> 0
        ([1, 0, 0, 0], 0),  # 有0 -> 0
        ([1, 1, 0, 0], 0),
        ([1, 1, 1, 0], 0),
        ([1, 1, 1, 1], 1),  # 全1 -> 1
    ]
    
    and_correct = 0
    for inputs, expected in and_cases:
        and_tree.reset()
        x = torch.tensor([[float(v) for v in inputs]], device=device)
        result = and_tree(x)
        if round(result.item()) == expected:
            and_correct += 1
    
    print(f"  ANDTree (IF): {and_correct}/5")
    
    # LIF 版本
    lif = SimpleLIFNode(beta=0.9)
    or_tree_lif = ORTree(n_inputs=4, neuron_template=lif).to(device)
    and_tree_lif = ANDTree(n_inputs=4, neuron_template=lif).to(device)
    
    or_lif_correct = 0
    for inputs, expected in or_cases:
        or_tree_lif.reset()
        x = torch.tensor([[float(v) for v in inputs]], device=device)
        result = or_tree_lif(x)
        if round(result.item()) == expected:
            or_lif_correct += 1
    
    and_lif_correct = 0
    for inputs, expected in and_cases:
        and_tree_lif.reset()
        x = torch.tensor([[float(v) for v in inputs]], device=device)
        result = and_tree_lif(x)
        if round(result.item()) == expected:
            and_lif_correct += 1
    
    print(f"  ORTree (LIF): {or_lif_correct}/6")
    print(f"  ANDTree (LIF): {and_lif_correct}/5")
    
    return (or_correct == 6 and and_correct == 5 and 
            or_lif_correct == 6 and and_lif_correct == 5)


def test_multiplier():
    """测试 4x4 阵列乘法器"""
    print("\n--- 4x4 阵列乘法器 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # IF 版本
    mul_if = ArrayMultiplier4x4_Strict().to(device)
    
    test_cases = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 3, 6),
        (3, 5, 15),
        (7, 7, 49),
        (15, 15, 225),
    ]
    
    mul_if_correct = 0
    for a_val, b_val, expected in test_cases:
        mul_if.reset()
        a = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
        b = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
        
        result = mul_if(a, b)
        result_val = sum(int(round(result[0, i].item())) << i for i in range(8))
        
        if result_val == expected:
            mul_if_correct += 1
    
    print(f"  ArrayMultiplier (IF): {mul_if_correct}/6")
    
    # LIF 版本
    lif = SimpleLIFNode(beta=0.95)
    mul_lif = ArrayMultiplier4x4_Strict(neuron_template=lif).to(device)
    
    mul_lif_correct = 0
    for a_val, b_val, expected in test_cases:
        mul_lif.reset()
        a = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
        b = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
        
        result = mul_lif(a, b)
        result_val = sum(int(round(result[0, i].item())) << i for i in range(8))
        
        if result_val == expected:
            mul_lif_correct += 1
    
    print(f"  ArrayMultiplier (LIF): {mul_lif_correct}/6")
    
    return mul_if_correct == 6 and mul_lif_correct == 6


def test_vec_subtractor():
    """测试向量化减法器"""
    print("\n--- 向量化减法器 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # IF 版本
    sub_if = VecSubtractor(bits=4).to(device)
    
    test_cases = [
        (5, 3, 2, 0),    # 5 - 3 = 2, 无借位
        (7, 2, 5, 0),    # 7 - 2 = 5, 无借位
        (0, 0, 0, 0),    # 0 - 0 = 0, 无借位
        (3, 5, 14, 1),   # 3 - 5 = -2 (补码 = 14), 有借位
        (0, 1, 15, 1),   # 0 - 1 = -1 (补码 = 15), 有借位
    ]
    
    sub_if_correct = 0
    for a_val, b_val, exp_diff, exp_borrow in test_cases:
        sub_if.reset()
        a = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
        b = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
        
        d, borrow = sub_if(a, b)
        diff_val = sum(int(round(d[0, i].item())) << i for i in range(4))
        borrow_val = int(round(borrow[0, 0].item()))
        
        if diff_val == exp_diff and borrow_val == exp_borrow:
            sub_if_correct += 1
    
    print(f"  VecSubtractor (IF): {sub_if_correct}/5")
    
    # LIF 版本
    lif = SimpleLIFNode(beta=0.9)
    sub_lif = VecSubtractor(bits=4, neuron_template=lif).to(device)
    
    sub_lif_correct = 0
    for a_val, b_val, exp_diff, exp_borrow in test_cases:
        sub_lif.reset()
        a = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
        b = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
        
        d, borrow = sub_lif(a, b)
        diff_val = sum(int(round(d[0, i].item())) << i for i in range(4))
        borrow_val = int(round(borrow[0, 0].item()))
        
        if diff_val == exp_diff and borrow_val == exp_borrow:
            sub_lif_correct += 1
    
    print(f"  VecSubtractor (LIF): {sub_lif_correct}/5")
    
    return sub_if_correct == 5 and sub_lif_correct == 5


def test_vec_comparator():
    """测试向量化比较器"""
    print("\n--- 向量化比较器 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # IF 版本 (注意: MSB first)
    comp_if = VecComparator(bits=4).to(device)
    
    def to_msb_first(val, bits=4):
        """转换为 MSB first"""
        return [float((val >> (bits - 1 - i)) & 1) for i in range(bits)]
    
    test_cases = [
        (5, 3, 1, 0),   # 5 > 3
        (3, 5, 0, 0),   # 3 < 5
        (5, 5, 0, 1),   # 5 == 5
        (0, 0, 0, 1),   # 0 == 0
        (15, 0, 1, 0),  # 15 > 0
        (0, 15, 0, 0),  # 0 < 15
    ]
    
    comp_if_correct = 0
    for a_val, b_val, exp_gt, exp_eq in test_cases:
        comp_if.reset()
        a = torch.tensor([to_msb_first(a_val)], device=device)
        b = torch.tensor([to_msb_first(b_val)], device=device)
        
        gt, eq = comp_if(a, b)
        gt_val = int(round(gt[0, 0].item()))
        eq_val = int(round(eq[0, 0].item()))
        
        if gt_val == exp_gt and eq_val == exp_eq:
            comp_if_correct += 1
    
    print(f"  VecComparator (IF): {comp_if_correct}/6")
    
    # LIF 版本
    lif = SimpleLIFNode(beta=0.9)
    comp_lif = VecComparator(bits=4, neuron_template=lif).to(device)
    
    comp_lif_correct = 0
    for a_val, b_val, exp_gt, exp_eq in test_cases:
        comp_lif.reset()
        a = torch.tensor([to_msb_first(a_val)], device=device)
        b = torch.tensor([to_msb_first(b_val)], device=device)
        
        gt, eq = comp_lif(a, b)
        gt_val = int(round(gt[0, 0].item()))
        eq_val = int(round(eq[0, 0].item()))
        
        if gt_val == exp_gt and eq_val == exp_eq:
            comp_lif_correct += 1
    
    print(f"  VecComparator (LIF): {comp_lif_correct}/6")
    
    return comp_if_correct == 6 and comp_lif_correct == 6


def test_batch_processing():
    """测试批量处理"""
    print("\n--- 批量处理 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 批量 AND 门测试
    and_gate = ANDGate().to(device)
    
    # 批量输入 [batch=4, 1]
    a = torch.tensor([[0.], [0.], [1.], [1.]], device=device)
    b = torch.tensor([[0.], [1.], [0.], [1.]], device=device)
    expected = torch.tensor([[0.], [0.], [0.], [1.]], device=device)
    
    and_gate.reset()
    result = and_gate(a, b)
    batch_and_ok = torch.allclose(result, expected)
    print(f"  批量 AND 门: {'✓' if batch_and_ok else '✗'}")
    
    # 批量加法器测试
    adder = VecAdder(bits=4).to(device)
    
    # batch=3, 每个样本 4-bit
    a_batch = torch.tensor([
        [1., 1., 0., 0.],  # 3
        [1., 0., 1., 0.],  # 5
        [1., 1., 1., 1.],  # 15
    ], device=device)
    b_batch = torch.tensor([
        [1., 0., 1., 0.],  # 5
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
    ], device=device)
    # 期望: 8, 6, 16
    
    adder.reset()
    s, cout = adder(a_batch, b_batch)
    
    results = []
    for i in range(3):
        val = sum(int(round(s[i, j].item())) << j for j in range(4))
        val += int(round(cout[i, 0].item())) << 4
        results.append(val)
    
    batch_add_ok = results == [8, 6, 16]
    print(f"  批量 VecAdder: {results} (期望 [8, 6, 16]) {'✓' if batch_add_ok else '✗'}")
    
    return batch_and_ok and batch_add_ok


def test_arithmetic_units():
    """测试算术单元"""
    print("\n--- 算术单元 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试全加器 (IF)
    fa = FullAdder().to(device)
    fa_cases = [
        (0, 0, 0, 0, 0),
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
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        cin_t = torch.tensor([[float(cin)]], device=device)
        s, c = fa(a_t, b_t, cin_t)
        if round(s.item()) == exp_s and round(c.item()) == exp_c:
            fa_correct += 1
    
    print(f"  FullAdder (IF): {fa_correct}/8")
    
    # 测试全加器 (LIF)
    lif = SimpleLIFNode(beta=0.95)
    fa_lif = FullAdder(neuron_template=lif).to(device)
    
    fa_lif_correct = 0
    for a, b, cin, exp_s, exp_c in fa_cases:
        fa_lif.reset()
        a_t = torch.tensor([[float(a)]], device=device)
        b_t = torch.tensor([[float(b)]], device=device)
        cin_t = torch.tensor([[float(cin)]], device=device)
        s, c = fa_lif(a_t, b_t, cin_t)
        if round(s.item()) == exp_s and round(c.item()) == exp_c:
            fa_lif_correct += 1
    
    print(f"  FullAdder (LIF): {fa_lif_correct}/8")
    
    return fa_correct == 8 and fa_lif_correct == 8


def test_vec_gates():
    """测试向量化门电路"""
    print("\n--- 向量化门电路 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 VecAdder (IF)
    adder = VecAdder(bits=4).to(device)
    
    # 测试 3 + 5 = 8
    a = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32, device=device)  # 3 (LSB first)
    b = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32, device=device)  # 5 (LSB first)
    
    s, cout = adder(a, b)
    result_val = sum(int(round(s[0, i].item())) << i for i in range(4))
    result_val += int(round(cout[0, 0].item())) << 4
    
    print(f"  VecAdder (IF): 3 + 5 = {result_val} (expected 8)")
    adder_if_ok = result_val == 8
    
    # 测试 VecAdder (LIF)
    lif = SimpleLIFNode(beta=0.9)
    adder_lif = VecAdder(bits=4, neuron_template=lif).to(device)
    
    s_lif, cout_lif = adder_lif(a, b)
    result_val_lif = sum(int(round(s_lif[0, i].item())) << i for i in range(4))
    result_val_lif += int(round(cout_lif[0, 0].item())) << 4
    
    print(f"  VecAdder (LIF): 3 + 5 = {result_val_lif} (expected 8)")
    adder_lif_ok = result_val_lif == 8
    
    return adder_if_ok and adder_lif_ok


def test_beta_sweep():
    """测试不同 β 值下的鲁棒性"""
    print("\n--- β 扫描 (LIF 鲁棒性) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    betas = [1.0, 0.99, 0.95, 0.9, 0.8, 0.5]
    
    for beta in betas:
        lif = SimpleLIFNode(beta=beta)
        and_gate = ANDGate(neuron_template=lif).to(device)
        
        # 测试所有输入组合
        correct = 0
        for a, b, exp in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]:
            and_gate.reset()
            a_t = torch.tensor([[float(a)]], device=device)
            b_t = torch.tensor([[float(b)]], device=device)
            result = and_gate(a_t, b_t)
            if round(result.item()) == exp:
                correct += 1
        
        status = "✓" if correct == 4 else "✗"
        print(f"  β={beta:.2f}: AND 门 {correct}/4 {status}")


def test_neuron_template_actually_used():
    """验证 neuron_template 参数是否真正生效"""
    print("\n--- 验证 neuron_template 实际生效 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_passed = True
    
    # 测试 1: 验证 SimpleLIFNode 的 beta 属性被正确复制
    lif_template = SimpleLIFNode(beta=0.7)
    and_gate = ANDGate(neuron_template=lif_template).to(device)
    
    # 检查内部神经元类型
    is_lif = isinstance(and_gate.node, SimpleLIFNode)
    beta_correct = hasattr(and_gate.node, 'beta') and and_gate.node.beta == 0.7
    threshold_correct = and_gate.node.v_threshold == 1.5  # AND 门阈值
    
    print(f"  AND 门使用 SimpleLIFNode: {is_lif} {'✓' if is_lif else '✗'}")
    print(f"  beta 值正确 (0.7): {beta_correct} {'✓' if beta_correct else '✗'}")
    print(f"  阈值正确 (1.5): {threshold_correct} {'✓' if threshold_correct else '✗'}")
    
    if not (is_lif and beta_correct and threshold_correct):
        all_passed = False
    
    # 测试 2: 验证不同门有不同阈值
    or_gate = ORGate(neuron_template=lif_template).to(device)
    or_threshold_correct = or_gate.node.v_threshold == 0.5  # OR 门阈值
    or_beta_correct = or_gate.node.beta == 0.7
    
    print(f"  OR 门阈值正确 (0.5): {or_threshold_correct} {'✓' if or_threshold_correct else '✗'}")
    print(f"  OR 门 beta 正确 (0.7): {or_beta_correct} {'✓' if or_beta_correct else '✗'}")
    
    if not (or_threshold_correct and or_beta_correct):
        all_passed = False
    
    # 测试 3: 验证默认情况下使用 IFNode
    from spikingjelly.activation_based import neuron as sj_neuron
    and_gate_default = ANDGate().to(device)  # 不传 neuron_template
    is_if = isinstance(and_gate_default.node, sj_neuron.IFNode)
    
    print(f"  默认使用 IFNode: {is_if} {'✓' if is_if else '✗'}")
    
    if not is_if:
        all_passed = False
    
    # 测试 4: 验证 deepcopy 独立性（修改模板不影响已创建的门）
    lif_template2 = SimpleLIFNode(beta=0.8)
    gate1 = ANDGate(neuron_template=lif_template2).to(device)
    lif_template2.beta = 0.3  # 修改原模板
    gate1_beta_unchanged = gate1.node.beta == 0.8  # 应该不受影响
    
    print(f"  deepcopy 独立性: {gate1_beta_unchanged} {'✓' if gate1_beta_unchanged else '✗'}")
    
    if not gate1_beta_unchanged:
        all_passed = False
    
    return all_passed


def test_template_propagation():
    """验证 neuron_template 在复合组件中正确传递"""
    print("\n--- 验证模板传递到子组件 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_passed = True
    
    lif_template = SimpleLIFNode(beta=0.85)
    
    # 测试 FullAdder 的子组件
    fa = FullAdder(neuron_template=lif_template).to(device)
    
    # 检查所有子组件
    components_ok = True
    
    # XOR 门内部有三个神经元 (双轨编码: and1, and2, or_out)
    xor1_and1_ok = isinstance(fa.xor1.and1, SimpleLIFNode) and fa.xor1.and1.beta == 0.85
    xor1_and2_ok = isinstance(fa.xor1.and2, SimpleLIFNode) and fa.xor1.and2.beta == 0.85
    xor1_or_ok = isinstance(fa.xor1.or_out, SimpleLIFNode) and fa.xor1.or_out.beta == 0.85
    components_ok = components_ok and xor1_and1_ok and xor1_and2_ok and xor1_or_ok
    
    # AND 门
    and1_ok = isinstance(fa.and1.node, SimpleLIFNode) and fa.and1.node.beta == 0.85
    and2_ok = isinstance(fa.and2.node, SimpleLIFNode) and fa.and2.node.beta == 0.85
    
    # OR 门
    or1_ok = isinstance(fa.or1.node, SimpleLIFNode) and fa.or1.node.beta == 0.85
    
    components_ok = components_ok and and1_ok and and2_ok and or1_ok
    
    print(f"  FullAdder 所有子组件使用 LIF: {components_ok} {'✓' if components_ok else '✗'}")
    
    if not components_ok:
        all_passed = False
    
    # 测试 RippleCarryAdder 的子组件
    rca = RippleCarryAdder(bits=4, neuron_template=lif_template).to(device)
    
    rca_ok = True
    for i, adder in enumerate(rca.adders):
        if not isinstance(adder.and1.node, SimpleLIFNode):
            rca_ok = False
            break
        if adder.and1.node.beta != 0.85:
            rca_ok = False
            break
    
    print(f"  RippleCarryAdder 所有子组件使用 LIF: {rca_ok} {'✓' if rca_ok else '✗'}")
    
    if not rca_ok:
        all_passed = False
    
    # 测试 VecAdder 的子组件
    va = VecAdder(bits=4, neuron_template=lif_template).to(device)
    
    va_xor1_ok = isinstance(va.xor1.and1, SimpleLIFNode) and va.xor1.and1.beta == 0.85
    va_and1_ok = isinstance(va.and1.node, SimpleLIFNode) and va.and1.node.beta == 0.85
    va_ok = va_xor1_ok and va_and1_ok
    
    print(f"  VecAdder 所有子组件使用 LIF: {va_ok} {'✓' if va_ok else '✗'}")
    
    if not va_ok:
        all_passed = False
    
    return all_passed


def test_lif_dynamics():
    """验证 LIF 神经元的动力学行为与 IF 不同"""
    print("\n--- 验证 LIF 动力学行为 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试膜电位泄漏
    # 使用极低的 beta，多次输入小于阈值的电流，观察是否累积
    lif_low_beta = SimpleLIFNode(beta=0.5, v_threshold=1.0).to(device)
    lif_high_beta = SimpleLIFNode(beta=0.99, v_threshold=1.0).to(device)
    
    # 输入 0.4 两次，理想情况：
    # - 高 beta (0.99): V = 0.4, 然后 V = 0.99*0.4 + 0.4 = 0.796 (不发放)
    # - 低 beta (0.5): V = 0.4, 然后 V = 0.5*0.4 + 0.4 = 0.6 (不发放)
    # 第三次：
    # - 高 beta: V = 0.99*0.796 + 0.4 = 1.188 (发放!)
    # - 低 beta: V = 0.5*0.6 + 0.4 = 0.7 (不发放)
    
    input_val = torch.tensor([[0.4]], device=device)
    
    # 高 beta 测试
    lif_high_beta.reset()
    for _ in range(3):
        out_high = lif_high_beta(input_val)
    high_fires = out_high.item() > 0.5
    
    # 低 beta 测试
    lif_low_beta.reset()
    for _ in range(3):
        out_low = lif_low_beta(input_val)
    low_fires = out_low.item() > 0.5
    
    # 高 beta 应该发放，低 beta 不应该（因为泄漏太多）
    dynamics_different = high_fires and not low_fires
    
    print(f"  高 β(0.99) 3次输入0.4后发放: {high_fires} {'✓' if high_fires else '✗'}")
    print(f"  低 β(0.5) 3次输入0.4后不发放: {not low_fires} {'✓' if not low_fires else '✗'}")
    print(f"  LIF 动力学行为验证: {dynamics_different} {'✓' if dynamics_different else '✗'}")
    
    return dynamics_different


def main():
    print("=" * 60)
    print("统一神经元架构测试 - 完整版")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    results = []
    
    # ========== 第一部分：功能正确性测试 ==========
    print("\n" + "=" * 60)
    print("第一部分：功能正确性测试")
    print("=" * 60)
    
    results.append(("基础门 (IF)", test_basic_gates_if()))
    results.append(("基础门 (LIF)", test_basic_gates_lif()))
    results.append(("MUX 选择器", test_mux_gate()))
    results.append(("算术单元", test_arithmetic_units()))
    results.append(("多位加法器", test_ripple_carry_adder()))
    results.append(("OR/AND 树", test_trees()))
    results.append(("4x4 乘法器", test_multiplier()))
    results.append(("向量化门", test_vec_gates()))
    results.append(("向量化减法器", test_vec_subtractor()))
    results.append(("向量化比较器", test_vec_comparator()))
    results.append(("批量处理", test_batch_processing()))
    
    # ========== 第二部分：核心验证 - neuron_template 真正生效 ==========
    print("\n" + "=" * 60)
    print("第二部分：核心验证 - neuron_template 机制")
    print("=" * 60)
    
    results.append(("neuron_template 实际生效", test_neuron_template_actually_used()))
    results.append(("模板传递到子组件", test_template_propagation()))
    results.append(("LIF 动力学行为", test_lif_dynamics()))
    
    # β 扫描（不计入通过/失败）
    test_beta_sweep()
    
    # 汇总结果
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    
    if failed > 0:
        print("\n⚠️  警告: 存在失败的测试！")
    else:
        print("\n✅ 所有测试通过！neuron_template 机制验证成功。")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

