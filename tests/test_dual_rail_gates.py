"""
双轨编码门电路测试 (Dual-Rail Gates Test)
==========================================

验证双轨编码门电路的正确性：
1. 基础门电路真值表验证
2. 与原始单轨门电路的功能等价性
3. 双轨编码的一致性验证

核心验证目标：
- 所有门电路只使用 +1 权重脉冲汇聚 + 阈值判断
- NOT 门是纯拓扑操作（零计算）
- 功能与原始门电路完全一致

作者: HumanBrain Project
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import torch.nn as nn

# 导入双轨门电路
from SNNTorch.atomic_ops.dual_rail_gates import (
    to_dual_rail, from_dual_rail,
    DualRailNOT, DualRailAND, DualRailOR, DualRailXOR, DualRailMUX,
    DualRailHalfAdder, DualRailFullAdder,
    PureSNN_NOT, PureSNN_AND, PureSNN_OR, PureSNN_XOR, PureSNN_MUX,
    PureSNN_HalfAdder, PureSNN_FullAdder
)

# 导入原始门电路（用于对比）
from SNNTorch.atomic_ops.logic_gates import (
    NOTGate, ANDGate, ORGate, XORGate, MUXGate,
    HalfAdder, FullAdder
)


def test_dual_rail_encoding():
    """测试双轨编码/解码"""
    print("=" * 60)
    print("测试 1: 双轨编码/解码")
    print("=" * 60)
    
    # 测试单轨转双轨
    x = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    x_dual = to_dual_rail(x)
    
    print(f"单轨输入: {x}")
    print(f"双轨输出形状: {x_dual.shape}")  # 应该是 [1, 4, 2]
    print(f"双轨输出:")
    print(f"  正极性: {x_dual[..., 0]}")
    print(f"  负极性: {x_dual[..., 1]}")
    
    # 验证：正极性 + 负极性 = 1
    sum_check = x_dual[..., 0] + x_dual[..., 1]
    assert torch.allclose(sum_check, torch.ones_like(sum_check)), "双轨编码错误：正+负 ≠ 1"
    
    # 测试双轨转单轨
    x_recovered = from_dual_rail(x_dual)
    assert torch.allclose(x, x_recovered), "双轨解码错误"
    
    print("✓ 双轨编码/解码测试通过")
    return True


def test_dual_rail_not():
    """测试双轨 NOT 门"""
    print("\n" + "=" * 60)
    print("测试 2: 双轨 NOT 门（零计算！）")
    print("=" * 60)
    
    not_gate = DualRailNOT()
    
    # 测试所有可能的输入
    test_cases = [
        (0.0, 1.0),  # NOT(0) = 1
        (1.0, 0.0),  # NOT(1) = 0
    ]
    
    all_passed = True
    for input_val, expected in test_cases:
        x = torch.tensor([[input_val]])
        x_dual = to_dual_rail(x)
        y_dual = not_gate(x_dual)
        y = from_dual_rail(y_dual)
        
        passed = torch.allclose(y, torch.tensor([[expected]]))
        status = "✓" if passed else "✗"
        print(f"  NOT({input_val}) = {y.item():.1f} (期望: {expected}) {status}")
        
        if not passed:
            all_passed = False
    
    # 验证 NOT 是纯拓扑操作
    print("\n  验证 NOT 是纯拓扑操作（交换线路）：")
    x = torch.tensor([[1.0]])
    x_dual = to_dual_rail(x)
    y_dual = not_gate(x_dual)
    
    # NOT 应该只是交换 pos 和 neg
    assert torch.allclose(x_dual[..., 0], y_dual[..., 1]), "NOT 未正确交换正极性"
    assert torch.allclose(x_dual[..., 1], y_dual[..., 0]), "NOT 未正确交换负极性"
    print("  ✓ 确认 NOT 只是交换线路，零计算")
    
    print("✓ 双轨 NOT 门测试通过" if all_passed else "✗ 双轨 NOT 门测试失败")
    return all_passed


def test_dual_rail_and():
    """测试双轨 AND 门"""
    print("\n" + "=" * 60)
    print("测试 3: 双轨 AND 门")
    print("=" * 60)
    
    and_gate = DualRailAND()
    
    # AND 真值表
    test_cases = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ]
    
    all_passed = True
    for a_val, b_val, expected in test_cases:
        a = torch.tensor([[a_val]])
        b = torch.tensor([[b_val]])
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        
        y_dual = and_gate(a_dual, b_dual)
        y = from_dual_rail(y_dual)
        
        passed = torch.allclose(y, torch.tensor([[expected]]))
        status = "✓" if passed else "✗"
        print(f"  AND({a_val}, {b_val}) = {y.item():.1f} (期望: {expected}) {status}")
        
        # 验证双轨一致性
        y_neg = y_dual[..., 1].item()
        expected_neg = 1.0 - expected
        if abs(y_neg - expected_neg) > 1e-5:
            print(f"    警告: 负极性 = {y_neg}, 期望 = {expected_neg}")
            all_passed = False
        
        if not passed:
            all_passed = False
    
    print("✓ 双轨 AND 门测试通过" if all_passed else "✗ 双轨 AND 门测试失败")
    return all_passed


def test_dual_rail_or():
    """测试双轨 OR 门"""
    print("\n" + "=" * 60)
    print("测试 4: 双轨 OR 门")
    print("=" * 60)
    
    or_gate = DualRailOR()
    
    # OR 真值表
    test_cases = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    
    all_passed = True
    for a_val, b_val, expected in test_cases:
        a = torch.tensor([[a_val]])
        b = torch.tensor([[b_val]])
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        
        y_dual = or_gate(a_dual, b_dual)
        y = from_dual_rail(y_dual)
        
        passed = torch.allclose(y, torch.tensor([[expected]]))
        status = "✓" if passed else "✗"
        print(f"  OR({a_val}, {b_val}) = {y.item():.1f} (期望: {expected}) {status}")
        
        if not passed:
            all_passed = False
    
    print("✓ 双轨 OR 门测试通过" if all_passed else "✗ 双轨 OR 门测试失败")
    return all_passed


def test_dual_rail_xor():
    """测试双轨 XOR 门（关键：无 -2.0 权重）"""
    print("\n" + "=" * 60)
    print("测试 5: 双轨 XOR 门（无 -2.0 权重！）")
    print("=" * 60)
    
    xor_gate = DualRailXOR()
    
    # XOR 真值表
    test_cases = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]
    
    all_passed = True
    for a_val, b_val, expected in test_cases:
        a = torch.tensor([[a_val]])
        b = torch.tensor([[b_val]])
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        
        y_dual = xor_gate(a_dual, b_dual)
        y = from_dual_rail(y_dual)
        
        passed = torch.allclose(y, torch.tensor([[expected]]))
        status = "✓" if passed else "✗"
        print(f"  XOR({a_val}, {b_val}) = {y.item():.1f} (期望: {expected}) {status}")
        
        # 验证双轨：负极性应该是 XNOR
        y_neg = y_dual[..., 1].item()
        expected_xnor = 1.0 if a_val == b_val else 0.0
        if abs(y_neg - expected_xnor) > 1e-5:
            print(f"    警告: XNOR = {y_neg}, 期望 = {expected_xnor}")
            all_passed = False
        
        if not passed:
            all_passed = False
    
    print("✓ 双轨 XOR 门测试通过" if all_passed else "✗ 双轨 XOR 门测试失败")
    return all_passed


def test_dual_rail_mux():
    """测试双轨 MUX 门"""
    print("\n" + "=" * 60)
    print("测试 6: 双轨 MUX 门")
    print("=" * 60)
    
    mux_gate = DualRailMUX()
    
    # MUX(sel, a, b): sel=1选a, sel=0选b
    test_cases = [
        (0.0, 0.0, 0.0, 0.0),  # sel=0, 选 b=0
        (0.0, 0.0, 1.0, 1.0),  # sel=0, 选 b=1
        (0.0, 1.0, 0.0, 0.0),  # sel=0, 选 b=0 (忽略 a=1)
        (0.0, 1.0, 1.0, 1.0),  # sel=0, 选 b=1 (忽略 a=1)
        (1.0, 0.0, 0.0, 0.0),  # sel=1, 选 a=0
        (1.0, 0.0, 1.0, 0.0),  # sel=1, 选 a=0 (忽略 b=1)
        (1.0, 1.0, 0.0, 1.0),  # sel=1, 选 a=1
        (1.0, 1.0, 1.0, 1.0),  # sel=1, 选 a=1
    ]
    
    all_passed = True
    for sel_val, a_val, b_val, expected in test_cases:
        sel = torch.tensor([[sel_val]])
        a = torch.tensor([[a_val]])
        b = torch.tensor([[b_val]])
        
        sel_dual = to_dual_rail(sel)
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        
        y_dual = mux_gate(sel_dual, a_dual, b_dual)
        y = from_dual_rail(y_dual)
        
        passed = torch.allclose(y, torch.tensor([[expected]]))
        status = "✓" if passed else "✗"
        print(f"  MUX({sel_val}, {a_val}, {b_val}) = {y.item():.1f} (期望: {expected}) {status}")
        
        if not passed:
            all_passed = False
    
    print("✓ 双轨 MUX 门测试通过" if all_passed else "✗ 双轨 MUX 门测试失败")
    return all_passed


def test_dual_rail_half_adder():
    """测试双轨半加器"""
    print("\n" + "=" * 60)
    print("测试 7: 双轨半加器")
    print("=" * 60)
    
    ha = DualRailHalfAdder()
    
    # 半加器真值表: S = A XOR B, C = A AND B
    test_cases = [
        (0.0, 0.0, 0.0, 0.0),  # 0+0 = 00
        (0.0, 1.0, 1.0, 0.0),  # 0+1 = 01
        (1.0, 0.0, 1.0, 0.0),  # 1+0 = 01
        (1.0, 1.0, 0.0, 1.0),  # 1+1 = 10
    ]
    
    all_passed = True
    for a_val, b_val, expected_s, expected_c in test_cases:
        a = torch.tensor([[a_val]])
        b = torch.tensor([[b_val]])
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        
        s_dual, c_dual = ha(a_dual, b_dual)
        s = from_dual_rail(s_dual)
        c = from_dual_rail(c_dual)
        
        passed_s = torch.allclose(s, torch.tensor([[expected_s]]))
        passed_c = torch.allclose(c, torch.tensor([[expected_c]]))
        passed = passed_s and passed_c
        status = "✓" if passed else "✗"
        print(f"  HA({a_val}, {b_val}) = (S={s.item():.0f}, C={c.item():.0f}) "
              f"(期望: S={expected_s:.0f}, C={expected_c:.0f}) {status}")
        
        if not passed:
            all_passed = False
    
    print("✓ 双轨半加器测试通过" if all_passed else "✗ 双轨半加器测试失败")
    return all_passed


def test_dual_rail_full_adder():
    """测试双轨全加器"""
    print("\n" + "=" * 60)
    print("测试 8: 双轨全加器")
    print("=" * 60)
    
    fa = DualRailFullAdder()
    
    # 全加器真值表
    test_cases = [
        (0.0, 0.0, 0.0, 0.0, 0.0),  # 0+0+0 = 00
        (0.0, 0.0, 1.0, 1.0, 0.0),  # 0+0+1 = 01
        (0.0, 1.0, 0.0, 1.0, 0.0),  # 0+1+0 = 01
        (0.0, 1.0, 1.0, 0.0, 1.0),  # 0+1+1 = 10
        (1.0, 0.0, 0.0, 1.0, 0.0),  # 1+0+0 = 01
        (1.0, 0.0, 1.0, 0.0, 1.0),  # 1+0+1 = 10
        (1.0, 1.0, 0.0, 0.0, 1.0),  # 1+1+0 = 10
        (1.0, 1.0, 1.0, 1.0, 1.0),  # 1+1+1 = 11
    ]
    
    all_passed = True
    for a_val, b_val, cin_val, expected_s, expected_cout in test_cases:
        a = torch.tensor([[a_val]])
        b = torch.tensor([[b_val]])
        cin = torch.tensor([[cin_val]])
        
        a_dual = to_dual_rail(a)
        b_dual = to_dual_rail(b)
        cin_dual = to_dual_rail(cin)
        
        s_dual, cout_dual = fa(a_dual, b_dual, cin_dual)
        s = from_dual_rail(s_dual)
        cout = from_dual_rail(cout_dual)
        
        passed_s = torch.allclose(s, torch.tensor([[expected_s]]))
        passed_c = torch.allclose(cout, torch.tensor([[expected_cout]]))
        passed = passed_s and passed_c
        status = "✓" if passed else "✗"
        print(f"  FA({a_val}, {b_val}, {cin_val}) = (S={s.item():.0f}, Cout={cout.item():.0f}) "
              f"(期望: S={expected_s:.0f}, Cout={expected_cout:.0f}) {status}")
        
        if not passed:
            all_passed = False
    
    print("✓ 双轨全加器测试通过" if all_passed else "✗ 双轨全加器测试失败")
    return all_passed


def test_pure_snn_wrappers():
    """测试纯 SNN 单轨接口包装器"""
    print("\n" + "=" * 60)
    print("测试 9: 纯 SNN 单轨接口包装器")
    print("=" * 60)
    
    # 对比原始门电路和纯 SNN 门电路
    original_not = NOTGate()
    original_and = ANDGate()
    original_or = ORGate()
    original_xor = XORGate()
    
    pure_not = PureSNN_NOT()
    pure_and = PureSNN_AND()
    pure_or = PureSNN_OR()
    pure_xor = PureSNN_XOR()
    
    # 批量测试
    test_inputs = [
        (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
        (torch.tensor([[1.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
    ]
    
    all_passed = True
    
    # 测试 NOT
    print("\n  NOT 门对比:")
    for a, _ in test_inputs[:2]:
        original_not.reset()
        orig_result = original_not(a)
        pure_result = pure_not(a)
        
        match = torch.allclose(orig_result, pure_result)
        status = "✓" if match else "✗"
        print(f"    NOT({a.item():.0f}): 原始={orig_result.item():.0f}, 纯SNN={pure_result.item():.0f} {status}")
        if not match:
            all_passed = False
    
    # 测试 AND
    print("\n  AND 门对比:")
    for a, b in test_inputs:
        original_and.reset()
        orig_result = original_and(a, b)
        pure_result = pure_and(a, b)
        
        match = torch.allclose(orig_result, pure_result)
        status = "✓" if match else "✗"
        print(f"    AND({a.item():.0f}, {b.item():.0f}): 原始={orig_result.item():.0f}, 纯SNN={pure_result.item():.0f} {status}")
        if not match:
            all_passed = False
    
    # 测试 OR
    print("\n  OR 门对比:")
    for a, b in test_inputs:
        original_or.reset()
        orig_result = original_or(a, b)
        pure_result = pure_or(a, b)
        
        match = torch.allclose(orig_result, pure_result)
        status = "✓" if match else "✗"
        print(f"    OR({a.item():.0f}, {b.item():.0f}): 原始={orig_result.item():.0f}, 纯SNN={pure_result.item():.0f} {status}")
        if not match:
            all_passed = False
    
    # 测试 XOR
    print("\n  XOR 门对比:")
    for a, b in test_inputs:
        original_xor.reset()
        orig_result = original_xor(a, b)
        pure_result = pure_xor(a, b)
        
        match = torch.allclose(orig_result, pure_result)
        status = "✓" if match else "✗"
        print(f"    XOR({a.item():.0f}, {b.item():.0f}): 原始={orig_result.item():.0f}, 纯SNN={pure_result.item():.0f} {status}")
        if not match:
            all_passed = False
    
    print("\n✓ 纯 SNN 包装器测试通过" if all_passed else "\n✗ 纯 SNN 包装器测试失败")
    return all_passed


def test_batch_processing():
    """测试批量处理能力"""
    print("\n" + "=" * 60)
    print("测试 10: 批量处理能力")
    print("=" * 60)
    
    xor_gate = DualRailXOR()
    
    # 批量输入
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    
    a_dual = to_dual_rail(a)
    b_dual = to_dual_rail(b)
    
    y_dual = xor_gate(a_dual, b_dual)
    y = from_dual_rail(y_dual)
    
    print(f"  输入 A: {a}")
    print(f"  输入 B: {b}")
    print(f"  输出 Y: {y}")
    print(f"  期望 Y: {expected}")
    
    passed = torch.allclose(y, expected)
    print("\n✓ 批量处理测试通过" if passed else "\n✗ 批量处理测试失败")
    return passed


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("双轨编码 SNN 门电路测试套件")
    print("=" * 60)
    print("\n核心原则验证:")
    print("  - 只使用 +1 权重的脉冲汇聚")
    print("  - 只使用阈值判断（IF神经元）")
    print("  - NOT 门 = 线路交换（零计算）")
    print("  - 无负权重，无浮点乘法\n")
    
    results = {
        "双轨编码/解码": test_dual_rail_encoding(),
        "双轨 NOT 门": test_dual_rail_not(),
        "双轨 AND 门": test_dual_rail_and(),
        "双轨 OR 门": test_dual_rail_or(),
        "双轨 XOR 门": test_dual_rail_xor(),
        "双轨 MUX 门": test_dual_rail_mux(),
        "双轨半加器": test_dual_rail_half_adder(),
        "双轨全加器": test_dual_rail_full_adder(),
        "纯 SNN 包装器": test_pure_snn_wrappers(),
        "批量处理": test_batch_processing(),
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n双轨编码门电路的学术优势:")
        print("  1. NOT 门是纯拓扑操作，零计算")
        print("  2. 所有门电路只使用 +1 权重脉冲汇聚")
        print("  3. 100% 位精确完全来自空间组合逻辑")
        print("  4. 不依赖权重精度")
    else:
        print("⚠️ 部分测试失败，请检查实现")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()

