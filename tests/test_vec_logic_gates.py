"""
向量化门电路测试 - 验证 VecAND, VecOR, VecXOR, VecNOT, VecMUX 等
============================================================

测试内容:
1. 基本逻辑正确性 (真值表验证)
2. 任意维度支持 (batch, 多维张量)
3. GPU 并行性能
4. 与非向量化版本的一致性

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time


def test_vec_and():
    """测试 VecAND"""
    from atomic_ops.vec_logic_gates import VecAND
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecAND().to(device)
    
    # 真值表测试
    a = torch.tensor([[0., 0., 1., 1.]], device=device)
    b = torch.tensor([[0., 1., 0., 1.]], device=device)
    expected = torch.tensor([[0., 0., 0., 1.]], device=device)
    
    result = gate(a, b)
    assert torch.allclose(result, expected), f"VecAND 真值表错误: {result} vs {expected}"
    
    # 批量测试
    batch_a = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_b = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_result = gate(batch_a, batch_b)
    
    # 验证: AND = min(a, b) for binary
    expected_batch = torch.minimum(batch_a, batch_b)
    assert torch.allclose(batch_result, expected_batch), "VecAND 批量测试错误"
    
    print("✓ VecAND 测试通过")
    return True


def test_vec_or():
    """测试 VecOR"""
    from atomic_ops.vec_logic_gates import VecOR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecOR().to(device)
    
    # 真值表测试
    a = torch.tensor([[0., 0., 1., 1.]], device=device)
    b = torch.tensor([[0., 1., 0., 1.]], device=device)
    expected = torch.tensor([[0., 1., 1., 1.]], device=device)
    
    result = gate(a, b)
    assert torch.allclose(result, expected), f"VecOR 真值表错误: {result} vs {expected}"
    
    # 批量测试
    batch_a = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_b = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_result = gate(batch_a, batch_b)
    
    # 验证: OR = max(a, b) for binary
    expected_batch = torch.maximum(batch_a, batch_b)
    assert torch.allclose(batch_result, expected_batch), "VecOR 批量测试错误"
    
    print("✓ VecOR 测试通过")
    return True


def test_vec_not():
    """测试 VecNOT"""
    from atomic_ops.vec_logic_gates import VecNOT
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecNOT().to(device)
    
    # 真值表测试
    a = torch.tensor([[0., 1.]], device=device)
    expected = torch.tensor([[1., 0.]], device=device)
    
    result = gate(a)
    assert torch.allclose(result, expected), f"VecNOT 真值表错误: {result} vs {expected}"
    
    # 批量测试
    batch_a = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_result = gate(batch_a)
    
    # 验证: NOT = 1 - a for binary
    expected_batch = 1.0 - batch_a
    assert torch.allclose(batch_result, expected_batch), "VecNOT 批量测试错误"
    
    print("✓ VecNOT 测试通过")
    return True


def test_vec_xor():
    """测试 VecXOR"""
    from atomic_ops.vec_logic_gates import VecXOR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecXOR().to(device)
    
    # 真值表测试
    a = torch.tensor([[0., 0., 1., 1.]], device=device)
    b = torch.tensor([[0., 1., 0., 1.]], device=device)
    expected = torch.tensor([[0., 1., 1., 0.]], device=device)
    
    result = gate(a, b)
    assert torch.allclose(result, expected), f"VecXOR 真值表错误: {result} vs {expected}"
    
    # 批量测试
    batch_a = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_b = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_result = gate(batch_a, batch_b)
    
    # 验证: XOR = (a + b) mod 2 for binary
    expected_batch = torch.abs(batch_a - batch_b)
    assert torch.allclose(batch_result, expected_batch), "VecXOR 批量测试错误"
    
    print("✓ VecXOR 测试通过")
    return True


def test_vec_mux():
    """测试 VecMUX"""
    from atomic_ops.vec_logic_gates import VecMUX
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecMUX().to(device)
    
    # MUX(sel, a, b) = a if sel=1 else b
    sel = torch.tensor([[0., 1., 0., 1.]], device=device)
    a = torch.tensor([[1., 1., 1., 1.]], device=device)
    b = torch.tensor([[0., 0., 0., 0.]], device=device)
    expected = torch.tensor([[0., 1., 0., 1.]], device=device)  # sel=0选b, sel=1选a
    
    result = gate(sel, a, b)
    assert torch.allclose(result, expected), f"VecMUX 真值表错误: {result} vs {expected}"
    
    # 批量测试
    batch_sel = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_a = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_b = torch.randint(0, 2, (100, 64), device=device, dtype=torch.float32)
    batch_result = gate(batch_sel, batch_a, batch_b)
    
    # 验证: MUX = sel*a + (1-sel)*b
    expected_batch = batch_sel * batch_a + (1 - batch_sel) * batch_b
    assert torch.allclose(batch_result, expected_batch), "VecMUX 批量测试错误"
    
    print("✓ VecMUX 测试通过")
    return True


def test_vec_or_tree():
    """测试 VecORTree"""
    from atomic_ops.vec_logic_gates import VecORTree
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecORTree().to(device)
    
    # 全0输入
    zeros = torch.zeros((2, 32), device=device)
    result = gate(zeros)
    assert torch.allclose(result, torch.zeros((2, 1), device=device)), "VecORTree 全0错误"
    
    # 有1输入
    has_one = torch.zeros((2, 32), device=device)
    has_one[0, 15] = 1.0
    has_one[1, 0] = 1.0
    result = gate(has_one)
    assert torch.allclose(result, torch.ones((2, 1), device=device)), "VecORTree 有1错误"
    
    print("✓ VecORTree 测试通过")
    return True


def test_vec_and_tree():
    """测试 VecANDTree"""
    from atomic_ops.vec_logic_gates import VecANDTree
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate = VecANDTree().to(device)
    
    # 全1输入
    ones = torch.ones((2, 32), device=device)
    result = gate(ones)
    assert torch.allclose(result, torch.ones((2, 1), device=device)), "VecANDTree 全1错误"
    
    # 有0输入
    has_zero = torch.ones((2, 32), device=device)
    has_zero[0, 15] = 0.0
    has_zero[1, 0] = 0.0
    result = gate(has_zero)
    assert torch.allclose(result, torch.zeros((2, 1), device=device)), "VecANDTree 有0错误"
    
    print("✓ VecANDTree 测试通过")
    return True


def test_vec_adder():
    """测试 VecAdder"""
    from atomic_ops.vec_logic_gates import VecAdder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adder = VecAdder(8).to(device)
    
    # 简单加法: 5 + 3 = 8
    # 5 = 00000101 (LSB first: 10100000)
    # 3 = 00000011 (LSB first: 11000000)
    # 8 = 00001000 (LSB first: 00010000)
    a = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0.]], device=device)  # 5 LSB first
    b = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]], device=device)  # 3 LSB first
    expected = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0.]], device=device)  # 8 LSB first
    
    result, carry = adder(a, b)
    assert torch.allclose(result, expected), f"VecAdder 加法错误: {result} vs {expected}"
    
    # 批量随机测试
    n_tests = 100
    for _ in range(n_tests):
        val_a = torch.randint(0, 128, (1,)).item()
        val_b = torch.randint(0, 128, (1,)).item()
        expected_sum = (val_a + val_b) & 0xFF
        
        bits_a = torch.tensor([[(val_a >> i) & 1 for i in range(8)]], device=device, dtype=torch.float32)
        bits_b = torch.tensor([[(val_b >> i) & 1 for i in range(8)]], device=device, dtype=torch.float32)
        
        result, _ = adder(bits_a, bits_b)
        result_val = sum(int(result[0, i].item()) << i for i in range(8))
        
        assert result_val == expected_sum, f"VecAdder 错误: {val_a}+{val_b}={result_val}, 期望{expected_sum}"
    
    print("✓ VecAdder 测试通过")
    return True


def test_vec_subtractor():
    """测试 VecSubtractor"""
    from atomic_ops.vec_logic_gates import VecSubtractor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sub = VecSubtractor(8).to(device)
    
    # 简单减法: 8 - 3 = 5
    a = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0.]], device=device)  # 8 LSB first
    b = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]], device=device)  # 3 LSB first
    expected = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0.]], device=device)  # 5 LSB first
    
    result, borrow = sub(a, b)
    assert torch.allclose(result, expected), f"VecSubtractor 减法错误: {result} vs {expected}"
    
    # 批量随机测试
    n_tests = 100
    for _ in range(n_tests):
        val_a = torch.randint(0, 256, (1,)).item()
        val_b = torch.randint(0, 256, (1,)).item()
        expected_diff = (val_a - val_b) & 0xFF
        
        bits_a = torch.tensor([[(val_a >> i) & 1 for i in range(8)]], device=device, dtype=torch.float32)
        bits_b = torch.tensor([[(val_b >> i) & 1 for i in range(8)]], device=device, dtype=torch.float32)
        
        result, _ = sub(bits_a, bits_b)
        result_val = sum(int(result[0, i].item()) << i for i in range(8))
        
        assert result_val == expected_diff, f"VecSubtractor 错误: {val_a}-{val_b}={result_val}, 期望{expected_diff}"
    
    print("✓ VecSubtractor 测试通过")
    return True


def test_performance():
    """测试向量化性能"""
    from atomic_ops.vec_logic_gates import VecAND, VecOR, VecXOR
    from atomic_ops.logic_gates import ANDGate, ORGate, XORGate
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备大批量数据
    batch_size = 1000
    bits = 64
    a = torch.randint(0, 2, (batch_size, bits), device=device, dtype=torch.float32)
    b = torch.randint(0, 2, (batch_size, bits), device=device, dtype=torch.float32)
    
    # 向量化版本
    vec_and = VecAND().to(device)
    
    # 预热
    for _ in range(3):
        _ = vec_and(a, b)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        _ = vec_and(a, b)
    if device == 'cuda':
        torch.cuda.synchronize()
    vec_time = time.time() - start
    
    print(f"✓ 向量化性能测试: {batch_size}x{bits} 位，10次迭代耗时 {vec_time*1000:.2f}ms")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("向量化门电路测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    tests = [
        ("VecAND", test_vec_and),
        ("VecOR", test_vec_or),
        ("VecNOT", test_vec_not),
        ("VecXOR", test_vec_xor),
        ("VecMUX", test_vec_mux),
        ("VecORTree", test_vec_or_tree),
        ("VecANDTree", test_vec_and_tree),
        ("VecAdder", test_vec_adder),
        ("VecSubtractor", test_vec_subtractor),
        ("性能测试", test_performance),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
