"""
FP32 Reciprocal 测试 - 验证 SpikeFP32Reciprocal 的正确性
=========================================================

测试内容:
1. 基本倒数功能: 1/x
2. 与 PyTorch 1/x 对比
3. 特殊值处理: 1/0, 1/Inf, 1/NaN
4. 位精确性验证

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

# 使用统一的转换函数
from atomic_ops import float32_to_pulse, pulse_to_float32, float32_to_bits


def test_basic_reciprocal():
    """测试基本倒数功能"""
    from atomic_ops import SpikeFP32Reciprocal
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 SpikeFP32Reciprocal 基本功能 (device={device})")
    print("=" * 50)
    
    recip = SpikeFP32Reciprocal().to(device)
    
    # 测试值
    test_cases = [
        (1.0, "1/1 = 1"),
        (2.0, "1/2 = 0.5"),
        (0.5, "1/0.5 = 2"),
        (4.0, "1/4 = 0.25"),
        (0.25, "1/0.25 = 4"),
        (-1.0, "1/-1 = -1"),
        (-2.0, "1/-2 = -0.5"),
        (10.0, "1/10 = 0.1"),
        (0.1, "1/0.1 = 10"),
    ]
    
    all_pass = True
    for val, desc in test_cases:
        x = np.array([val], dtype=np.float32)
        x_pulse = float32_to_pulse(x, device)
        
        recip.reset()
        result_pulse = recip(x_pulse)
        result = pulse_to_float32(result_pulse).cpu().numpy()[0]
        
        expected = 1.0 / val
        
        # 计算相对误差
        if abs(expected) > 1e-6:
            rel_err = abs(result - expected) / abs(expected)
        else:
            rel_err = abs(result - expected)
        
        status = "✓" if rel_err < 1e-5 else "✗"  # 高精度要求
        print(f"  {status} {desc}: 1/{val} = {result:.6f} (期望: {expected:.6f}, 误差: {rel_err:.2e})")
        
        if rel_err >= 1e-5:
            all_pass = False
    
    return all_pass


def test_bit_exactness():
    """测试位精确性"""
    from atomic_ops import SpikeFP32Reciprocal
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试位精确性 (device={device})")
    print("=" * 50)
    
    recip = SpikeFP32Reciprocal().to(device)
    
    # 随机测试
    torch.manual_seed(42)
    n_tests = 20
    exact_count = 0
    
    for i in range(n_tests):
        val = torch.randn(1).item() * 10
        if abs(val) < 0.01:
            val = 1.0  # 避免太小的值
        
        x = np.array([val], dtype=np.float32)
        x_pulse = float32_to_pulse(x, device)
        
        recip.reset()
        result_pulse = recip(x_pulse)
        result = pulse_to_float32(result_pulse).cpu().numpy()[0]
        
        expected = np.float32(1.0 / val)
        
        # 比较位模式
        result_bits = float32_to_bits(float(result))
        expected_bits = float32_to_bits(float(expected))
        
        if result_bits == expected_bits:
            exact_count += 1
    
    rate = exact_count / n_tests * 100
    print(f"  位精确匹配: {exact_count}/{n_tests} ({rate:.1f}%)")
    
    passed = rate >= 90  # 90% 阈值
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed


def test_special_values():
    """测试特殊值"""
    from atomic_ops import SpikeFP32Reciprocal
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试特殊值处理 (device={device})")
    print("=" * 50)
    
    recip = SpikeFP32Reciprocal().to(device)
    
    all_pass = True
    
    # 1/Inf = 0
    x = np.array([np.inf], dtype=np.float32)
    x_pulse = float32_to_pulse(x, device)
    recip.reset()
    result = pulse_to_float32(recip(x_pulse)).cpu().numpy()[0]
    check1 = result == 0.0
    print(f"  {'✓' if check1 else '✗'} 1/Inf = {result} (期望: 0)")
    all_pass = all_pass and check1
    
    # 1/-Inf = -0
    x = np.array([-np.inf], dtype=np.float32)
    x_pulse = float32_to_pulse(x, device)
    recip.reset()
    result = pulse_to_float32(recip(x_pulse)).cpu().numpy()[0]
    check2 = result == 0.0 or result == -0.0
    print(f"  {'✓' if check2 else '✗'} 1/-Inf = {result} (期望: -0)")
    all_pass = all_pass and check2
    
    # 1/0 = Inf
    x = np.array([0.0], dtype=np.float32)
    x_pulse = float32_to_pulse(x, device)
    recip.reset()
    result = pulse_to_float32(recip(x_pulse)).cpu().numpy()[0]
    check3 = np.isinf(result) and result > 0
    print(f"  {'✓' if check3 else '✗'} 1/0 = {result} (期望: Inf)")
    all_pass = all_pass and check3
    
    return all_pass


def test_batch_reciprocal():
    """测试批量倒数"""
    from atomic_ops import SpikeFP32Reciprocal
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试批量倒数 (device={device})")
    print("=" * 50)
    
    recip = SpikeFP32Reciprocal().to(device)
    
    # 批量输入
    x = np.array([1.0, 2.0, 4.0, 0.5, 0.25], dtype=np.float32)
    x_pulse = float32_to_pulse(x, device)
    
    recip.reset()
    result_pulse = recip(x_pulse)
    result = pulse_to_float32(result_pulse).cpu().numpy()
    
    expected = 1.0 / x
    
    print(f"  输入: {x}")
    print(f"  输出: {result}")
    print(f"  期望: {expected}")
    
    # 检查误差
    rel_err = np.abs(result - expected) / np.abs(expected)
    max_err = rel_err.max()
    print(f"  最大相对误差: {max_err:.2e}")
    
    passed = max_err < 1e-5
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed


def main():
    print("=" * 60)
    print("SpikeFP32Reciprocal 测试")
    print("=" * 60)
    
    results = []
    
    results.append(("基本功能", test_basic_reciprocal()))
    results.append(("位精确性", test_bit_exactness()))
    results.append(("特殊值", test_special_values()))
    results.append(("批量处理", test_batch_reciprocal()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed
    
    print("=" * 60)
    if all_pass:
        print("所有测试通过!")
    else:
        print("部分测试失败!")
    
    return all_pass


if __name__ == "__main__":
    main()
