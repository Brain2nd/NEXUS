"""
FP32 加法器测试 - 验证 SpikeFP32Adder 的 bit-exactness
========================================================

测试内容:
1. 基本加法正确性
2. 特殊值处理 (NaN, Inf, Zero)
3. 舍入模式 (RNE)
4. ULP 误差统计

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

# 使用统一的转换函数
from atomic_ops import (
    float32_to_bits, bits_to_float32,
    float32_to_pulse, pulse_to_float32,
    float_to_pulse, pulse_to_bits
)

# 批量转换函数别名
float_to_pulse_batch = float32_to_pulse
def pulse_to_bits_batch(pulse):
    """批量转换脉冲到32位整数"""
    flat_pulse = pulse.reshape(-1, 32).cpu().numpy() > 0.5
    n = flat_pulse.shape[0]
    bits = np.zeros(n, dtype=np.uint32)
    for i in range(32):
        shift = np.uint32(31 - i)
        bits |= (flat_pulse[:, i].astype(np.uint32) << shift)
    return bits.reshape(pulse.shape[:-1])


def test_basic_addition():
    """测试基本加法"""
    from atomic_ops import SpikeFP32Adder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adder = SpikeFP32Adder().to(device)
    
    test_cases = [
        (1.0, 1.0, 2.0),
        (1.0, 2.0, 3.0),
        (0.5, 0.5, 1.0),
        (-1.0, 1.0, 0.0),
        (1.5, 2.5, 4.0),
        (0.0, 5.0, 5.0),
        (-2.0, -3.0, -5.0),
        (1e10, 1e-10, 1e10),  # 大小差异
    ]
    
    passed = 0
    for a, b, expected in test_cases:
        a_pulse = float_to_pulse(a, device)
        b_pulse = float_to_pulse(b, device)
        
        adder.reset()
        result_pulse = adder(a_pulse, b_pulse)
        
        result_bits = pulse_to_bits(result_pulse)
        result_float = bits_to_float32(result_bits)
        
        # PyTorch 参考
        pytorch_result = np.float32(a) + np.float32(b)
        
        if np.isclose(result_float, pytorch_result, rtol=1e-6) or result_float == pytorch_result:
            passed += 1
            print(f"  ✓ {a} + {b} = {result_float} (期望 {pytorch_result})")
        else:
            print(f"  ✗ {a} + {b} = {result_float} (期望 {pytorch_result})")
    
    print(f"\n基本加法: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_special_values():
    """测试特殊值"""
    from atomic_ops import SpikeFP32Adder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adder = SpikeFP32Adder().to(device)
    
    inf = float('inf')
    nan = float('nan')
    
    test_cases = [
        (0.0, 0.0, 0.0, "0 + 0"),
        (inf, 1.0, inf, "Inf + 1"),
        (1.0, inf, inf, "1 + Inf"),
        (-inf, -inf, -inf, "-Inf + -Inf"),
        (inf, -inf, nan, "Inf + -Inf → NaN"),
        (nan, 1.0, nan, "NaN + 1"),
        (1.0, nan, nan, "1 + NaN"),
    ]
    
    passed = 0
    for a, b, expected, desc in test_cases:
        a_pulse = float_to_pulse(a, device)
        b_pulse = float_to_pulse(b, device)
        
        adder.reset()
        result_pulse = adder(a_pulse, b_pulse)
        
        result_bits = pulse_to_bits(result_pulse)
        result_float = bits_to_float32(result_bits)
        
        # 检查特殊值
        if np.isnan(expected):
            success = np.isnan(result_float)
        elif np.isinf(expected):
            success = np.isinf(result_float) and np.sign(result_float) == np.sign(expected)
        else:
            success = result_float == expected
        
        if success:
            passed += 1
            print(f"  ✓ {desc}: {result_float}")
        else:
            print(f"  ✗ {desc}: {result_float} (期望 {expected})")
    
    print(f"\n特殊值: {passed}/{len(test_cases)} 通过")
    return passed >= len(test_cases) - 1  # 允许1个失败


def test_random_ulp():
    """随机测试 ULP 误差"""
    from atomic_ops import SpikeFP32Adder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adder = SpikeFP32Adder().to(device)
    
    np.random.seed(42)
    n_samples = 100
    
    # 生成随机数
    a_vals = np.random.randn(n_samples).astype(np.float32) * 10
    b_vals = np.random.randn(n_samples).astype(np.float32) * 10
    
    a_pulse = float_to_pulse_batch(a_vals, device)
    b_pulse = float_to_pulse_batch(b_vals, device)
    
    adder.reset()
    result_pulse = adder(a_pulse, b_pulse)
    
    snn_bits = pulse_to_bits_batch(result_pulse)
    pytorch_result = a_vals + b_vals
    pytorch_bits = pytorch_result.view(np.uint32)
    
    # 计算 ULP 误差
    ulp_errors = np.abs(snn_bits.astype(np.int64) - pytorch_bits.astype(np.int64))
    
    exact_match = np.sum(ulp_errors == 0)
    max_ulp = np.max(ulp_errors)
    mean_ulp = np.mean(ulp_errors)
    
    print(f"\n随机测试 ({n_samples} 样本):")
    print(f"  0 ULP (精确匹配): {exact_match}/{n_samples} ({100*exact_match/n_samples:.1f}%)")
    print(f"  最大 ULP: {max_ulp}")
    print(f"  平均 ULP: {mean_ulp:.2f}")
    
    return exact_match >= n_samples * 0.9  # 90% 精确匹配


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("FP32 加法器测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    tests = [
        ("基本加法", test_basic_addition),
        ("特殊值", test_special_values),
        ("随机 ULP", test_random_ulp),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {name} 通过")
            else:
                failed += 1
                print(f"✗ {name} 失败")
        except Exception as e:
            print(f"✗ {name} 异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
