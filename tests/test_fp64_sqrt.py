"""
FP64 平方根测试 - 验证 SpikeFP64Sqrt 的正确性
==============================================

测试内容:
1. 基本平方根正确性
2. 特殊值处理 (NaN, Inf, Zero, 负数)
3. 精度验证

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

# 使用统一的转换函数
from atomic_ops import (
    float64_to_bits, bits_to_float64,
    float64_to_pulse, pulse_to_float64
)

# 兼容旧接口的别名
def float_to_pulse(val, device):
    """单个FP64值转脉冲"""
    return float64_to_pulse(np.array([val], dtype=np.float64), device)

def pulse_to_bits(pulse):
    """脉冲转64位整数"""
    bits = 0
    for i in range(64):
        if pulse[0, i].item() > 0.5:
            bits |= (1 << (63 - i))
    return bits

float_to_pulse_batch = float64_to_pulse
    
    return torch.from_numpy(pulses.reshape(original_shape + (64,))).to(device)


def pulse_to_bits_batch(pulse):
    """批量转换脉冲到64位整数"""
    flat_pulse = pulse.reshape(-1, 64).cpu().numpy() > 0.5
    n = flat_pulse.shape[0]
    
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(64):
        shift = np.uint64(63 - i)
        bits |= (flat_pulse[:, i].astype(np.uint64) << shift)
    
    return bits.reshape(pulse.shape[:-1])


def test_basic_sqrt():
    """测试基本平方根"""
    from atomic_ops import SpikeFP64Sqrt
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sqrt_mod = SpikeFP64Sqrt().to(device)
    
    test_cases = [
        (1.0, 1.0),
        (4.0, 2.0),
        (9.0, 3.0),
        (16.0, 4.0),
        (25.0, 5.0),
        (2.0, np.sqrt(2.0)),
        (0.25, 0.5),
        (0.0, 0.0),
    ]
    
    passed = 0
    for x, expected in test_cases:
        x_pulse = float_to_pulse(x, device)
        
        sqrt_mod.reset()
        result_pulse = sqrt_mod(x_pulse)
        
        result_bits = pulse_to_bits(result_pulse)
        result_float = bits_to_float64(result_bits)
        
        # PyTorch 参考
        pytorch_result = np.sqrt(np.float64(x))
        
        # 允许小误差
        if np.isclose(result_float, pytorch_result, rtol=1e-10) or result_float == pytorch_result:
            passed += 1
            print(f"  ✓ sqrt({x}) = {result_float} (期望 {pytorch_result})")
        else:
            rel_err = abs(result_float - pytorch_result) / max(abs(pytorch_result), 1e-10)
            print(f"  ✗ sqrt({x}) = {result_float} (期望 {pytorch_result}, 相对误差 {rel_err:.2e})")
    
    print(f"\n基本平方根: {passed}/{len(test_cases)} 通过")
    return passed >= len(test_cases) - 1


def test_special_values():
    """测试特殊值"""
    from atomic_ops import SpikeFP64Sqrt
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sqrt_mod = SpikeFP64Sqrt().to(device)
    
    inf = float('inf')
    nan = float('nan')
    
    test_cases = [
        (0.0, 0.0, "sqrt(0)"),
        (inf, inf, "sqrt(Inf)"),
        (-1.0, nan, "sqrt(-1) → NaN"),
        (-inf, nan, "sqrt(-Inf) → NaN"),
        (nan, nan, "sqrt(NaN)"),
    ]
    
    passed = 0
    for x, expected, desc in test_cases:
        x_pulse = float_to_pulse(x, device)
        
        sqrt_mod.reset()
        result_pulse = sqrt_mod(x_pulse)
        
        result_bits = pulse_to_bits(result_pulse)
        result_float = bits_to_float64(result_bits)
        
        # 检查特殊值
        if np.isnan(expected):
            success = np.isnan(result_float)
        elif np.isinf(expected):
            success = np.isinf(result_float) and result_float > 0
        else:
            success = result_float == expected
        
        if success:
            passed += 1
            print(f"  ✓ {desc}: {result_float}")
        else:
            print(f"  ✗ {desc}: {result_float} (期望 {expected})")
    
    print(f"\n特殊值: {passed}/{len(test_cases)} 通过")
    return passed >= len(test_cases) - 1


def test_random_precision():
    """随机测试精度"""
    from atomic_ops import SpikeFP64Sqrt
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sqrt_mod = SpikeFP64Sqrt().to(device)
    
    np.random.seed(42)
    n_samples = 10  # FP64 sqrt 较慢
    
    # 生成随机正数
    x_vals = np.abs(np.random.randn(n_samples).astype(np.float64)) * 100 + 0.1
    
    x_pulse = float_to_pulse_batch(x_vals, device)
    
    sqrt_mod.reset()
    result_pulse = sqrt_mod(x_pulse)
    
    snn_bits = pulse_to_bits_batch(result_pulse)
    snn_floats = np.array([bits_to_float64(b) for b in snn_bits.flatten()])
    
    pytorch_result = np.sqrt(x_vals)
    
    # 计算相对误差
    rel_errors = np.abs(snn_floats - pytorch_result) / np.maximum(np.abs(pytorch_result), 1e-15)
    
    max_rel_err = np.max(rel_errors)
    mean_rel_err = np.mean(rel_errors)
    
    # 计算 ULP 误差
    pytorch_bits = pytorch_result.view(np.uint64)
    ulp_errors = np.abs(snn_bits.flatten().astype(np.int64) - pytorch_bits.astype(np.int64))
    
    exact_match = np.sum(ulp_errors == 0)
    
    print(f"\n随机测试 ({n_samples} 样本):")
    print(f"  0 ULP (精确匹配): {exact_match}/{n_samples} ({100*exact_match/n_samples:.1f}%)")
    print(f"  最大相对误差: {max_rel_err:.2e}")
    print(f"  平均相对误差: {mean_rel_err:.2e}")
    
    return max_rel_err < 1e-10  # FP64 精度要求


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("FP64 平方根测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    tests = [
        ("基本平方根", test_basic_sqrt),
        ("特殊值", test_special_values),
        ("随机精度", test_random_precision),
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
