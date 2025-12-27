"""
Converters 模块测试 - 验证脉冲转换函数的正确性
================================================

测试内容:
1. FP8 转换: float_to_fp8_bits / fp8_bits_to_float
2. FP32 转换: float32_to_pulse / pulse_to_float32
3. FP64 转换: float64_to_pulse / pulse_to_float64
4. 批量处理
5. 特殊值处理

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def test_fp8_conversion():
    """测试 FP8 转换函数"""
    from atomic_ops import float_to_fp8_bits, fp8_bits_to_float
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 FP8 转换 (device={device})")
    print("=" * 50)
    
    # 测试值
    test_values = [
        0.0, 1.0, -1.0, 0.5, -0.5,
        2.0, -2.0, 0.125, -0.125,
        448.0,  # FP8 E4M3 最大正常值
        -448.0,
    ]
    
    x = torch.tensor(test_values, device=device)
    
    # 转换为脉冲
    pulse = float_to_fp8_bits(x, device)
    assert pulse.shape == (len(test_values), 8), f"形状错误: {pulse.shape}"
    
    # 转换回浮点
    recovered = fp8_bits_to_float(pulse)
    
    # FP8 精度有限，需要先量化原始值
    x_fp8 = x.to(torch.float8_e4m3fn).to(torch.float32)
    
    match = (recovered == x_fp8).all()
    print(f"✓ FP8 往返转换: {match.item()}")
    
    # 批量测试
    batch = torch.randn(100, 50, device=device)
    pulse_batch = float_to_fp8_bits(batch, device)
    assert pulse_batch.shape == (100, 50, 8), f"批量形状错误: {pulse_batch.shape}"
    
    recovered_batch = fp8_bits_to_float(pulse_batch)
    batch_fp8 = batch.to(torch.float8_e4m3fn).to(torch.float32)
    batch_match = (recovered_batch == batch_fp8).all()
    print(f"✓ FP8 批量转换: {batch_match.item()}")
    
    return match.item() and batch_match.item()


def test_fp32_conversion():
    """测试 FP32 转换函数"""
    from atomic_ops import float32_to_pulse, pulse_to_float32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 FP32 转换 (device={device})")
    print("=" * 50)
    
    # 测试值
    test_values = [
        0.0, 1.0, -1.0,
        3.14159, -2.71828,
        1e10, -1e10,
        1e-10, -1e-10,
        float('inf'), float('-inf'),
    ]
    
    x = np.array(test_values, dtype=np.float32)
    
    # 转换为脉冲
    pulse = float32_to_pulse(x, device)
    assert pulse.shape == (len(test_values), 32), f"形状错误: {pulse.shape}"
    
    # 转换回浮点
    recovered = pulse_to_float32(pulse).cpu().numpy()
    
    # 检查 (NaN 需要特殊处理)
    match_count = 0
    for i, (orig, rec) in enumerate(zip(x, recovered)):
        if np.isnan(orig) and np.isnan(rec):
            match_count += 1
        elif orig == rec:
            match_count += 1
        else:
            print(f"  不匹配: {orig} vs {rec}")
    
    print(f"✓ FP32 往返转换: {match_count}/{len(test_values)}")
    
    # 批量测试
    batch = np.random.randn(50, 30).astype(np.float32)
    pulse_batch = float32_to_pulse(batch, device)
    assert pulse_batch.shape == (50, 30, 32), f"批量形状错误: {pulse_batch.shape}"
    
    recovered_batch = pulse_to_float32(pulse_batch).cpu().numpy()
    batch_match = np.allclose(batch, recovered_batch, equal_nan=True)
    print(f"✓ FP32 批量转换: {batch_match}")
    
    return match_count == len(test_values) and batch_match


def test_fp64_conversion():
    """测试 FP64 转换函数"""
    from atomic_ops import float64_to_pulse, pulse_to_float64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 FP64 转换 (device={device})")
    print("=" * 50)
    
    # 测试值
    test_values = [
        0.0, 1.0, -1.0,
        3.141592653589793, -2.718281828459045,
        1e100, -1e100,
        1e-100, -1e-100,
        float('inf'), float('-inf'),
    ]
    
    x = np.array(test_values, dtype=np.float64)
    
    # 转换为脉冲
    pulse = float64_to_pulse(x, device)
    assert pulse.shape == (len(test_values), 64), f"形状错误: {pulse.shape}"
    
    # 转换回浮点
    recovered = pulse_to_float64(pulse).cpu().numpy()
    
    # 检查
    match_count = 0
    for i, (orig, rec) in enumerate(zip(x, recovered)):
        if np.isnan(orig) and np.isnan(rec):
            match_count += 1
        elif orig == rec:
            match_count += 1
        else:
            print(f"  不匹配: {orig} vs {rec}")
    
    print(f"✓ FP64 往返转换: {match_count}/{len(test_values)}")
    
    # 批量测试
    batch = np.random.randn(30, 20).astype(np.float64)
    pulse_batch = float64_to_pulse(batch, device)
    assert pulse_batch.shape == (30, 20, 64), f"批量形状错误: {pulse_batch.shape}"
    
    recovered_batch = pulse_to_float64(pulse_batch).cpu().numpy()
    batch_match = np.allclose(batch, recovered_batch, equal_nan=True)
    print(f"✓ FP64 批量转换: {batch_match}")
    
    return match_count == len(test_values) and batch_match


def test_helper_functions():
    """测试辅助函数"""
    from atomic_ops import float32_to_bits, bits_to_float32, float64_to_bits, bits_to_float64
    
    print(f"\n测试辅助函数")
    print("=" * 50)
    
    # FP32
    val32 = 3.14
    bits32 = float32_to_bits(val32)
    recovered32 = bits_to_float32(bits32)
    match32 = abs(val32 - recovered32) < 1e-6
    print(f"✓ FP32 bits: {val32} -> 0x{bits32:08X} -> {recovered32} (match={match32})")
    
    # FP64
    val64 = 3.141592653589793
    bits64 = float64_to_bits(val64)
    recovered64 = bits_to_float64(bits64)
    match64 = val64 == recovered64
    print(f"✓ FP64 bits: {val64} -> 0x{bits64:016X} -> {recovered64} (match={match64})")
    
    return match32 and match64


def main():
    print("=" * 60)
    print("Converters 模块测试")
    print("=" * 60)
    
    results = []
    
    results.append(("FP8 转换", test_fp8_conversion()))
    results.append(("FP32 转换", test_fp32_conversion()))
    results.append(("FP64 转换", test_fp64_conversion()))
    results.append(("辅助函数", test_helper_functions()))
    
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
