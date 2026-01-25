"""
FP8 到 FP32 乘法器测试 (FP8-to-FP32 Multiplier Test)
===================================================

测试 SpikeFP8MulToFP32 的正确性。

此乘法器将两个 FP8 输入相乘，输出 FP32 精度结果，
用于 FP32 累加模式，避免中间舍入误差。

数学公式
--------
FP8 × FP8 → FP32

输出格式: [S | E7...E0 | M22...M0] (32位)

测试内容
--------
1. 基本乘法正确性
2. 与 PyTorch 参考对比
3. 边界情况 (零、subnormal)

作者: MofNeuroSim Project
"""
import torch
import sys
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    PulseFloatingPointEncoder,
    PulseFloatingPointDecoder,
    PulseFP32Decoder,
)
from atomic_ops.arithmetic.fp8.fp8_mul_to_fp32 import SpikeFP8MulToFP32

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_basic_multiplication():
    """测试基本乘法正确性"""
    print("\n" + "=" * 60)
    print("测试 1: 基本乘法正确性")
    print("=" * 60)
    print(f"Device: {device}")

    encoder = PulseFloatingPointEncoder().to(device)
    mul = SpikeFP8MulToFP32().to(device)
    decoder = PulseFP32Decoder().to(device)
    
    test_cases = [
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 4.0),
        (2.0, 3.0, 6.0),
        (0.5, 2.0, 1.0),
        (0.5, 0.5, 0.25),
        (-1.0, 2.0, -2.0),
        (-2.0, -2.0, 4.0),
        (1.5, 2.0, 3.0),
        (4.0, 4.0, 16.0),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for a, b, expected in test_cases:
        a_fp8 = torch.tensor([a], device=device).to(torch.float8_e4m3fn).float()
        b_fp8 = torch.tensor([b], device=device).to(torch.float8_e4m3fn).float()
        expected_fp32 = a_fp8 * b_fp8  # PyTorch 参考
        
        a_pulse = encoder(a_fp8)
        b_pulse = encoder(b_fp8)
        
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)
        result = decoder(result_pulse)
        
        match = torch.isclose(result, expected_fp32, rtol=1e-5).item()
        status = "✓" if match else "✗"
        print(f"  {status} {a} × {b} = {result.item():.6f} (预期: {expected_fp32.item():.6f})")
        
        if match:
            passed += 1
    
    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_zero_multiplication():
    """测试零乘法"""
    print("\n" + "=" * 60)
    print("测试 2: 零乘法")
    print("=" * 60)

    encoder = PulseFloatingPointEncoder().to(device)
    mul = SpikeFP8MulToFP32().to(device)
    decoder = PulseFP32Decoder().to(device)

    test_cases = [
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, -5.0),
    ]

    passed = 0
    total = len(test_cases)

    for a, b in test_cases:
        a_fp8 = torch.tensor([a], device=device).to(torch.float8_e4m3fn).float()
        b_fp8 = torch.tensor([b], device=device).to(torch.float8_e4m3fn).float()
        
        a_pulse = encoder(a_fp8)
        b_pulse = encoder(b_fp8)
        
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)
        result = decoder(result_pulse)
        
        is_zero = (result.item() == 0.0)
        status = "✓" if is_zero else "✗"
        print(f"  {status} {a} × {b} = {result.item():.6f} (预期: 0.0)")
        
        if is_zero:
            passed += 1
    
    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_random_alignment():
    """测试随机数据与 PyTorch 对齐"""
    print("\n" + "=" * 60)
    print("测试 3: 随机数据对齐")
    print("=" * 60)

    encoder = PulseFloatingPointEncoder().to(device)
    mul = SpikeFP8MulToFP32().to(device)
    decoder = PulseFP32Decoder().to(device)

    torch.manual_seed(42)

    n_tests = 100
    match_count = 0

    for i in range(n_tests):
        a = torch.randn(1, device=device) * 2
        b = torch.randn(1, device=device) * 2
        
        a_fp8 = a.to(torch.float8_e4m3fn).float()
        b_fp8 = b.to(torch.float8_e4m3fn).float()
        
        # PyTorch 参考
        expected = a_fp8 * b_fp8
        
        # SNN 计算
        a_pulse = encoder(a_fp8)
        b_pulse = encoder(b_fp8)
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)
        result = decoder(result_pulse)
        
        # 100%位精确比较 - 比较原始位表示
        import struct
        result_bits = struct.unpack('>I', struct.pack('>f', result.item()))[0]
        expected_bits = struct.unpack('>I', struct.pack('>f', expected.item()))[0]
        if result_bits == expected_bits:
            match_count += 1
    
    rate = match_count / n_tests * 100
    print(f"  位精确率: {rate:.1f}% ({match_count}/{n_tests}) [要求100%]")
    
    return rate == 100  # 必须100%位精确


def test_batch_operation():
    """测试批量操作"""
    print("\n" + "=" * 60)
    print("测试 4: 批量操作")
    print("=" * 60)

    encoder = PulseFloatingPointEncoder().to(device)
    mul = SpikeFP8MulToFP32().to(device)
    decoder = PulseFP32Decoder().to(device)

    # 批量输入
    a = torch.tensor([1.0, 2.0, 0.5, -1.0, 4.0], device=device)
    b = torch.tensor([2.0, 3.0, 0.5, -2.0, 0.25], device=device)
    
    a_fp8 = a.to(torch.float8_e4m3fn).float()
    b_fp8 = b.to(torch.float8_e4m3fn).float()
    
    # PyTorch 参考
    expected = a_fp8 * b_fp8
    
    # SNN 计算
    a_pulse = encoder(a_fp8)
    b_pulse = encoder(b_fp8)
    mul.reset()
    result_pulse = mul(a_pulse, b_pulse)
    result = decoder(result_pulse)
    
    match = torch.isclose(result, expected, rtol=1e-5)
    match_count = match.sum().item()
    
    print(f"  输入形状: {a.shape}")
    print(f"  输出形状: {result.shape}")
    print(f"  对齐: {match_count}/{len(a)}")
    
    for i in range(len(a)):
        status = "✓" if match[i] else "✗"
        print(f"    {status} {a_fp8[i].item():.2f} × {b_fp8[i].item():.2f} = {result[i].item():.4f} (预期: {expected[i].item():.4f})")
    
    return match_count == len(a)


def main():
    print("=" * 60)
    print("FP8-to-FP32 乘法器测试")
    print("=" * 60)
    
    results = []
    results.append(("基本乘法", test_basic_multiplication()))
    results.append(("零乘法", test_zero_multiplication()))
    results.append(("随机对齐", test_random_alignment()))
    results.append(("批量操作", test_batch_operation()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ 所有测试通过!")
    else:
        print("\n✗ 存在失败测试")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

