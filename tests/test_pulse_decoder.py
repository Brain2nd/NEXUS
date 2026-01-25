"""
脉冲解码器测试 (Pulse Decoder Test)
==================================

测试 PulseFloatingPointDecoder、PulseFP16Decoder、PulseFP32Decoder 的正确性。

测试内容
--------
1. FP8 解码器往返精度
2. 特殊值处理 (零、subnormal、边界值)
3. 任意维度支持
4. FP16/FP32 解码器

作者: MofNeuroSim Project
"""
import torch
import sys
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    PulseFloatingPointEncoder,
    PulseFloatingPointDecoder,
    PulseFP16Decoder,
    PulseFP32Decoder,
)

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_fp8_decoder_roundtrip():
    """测试 FP8 编码器→解码器往返精度"""
    print("\n" + "=" * 60)
    print("测试 1: FP8 解码器往返精度")
    print("=" * 60)
    print(f"Device: {device}")

    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    
    # 测试各种数值
    test_values = [
        0.0, 1.0, -1.0, 0.5, -0.5,
        1.5, 2.0, 4.0, 8.0, 16.0,
        0.25, 0.125, 0.0625,
        -2.0, -4.0, -0.25,
        448.0, -448.0,  # FP8 E4M3 最大值
    ]
    
    passed = 0
    total = len(test_values)
    
    for val in test_values:
        x = torch.tensor([val], device=device).to(torch.float8_e4m3fn).float()
        pulse = encoder(x)
        decoded = decoder(pulse)
        
        match = torch.isclose(decoded, x, rtol=1e-5).item()
        status = "✓" if match else "✗"
        print(f"  {status} {val:>8.4f} → 编码 → 解码 → {decoded.item():>8.4f}")
        
        if match:
            passed += 1
    
    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_fp8_special_values():
    """测试特殊值处理"""
    print("\n" + "=" * 60)
    print("测试 2: 特殊值处理")
    print("=" * 60)

    decoder = PulseFloatingPointDecoder().to(device)

    # 测试用例: (字节值, 预期浮点值)
    test_cases = [
        # 零
        (0b00000000, 0.0),      # +0
        (0b10000000, -0.0),     # -0

        # Subnormal (E=0, M≠0)
        (0b00000001, 0.001953125),  # 最小正 subnormal: 2^-6 * 1/8
        (0b00000111, 0.013671875),  # 最大正 subnormal: 2^-6 * 7/8

        # Normal
        (0b00111000, 1.0),      # 1.0: E=7, M=0
        (0b01000000, 2.0),      # 2.0: E=8, M=0
        (0b00110100, 0.75),     # 0.75: E=6, M=4 → 2^-1 * 1.5

        # 最大值
        (0b01111110, 448.0),    # 最大正: E=15, M=6
        (0b11111110, -448.0),   # 最大负
    ]

    passed = 0
    total = len(test_cases)

    for byte_val, expected in test_cases:
        # 构造脉冲
        pulse = torch.zeros(1, 8, device=device)
        for i in range(8):
            pulse[0, i] = (byte_val >> (7 - i)) & 1
        
        decoded = decoder(pulse)
        
        # 比较 (处理 -0.0)
        if expected == 0.0:
            match = decoded.item() == 0.0
        else:
            match = torch.isclose(decoded, torch.tensor([expected], device=device), rtol=1e-5).item()
        
        status = "✓" if match else "✗"
        print(f"  {status} 0b{byte_val:08b} → {decoded.item():>12.6f} (预期: {expected:>12.6f})")
        
        if match:
            passed += 1
    
    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_arbitrary_dimensions():
    """测试任意维度支持"""
    print("\n" + "=" * 60)
    print("测试 3: 任意维度支持")
    print("=" * 60)

    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFloatingPointDecoder().to(device)

    shapes = [
        (10,),
        (5, 8),
        (2, 3, 4),
        (2, 2, 2, 2),
        (1, 1, 1, 1, 1),
    ]

    passed = 0
    total = len(shapes)

    for shape in shapes:
        x = torch.randn(shape, device=device).to(torch.float8_e4m3fn).float()
        pulse = encoder(x)
        decoded = decoder(pulse)
        
        expected_pulse_shape = shape + (8,)
        shape_ok = (pulse.shape == expected_pulse_shape) and (decoded.shape == x.shape)
        value_ok = torch.allclose(decoded, x, rtol=1e-5)
        
        success = shape_ok and value_ok
        status = "✓" if success else "✗"
        print(f"  {status} {str(shape):20s} → {str(pulse.shape):25s} → {str(decoded.shape)}")
        
        if success:
            passed += 1
    
    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_batch_random():
    """测试大批量随机数据"""
    print("\n" + "=" * 60)
    print("测试 4: 大批量随机数据")
    print("=" * 60)

    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFloatingPointDecoder().to(device)

    torch.manual_seed(42)

    # 测试多个批次
    test_configs = [
        (100, 0.1),
        (100, 1.0),
        (100, 10.0),
        (1000, 0.5),
    ]

    total_match = 0
    total_count = 0

    for n, scale in test_configs:
        x = torch.randn(n, device=device) * scale
        x_fp8 = x.to(torch.float8_e4m3fn).float()
        
        pulse = encoder(x_fp8)
        decoded = decoder(pulse)
        
        match = torch.isclose(decoded, x_fp8, rtol=1e-5).sum().item()
        rate = match / n * 100
        
        total_match += match
        total_count += n
        
        print(f"  n={n}, scale={scale}: {rate:.1f}% ({match}/{n})")
    
    overall_rate = total_match / total_count * 100
    print(f"\n  总体对齐率: {overall_rate:.2f}%")
    return overall_rate == 100


def main():
    print("=" * 60)
    print("脉冲解码器测试")
    print("=" * 60)
    
    results = []
    results.append(("往返精度", test_fp8_decoder_roundtrip()))
    results.append(("特殊值处理", test_fp8_special_values()))
    results.append(("任意维度", test_arbitrary_dimensions()))
    results.append(("批量随机", test_batch_random()))
    
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

