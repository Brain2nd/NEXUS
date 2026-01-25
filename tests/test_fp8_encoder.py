"""
FP8 编码器测试 (CLAUDE.md #8: 随机+边界值)
==========================================

测试 PulseFloatingPointEncoder 的正确性。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from atomic_ops import PulseFloatingPointEncoder
from atomic_ops.encoding.pulse_decoder import PulseFloatingPointDecoder

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_fp_encoder():
    """测试基本编码功能"""
    print("=== 基本编码测试 (FP8: E3M4) ===")
    print(f"Device: {device}")
    encoder = PulseFloatingPointEncoder(exponent_bits=3, mantissa_bits=4).to(device)

    # Input Shape: [2, 3] (Batch=2, Channel=3)
    inputs = torch.tensor([
        [1.5, -0.25, 0.0],
        [6.0, -1.5, 0.0625]
    ], device=device)
    print(f"Input Shape: {inputs.shape}")
    print(f"Input Values:\n{inputs}")

    output = encoder(inputs)

    print("-" * 30)
    print(f"Output Shape: {output.shape}")
    print(f"Output Values (Last dim is [S, E, M]):\n{output.int()}")

    # 验证 1.5 -> 0 011 1000
    case1 = output[0, 0].int().tolist()
    passed1 = case1 == [0, 0, 1, 1, 1, 0, 0, 0]
    if passed1:
        print("Case 1.5 Passed ✓")
    else:
        print(f"Case 1.5 Failed: {case1} ✗")

    # 验证 -0.25 -> 1 001 0000
    case2 = output[0, 1].int().tolist()
    passed2 = case2 == [1, 0, 0, 1, 0, 0, 0, 0]
    if passed2:
        print("Case -0.25 Passed ✓")
    else:
        print(f"Case -0.25 Failed: {case2} ✗")

    return passed1 and passed2


def test_boundary_values():
    """测试边界值"""
    print("\n=== 边界值测试 (FP8: E4M3) ===")
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    decoder = PulseFloatingPointDecoder().to(device)

    # FP8 E4M3 边界值
    boundary_values = [
        0.0,                    # 零
        -0.0,                   # 负零
        1.0,                    # 单位正
        -1.0,                   # 单位负
        0.5,                    # 小正数
        -0.5,                   # 小负数
        448.0,                  # FP8 E4M3 最大值
        -448.0,                 # FP8 E4M3 最小值
        0.001953125,            # 最小正 subnormal
        0.125,                  # 2^-3
        2.0,                    # 2^1
        4.0,                    # 2^2
    ]

    passed = 0
    total = len(boundary_values)

    for val in boundary_values:
        x = torch.tensor([val], device=device).to(torch.float8_e4m3fn).float()
        pulse = encoder(x)
        decoded = decoder(pulse)

        match = torch.isclose(decoded, x, rtol=1e-5).item()
        status = "✓" if match else "✗"
        print(f"  {status} {val:>12.6f} → 编码 → 解码 → {decoded.item():>12.6f}")

        if match:
            passed += 1

    print(f"\n边界值测试: {passed}/{total} 通过")
    return passed == total


def test_random_values():
    """测试随机值"""
    print("\n=== 随机值测试 (FP8: E4M3) ===")
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    decoder = PulseFloatingPointDecoder().to(device)

    torch.manual_seed(42)

    # 多种范围的随机测试
    test_configs = [
        (50, 0.5, "小范围"),
        (50, 2.0, "中范围"),
        (50, 10.0, "大范围"),
    ]

    total_passed = 0
    total_count = 0

    for n, scale, desc in test_configs:
        x = torch.randn(n, device=device) * scale
        x_fp8 = x.to(torch.float8_e4m3fn).float()

        pulse = encoder(x_fp8)
        decoded = decoder(pulse)

        match = torch.isclose(decoded, x_fp8, rtol=1e-5).sum().item()
        rate = match / n * 100

        total_passed += match
        total_count += n

        print(f"  {desc}: {match}/{n} ({rate:.1f}%)")

    overall_rate = total_passed / total_count * 100
    print(f"\n随机值测试: {total_passed}/{total_count} ({overall_rate:.1f}%)")
    return overall_rate >= 95


def test_batch_shapes():
    """测试不同批量形状"""
    print("\n=== 批量形状测试 ===")
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    decoder = PulseFloatingPointDecoder().to(device)

    torch.manual_seed(123)

    shapes = [
        (10,),
        (5, 8),
        (2, 3, 4),
        (2, 2, 2, 2),
    ]

    passed = 0
    total = len(shapes)

    for shape in shapes:
        x = torch.randn(shape, device=device).to(torch.float8_e4m3fn).float()
        pulse = encoder(x)
        decoded = decoder(pulse)

        expected_pulse_shape = shape + (8,)
        shape_ok = pulse.shape == expected_pulse_shape
        value_ok = torch.allclose(decoded, x, rtol=1e-5)

        success = shape_ok and value_ok
        status = "✓" if success else "✗"
        print(f"  {status} {str(shape):20s} → {str(pulse.shape):25s}")

        if success:
            passed += 1

    print(f"\n批量形状测试: {passed}/{total} 通过")
    return passed == total


def main():
    print("=" * 60)
    print("FP8 编码器测试")
    print("=" * 60)

    results = []
    results.append(("基本编码", test_fp_encoder()))
    results.append(("边界值", test_boundary_values()))
    results.append(("随机值", test_random_values()))
    results.append(("批量形状", test_batch_shapes()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("所有测试通过!")
    else:
        print("部分测试失败!")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
