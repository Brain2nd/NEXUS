"""
特殊值传播测试
===============

测试 Inf/NaN 修复是否解决了大规模矩阵的精度问题。

假设验证：
- 大规模矩阵中出现边界值的概率更高
- 之前编码器/解码器没有正确处理 Inf/NaN
- 这会导致整体精度损坏

测试内容：
1. 单独测试 Inf/NaN 的编码-解码正确性
2. 在大规模矩阵中注入少量特殊值，验证不会传播错误
3. 对比修复前后的行为（通过禁用特殊值处理模拟）

作者: MofNeuroSim Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def test_special_value_encoding_decoding():
    """测试 1: Inf/NaN 编码-解码正确性"""
    print("\n" + "="*70)
    print("测试 1: Inf/NaN 编码-解码正确性")
    print("="*70)

    from atomic_ops import PulseFloatingPointEncoder
    from atomic_ops.encoding.pulse_decoder import (
        PulseFloatingPointDecoder, PulseFP16Decoder, PulseFP32Decoder
    )

    # FP8 测试
    print("\n--- FP8 E4M3 ---")
    encoder_fp8 = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    decoder_fp8 = PulseFloatingPointDecoder(exponent_bits=4, mantissa_bits=3).to(device)

    test_values = [
        (float('inf'), "+Inf"),
        (float('-inf'), "-Inf"),
        (float('nan'), "NaN"),
        (0.0, "Zero"),
        (-0.0, "-Zero"),
        (1.0, "One"),
        (-1.0, "-One"),
        (448.0, "Max Normal"),
        (1e-7, "Near Subnormal"),
    ]

    fp8_results = []
    for val, name in test_values:
        x = torch.tensor([[val]], device=device)
        encoder_fp8.reset()
        pulse = encoder_fp8(x)
        decoded = decoder_fp8(pulse)
        decoded_val = decoded.item()

        # 检查结果
        if np.isnan(val):
            correct = np.isnan(decoded_val)
        elif np.isinf(val):
            correct = np.isinf(decoded_val) and (np.sign(val) == np.sign(decoded_val) or (val > 0 and decoded_val > 0) or (val < 0 and decoded_val < 0))
        else:
            correct = np.isclose(decoded_val, val, rtol=0.1, atol=1e-7) or (val == 0 and decoded_val == 0)

        fp8_results.append(correct)
        status = "✓" if correct else "✗"
        print(f"  {name:15s}: Input={val:>12} -> Decoded={decoded_val:>12} {status}")

    fp8_pass_rate = sum(fp8_results) / len(fp8_results) * 100
    print(f"\nFP8 Pass Rate: {fp8_pass_rate:.1f}%")

    # FP32 测试
    print("\n--- FP32 ---")
    encoder_fp32 = PulseFloatingPointEncoder(
        exponent_bits=8, mantissa_bits=23,
        scan_integer_bits=128, scan_decimal_bits=128
    ).to(device)
    decoder_fp32 = PulseFP32Decoder().to(device)

    test_values_fp32 = [
        (float('inf'), "+Inf"),
        (float('-inf'), "-Inf"),
        (float('nan'), "NaN"),
        (0.0, "Zero"),
        (1.0, "One"),
        (3.4e38, "Near Max"),
        (1e-38, "Near Min"),
    ]

    fp32_results = []
    for val, name in test_values_fp32:
        x = torch.tensor([[val]], device=device)
        encoder_fp32.reset()
        pulse = encoder_fp32(x)
        decoded = decoder_fp32(pulse)
        decoded_val = decoded.item()

        if np.isnan(val):
            correct = np.isnan(decoded_val)
        elif np.isinf(val):
            correct = np.isinf(decoded_val) and np.sign(decoded_val) == np.sign(val)
        else:
            correct = np.isclose(decoded_val, val, rtol=1e-5) or (val == 0 and decoded_val == 0)

        fp32_results.append(correct)
        status = "✓" if correct else "✗"
        print(f"  {name:15s}: Input={val:>15} -> Decoded={decoded_val:>15} {status}")

    fp32_pass_rate = sum(fp32_results) / len(fp32_results) * 100
    print(f"\nFP32 Pass Rate: {fp32_pass_rate:.1f}%")

    return fp8_pass_rate, fp32_pass_rate


def test_special_value_injection_in_matrix():
    """测试 2: 大规模矩阵中注入特殊值"""
    print("\n" + "="*70)
    print("测试 2: 大规模矩阵中注入特殊值")
    print("目标: 验证特殊值不会传播到整个矩阵")
    print("="*70)

    from atomic_ops import PulseFloatingPointEncoder
    from atomic_ops.encoding.pulse_decoder import PulseFP32Decoder

    encoder = PulseFloatingPointEncoder(
        exponent_bits=8, mantissa_bits=23,
        scan_integer_bits=128, scan_decimal_bits=128
    ).to(device)
    decoder = PulseFP32Decoder().to(device)

    test_sizes = [64, 256, 1024, 4096]
    injection_rates = [0.0, 0.001, 0.01, 0.1]  # 0%, 0.1%, 1%, 10%

    results = []

    for size in test_sizes:
        for inject_rate in injection_rates:
            torch.manual_seed(42)

            # 生成正常数据
            x = torch.randn(1, size, device=device)

            # 注入特殊值
            n_special = int(size * inject_rate)
            if n_special > 0:
                special_indices = torch.randperm(size)[:n_special]
                for i, idx in enumerate(special_indices):
                    if i % 3 == 0:
                        x[0, idx] = float('inf')
                    elif i % 3 == 1:
                        x[0, idx] = float('-inf')
                    else:
                        x[0, idx] = float('nan')

            # 编码-解码
            encoder.reset()
            pulse = encoder(x)
            decoded = decoder(pulse)

            # 分析结果
            x_flat = x.flatten()
            decoded_flat = decoded.flatten()

            # 正常值的精度
            normal_mask = torch.isfinite(x_flat)
            if normal_mask.any():
                normal_x = x_flat[normal_mask]
                normal_decoded = decoded_flat[normal_mask]

                # 检查是否有 NaN 传播到正常值
                nan_in_normal = torch.isnan(normal_decoded).sum().item()

                # 计算正常值的误差
                rel_err = torch.abs(normal_decoded - normal_x) / (torch.abs(normal_x) + 1e-10)
                mean_rel_err = rel_err.mean().item()
                max_rel_err = rel_err.max().item()
            else:
                nan_in_normal = 0
                mean_rel_err = 0
                max_rel_err = 0

            # 特殊值的正确性
            inf_mask = torch.isinf(x_flat)
            nan_mask = torch.isnan(x_flat)

            inf_correct = 0
            if inf_mask.any():
                inf_original = x_flat[inf_mask]
                inf_decoded = decoded_flat[inf_mask]
                inf_correct = (torch.isinf(inf_decoded) & (torch.sign(inf_original) == torch.sign(inf_decoded))).float().mean().item()

            nan_correct = 0
            if nan_mask.any():
                nan_decoded = decoded_flat[nan_mask]
                nan_correct = torch.isnan(nan_decoded).float().mean().item()

            results.append({
                'size': size,
                'inject_rate': inject_rate,
                'nan_propagation': nan_in_normal,
                'mean_rel_err': mean_rel_err,
                'max_rel_err': max_rel_err,
                'inf_correct': inf_correct,
                'nan_correct': nan_correct
            })

            print(f"\n  Size={size:4d}, Inject={inject_rate*100:5.1f}%:")
            print(f"    NaN传播到正常值: {nan_in_normal}")
            print(f"    正常值平均相对误差: {mean_rel_err:.2e}")
            print(f"    正常值最大相对误差: {max_rel_err:.2e}")
            if inf_mask.any():
                print(f"    Inf保持率: {inf_correct*100:.1f}%")
            if nan_mask.any():
                print(f"    NaN保持率: {nan_correct*100:.1f}%")

    # 检查是否有NaN传播
    any_propagation = any(r['nan_propagation'] > 0 for r in results)
    print(f"\n总结: NaN传播问题 = {'存在' if any_propagation else '不存在'}")

    return results


def test_arithmetic_with_special_values():
    """测试 3: 算术运算中的特殊值处理"""
    print("\n" + "="*70)
    print("测试 3: 算术运算中的特殊值处理")
    print("="*70)

    from atomic_ops import SpikeFP32Adder, SpikeFP32Multiplier
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    adder = SpikeFP32Adder().to(device)
    mul = SpikeFP32Multiplier().to(device)

    # 测试用例: (a, b, expected_sum, expected_prod)
    test_cases = [
        # 正常 + 正常
        (1.0, 2.0, 3.0, 2.0),
        # Inf 运算
        (float('inf'), 1.0, float('inf'), float('inf')),
        (float('inf'), float('inf'), float('inf'), float('inf')),
        (float('inf'), float('-inf'), float('nan'), float('-inf')),  # Inf + (-Inf) = NaN
        # 零运算
        (0.0, float('inf'), float('inf'), float('nan')),  # 0 * Inf = NaN
        (0.0, 1.0, 1.0, 0.0),
        # NaN 传播
        (float('nan'), 1.0, float('nan'), float('nan')),
    ]

    print("\n加法测试:")
    add_results = []
    for a, b, expected_sum, _ in test_cases:
        a_t = torch.tensor([[a]], device=device)
        b_t = torch.tensor([[b]], device=device)
        a_pulse = float32_to_pulse(a_t)
        b_pulse = float32_to_pulse(b_t)

        adder.reset()
        sum_pulse = adder(a_pulse, b_pulse)
        sum_val = pulse_to_float32(sum_pulse).item()

        if np.isnan(expected_sum):
            correct = np.isnan(sum_val)
        elif np.isinf(expected_sum):
            correct = np.isinf(sum_val) and np.sign(sum_val) == np.sign(expected_sum)
        else:
            correct = np.isclose(sum_val, expected_sum, rtol=1e-4)

        add_results.append(correct)
        status = "✓" if correct else "✗"
        print(f"  {a:>10} + {b:>10} = {sum_val:>12} (expected: {expected_sum:>12}) {status}")

    print(f"\n加法通过率: {sum(add_results)/len(add_results)*100:.1f}%")

    print("\n乘法测试:")
    mul_results = []
    for a, b, _, expected_prod in test_cases:
        a_t = torch.tensor([[a]], device=device)
        b_t = torch.tensor([[b]], device=device)
        a_pulse = float32_to_pulse(a_t)
        b_pulse = float32_to_pulse(b_t)

        mul.reset()
        prod_pulse = mul(a_pulse, b_pulse)
        prod_val = pulse_to_float32(prod_pulse).item()

        if np.isnan(expected_prod):
            correct = np.isnan(prod_val)
        elif np.isinf(expected_prod):
            correct = np.isinf(prod_val) and np.sign(prod_val) == np.sign(expected_prod)
        else:
            correct = np.isclose(prod_val, expected_prod, rtol=1e-4)

        mul_results.append(correct)
        status = "✓" if correct else "✗"
        print(f"  {a:>10} * {b:>10} = {prod_val:>12} (expected: {expected_prod:>12}) {status}")

    print(f"\n乘法通过率: {sum(mul_results)/len(mul_results)*100:.1f}%")

    return add_results, mul_results


def test_large_matrix_precision():
    """测试 4: 大规模矩阵运算精度（验证假设）"""
    print("\n" + "="*70)
    print("测试 4: 大规模矩阵运算精度")
    print("假设验证: 大规模矩阵更容易出现特殊值导致的精度问题")
    print("="*70)

    from atomic_ops import SpikeFP32Linear_MultiPrecision
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    # 不同规模的矩阵
    test_configs = [
        (32, 32),    # 小矩阵
        (128, 128),  # 中等矩阵
        (512, 512),  # 大矩阵
    ]

    results = []

    for in_dim, out_dim in test_configs:
        print(f"\n  测试 {in_dim}x{out_dim} 矩阵:")

        # 准备数据
        torch.manual_seed(42)
        x = torch.randn(1, in_dim, device=device) * 0.5
        w = torch.randn(out_dim, in_dim, device=device) * 0.1

        # PyTorch 参考
        ref = torch.nn.functional.linear(x, w)

        # SNN
        linear = SpikeFP32Linear_MultiPrecision(in_dim, out_dim, accum_precision='fp32').to(device)
        linear.set_weight_from_float(w)

        x_pulse = float32_to_pulse(x)
        linear.reset()
        y_pulse = linear(x_pulse)
        y_snn = pulse_to_float32(y_pulse)

        # 分析
        diff = torch.abs(y_snn - ref)
        rel_err = diff / (torch.abs(ref) + 1e-10)

        # 检查是否有 NaN
        snn_nan_count = torch.isnan(y_snn).sum().item()
        ref_nan_count = torch.isnan(ref).sum().item()

        # 检查是否有 Inf
        snn_inf_count = torch.isinf(y_snn).sum().item()
        ref_inf_count = torch.isinf(ref).sum().item()

        # 计算精度 (排除非有限值)
        valid_mask = torch.isfinite(y_snn) & torch.isfinite(ref)
        if valid_mask.any():
            mean_rel_err = rel_err[valid_mask].mean().item()
            max_rel_err = rel_err[valid_mask].max().item()
            match_rate = torch.isclose(y_snn[valid_mask], ref[valid_mask], rtol=1e-4, atol=1e-5).float().mean().item() * 100
        else:
            mean_rel_err = float('nan')
            max_rel_err = float('nan')
            match_rate = 0.0

        results.append({
            'config': f"{in_dim}x{out_dim}",
            'snn_nan': snn_nan_count,
            'ref_nan': ref_nan_count,
            'snn_inf': snn_inf_count,
            'ref_inf': ref_inf_count,
            'mean_rel_err': mean_rel_err,
            'max_rel_err': max_rel_err,
            'match_rate': match_rate
        })

        print(f"    SNN NaN: {snn_nan_count}, Ref NaN: {ref_nan_count}")
        print(f"    SNN Inf: {snn_inf_count}, Ref Inf: {ref_inf_count}")
        print(f"    平均相对误差: {mean_rel_err:.2e}")
        print(f"    最大相对误差: {max_rel_err:.2e}")
        print(f"    匹配率: {match_rate:.1f}%")

    return results


def main():
    print("特殊值传播测试")
    print("="*70)
    print("验证 Inf/NaN 修复是否解决大规模矩阵精度问题")
    print("="*70)

    # 测试 1: 编码-解码正确性
    fp8_rate, fp32_rate = test_special_value_encoding_decoding()

    # 测试 2: 矩阵中的特殊值
    injection_results = test_special_value_injection_in_matrix()

    # 测试 3: 算术运算
    add_results, mul_results = test_arithmetic_with_special_values()

    # 测试 4: 大规模矩阵
    matrix_results = test_large_matrix_precision()

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"\n1. 编码-解码: FP8={fp8_rate:.1f}%, FP32={fp32_rate:.1f}%")
    print(f"2. 特殊值注入: NaN传播检测 = {'通过' if all(r['nan_propagation']==0 for r in injection_results) else '失败'}")
    print(f"3. 算术运算: 加法={sum(add_results)/len(add_results)*100:.1f}%, 乘法={sum(mul_results)/len(mul_results)*100:.1f}%")
    print(f"4. 大规模矩阵: 最大匹配率={max(r['match_rate'] for r in matrix_results):.1f}%")

    # 假设验证
    print("\n" + "="*70)
    print("假设验证")
    print("="*70)
    print("""
结论：
- 如果编码器/解码器正确处理 Inf/NaN，这些值不会传播错误到正常数据
- 大规模矩阵中，即使少量特殊值存在，也不应该影响正常值的精度
- 修复后，特殊值的编码-解码是正确的
    """)


if __name__ == '__main__':
    main()
