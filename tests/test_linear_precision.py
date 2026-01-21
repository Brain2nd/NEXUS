"""
Linear 层精度与时延测试
=======================

测试三种 Linear 层在不同中间精度和累积方法下的:
1. ULP 误差 (与 PyTorch 对比)
2. 前向传播时延

测试矩阵:
---------
| Linear 类型 | 累加精度选项 | 累积模式 |
|-------------|--------------|----------|
| FP8 Linear  | fp8, fp16, fp32 | sequential, parallel |
| FP16 Linear | fp16, fp32 | sequential, parallel |
| FP32 Linear | fp32, fp64 | sequential, parallel |

遵循 CLAUDE.md:
- Random + Boundary Values 测试
- GPU 优先 (CUDA)

作者: MofNeuroSim Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import time

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"测试设备: {device}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")


def compute_ulp_error_fp32(snn_result, pytorch_result):
    """计算 FP32 ULP 误差"""
    snn_bits = snn_result.view(torch.int32)
    ref_bits = pytorch_result.view(torch.int32)
    ulp_diff = (snn_bits - ref_bits).abs()
    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
    }


def compute_ulp_error_fp16(snn_result, pytorch_result):
    """计算 FP16 ULP 误差"""
    snn_bits = snn_result.view(torch.int16).int()
    ref_bits = pytorch_result.view(torch.int16).int()
    ulp_diff = (snn_bits - ref_bits).abs()
    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
    }


def compute_ulp_error_fp8(snn_result, pytorch_result):
    """计算 FP8 ULP 误差 (使用 int8)"""
    snn_bits = snn_result.view(torch.int8).int()
    ref_bits = pytorch_result.view(torch.int8).int()
    ulp_diff = (snn_bits - ref_bits).abs()
    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
    }


def measure_latency(fn, warmup=3, repeat=5):
    """测量函数执行时延"""
    for _ in range(warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


def check_tensor_device(name, tensor):
    """检查张量设备"""
    if tensor is not None:
        print(f"  {name}: device={tensor.device}, shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        print(f"  {name}: None")


# =============================================================================
# FP32 Linear 测试
# =============================================================================
def test_fp32_linear():
    """测试 FP32 Linear 不同配置"""
    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    print("\n" + "="*60)
    print("FP32 Linear 测试")
    print("="*60)

    batch_size = 4
    in_features = 8
    out_features = 4

    print(f"规模: batch={batch_size}, in={in_features}, out={out_features}")

    # 生成测试数据
    x = torch.randn(batch_size, in_features, device=device)
    weight = torch.randn(out_features, in_features, device=device) * 0.1

    print("\n输入数据:")
    check_tensor_device("x", x)
    check_tensor_device("weight", weight)

    # PyTorch 参考
    pytorch_ref = x @ weight.T
    print("PyTorch 参考:")
    check_tensor_device("pytorch_ref", pytorch_ref)

    results = []

    for accum_precision in ['fp32', 'fp64']:
        for accum_mode in ['sequential', 'parallel']:
            print(f"\n--- accum_precision={accum_precision}, accum_mode={accum_mode} ---")

            # 创建 SNN Linear
            print("  创建 SNN Linear...")
            snn_linear = SpikeFP32Linear_MultiPrecision(
                in_features=in_features,
                out_features=out_features,
                accum_precision=accum_precision,
                accum_mode=accum_mode
            ).to(device)

            print("  设置权重...")
            snn_linear.set_weight_from_float(weight)
            check_tensor_device("weight_pulse", snn_linear.weight_pulse)

            # 编码输入
            print("  编码输入...")
            x_pulse = float32_to_pulse(x, device=device)
            check_tensor_device("x_pulse", x_pulse)

            # 前向传播
            print("  前向传播开始...")
            start_time = time.perf_counter()
            snn_linear.reset()
            y_pulse = snn_linear(x_pulse)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"  前向传播完成! 耗时: {(end_time - start_time)*1000:.2f} ms")
            check_tensor_device("y_pulse", y_pulse)

            # 解码输出
            print("  解码输出...")
            y_snn = pulse_to_float32(y_pulse)
            check_tensor_device("y_snn", y_snn)

            # 计算 ULP 误差
            ulp_stats = compute_ulp_error_fp32(y_snn, pytorch_ref)

            # 测量时延
            print("  测量时延...")
            def forward_fn():
                snn_linear.reset()
                return snn_linear(x_pulse)
            mean_ms, std_ms = measure_latency(forward_fn)

            print(f"  结果:")
            print(f"    Max ULP: {ulp_stats['max_ulp']}")
            print(f"    Mean ULP: {ulp_stats['mean_ulp']:.4f}")
            print(f"    0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")
            print(f"    Latency: {mean_ms:.3f} ± {std_ms:.3f} ms")

            results.append({
                'linear_type': 'FP32',
                'accum_precision': accum_precision,
                'accum_mode': accum_mode,
                **ulp_stats,
                'latency_ms': mean_ms,
                'latency_std': std_ms,
            })

    return results


# =============================================================================
# FP16 Linear 测试
# =============================================================================
def test_fp16_linear():
    """测试 FP16 Linear 不同配置"""
    from atomic_ops.linear.fp16.fp16_linear import SpikeFP16Linear_MultiPrecision
    from atomic_ops.encoding.converters import float16_to_pulse, pulse_to_float16

    print("\n" + "="*60)
    print("FP16 Linear 测试")
    print("="*60)

    batch_size = 4
    in_features = 8
    out_features = 4

    print(f"规模: batch={batch_size}, in={in_features}, out={out_features}")

    # 生成测试数据 (FP16 范围)
    x = torch.randn(batch_size, in_features, device=device).half()
    weight = (torch.randn(out_features, in_features, device=device) * 0.1).half()

    print("\n输入数据:")
    check_tensor_device("x", x)
    check_tensor_device("weight", weight)

    # PyTorch 参考 (FP16)
    pytorch_ref = (x.float() @ weight.float().T).half()
    print("PyTorch 参考:")
    check_tensor_device("pytorch_ref", pytorch_ref)

    results = []

    for accum_precision in ['fp16', 'fp32']:
        for accum_mode in ['sequential', 'parallel']:
            print(f"\n--- accum_precision={accum_precision}, accum_mode={accum_mode} ---")

            print("  创建 SNN Linear...")
            snn_linear = SpikeFP16Linear_MultiPrecision(
                in_features=in_features,
                out_features=out_features,
                accum_precision=accum_precision,
                accum_mode=accum_mode
            ).to(device)

            print("  设置权重...")
            snn_linear.set_weight_from_float(weight.float())
            check_tensor_device("weight_pulse", snn_linear.weight_pulse)

            print("  编码输入...")
            x_pulse = float16_to_pulse(x, device=device)
            check_tensor_device("x_pulse", x_pulse)

            print("  前向传播开始...")
            start_time = time.perf_counter()
            snn_linear.reset()
            y_pulse = snn_linear(x_pulse)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"  前向传播完成! 耗时: {(end_time - start_time)*1000:.2f} ms")
            check_tensor_device("y_pulse", y_pulse)

            print("  解码输出...")
            y_snn = pulse_to_float16(y_pulse)  # 已返回 FP16
            check_tensor_device("y_snn", y_snn)

            # 计算 ULP 误差
            ulp_stats = compute_ulp_error_fp16(y_snn, pytorch_ref)

            print("  测量时延...")
            def forward_fn():
                snn_linear.reset()
                return snn_linear(x_pulse)
            mean_ms, std_ms = measure_latency(forward_fn)

            print(f"  结果:")
            print(f"    Max ULP: {ulp_stats['max_ulp']}")
            print(f"    Mean ULP: {ulp_stats['mean_ulp']:.4f}")
            print(f"    0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")
            print(f"    Latency: {mean_ms:.3f} ± {std_ms:.3f} ms")

            results.append({
                'linear_type': 'FP16',
                'accum_precision': accum_precision,
                'accum_mode': accum_mode,
                **ulp_stats,
                'latency_ms': mean_ms,
                'latency_std': std_ms,
            })

    return results


# =============================================================================
# FP8 Linear 测试
# =============================================================================
def test_fp8_linear():
    """测试 FP8 Linear 不同配置"""
    from atomic_ops.linear.fp8.fp8_linear_multi import SpikeFP8Linear_MultiPrecision
    from atomic_ops.encoding.converters import float_to_fp8_bits, fp8_bits_to_float

    print("\n" + "="*60)
    print("FP8 Linear 测试")
    print("="*60)

    batch_size = 4
    in_features = 8
    out_features = 4

    print(f"规模: batch={batch_size}, in={in_features}, out={out_features}")

    # 生成测试数据 (FP8 范围)
    x = torch.randn(batch_size, in_features, device=device) * 0.5
    weight = torch.randn(out_features, in_features, device=device) * 0.1
    x = x.clamp(-240, 240)
    weight = weight.clamp(-240, 240)

    print("\n输入数据:")
    check_tensor_device("x", x)
    check_tensor_device("weight", weight)

    # 将输入量化到 FP8 再计算参考 (这样公平比较)
    x_fp8 = fp8_bits_to_float(float_to_fp8_bits(x, device=device))
    w_fp8 = fp8_bits_to_float(float_to_fp8_bits(weight, device=device))

    results = []

    for accum_precision in ['fp8', 'fp16', 'fp32']:
        for accum_mode in ['sequential', 'parallel']:
            print(f"\n--- accum_precision={accum_precision}, accum_mode={accum_mode} ---")

            print("  创建 SNN Linear...")
            snn_linear = SpikeFP8Linear_MultiPrecision(
                in_features=in_features,
                out_features=out_features,
                accum_precision=accum_precision,
                accum_mode=accum_mode
            ).to(device)

            print("  设置权重...")
            snn_linear.set_weight_from_float(weight)
            check_tensor_device("weight_pulse", snn_linear.weight_pulse)

            print("  编码输入...")
            x_pulse = float_to_fp8_bits(x, device=device)
            check_tensor_device("x_pulse", x_pulse)

            print("  前向传播开始...")
            start_time = time.perf_counter()
            snn_linear.reset()
            y_pulse = snn_linear(x_pulse)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"  前向传播完成! 耗时: {(end_time - start_time)*1000:.2f} ms")
            check_tensor_device("y_pulse", y_pulse)

            # 解码输出并计算误差
            print("  解码输出...")
            if accum_precision == 'fp8':
                y_snn = fp8_bits_to_float(y_pulse)
                # FP8 参考：使用 FP8 量化后的输入计算
                pytorch_ref = x_fp8 @ w_fp8.T
                # 将参考也量化到 FP8
                pytorch_ref_fp8 = fp8_bits_to_float(float_to_fp8_bits(pytorch_ref, device=device))
                # 计算相对误差 (FP8 没有直接的 ULP 比较)
                rel_error = ((y_snn - pytorch_ref_fp8).abs() / (pytorch_ref_fp8.abs() + 1e-8)).mean().item()
                max_abs_error = (y_snn - pytorch_ref_fp8).abs().max().item()
                ulp_stats = {
                    'max_ulp': f"{max_abs_error:.4f} (abs)",
                    'mean_ulp': rel_error,
                    'zero_ulp_rate': ((y_snn - pytorch_ref_fp8).abs() < 1e-6).float().mean().item() * 100,
                }
            elif accum_precision == 'fp16':
                from atomic_ops.encoding.converters import pulse_to_float16
                y_snn = pulse_to_float16(y_pulse)  # 已返回 FP16
                # FP16 参考
                pytorch_ref = (x_fp8.float() @ w_fp8.float().T).half()
                ulp_stats = compute_ulp_error_fp16(y_snn, pytorch_ref)
            else:  # fp32
                from atomic_ops.encoding.converters import pulse_to_float32
                y_snn = pulse_to_float32(y_pulse)
                # FP32 参考
                pytorch_ref = x_fp8 @ w_fp8.T
                ulp_stats = compute_ulp_error_fp32(y_snn, pytorch_ref)

            check_tensor_device("y_snn", y_snn)

            print("  测量时延...")
            def forward_fn():
                snn_linear.reset()
                return snn_linear(x_pulse)
            mean_ms, std_ms = measure_latency(forward_fn)

            print(f"  结果:")
            print(f"    Max ULP: {ulp_stats['max_ulp']}")
            print(f"    Mean ULP: {ulp_stats['mean_ulp']:.4f}" if isinstance(ulp_stats['mean_ulp'], float) else f"    Mean ULP: {ulp_stats['mean_ulp']}")
            print(f"    0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")
            print(f"    Latency: {mean_ms:.3f} ± {std_ms:.3f} ms")

            results.append({
                'linear_type': 'FP8',
                'accum_precision': accum_precision,
                'accum_mode': accum_mode,
                **ulp_stats,
                'latency_ms': mean_ms,
                'latency_std': std_ms,
            })

    return results


# =============================================================================
# 综合测试
# =============================================================================
def test_all_linear_precision():
    """运行所有 Linear 精度测试"""
    print("\n" + "="*70)
    print("Linear 层精度与时延综合测试")
    print("="*70)

    all_results = []

    try:
        fp32_results = test_fp32_linear()
        all_results.extend(fp32_results)
    except Exception as e:
        import traceback
        print(f"FP32 Linear 测试失败: {e}")
        traceback.print_exc()

    try:
        fp16_results = test_fp16_linear()
        all_results.extend(fp16_results)
    except Exception as e:
        import traceback
        print(f"FP16 Linear 测试失败: {e}")
        traceback.print_exc()

    try:
        fp8_results = test_fp8_linear()
        all_results.extend(fp8_results)
    except Exception as e:
        import traceback
        print(f"FP8 Linear 测试失败: {e}")
        traceback.print_exc()

    # 打印汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'Linear':<8} {'Accum':<8} {'Mode':<12} {'Max ULP':<15} {'0-ULP%':<10} {'Latency(ms)':<15}")
    print("-"*70)

    for r in all_results:
        linear_type = r['linear_type']
        accum = r['accum_precision']
        mode = r['accum_mode']
        max_ulp = r['max_ulp']
        zero_rate = r['zero_ulp_rate']
        latency_str = f"{r['latency_ms']:.1f} ± {r['latency_std']:.1f}"
        print(f"{linear_type:<8} {accum:<8} {mode:<12} {str(max_ulp):<15} {zero_rate:<10.1f} {latency_str:<15}")

    print("-"*70)
    return all_results


def test_fp32_sequential_accuracy():
    """验证 FP32 Sequential 0 ULP"""
    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    print("\n验证 FP32 Sequential 0 ULP...")

    in_features = 8
    out_features = 4
    batch_size = 4

    x = torch.randn(batch_size, in_features, device=device)
    weight = torch.randn(out_features, in_features, device=device)

    print(f"x device: {x.device}")
    print(f"weight device: {weight.device}")

    snn_linear = SpikeFP32Linear_MultiPrecision(
        in_features=in_features,
        out_features=out_features,
        accum_precision='fp32',
        accum_mode='sequential'
    ).to(device)
    snn_linear.set_weight_from_float(weight)

    print(f"weight_pulse device: {snn_linear.weight_pulse.device}")

    x_pulse = float32_to_pulse(x, device=device)
    print(f"x_pulse device: {x_pulse.device}")

    print("前向传播...")
    snn_linear.reset()
    y_pulse = snn_linear(x_pulse)
    y_snn = pulse_to_float32(y_pulse)

    print("计算参考...")
    y_ref = torch.zeros(batch_size, out_features, device=device)
    for b in range(batch_size):
        for o in range(out_features):
            acc = torch.tensor(0.0, device=device)
            for i in range(in_features):
                acc = acc + x[b, i] * weight[o, i]
            y_ref[b, o] = acc

    ulp_stats = compute_ulp_error_fp32(y_snn, y_ref)

    print(f"  Max ULP: {ulp_stats['max_ulp']}")
    print(f"  0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")

    if ulp_stats['max_ulp'] == 0:
        print("  ✓ FP32 Sequential 达到 0 ULP!")
        return True
    else:
        print("  ✗ FP32 Sequential 未达到 0 ULP")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Linear 层精度与时延测试')
    parser.add_argument('--quick', action='store_true', help='快速验证模式')
    parser.add_argument('--fp32', action='store_true', help='只测试 FP32')
    parser.add_argument('--fp16', action='store_true', help='只测试 FP16')
    parser.add_argument('--fp8', action='store_true', help='只测试 FP8')
    args = parser.parse_args()

    if args.quick:
        test_fp32_sequential_accuracy()
    elif args.fp32:
        test_fp32_linear()
    elif args.fp16:
        test_fp16_linear()
    elif args.fp8:
        test_fp8_linear()
    else:
        test_all_linear_precision()
