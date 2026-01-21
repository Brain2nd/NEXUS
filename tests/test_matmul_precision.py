"""
MatMul 精度测试 - FP8/FP16 激活值×激活值矩阵乘法
================================================

测试 SpikeFP8MatMul 和 SpikeFP16MatMul 在不同累加精度下的:
1. ULP 误差 (与 PyTorch 对比)
2. 前向传播时延

遵循 CLAUDE.md:
- Random + Boundary Values 测试
- GPU 优先 (CUDA)

作者: MofNeuroSim Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"测试设备: {device}")


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


# =============================================================================
# FP32 MatMul 测试
# =============================================================================
def test_fp32_matmul():
    """测试 FP32 MatMul (A @ B^T)"""
    from atomic_ops.arithmetic.fp32.fp32_matmul import SpikeFP32MatMulTransposed
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    print("\n" + "="*60)
    print("FP32 MatMul 测试 (A @ B^T)")
    print("="*60)

    M, K, N = 4, 8, 4

    print(f"规模: M={M}, K={K}, N={N}")

    torch.manual_seed(42)
    A = torch.randn(M, K, device=device)
    B = torch.randn(N, K, device=device)

    print(f"A: {A.shape}, dtype={A.dtype}")
    print(f"B: {B.shape}, dtype={B.dtype}")

    # PyTorch 参考
    pytorch_ref = A @ B.T
    print(f"PyTorch ref: {pytorch_ref.shape}")

    results = []

    # FP32 MatMul 只有一种累加精度 (fp32)
    print(f"\n--- FP32 MatMul (现有实现，循环累加) ---")

    matmul = SpikeFP32MatMulTransposed().to(device)

    A_pulse = float32_to_pulse(A, device=device)
    B_pulse = float32_to_pulse(B, device=device)
    print(f"  A_pulse: {A_pulse.shape}")
    print(f"  B_pulse: {B_pulse.shape}")

    print("  前向传播...")
    start = time.perf_counter()
    matmul.reset()
    C_pulse = matmul(A_pulse, B_pulse)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  完成! 耗时: {elapsed:.2f} ms")
    print(f"  C_pulse: {C_pulse.shape}")

    C_snn = pulse_to_float32(C_pulse)
    ulp_stats = compute_ulp_error_fp32(C_snn, pytorch_ref)

    print(f"  结果:")
    print(f"    Max ULP: {ulp_stats['max_ulp']}")
    print(f"    Mean ULP: {ulp_stats['mean_ulp']:.4f}")
    print(f"    0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")

    results.append({
        'type': 'FP32 MatMul',
        'accum': 'fp32',
        'mode': 'sequential',
        **ulp_stats,
        'latency_ms': elapsed,
    })

    return results


# =============================================================================
# FP16 MatMul 测试
# =============================================================================
def test_fp16_matmul():
    """测试 FP16 MatMul (A @ B^T)"""
    from atomic_ops.arithmetic.fp16.fp16_matmul import SpikeFP16MatMul
    from atomic_ops.encoding.converters import float16_to_pulse, pulse_to_float16, pulse_to_float32

    print("\n" + "="*60)
    print("FP16 MatMul 测试 (A @ B^T)")
    print("="*60)

    # 测试规模: 模拟 Attention QK 乘法
    M, K, N = 4, 8, 4  # seq_q=4, head_dim=8, seq_k=4

    print(f"规模: M={M}, K={K}, N={N}")

    # 生成测试数据 (FP16 范围)
    torch.manual_seed(42)
    A = (torch.randn(M, K, device=device) * 0.5).half()
    B = (torch.randn(N, K, device=device) * 0.5).half()

    print(f"A: {A.shape}, dtype={A.dtype}")
    print(f"B: {B.shape}, dtype={B.dtype}")

    # PyTorch 参考: A @ B^T
    pytorch_ref = (A.float() @ B.float().T).half()
    print(f"PyTorch ref: {pytorch_ref.shape}")

    results = []

    for accum_precision in ['fp16', 'fp32']:
        for accum_mode in ['sequential', 'parallel']:
            print(f"\n--- accum_precision={accum_precision}, accum_mode={accum_mode} ---")

            matmul = SpikeFP16MatMul(
                accum_precision=accum_precision,
                accum_mode=accum_mode
            ).to(device)

            # 编码输入
            A_pulse = float16_to_pulse(A, device=device)
            B_pulse = float16_to_pulse(B, device=device)
            print(f"  A_pulse: {A_pulse.shape}")
            print(f"  B_pulse: {B_pulse.shape}")

            # 前向传播
            print("  前向传播...")
            start = time.perf_counter()
            matmul.reset()
            C_pulse = matmul(A_pulse, B_pulse)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  完成! 耗时: {elapsed:.2f} ms")
            print(f"  C_pulse: {C_pulse.shape}")

            # 解码输出
            if accum_precision == 'fp32':
                C_snn = pulse_to_float32(C_pulse)
                # FP32 参考: 使用 FP16 输入的 FP32 精度计算
                pytorch_ref_fp32 = A.float() @ B.float().T
                ulp_stats = compute_ulp_error_fp32(C_snn, pytorch_ref_fp32)
            else:
                C_snn = pulse_to_float16(C_pulse)
                ulp_stats = compute_ulp_error_fp16(C_snn, pytorch_ref)

            print(f"  结果:")
            print(f"    Max ULP: {ulp_stats['max_ulp']}")
            print(f"    Mean ULP: {ulp_stats['mean_ulp']:.4f}")
            print(f"    0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")

            results.append({
                'type': 'FP16 MatMul',
                'accum': accum_precision,
                'mode': accum_mode,
                **ulp_stats,
                'latency_ms': elapsed,
            })

    return results


# =============================================================================
# FP8 MatMul 测试
# =============================================================================
def test_fp8_matmul():
    """测试 FP8 MatMul (A @ B^T)"""
    from atomic_ops.arithmetic.fp8.fp8_matmul import SpikeFP8MatMul
    from atomic_ops.encoding.converters import float_to_fp8_bits, fp8_bits_to_float, pulse_to_float16, pulse_to_float32

    print("\n" + "="*60)
    print("FP8 MatMul 测试 (A @ B^T)")
    print("="*60)

    M, K, N = 4, 8, 4

    print(f"规模: M={M}, K={K}, N={N}")

    # 生成测试数据 (FP8 范围)
    torch.manual_seed(42)
    A = torch.randn(M, K, device=device) * 0.5
    B = torch.randn(N, K, device=device) * 0.5
    A = A.clamp(-240, 240)
    B = B.clamp(-240, 240)

    # 量化到 FP8
    A_fp8 = fp8_bits_to_float(float_to_fp8_bits(A, device=device))
    B_fp8 = fp8_bits_to_float(float_to_fp8_bits(B, device=device))

    print(f"A_fp8: {A_fp8.shape}")
    print(f"B_fp8: {B_fp8.shape}")

    results = []

    for accum_precision in ['fp8', 'fp16', 'fp32']:
        for accum_mode in ['sequential', 'parallel']:
            print(f"\n--- accum_precision={accum_precision}, accum_mode={accum_mode} ---")

            matmul = SpikeFP8MatMul(
                accum_precision=accum_precision,
                accum_mode=accum_mode
            ).to(device)

            # 编码输入
            A_pulse = float_to_fp8_bits(A, device=device)
            B_pulse = float_to_fp8_bits(B, device=device)
            print(f"  A_pulse: {A_pulse.shape}")
            print(f"  B_pulse: {B_pulse.shape}")

            # 前向传播
            print("  前向传播...")
            start = time.perf_counter()
            matmul.reset()
            C_pulse = matmul(A_pulse, B_pulse)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  完成! 耗时: {elapsed:.2f} ms")
            print(f"  C_pulse: {C_pulse.shape}")

            # 解码输出并计算参考
            if accum_precision == 'fp32':
                C_snn = pulse_to_float32(C_pulse)
                pytorch_ref = A_fp8 @ B_fp8.T
                ulp_stats = compute_ulp_error_fp32(C_snn, pytorch_ref)
            elif accum_precision == 'fp16':
                C_snn = pulse_to_float16(C_pulse)
                pytorch_ref = (A_fp8.float() @ B_fp8.float().T).half()
                ulp_stats = compute_ulp_error_fp16(C_snn, pytorch_ref)
            else:
                C_snn = fp8_bits_to_float(C_pulse)
                pytorch_ref = A_fp8 @ B_fp8.T
                pytorch_ref_fp8 = fp8_bits_to_float(float_to_fp8_bits(pytorch_ref, device=device))
                rel_error = ((C_snn - pytorch_ref_fp8).abs() / (pytorch_ref_fp8.abs() + 1e-8)).mean().item()
                max_abs = (C_snn - pytorch_ref_fp8).abs().max().item()
                ulp_stats = {
                    'max_ulp': f"{max_abs:.4f} (abs)",
                    'mean_ulp': rel_error,
                    'zero_ulp_rate': ((C_snn - pytorch_ref_fp8).abs() < 1e-6).float().mean().item() * 100,
                }

            print(f"  结果:")
            print(f"    Max ULP: {ulp_stats['max_ulp']}")
            if isinstance(ulp_stats['mean_ulp'], float):
                print(f"    Mean ULP: {ulp_stats['mean_ulp']:.4f}")
            print(f"    0-ULP Rate: {ulp_stats['zero_ulp_rate']:.2f}%")

            results.append({
                'type': 'FP8 MatMul',
                'accum': accum_precision,
                'mode': accum_mode,
                **ulp_stats,
                'latency_ms': elapsed,
            })

    return results


# =============================================================================
# 综合测试
# =============================================================================
def test_all_matmul():
    """运行所有 MatMul 测试"""
    print("\n" + "="*70)
    print("MatMul 精度与时延综合测试")
    print("="*70)

    all_results = []

    try:
        fp32_results = test_fp32_matmul()
        all_results.extend(fp32_results)
    except Exception as e:
        import traceback
        print(f"FP32 MatMul 测试失败: {e}")
        traceback.print_exc()

    try:
        fp16_results = test_fp16_matmul()
        all_results.extend(fp16_results)
    except Exception as e:
        import traceback
        print(f"FP16 MatMul 测试失败: {e}")
        traceback.print_exc()

    try:
        fp8_results = test_fp8_matmul()
        all_results.extend(fp8_results)
    except Exception as e:
        import traceback
        print(f"FP8 MatMul 测试失败: {e}")
        traceback.print_exc()

    # 打印汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'Type':<15} {'Accum':<8} {'Mode':<12} {'Max ULP':<15} {'0-ULP%':<10} {'Latency(ms)':<12}")
    print("-"*70)

    for r in all_results:
        typ = r['type']
        accum = r['accum']
        mode = r['mode']
        max_ulp = r['max_ulp']
        zero_rate = r['zero_ulp_rate']
        latency = r['latency_ms']
        print(f"{typ:<15} {accum:<8} {mode:<12} {str(max_ulp):<15} {zero_rate:<10.1f} {latency:<12.1f}")

    print("-"*70)
    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MatMul 精度测试')
    parser.add_argument('--fp32', action='store_true', help='只测试 FP32')
    parser.add_argument('--fp16', action='store_true', help='只测试 FP16')
    parser.add_argument('--fp8', action='store_true', help='只测试 FP8')
    args = parser.parse_args()

    if args.fp32:
        test_fp32_matmul()
    elif args.fp16:
        test_fp16_matmul()
    elif args.fp8:
        test_fp8_matmul()
    else:
        test_all_matmul()
