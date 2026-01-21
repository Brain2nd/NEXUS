"""
MatMul 和 Linear 精度测试
========================

对比 SNN 实现与 PyTorch 的 @ 运算符，统计 ULP 误差和数值误差。

Baseline: PyTorch @ 运算符 (GPU 使用 cuBLAS)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"测试设备: {device}")


def compute_ulp_error_fp32(snn_result, pytorch_result):
    """计算 FP32 ULP 误差"""
    snn_bits = snn_result.view(torch.int32)
    ref_bits = pytorch_result.view(torch.int32)
    ulp_diff = (snn_bits - ref_bits).abs()

    abs_err = (snn_result - pytorch_result).abs()
    rel_err = abs_err / (pytorch_result.abs() + 1e-10)

    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
        'max_abs_err': abs_err.max().item(),
        'mean_abs_err': abs_err.mean().item(),
        'max_rel_err': rel_err.max().item(),
        'mean_rel_err': rel_err.mean().item(),
    }


def compute_ulp_error_fp16(snn_result, pytorch_result):
    """计算 FP16 ULP 误差"""
    snn_bits = snn_result.view(torch.int16).int()
    ref_bits = pytorch_result.view(torch.int16).int()
    ulp_diff = (snn_bits - ref_bits).abs()

    abs_err = (snn_result.float() - pytorch_result.float()).abs()
    rel_err = abs_err / (pytorch_result.float().abs() + 1e-10)

    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
        'max_abs_err': abs_err.max().item(),
        'mean_abs_err': abs_err.mean().item(),
        'max_rel_err': rel_err.max().item(),
        'mean_rel_err': rel_err.mean().item(),
    }


def print_result(name, stats):
    """打印测试结果"""
    print(f"  {name}:")
    print(f"    Max ULP: {stats['max_ulp']}, Mean ULP: {stats['mean_ulp']:.2f}, 0-ULP Rate: {stats['zero_ulp_rate']:.1f}%")
    print(f"    Max Abs Err: {stats['max_abs_err']:.2e}, Max Rel Err: {stats['max_rel_err']:.2e}")


# =============================================================================
# FP32 MatMul 测试
# =============================================================================
def test_fp32_matmul():
    """测试 FP32 MatMul vs PyTorch @"""
    from atomic_ops.arithmetic.fp32.fp32_matmul import SpikeFP32MatMulTransposed
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    print("\n" + "="*70)
    print("FP32 MatMul 测试 (SNN vs PyTorch @)")
    print("="*70)

    results = []
    for M, K, N in [(4, 8, 4), (8, 16, 8), (16, 32, 16), (32, 64, 32)]:
        torch.manual_seed(42)
        A = torch.randn(M, K, device=device)
        B = torch.randn(N, K, device=device)

        # PyTorch baseline: A @ B^T
        pytorch_ref = A @ B.T

        # SNN
        matmul = SpikeFP32MatMulTransposed().to(device)
        A_pulse = float32_to_pulse(A, device=device)
        B_pulse = float32_to_pulse(B, device=device)
        matmul.reset()
        C_pulse = matmul(A_pulse, B_pulse)
        C_snn = pulse_to_float32(C_pulse)

        stats = compute_ulp_error_fp32(C_snn, pytorch_ref)
        print_result(f"({M}x{K}x{N})", stats)
        results.append({'size': (M, K, N), **stats})

    return results


# =============================================================================
# FP32 Linear 测试
# =============================================================================
def test_fp32_linear():
    """测试 FP32 Linear vs PyTorch @"""
    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    print("\n" + "="*70)
    print("FP32 Linear 测试 (SNN vs PyTorch @)")
    print("="*70)

    results = []
    for batch, in_f, out_f in [(4, 64, 32), (8, 128, 64), (16, 256, 128)]:
        torch.manual_seed(42)
        x = torch.randn(batch, in_f, device=device)
        W = torch.randn(out_f, in_f, device=device)

        # PyTorch baseline: x @ W^T
        pytorch_ref = x @ W.T

        # SNN
        linear = SpikeFP32Linear(in_f, out_f, accum_precision='fp32').to(device)
        linear.set_weight_from_float(W)
        x_pulse = float32_to_pulse(x, device=device)
        linear.reset()
        y_pulse = linear(x_pulse)
        y_snn = pulse_to_float32(y_pulse)

        stats = compute_ulp_error_fp32(y_snn, pytorch_ref)
        print_result(f"({batch}x{in_f}->{out_f})", stats)
        results.append({'size': (batch, in_f, out_f), **stats})

    return results


# =============================================================================
# FP16 MatMul 测试
# =============================================================================
def test_fp16_matmul():
    """测试 FP16 MatMul vs PyTorch @ (不同累加精度)"""
    from atomic_ops.arithmetic.fp16.fp16_matmul import SpikeFP16MatMul
    from atomic_ops.encoding.converters import float16_to_pulse, pulse_to_float32, pulse_to_float16

    print("\n" + "="*70)
    print("FP16 MatMul 测试 (SNN vs PyTorch @)")
    print("="*70)

    results = []

    # 测试不同累加精度
    for accum_precision in ['fp32', 'fp16']:
        print(f"\n--- 累加精度: {accum_precision} ---")

        for M, K, N in [(4, 8, 4), (8, 16, 8), (16, 32, 16)]:
            torch.manual_seed(42)
            A = (torch.randn(M, K, device=device) * 0.5).half()
            B = (torch.randn(N, K, device=device) * 0.5).half()

            # SNN
            matmul = SpikeFP16MatMul(accum_precision=accum_precision, accum_mode='sequential').to(device)
            A_pulse = float16_to_pulse(A, device=device)
            B_pulse = float16_to_pulse(B, device=device)
            matmul.reset()
            C_pulse = matmul(A_pulse, B_pulse)

            if accum_precision == 'fp32':
                # PyTorch baseline: FP32 累加
                pytorch_ref = A.float() @ B.float().T
                C_snn = pulse_to_float32(C_pulse)
                stats = compute_ulp_error_fp32(C_snn, pytorch_ref)
            else:
                # PyTorch baseline: FP16 累加
                pytorch_ref = (A @ B.T)  # FP16 matmul
                C_snn = pulse_to_float16(C_pulse)
                stats = compute_ulp_error_fp16(C_snn, pytorch_ref)

            print_result(f"({M}x{K}x{N})", stats)
            results.append({'size': (M, K, N), 'accum': accum_precision, **stats})

    return results


# =============================================================================
# FP8 MatMul 测试
# =============================================================================
def test_fp8_matmul():
    """测试 FP8 MatMul vs PyTorch @ (不同累加精度)"""
    from atomic_ops.arithmetic.fp8.fp8_matmul import SpikeFP8MatMul
    from atomic_ops.encoding.converters import float_to_fp8_bits, fp8_bits_to_float, pulse_to_float32, pulse_to_float16

    print("\n" + "="*70)
    print("FP8 MatMul 测试 (SNN vs PyTorch @)")
    print("="*70)

    results = []

    # 测试不同累加精度
    for accum_precision in ['fp32', 'fp16', 'fp8']:
        print(f"\n--- 累加精度: {accum_precision} ---")

        for M, K, N in [(4, 8, 4), (8, 16, 8), (16, 32, 16)]:
            torch.manual_seed(42)
            A = torch.randn(M, K, device=device) * 0.5
            B = torch.randn(N, K, device=device) * 0.5
            A = A.clamp(-240, 240)
            B = B.clamp(-240, 240)

            # 量化到 FP8
            A_fp8 = fp8_bits_to_float(float_to_fp8_bits(A, device=device))
            B_fp8 = fp8_bits_to_float(float_to_fp8_bits(B, device=device))

            # SNN
            matmul = SpikeFP8MatMul(accum_precision=accum_precision, accum_mode='sequential').to(device)
            A_pulse = float_to_fp8_bits(A, device=device)
            B_pulse = float_to_fp8_bits(B, device=device)
            matmul.reset()
            C_pulse = matmul(A_pulse, B_pulse)

            if accum_precision == 'fp32':
                # PyTorch baseline: FP32 累加
                pytorch_ref = A_fp8 @ B_fp8.T
                C_snn = pulse_to_float32(C_pulse)
                stats = compute_ulp_error_fp32(C_snn, pytorch_ref)
            elif accum_precision == 'fp16':
                # PyTorch baseline: FP16 累加
                pytorch_ref = (A_fp8.half() @ B_fp8.half().T)
                C_snn = pulse_to_float16(C_pulse)
                stats = compute_ulp_error_fp16(C_snn, pytorch_ref)
            else:
                # FP8 累加 - 使用相对误差
                pytorch_ref = A_fp8 @ B_fp8.T
                pytorch_ref_fp8 = fp8_bits_to_float(float_to_fp8_bits(pytorch_ref, device=device))
                C_snn = fp8_bits_to_float(C_pulse)
                abs_err = (C_snn - pytorch_ref_fp8).abs()
                rel_err = abs_err / (pytorch_ref_fp8.abs() + 1e-8)
                stats = {
                    'max_ulp': f"{abs_err.max().item():.4f}(abs)",
                    'mean_ulp': rel_err.mean().item(),
                    'zero_ulp_rate': ((abs_err < 1e-6).float().mean().item() * 100),
                    'max_abs_err': abs_err.max().item(),
                    'mean_abs_err': abs_err.mean().item(),
                    'max_rel_err': rel_err.max().item(),
                    'mean_rel_err': rel_err.mean().item(),
                }

            print_result(f"({M}x{K}x{N})", stats)
            results.append({'size': (M, K, N), 'accum': accum_precision, **stats})

    return results


# =============================================================================
# 汇总
# =============================================================================
def test_all():
    """运行所有测试并汇总"""
    print("\n" + "="*70)
    print("MatMul 和 Linear 精度测试汇总")
    print("Baseline: PyTorch @ 运算符")
    print("="*70)

    all_results = {}

    try:
        all_results['fp32_matmul'] = test_fp32_matmul()
    except Exception as e:
        print(f"FP32 MatMul 测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['fp32_linear'] = test_fp32_linear()
    except Exception as e:
        print(f"FP32 Linear 测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['fp16_matmul'] = test_fp16_matmul()
    except Exception as e:
        print(f"FP16 MatMul 测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['fp8_matmul'] = test_fp8_matmul()
    except Exception as e:
        print(f"FP8 MatMul 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 打印汇总表格
    print("\n" + "="*70)
    print("汇总表格")
    print("="*70)
    print(f"{'类型':<20} {'规模':<20} {'Max ULP':<12} {'Max Abs Err':<15} {'Max Rel Err':<15}")
    print("-"*70)

    for test_name, results in all_results.items():
        for r in results:
            size_str = str(r['size'])
            print(f"{test_name:<20} {size_str:<20} {r['max_ulp']:<12} {r['max_abs_err']:<15.2e} {r['max_rel_err']:<15.2e}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MatMul/Linear 精度测试')
    parser.add_argument('--fp32-matmul', action='store_true', help='只测试 FP32 MatMul')
    parser.add_argument('--fp32-linear', action='store_true', help='只测试 FP32 Linear')
    parser.add_argument('--fp16-matmul', action='store_true', help='只测试 FP16 MatMul')
    parser.add_argument('--fp8-matmul', action='store_true', help='只测试 FP8 MatMul')
    args = parser.parse_args()

    if args.fp32_matmul:
        test_fp32_matmul()
    elif args.fp32_linear:
        test_fp32_linear()
    elif args.fp16_matmul:
        test_fp16_matmul()
    elif args.fp8_matmul:
        test_fp8_matmul()
    else:
        test_all()
