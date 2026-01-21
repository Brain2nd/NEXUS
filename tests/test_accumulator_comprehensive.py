"""
累加器全面测试 - 覆盖不同精度、组件、累加顺序
==================================================

测试维度：
1. 不同精度组件：FP8/FP16/FP32 Linear
2. 不同中间精度：FP32/FP64 内部计算
3. 不同累加顺序：Sequential vs Parallel

作者: MofNeuroSim Project
"""

import torch
import torch.nn as nn
import numpy as np
import struct
from collections import defaultdict

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_ulp_error(snn_val, ref_val, precision='fp32'):
    """计算 ULP 误差"""
    if not np.isfinite(ref_val) or not np.isfinite(snn_val):
        return 0 if (np.isnan(ref_val) and np.isnan(snn_val)) or (ref_val == snn_val) else float('inf')

    if ref_val == snn_val:
        return 0

    if precision == 'fp32':
        ref_bits = struct.unpack('>I', struct.pack('>f', float(ref_val)))[0]
        snn_bits = struct.unpack('>I', struct.pack('>f', float(snn_val)))[0]
    elif precision == 'fp16':
        ref_bits = int(np.float16(ref_val).view(np.uint16))
        snn_bits = int(np.float16(snn_val).view(np.uint16))
    elif precision == 'fp8':
        # FP8 E4M3: 简化处理
        return abs(float(snn_val) - float(ref_val)) / max(abs(float(ref_val)), 1e-10)
    else:
        ref_bits = struct.unpack('>I', struct.pack('>f', float(ref_val)))[0]
        snn_bits = struct.unpack('>I', struct.pack('>f', float(snn_val)))[0]

    return abs(int(ref_bits) - int(snn_bits))


def test_accumulator_modes():
    """测试累加器两种模式的差异"""
    print("\n" + "="*70)
    print("测试 1: 累加器模式对比 (Sequential vs Parallel)")
    print("="*70)

    from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
    from atomic_ops.core.accumulator import SequentialAccumulator, ParallelAccumulator
    from atomic_ops.encoding.converters import float64_to_pulse, pulse_to_float64

    adder_seq = SpikeFP64Adder().to(device)
    adder_par = SpikeFP64Adder().to(device)

    seq_acc = SequentialAccumulator(adder_seq)
    par_acc = ParallelAccumulator(adder_par)

    test_dims = [8, 16, 32, 64, 128, 256]

    results = []
    for dim in test_dims:
        # 生成随机数据
        torch.manual_seed(42)
        data = torch.randn(dim).to(device)

        # PyTorch 参考
        ref_sum = data.sum().item()

        # 转换为脉冲
        data_pulse = float64_to_pulse(data.double(), device=device)  # [dim, 64]

        # Sequential 累加
        adder_seq.reset()
        seq_result_pulse = seq_acc.reduce(data_pulse, dim=-2)
        seq_result = pulse_to_float64(seq_result_pulse).item()

        # Parallel 累加
        adder_par.reset()
        par_result_pulse = par_acc.reduce(data_pulse, dim=-2)
        par_result = pulse_to_float64(par_result_pulse).item()

        # 计算误差
        seq_ulp = compute_ulp_error(seq_result, ref_sum, 'fp32')
        par_ulp = compute_ulp_error(par_result, ref_sum, 'fp32')
        diff_ulp = compute_ulp_error(seq_result, par_result, 'fp32')

        results.append({
            'dim': dim,
            'ref': ref_sum,
            'seq': seq_result,
            'par': par_result,
            'seq_ulp': seq_ulp,
            'par_ulp': par_ulp,
            'diff_ulp': diff_ulp
        })

        print(f"  dim={dim:4d}: ref={ref_sum:12.6f}, seq={seq_result:12.6f}, par={par_result:12.6f}")
        print(f"           seq_ulp={seq_ulp:6.0f}, par_ulp={par_ulp:6.0f}, seq_vs_par_ulp={diff_ulp:6.0f}")

    return results


def test_rmsnorm_accumulator_modes():
    """测试 RMSNorm 在不同累加模式下的表现"""
    print("\n" + "="*70)
    print("测试 2: RMSNorm 累加模式对比")
    print("="*70)

    from atomic_ops.normalization.fp32.fp32_rmsnorm import SpikeFP32RMSNormFullFP64
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    test_dims = [64, 128, 256, 512, 1024]

    results = []
    for dim in test_dims:
        print(f"\n  dim={dim}:")

        # 生成随机数据
        torch.manual_seed(42)
        x = torch.randn(2, dim).to(device)  # batch=2

        # PyTorch 参考
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
        ref = (x / rms).float()

        # 转换为脉冲
        x_pulse = float32_to_pulse(x, device=device)

        # Sequential 模式
        rmsnorm_seq = SpikeFP32RMSNormFullFP64(dim, accumulator_mode='sequential').to(device)
        rmsnorm_seq.weight.data.fill_(1.0)
        rmsnorm_seq.reset()
        seq_pulse = rmsnorm_seq(x_pulse)
        seq_result = pulse_to_float32(seq_pulse)

        # Parallel 模式
        rmsnorm_par = SpikeFP32RMSNormFullFP64(dim, accumulator_mode='parallel').to(device)
        rmsnorm_par.weight.data.fill_(1.0)
        rmsnorm_par.reset()
        par_pulse = rmsnorm_par(x_pulse)
        par_result = pulse_to_float32(par_pulse)

        # 计算误差
        seq_ulps = []
        par_ulps = []
        diff_ulps = []

        for i in range(ref.numel()):
            seq_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), ref.flatten()[i].item()))
            par_ulps.append(compute_ulp_error(par_result.flatten()[i].item(), ref.flatten()[i].item()))
            diff_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), par_result.flatten()[i].item()))

        seq_max_ulp = max(seq_ulps)
        par_max_ulp = max(par_ulps)
        diff_max_ulp = max(diff_ulps)

        results.append({
            'dim': dim,
            'seq_max_ulp': seq_max_ulp,
            'par_max_ulp': par_max_ulp,
            'diff_max_ulp': diff_max_ulp
        })

        print(f"    Sequential: max_ulp={seq_max_ulp:.0f}")
        print(f"    Parallel:   max_ulp={par_max_ulp:.0f}")
        print(f"    Seq vs Par: max_ulp={diff_max_ulp:.0f}")

    return results


def test_layernorm_accumulator_modes():
    """测试 LayerNorm 在不同累加模式下的表现"""
    print("\n" + "="*70)
    print("测试 3: LayerNorm 累加模式对比")
    print("="*70)

    from atomic_ops.normalization.fp32.fp32_layernorm import SpikeFP32LayerNorm
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    test_dims = [64, 128, 256, 512]

    results = []
    for dim in test_dims:
        print(f"\n  dim={dim}:")

        # 生成随机数据
        torch.manual_seed(42)
        x = torch.randn(2, dim).to(device)

        # PyTorch 参考
        ln = nn.LayerNorm(dim, elementwise_affine=False).to(device)
        ref = ln(x)

        # 转换为脉冲
        x_pulse = float32_to_pulse(x, device=device)

        # Sequential 模式
        ln_seq = SpikeFP32LayerNorm(accumulator_mode='sequential').to(device)
        ln_seq.reset()
        seq_pulse = ln_seq(x_pulse)
        seq_result = pulse_to_float32(seq_pulse)

        # Parallel 模式
        ln_par = SpikeFP32LayerNorm(accumulator_mode='parallel').to(device)
        ln_par.reset()
        par_pulse = ln_par(x_pulse)
        par_result = pulse_to_float32(par_pulse)

        # 计算误差
        seq_ulps = []
        par_ulps = []
        diff_ulps = []

        for i in range(ref.numel()):
            seq_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), ref.flatten()[i].item()))
            par_ulps.append(compute_ulp_error(par_result.flatten()[i].item(), ref.flatten()[i].item()))
            diff_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), par_result.flatten()[i].item()))

        seq_max_ulp = max(seq_ulps)
        par_max_ulp = max(par_ulps)
        diff_max_ulp = max(diff_ulps)

        results.append({
            'dim': dim,
            'seq_max_ulp': seq_max_ulp,
            'par_max_ulp': par_max_ulp,
            'diff_max_ulp': diff_max_ulp
        })

        print(f"    Sequential: max_ulp={seq_max_ulp:.0f}")
        print(f"    Parallel:   max_ulp={par_max_ulp:.0f}")
        print(f"    Seq vs Par: max_ulp={diff_max_ulp:.0f}")

    return results


def test_fp32_linear_accumulator_modes():
    """测试 FP32 Linear 在不同累加模式下的表现"""
    print("\n" + "="*70)
    print("测试 4: FP32 Linear 累加模式对比")
    print("="*70)

    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    test_configs = [(64, 64), (128, 128), (256, 256), (512, 512)]

    results = []
    for in_dim, out_dim in test_configs:
        print(f"\n  Linear({in_dim}, {out_dim}):")

        # 生成随机数据和权重
        torch.manual_seed(42)
        x = torch.randn(2, in_dim).to(device)
        weight = torch.randn(out_dim, in_dim).to(device) * 0.1

        # PyTorch 参考
        ref = torch.mm(x, weight.t())

        # 转换为脉冲
        x_pulse = float32_to_pulse(x, device=device)

        # Sequential 模式
        linear_seq = SpikeFP32Linear(in_dim, out_dim, bias=False, accumulator_mode='sequential').to(device)
        linear_seq.set_weight_from_float(weight)
        linear_seq.reset()
        seq_pulse = linear_seq(x_pulse)
        seq_result = pulse_to_float32(seq_pulse)

        # Parallel 模式
        linear_par = SpikeFP32Linear(in_dim, out_dim, bias=False, accumulator_mode='parallel').to(device)
        linear_par.set_weight_from_float(weight)
        linear_par.reset()
        par_pulse = linear_par(x_pulse)
        par_result = pulse_to_float32(par_pulse)

        # 计算误差
        seq_ulps = []
        par_ulps = []
        diff_ulps = []

        for i in range(ref.numel()):
            seq_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), ref.flatten()[i].item()))
            par_ulps.append(compute_ulp_error(par_result.flatten()[i].item(), ref.flatten()[i].item()))
            diff_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), par_result.flatten()[i].item()))

        seq_max_ulp = max(seq_ulps)
        par_max_ulp = max(par_ulps)
        diff_max_ulp = max(diff_ulps)
        seq_mean_ulp = np.mean(seq_ulps)
        par_mean_ulp = np.mean(par_ulps)

        results.append({
            'config': f'{in_dim}x{out_dim}',
            'seq_max_ulp': seq_max_ulp,
            'seq_mean_ulp': seq_mean_ulp,
            'par_max_ulp': par_max_ulp,
            'par_mean_ulp': par_mean_ulp,
            'diff_max_ulp': diff_max_ulp
        })

        print(f"    Sequential: max_ulp={seq_max_ulp:.0f}, mean_ulp={seq_mean_ulp:.1f}")
        print(f"    Parallel:   max_ulp={par_max_ulp:.0f}, mean_ulp={par_mean_ulp:.1f}")
        print(f"    Seq vs Par: max_ulp={diff_max_ulp:.0f}")

    return results


def test_fp16_linear():
    """测试 FP16 Linear"""
    print("\n" + "="*70)
    print("测试 5: FP16 Linear")
    print("="*70)

    from atomic_ops.linear.fp16.fp16_linear import SpikeFP16Linear
    from atomic_ops.encoding.converters import float16_to_pulse, pulse_to_float16

    test_configs = [(64, 64), (128, 128), (256, 256)]

    results = []
    for in_dim, out_dim in test_configs:
        print(f"\n  FP16 Linear({in_dim}, {out_dim}):")

        # 生成随机数据和权重
        torch.manual_seed(42)
        x = torch.randn(2, in_dim).to(device).half()
        weight = (torch.randn(out_dim, in_dim).to(device) * 0.1).half()

        # PyTorch 参考
        ref = torch.mm(x, weight.t())

        # 转换为脉冲
        x_pulse = float16_to_pulse(x, device=device)

        # SNN Linear
        linear = SpikeFP16Linear(in_dim, out_dim, bias=False).to(device)
        linear.set_weight_from_float(weight)
        linear.reset()
        result_pulse = linear(x_pulse)
        result = pulse_to_float16(result_pulse)

        # 计算误差
        ulps = []
        for i in range(ref.numel()):
            ulps.append(compute_ulp_error(result.flatten()[i].item(), ref.flatten()[i].item(), 'fp16'))

        max_ulp = max(ulps)
        mean_ulp = np.mean(ulps)

        results.append({
            'config': f'{in_dim}x{out_dim}',
            'max_ulp': max_ulp,
            'mean_ulp': mean_ulp
        })

        print(f"    max_ulp={max_ulp:.0f}, mean_ulp={mean_ulp:.1f}")

    return results


def test_fp8_linear():
    """测试 FP8 Linear"""
    print("\n" + "="*70)
    print("测试 6: FP8 Linear (多种累加精度)")
    print("="*70)

    from atomic_ops.linear.fp8.fp8_linear_multi import SpikeFP8LinearMultiPrecision
    from atomic_ops.encoding.converters import float_to_fp8_pulse, fp8_pulse_to_float

    test_configs = [(64, 64), (128, 128)]
    accum_precisions = ['fp8', 'fp16', 'fp32']

    results = []
    for in_dim, out_dim in test_configs:
        print(f"\n  FP8 Linear({in_dim}, {out_dim}):")

        # 生成随机数据和权重 (范围限制以适应 FP8)
        torch.manual_seed(42)
        x = (torch.randn(2, in_dim).to(device) * 0.5).clamp(-1.5, 1.5)
        weight = (torch.randn(out_dim, in_dim).to(device) * 0.1).clamp(-0.5, 0.5)

        # PyTorch 参考 (FP32)
        ref = torch.mm(x, weight.t())

        for accum_prec in accum_precisions:
            try:
                # 转换为 FP8 脉冲
                x_pulse = float_to_fp8_pulse(x, device=device)

                # SNN Linear
                linear = SpikeFP8LinearMultiPrecision(
                    in_dim, out_dim, bias=False,
                    accumulation_precision=accum_prec
                ).to(device)
                linear.set_weight_from_float(weight)
                linear.reset()
                result_pulse = linear(x_pulse)
                result = fp8_pulse_to_float(result_pulse)

                # 计算相对误差 (FP8 精度低，用相对误差)
                rel_errors = []
                for i in range(ref.numel()):
                    ref_val = ref.flatten()[i].item()
                    snn_val = result.flatten()[i].item()
                    if abs(ref_val) > 1e-6:
                        rel_errors.append(abs(snn_val - ref_val) / abs(ref_val))
                    else:
                        rel_errors.append(abs(snn_val - ref_val))

                max_rel_err = max(rel_errors) * 100
                mean_rel_err = np.mean(rel_errors) * 100

                results.append({
                    'config': f'{in_dim}x{out_dim}',
                    'accum_prec': accum_prec,
                    'max_rel_err': max_rel_err,
                    'mean_rel_err': mean_rel_err
                })

                print(f"    accum={accum_prec}: max_rel_err={max_rel_err:.2f}%, mean_rel_err={mean_rel_err:.2f}%")
            except Exception as e:
                print(f"    accum={accum_prec}: ERROR - {e}")
                results.append({
                    'config': f'{in_dim}x{out_dim}',
                    'accum_prec': accum_prec,
                    'error': str(e)
                })

    return results


def test_softmax_accumulator_modes():
    """测试 Softmax 在不同累加模式下的表现"""
    print("\n" + "="*70)
    print("测试 7: Softmax 累加模式对比")
    print("="*70)

    from atomic_ops.activation.fp32.fp32_softmax import SpikeFP32SoftmaxFullFP64
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    test_dims = [64, 128, 256, 512]

    results = []
    for dim in test_dims:
        print(f"\n  dim={dim}:")

        # 生成随机数据
        torch.manual_seed(42)
        x = torch.randn(2, dim).to(device)

        # PyTorch 参考
        ref = torch.softmax(x, dim=-1)

        # 转换为脉冲
        x_pulse = float32_to_pulse(x, device=device)

        # Sequential 模式
        softmax_seq = SpikeFP32SoftmaxFullFP64(accumulator_mode='sequential').to(device)
        softmax_seq.reset()
        seq_pulse = softmax_seq(x_pulse)
        seq_result = pulse_to_float32(seq_pulse)

        # Parallel 模式
        softmax_par = SpikeFP32SoftmaxFullFP64(accumulator_mode='parallel').to(device)
        softmax_par.reset()
        par_pulse = softmax_par(x_pulse)
        par_result = pulse_to_float32(par_pulse)

        # 计算误差
        seq_ulps = []
        par_ulps = []
        diff_ulps = []

        for i in range(ref.numel()):
            seq_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), ref.flatten()[i].item()))
            par_ulps.append(compute_ulp_error(par_result.flatten()[i].item(), ref.flatten()[i].item()))
            diff_ulps.append(compute_ulp_error(seq_result.flatten()[i].item(), par_result.flatten()[i].item()))

        seq_max_ulp = max(seq_ulps)
        par_max_ulp = max(par_ulps)
        diff_max_ulp = max(diff_ulps)

        results.append({
            'dim': dim,
            'seq_max_ulp': seq_max_ulp,
            'par_max_ulp': par_max_ulp,
            'diff_max_ulp': diff_max_ulp
        })

        print(f"    Sequential: max_ulp={seq_max_ulp:.0f}")
        print(f"    Parallel:   max_ulp={par_max_ulp:.0f}")
        print(f"    Seq vs Par: max_ulp={diff_max_ulp:.0f}")

    return results


def print_summary(all_results):
    """打印测试汇总"""
    print("\n" + "="*70)
    print("测试汇总")
    print("="*70)

    print("\n1. 累加器模式对比 (FP64 Adder):")
    print("   维度  | Sequential ULP | Parallel ULP | Seq vs Par ULP")
    print("   " + "-"*55)
    for r in all_results.get('accumulator', []):
        print(f"   {r['dim']:4d}  | {r['seq_ulp']:14.0f} | {r['par_ulp']:12.0f} | {r['diff_ulp']:14.0f}")

    print("\n2. RMSNorm 累加模式对比:")
    print("   维度  | Sequential ULP | Parallel ULP | Seq vs Par ULP")
    print("   " + "-"*55)
    for r in all_results.get('rmsnorm', []):
        print(f"   {r['dim']:4d}  | {r['seq_max_ulp']:14.0f} | {r['par_max_ulp']:12.0f} | {r['diff_max_ulp']:14.0f}")

    print("\n3. LayerNorm 累加模式对比:")
    print("   维度  | Sequential ULP | Parallel ULP | Seq vs Par ULP")
    print("   " + "-"*55)
    for r in all_results.get('layernorm', []):
        print(f"   {r['dim']:4d}  | {r['seq_max_ulp']:14.0f} | {r['par_max_ulp']:12.0f} | {r['diff_max_ulp']:14.0f}")

    print("\n4. FP32 Linear 累加模式对比:")
    print("   配置      | Sequential ULP | Parallel ULP | Seq vs Par ULP")
    print("   " + "-"*60)
    for r in all_results.get('fp32_linear', []):
        print(f"   {r['config']:9s} | {r['seq_max_ulp']:14.0f} | {r['par_max_ulp']:12.0f} | {r['diff_max_ulp']:14.0f}")

    print("\n5. FP16 Linear:")
    print("   配置      | Max ULP | Mean ULP")
    print("   " + "-"*35)
    for r in all_results.get('fp16_linear', []):
        print(f"   {r['config']:9s} | {r['max_ulp']:7.0f} | {r['mean_ulp']:8.1f}")

    print("\n6. FP8 Linear (不同累加精度):")
    print("   配置      | 累加精度 | Max Rel Err | Mean Rel Err")
    print("   " + "-"*55)
    for r in all_results.get('fp8_linear', []):
        if 'error' in r:
            print(f"   {r['config']:9s} | {r['accum_prec']:8s} | ERROR: {r['error'][:30]}")
        else:
            print(f"   {r['config']:9s} | {r['accum_prec']:8s} | {r['max_rel_err']:10.2f}% | {r['mean_rel_err']:11.2f}%")

    print("\n7. Softmax 累加模式对比:")
    print("   维度  | Sequential ULP | Parallel ULP | Seq vs Par ULP")
    print("   " + "-"*55)
    for r in all_results.get('softmax', []):
        print(f"   {r['dim']:4d}  | {r['seq_max_ulp']:14.0f} | {r['par_max_ulp']:12.0f} | {r['diff_max_ulp']:14.0f}")


def main():
    print("="*70)
    print("累加器全面测试")
    print("="*70)

    all_results = {}

    # 测试 1: 累加器模式
    try:
        all_results['accumulator'] = test_accumulator_modes()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 测试 2: RMSNorm
    try:
        all_results['rmsnorm'] = test_rmsnorm_accumulator_modes()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 测试 3: LayerNorm
    try:
        all_results['layernorm'] = test_layernorm_accumulator_modes()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 测试 4: FP32 Linear
    try:
        all_results['fp32_linear'] = test_fp32_linear_accumulator_modes()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 测试 5: FP16 Linear
    try:
        all_results['fp16_linear'] = test_fp16_linear()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 测试 6: FP8 Linear
    try:
        all_results['fp8_linear'] = test_fp8_linear()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 测试 7: Softmax
    try:
        all_results['softmax'] = test_softmax_accumulator_modes()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 打印汇总
    print_summary(all_results)

    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == '__main__':
    main()
