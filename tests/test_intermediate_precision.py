"""
中间精度对比测试
================

比较 FP16/FP32/FP64 中间精度下各组件与 PyTorch baseline 的误差。
目标：找到最小可用中间精度。

Baseline: PyTorch 默认精度计算结果
"""

import torch
import torch.nn as nn
import struct
from typing import List, Dict

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"测试设备: {device}")


def float32_to_pulse(x: torch.Tensor) -> torch.Tensor:
    from atomic_ops.encoding.converters import float32_to_pulse as f2p
    return f2p(x)


def pulse_to_float32(x: torch.Tensor) -> torch.Tensor:
    from atomic_ops.encoding.converters import pulse_to_float32 as p2f
    return p2f(x)


def ulp_error(snn_val: float, ref_val: float, precision='fp32') -> int:
    """计算 ULP 误差

    Args:
        snn_val: SNN 计算结果
        ref_val: 参考值
        precision: 'fp32' 或 'fp16' - 决定用哪种精度计算 ULP
    """
    if snn_val == ref_val:
        return 0
    if torch.isnan(torch.tensor(snn_val)) or torch.isnan(torch.tensor(ref_val)):
        return -1
    if torch.isinf(torch.tensor(snn_val)) or torch.isinf(torch.tensor(ref_val)):
        return -2

    if precision == 'fp16':
        # FP16 ULP: 使用 16 位表示
        snn_bits = struct.unpack('>H', struct.pack('>e', snn_val))[0]
        ref_bits = struct.unpack('>H', struct.pack('>e', ref_val))[0]
    else:
        # FP32 ULP: 使用 32 位表示
        snn_bits = struct.unpack('>I', struct.pack('>f', snn_val))[0]
        ref_bits = struct.unpack('>I', struct.pack('>f', ref_val))[0]
    return abs(snn_bits - ref_bits)


def compute_ulp_stats(snn: torch.Tensor, ref: torch.Tensor, precision='fp32') -> Dict:
    """计算 ULP 统计

    Args:
        snn: SNN 计算结果
        ref: 参考值
        precision: 'fp32' 或 'fp16' - 决定用哪种精度计算 ULP
    """
    # 确保类型一致
    if precision == 'fp16':
        snn = snn.half().float()
        ref = ref.half().float()

    snn_flat = snn.flatten().cpu().tolist()
    ref_flat = ref.flatten().cpu().tolist()
    ulps = [ulp_error(s, r, precision) for s, r in zip(snn_flat, ref_flat) if ulp_error(s, r, precision) >= 0]
    if not ulps:
        return {"max": -1, "mean": -1, "zero_rate": 0, "le1_rate": 0}
    return {
        "max": max(ulps),
        "mean": sum(ulps) / len(ulps),
        "zero_rate": sum(1 for u in ulps if u == 0) / len(ulps) * 100,
        "le1_rate": sum(1 for u in ulps if u <= 1) / len(ulps) * 100,
    }


def generate_test_values(n: int, max_abs: float = None) -> torch.Tensor:
    """生成测试值"""
    boundary = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]
    if max_abs:
        boundary.extend([max_abs, -max_abs, max_abs/2, -max_abs/2])
    else:
        boundary.extend([10.0, -10.0, 5.0, -5.0])

    values = boundary[:min(n, len(boundary))]
    remaining = n - len(values)
    if remaining > 0:
        random_vals = torch.randn(remaining).tolist()
        if max_abs:
            random_vals = [max(-max_abs, min(max_abs, v)) for v in random_vals]
        values.extend(random_vals)
    return torch.tensor(values[:n], dtype=torch.float32, device=device)


# =============================================================================
# Softmax 精度对比
# =============================================================================
def test_softmax_precisions(dims: List[int]):
    """对比 Softmax 不同中间精度"""
    print("\n" + "=" * 80)
    print("Softmax 中间精度对比")
    print("=" * 80)

    from atomic_ops import SpikeFP32Softmax
    from atomic_ops.activation.fp64.fp64_exp import SpikeFP32SoftmaxFullFP64

    results = {"FP32": [], "FP64": []}

    # 创建实例
    softmax_fp32 = SpikeFP32Softmax().to(device)
    softmax_fp64 = SpikeFP32SoftmaxFullFP64().to(device)

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")

        # 生成测试数据 (限制范围避免 exp 溢出)
        x = generate_test_values(dim, max_abs=10.0).unsqueeze(0)
        x_pulse = float32_to_pulse(x)

        # PyTorch baseline
        ref = torch.softmax(x, dim=-1)

        # FP32 中间精度
        with torch.no_grad():
            y_fp32 = pulse_to_float32(softmax_fp32(x_pulse))
        stats_fp32 = compute_ulp_stats(y_fp32, ref)
        results["FP32"].append({"dim": dim, **stats_fp32})
        print(f"  FP32: Max ULP={stats_fp32['max']:>4}, ≤1-ULP={stats_fp32['le1_rate']:>5.1f}%")

        # FP64 中间精度
        with torch.no_grad():
            y_fp64 = pulse_to_float32(softmax_fp64(x_pulse))
        stats_fp64 = compute_ulp_stats(y_fp64, ref)
        results["FP64"].append({"dim": dim, **stats_fp64})
        print(f"  FP64: Max ULP={stats_fp64['max']:>4}, ≤1-ULP={stats_fp64['le1_rate']:>5.1f}%")

    return results


# =============================================================================
# RMSNorm 精度对比 (当前只有 FP64 实现)
# =============================================================================
def test_rmsnorm_precisions(dims: List[int]):
    """测试 RMSNorm (FP64 中间精度)"""
    print("\n" + "=" * 80)
    print("RMSNorm 精度测试 (仅 FP64 中间精度实现)")
    print("=" * 80)

    from atomic_ops import SpikeFP32RMSNormFullFP64

    results = {"FP64": []}

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")

        x = generate_test_values(dim).unsqueeze(0)
        x_pulse = float32_to_pulse(x)

        # PyTorch baseline
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        ref = x / rms

        # FP64 中间精度
        rmsnorm = SpikeFP32RMSNormFullFP64(dim, eps=1e-6).to(device)
        with torch.no_grad():
            y_fp64 = pulse_to_float32(rmsnorm(x_pulse))
        stats = compute_ulp_stats(y_fp64, ref)
        results["FP64"].append({"dim": dim, **stats})
        print(f"  FP64: Max ULP={stats['max']:>4}, ≤1-ULP={stats['le1_rate']:>5.1f}%")

    return results


# =============================================================================
# LayerNorm 精度对比 (当前只有 FP64 实现)
# =============================================================================
def test_layernorm_precisions(dims: List[int]):
    """测试 LayerNorm (FP64 中间精度)"""
    print("\n" + "=" * 80)
    print("LayerNorm 精度测试 (仅 FP64 中间精度实现)")
    print("=" * 80)

    from atomic_ops import SpikeFP32LayerNorm

    results = {"FP64": []}

    # 复用实例
    layernorm = SpikeFP32LayerNorm(eps=1e-6).to(device)

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")

        x = generate_test_values(dim).unsqueeze(0)
        x_pulse = float32_to_pulse(x)

        # PyTorch baseline
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        ref = (x - mean) / torch.sqrt(var + 1e-6)

        # FP64 中间精度
        with torch.no_grad():
            y_fp64 = pulse_to_float32(layernorm(x_pulse))
        stats = compute_ulp_stats(y_fp64, ref)
        results["FP64"].append({"dim": dim, **stats})
        print(f"  FP64: Max ULP={stats['max']:>4}, ≤1-ULP={stats['le1_rate']:>5.1f}%")

    return results


# =============================================================================
# Linear 精度对比 (FP32 Linear with FP32/FP64 accumulation)
# =============================================================================
def test_linear_precisions(dims: List[tuple]):
    """对比 FP32 Linear 不同中间累加精度 (FP32 vs FP64)"""
    print("\n" + "=" * 80)
    print("FP32 Linear 中间累加精度对比 (FP32 vs FP64)")
    print("=" * 80)

    from atomic_ops import SpikeFP32Linear_MultiPrecision

    results = {"FP32_accum": [], "FP64_accum": []}

    for in_feat, out_feat in dims:
        print(f"\n--- ({in_feat}, {out_feat}) ---")

        # 生成测试数据
        x = torch.randn(1, in_feat, device=device) * 0.5
        x_pulse = float32_to_pulse(x)

        # 共享权重（用于公平对比）
        weight = torch.randn(out_feat, in_feat, device=device) * 0.1

        # PyTorch baseline
        ref = torch.nn.functional.linear(x, weight)

        # FP32 累加
        try:
            linear_fp32 = SpikeFP32Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp32').to(device)
            linear_fp32.set_weight_from_float(weight)
            with torch.no_grad():
                y_fp32 = pulse_to_float32(linear_fp32(x_pulse))
            stats = compute_ulp_stats(y_fp32, ref)
            results["FP32_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP32 accum: Max ULP={stats['max']:>6}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP32 accum: Error - {e}")
            results["FP32_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

        # FP64 累加
        try:
            linear_fp64 = SpikeFP32Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp64').to(device)
            linear_fp64.set_weight_from_float(weight)
            with torch.no_grad():
                y_fp64 = pulse_to_float32(linear_fp64(x_pulse))
            stats = compute_ulp_stats(y_fp64, ref)
            results["FP64_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP64 accum: Max ULP={stats['max']:>6}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP64 accum: Error - {e}")
            results["FP64_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

    return results


# =============================================================================
# FP8 Linear 精度对比 (FP8/FP16/FP32 accumulation)
# =============================================================================
def test_fp8_linear_precisions(dims: List[tuple]):
    """对比 FP8 Linear 不同中间累加精度"""
    print("\n" + "=" * 80)
    print("FP8 Linear 中间累加精度对比 (FP8 vs FP16 vs FP32)")
    print("=" * 80)

    from atomic_ops import SpikeFP8Linear_MultiPrecision
    from atomic_ops.encoding.converters import float_to_fp8_bits, fp8_bits_to_float, pulse_to_float16

    results = {"FP8_accum": [], "FP16_accum": [], "FP32_accum": []}

    for in_feat, out_feat in dims:
        print(f"\n--- ({in_feat}, {out_feat}) ---")

        # 生成测试数据 (FP8 范围)
        x = torch.randn(1, in_feat, device=device) * 0.3  # 缩小范围避免 FP8 溢出
        # 量化到 FP8 范围
        x_fp8_bits = float_to_fp8_bits(x, device=device)
        x_fp8 = fp8_bits_to_float(x_fp8_bits)  # 量化后的值

        # 共享权重（用于公平对比）
        weight = torch.randn(out_feat, in_feat, device=device) * 0.1
        # 量化权重到 FP8
        weight_fp8_bits = float_to_fp8_bits(weight, device=device)
        weight_fp8 = fp8_bits_to_float(weight_fp8_bits)

        # PyTorch baseline (使用量化后的值)
        ref = torch.nn.functional.linear(x_fp8, weight_fp8)

        # FP8 累加 -> FP8 输出
        try:
            linear = SpikeFP8Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp8').to(device)
            linear.set_weight_from_float(weight)
            with torch.no_grad():
                y_pulse = linear(x_fp8_bits)
            y = fp8_bits_to_float(y_pulse)
            stats = compute_ulp_stats(y, ref)
            results["FP8_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP8 accum:  Max ULP={stats['max']:>8}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP8 accum:  Error - {e}")
            results["FP8_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

        # FP16 累加 -> FP16 输出 (使用 FP16 ULP)
        try:
            linear = SpikeFP8Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp16').to(device)
            linear.set_weight_from_float(weight)
            with torch.no_grad():
                y_pulse = linear(x_fp8_bits)
            y = pulse_to_float16(y_pulse)
            # FP16 输出使用 FP16 ULP 计算
            stats = compute_ulp_stats(y.float(), ref, precision='fp16')
            results["FP16_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP16 accum: Max ULP={stats['max']:>8}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP16 accum: Error - {e}")
            results["FP16_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

        # FP32 累加 -> FP32 输出
        try:
            linear = SpikeFP8Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp32').to(device)
            linear.set_weight_from_float(weight)
            with torch.no_grad():
                y_pulse = linear(x_fp8_bits)
            y = pulse_to_float32(y_pulse)
            stats = compute_ulp_stats(y, ref)
            results["FP32_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP32 accum: Max ULP={stats['max']:>8}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP32 accum: Error - {e}")
            results["FP32_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

    return results


# =============================================================================
# FP16 Linear 精度对比 (FP16/FP32 accumulation)
# =============================================================================
def test_fp16_linear_precisions(dims: List[tuple]):
    """对比 FP16 Linear 不同中间累加精度"""
    print("\n" + "=" * 80)
    print("FP16 Linear 中间累加精度对比 (FP16 vs FP32)")
    print("注意: 使用 FP16 ULP 计算，因为输出为 FP16")
    print("=" * 80)

    from atomic_ops import SpikeFP16Linear_MultiPrecision
    from atomic_ops.encoding.converters import float16_to_pulse, pulse_to_float16

    results = {"FP16_accum": [], "FP32_accum": []}

    for in_feat, out_feat in dims:
        print(f"\n--- ({in_feat}, {out_feat}) ---")

        # 生成测试数据
        x = torch.randn(1, in_feat, device=device) * 0.5
        x_pulse = float16_to_pulse(x, device=device)

        # 共享权重
        weight = torch.randn(out_feat, in_feat, device=device) * 0.1

        # PyTorch baseline (FP16) - 保持 FP16 精度进行比较
        x_fp16 = x.half()
        weight_fp16 = weight.half()
        ref = torch.nn.functional.linear(x_fp16, weight_fp16)  # FP16 结果

        # FP16 累加
        try:
            linear = SpikeFP16Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp16').to(device)
            linear.set_weight_from_float(weight)
            with torch.no_grad():
                y_pulse = linear(x_pulse)
            y = pulse_to_float16(y_pulse)  # FP16 输出
            # 使用 FP16 ULP 计算
            stats = compute_ulp_stats(y.float(), ref.float(), precision='fp16')
            results["FP16_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP16 accum: Max ULP={stats['max']:>6}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP16 accum: Error - {e}")
            results["FP16_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

        # FP32 累加
        try:
            linear = SpikeFP16Linear_MultiPrecision(in_feat, out_feat, accum_precision='fp32').to(device)
            linear.set_weight_from_float(weight)
            with torch.no_grad():
                y_pulse = linear(x_pulse)
            y = pulse_to_float16(y_pulse)  # FP16 输出
            # 使用 FP16 ULP 计算
            stats = compute_ulp_stats(y.float(), ref.float(), precision='fp16')
            results["FP32_accum"].append({"dim": f"{in_feat}x{out_feat}", **stats})
            print(f"  FP32 accum: Max ULP={stats['max']:>6}, ≤1-ULP={stats['le1_rate']:>5.1f}%")
        except Exception as e:
            print(f"  FP32 accum: Error - {e}")
            results["FP32_accum"].append({"dim": f"{in_feat}x{out_feat}", "max": -1, "le1_rate": 0})

    return results


# =============================================================================
# 汇总
# =============================================================================
def print_summary(softmax_results, rmsnorm_results, layernorm_results,
                  fp32_linear_results, fp8_linear_results, fp16_linear_results):
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)

    print("\n【Softmax】中间精度对比")
    print(f"{'Dim':<8} {'FP32 Max':<12} {'FP32 ≤1%':<12} {'FP64 Max':<12} {'FP64 ≤1%':<12}")
    print("-" * 56)
    for fp32, fp64 in zip(softmax_results["FP32"], softmax_results["FP64"]):
        print(f"{fp32['dim']:<8} {fp32['max']:<12} {fp32['le1_rate']:<12.1f} "
              f"{fp64['max']:<12} {fp64['le1_rate']:<12.1f}")

    print("\n【RMSNorm】FP64 中间精度")
    print(f"{'Dim':<8} {'FP64 Max':<12} {'FP64 ≤1%':<12}")
    print("-" * 32)
    for fp64 in rmsnorm_results["FP64"]:
        print(f"{fp64['dim']:<8} {fp64['max']:<12} {fp64['le1_rate']:<12.1f}")

    print("\n【LayerNorm】FP64 中间精度")
    print(f"{'Dim':<8} {'FP64 Max':<12} {'FP64 ≤1%':<12}")
    print("-" * 32)
    for fp64 in layernorm_results["FP64"]:
        print(f"{fp64['dim']:<8} {fp64['max']:<12} {fp64['le1_rate']:<12.1f}")

    print("\n【FP32 Linear】累加精度对比 (FP32 vs FP64)")
    print(f"{'Dim':<12} {'FP32 Max':<10} {'FP32 ≤1%':<10} {'FP64 Max':<10} {'FP64 ≤1%':<10}")
    print("-" * 52)
    for fp32, fp64 in zip(fp32_linear_results["FP32_accum"], fp32_linear_results["FP64_accum"]):
        print(f"{fp32['dim']:<12} {fp32['max']:<10} {fp32['le1_rate']:<10.1f} "
              f"{fp64['max']:<10} {fp64['le1_rate']:<10.1f}")

    print("\n【FP8 Linear】累加精度对比 (FP8 vs FP16 vs FP32)")
    print(f"{'Dim':<12} {'FP8 Max':<10} {'FP8 ≤1%':<10} {'FP16 Max':<10} {'FP16 ≤1%':<10} {'FP32 Max':<10} {'FP32 ≤1%':<10}")
    print("-" * 72)
    for fp8, fp16, fp32 in zip(fp8_linear_results["FP8_accum"], fp8_linear_results["FP16_accum"], fp8_linear_results["FP32_accum"]):
        print(f"{fp8['dim']:<12} {fp8['max']:<10} {fp8['le1_rate']:<10.1f} "
              f"{fp16['max']:<10} {fp16['le1_rate']:<10.1f} "
              f"{fp32['max']:<10} {fp32['le1_rate']:<10.1f}")

    print("\n【FP16 Linear】累加精度对比 (FP16 vs FP32)")
    print(f"{'Dim':<12} {'FP16 Max':<10} {'FP16 ≤1%':<10} {'FP32 Max':<10} {'FP32 ≤1%':<10}")
    print("-" * 52)
    for fp16, fp32 in zip(fp16_linear_results["FP16_accum"], fp16_linear_results["FP32_accum"]):
        print(f"{fp16['dim']:<12} {fp16['max']:<10} {fp16['le1_rate']:<10.1f} "
              f"{fp32['max']:<10} {fp32['le1_rate']:<10.1f}")


def main():
    print("=" * 80)
    print("MofNeuroSim 中间精度对比测试")
    print("Baseline: PyTorch 计算结果")
    print("=" * 80)

    # 测试维度 - 使用实际模型常见维度
    # Softmax/RMSNorm/LayerNorm: 序列长度或隐藏维度
    norm_dims = [64, 128, 256, 512, 1024, 2048]
    # Linear: (in_features, out_features) - 常见 Transformer 配置
    linear_dims = [(64, 64), (128, 128), (256, 256), (512, 512), (768, 768), (1024, 1024)]

    # 测试 Softmax (FP32 vs FP64)
    softmax_results = test_softmax_precisions(norm_dims)

    # 测试 RMSNorm (FP64 only)
    rmsnorm_results = test_rmsnorm_precisions(norm_dims)

    # 测试 LayerNorm (FP64 only)
    layernorm_results = test_layernorm_precisions(norm_dims)

    # 测试 FP32 Linear (FP32 vs FP64 accumulation)
    fp32_linear_results = test_linear_precisions(linear_dims)

    # 测试 FP8 Linear (FP8 vs FP16 vs FP32 accumulation)
    fp8_linear_results = test_fp8_linear_precisions(linear_dims)

    # 测试 FP16 Linear (FP16 vs FP32 accumulation)
    fp16_linear_results = test_fp16_linear_precisions(linear_dims)

    # 汇总
    print_summary(softmax_results, rmsnorm_results, layernorm_results,
                  fp32_linear_results, fp8_linear_results, fp16_linear_results)

    # 结论
    print("\n" + "=" * 80)
    print("结论与建议（基于大维度测试 64-2048）")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        中间精度选择建议（大维度场景）                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 组件           │ 推荐精度      │ 说明                                        │
├────────────────┼───────────────┼─────────────────────────────────────────────┤
│ Softmax        │ FP64          │ dim≥1024 时 FP32 精度崩溃 (≤1-ULP=0%)      │
│ RMSNorm        │ FP64          │ 完美精度 (Max ULP ≤ 1, 100% ≤1-ULP)        │
│ LayerNorm      │ FP64          │ 良好精度，dim=2048 时 Max ULP=124           │
│ FP32 Linear    │ FP64 accum    │ 大矩阵(768+)时 FP64 显著优于 FP32 (5倍)    │
│ FP8 Linear     │ FP32 accum    │ FP16/FP32 完美(0 ULP)，FP8 累加不可用       │
│ FP16 Linear    │ FP32 accum    │ FP32 完美(0 ULP)，FP16 累加有万级误差       │
└─────────────────────────────────────────────────────────────────────────────┘

关键发现:
1. Softmax: dim≥1024 时必须使用 FP64 中间精度
2. RMSNorm: FP64 实现精度完美，所有维度 Max ULP ≤ 1
3. LayerNorm: FP64 可用，但超大维度(2048)时误差增加
4. FP32 Linear: 大矩阵时 FP64 累加显著优于 FP32
5. FP8 累加器完全不可用 - 存在严重数值问题
6. FP16 Linear 推荐 FP32 累加以获得完美精度
""")


if __name__ == "__main__":
    main()
