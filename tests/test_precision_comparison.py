"""
中间精度对比测试
================

比较不同中间精度实现与 PyTorch baseline 的误差。
目标：找到最小中间精度但保证结果可靠。

测试组件：
- RMSNorm: FP64 中间精度 vs FP32 中间精度
- LayerNorm: FP64 中间精度 vs FP32 中间精度
- Softmax: FP32 中间精度

Baseline: PyTorch 计算结果 (默认精度)
"""

import torch
import torch.nn as nn
import struct
import time
from typing import List, Tuple, Dict

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"测试设备: {device}")


def float32_to_pulse(x: torch.Tensor) -> torch.Tensor:
    """FP32 转脉冲"""
    from atomic_ops.encoding.converters import float32_to_pulse as f2p
    return f2p(x)


def pulse_to_float32(x: torch.Tensor) -> torch.Tensor:
    """脉冲转 FP32"""
    from atomic_ops.encoding.converters import pulse_to_float32 as p2f
    return p2f(x)


def ulp_error(snn_val: float, ref_val: float) -> int:
    """计算 ULP 误差"""
    if snn_val == ref_val:
        return 0
    # 处理特殊值
    if torch.isnan(torch.tensor(snn_val)) or torch.isnan(torch.tensor(ref_val)):
        return -1  # NaN
    if torch.isinf(torch.tensor(snn_val)) or torch.isinf(torch.tensor(ref_val)):
        return -2  # Inf

    snn_bits = struct.unpack('>I', struct.pack('>f', snn_val))[0]
    ref_bits = struct.unpack('>I', struct.pack('>f', ref_val))[0]
    return abs(snn_bits - ref_bits)


def compute_ulp_stats(snn: torch.Tensor, ref: torch.Tensor) -> Dict:
    """计算 ULP 统计"""
    snn_flat = snn.flatten().cpu().tolist()
    ref_flat = ref.flatten().cpu().tolist()

    ulps = []
    for s, r in zip(snn_flat, ref_flat):
        u = ulp_error(s, r)
        if u >= 0:
            ulps.append(u)

    if not ulps:
        return {"max": -1, "mean": -1, "zero_rate": 0}

    return {
        "max": max(ulps),
        "mean": sum(ulps) / len(ulps),
        "zero_rate": sum(1 for u in ulps if u == 0) / len(ulps) * 100,
        "le1_rate": sum(1 for u in ulps if u <= 1) / len(ulps) * 100,
    }


def generate_test_values(n: int, include_boundary: bool = True,
                         max_abs: float = None) -> torch.Tensor:
    """生成测试值：随机 + 边界值

    Args:
        n: 值的数量
        include_boundary: 是否包含边界值
        max_abs: 最大绝对值限制 (用于避免 exp 溢出等)
    """
    values = []

    if include_boundary:
        if max_abs is None:
            # 默认边界值 (包含极端值)
            boundary = [
                0.0, -0.0, 1.0, -1.0,
                0.5, -0.5, 2.0, -2.0,
                1e-6, -1e-6, 1e6, -1e6,
                0.1, -0.1, 10.0, -10.0,
            ]
        else:
            # 受限边界值 (避免溢出)
            boundary = [
                0.0, -0.0, 1.0, -1.0,
                0.5, -0.5, 2.0, -2.0,
                1e-6, -1e-6, max_abs, -max_abs,
                0.1, -0.1, max_abs/2, -max_abs/2,
            ]
        values.extend(boundary[:min(n, len(boundary))])

    # 填充随机值
    remaining = n - len(values)
    if remaining > 0:
        random_vals = torch.randn(remaining).tolist()
        if max_abs is not None:
            random_vals = [max(-max_abs, min(max_abs, v)) for v in random_vals]
        values.extend(random_vals)

    return torch.tensor(values[:n], dtype=torch.float32, device=device)


# =============================================================================
# RMSNorm 测试
# =============================================================================
def test_rmsnorm_precision(dims: List[int], batch_size: int = 1):
    """测试 RMSNorm 不同精度"""
    print("\n" + "=" * 70)
    print("RMSNorm 精度测试 (中间精度: FP64)")
    print("=" * 70)

    from atomic_ops import SpikeFP32RMSNormFullFP64

    results = []

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")

        # 生成测试数据
        x = generate_test_values(dim).unsqueeze(0)  # [1, dim]

        # PyTorch baseline
        x_pt = x.clone()
        rms = torch.sqrt(torch.mean(x_pt ** 2, dim=-1, keepdim=True) + 1e-6)
        ref = x_pt / rms

        # SNN (FP64 中间精度)
        rmsnorm = SpikeFP32RMSNormFullFP64(dim, eps=1e-6).to(device)
        x_pulse = float32_to_pulse(x)

        with torch.no_grad():
            y_pulse = rmsnorm(x_pulse)

        y_snn = pulse_to_float32(y_pulse)

        # 统计
        stats = compute_ulp_stats(y_snn, ref)
        results.append({
            "dim": dim,
            "precision": "FP64",
            **stats
        })

        print(f"  Max ULP: {stats['max']}, Mean ULP: {stats['mean']:.4f}, "
              f"0-ULP: {stats['zero_rate']:.1f}%, ≤1-ULP: {stats['le1_rate']:.1f}%")

    return results


# =============================================================================
# LayerNorm 测试
# =============================================================================
def test_layernorm_precision(dims: List[int], batch_size: int = 1):
    """测试 LayerNorm 不同精度"""
    print("\n" + "=" * 70)
    print("LayerNorm 精度测试 (中间精度: FP64)")
    print("=" * 70)

    from atomic_ops import SpikeFP32LayerNorm

    results = []

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")

        # 生成测试数据
        x = generate_test_values(dim).unsqueeze(0)  # [1, dim]

        # PyTorch baseline
        x_pt = x.clone()
        mean = x_pt.mean(dim=-1, keepdim=True)
        var = x_pt.var(dim=-1, unbiased=False, keepdim=True)
        ref = (x_pt - mean) / torch.sqrt(var + 1e-6)

        # SNN (FP64 中间精度)
        layernorm = SpikeFP32LayerNorm(eps=1e-6).to(device)
        x_pulse = float32_to_pulse(x)

        with torch.no_grad():
            y_pulse = layernorm(x_pulse)

        y_snn = pulse_to_float32(y_pulse)

        # 统计
        stats = compute_ulp_stats(y_snn, ref)
        results.append({
            "dim": dim,
            "precision": "FP64",
            **stats
        })

        print(f"  Max ULP: {stats['max']}, Mean ULP: {stats['mean']:.4f}, "
              f"0-ULP: {stats['zero_rate']:.1f}%, ≤1-ULP: {stats['le1_rate']:.1f}%")

    return results


# =============================================================================
# Softmax 测试
# =============================================================================
def test_softmax_precision(dims: List[int], batch_size: int = 1):
    """测试 Softmax 精度"""
    print("\n" + "=" * 70)
    print("Softmax 精度测试 (中间精度: FP32)")
    print("注意: 输入范围限制在 [-10, 10] 避免 exp 溢出")
    print("=" * 70)

    from atomic_ops import SpikeFP32Softmax

    results = []

    for dim in dims:
        print(f"\n--- Dim = {dim} ---")

        # 生成测试数据 (限制范围避免 exp 溢出)
        # exp(88) ≈ 1.6e38 接近 FP32 上限，所以限制在 [-10, 10]
        x = generate_test_values(dim, max_abs=10.0).unsqueeze(0)  # [1, dim]

        # PyTorch baseline
        ref = torch.softmax(x, dim=-1)

        # SNN (FP32 中间精度) - 复用实例，依赖形状自动检测
        if dim == dims[0]:
            softmax = SpikeFP32Softmax().to(device)
        x_pulse = float32_to_pulse(x)

        with torch.no_grad():
            y_pulse = softmax(x_pulse)

        y_snn = pulse_to_float32(y_pulse)

        # 统计
        stats = compute_ulp_stats(y_snn, ref)
        results.append({
            "dim": dim,
            "precision": "FP32",
            **stats
        })

        print(f"  Max ULP: {stats['max']}, Mean ULP: {stats['mean']:.4f}, "
              f"0-ULP: {stats['zero_rate']:.1f}%, ≤1-ULP: {stats['le1_rate']:.1f}%")

    return results


# =============================================================================
# FP64 Multiplier 测试
# =============================================================================
def test_fp64_mul_precision(n_tests: int = 20):
    """测试 FP64 乘法器精度"""
    print("\n" + "=" * 70)
    print("FP64 Multiplier 精度测试")
    print("=" * 70)

    from atomic_ops import SpikeFP64Multiplier
    from atomic_ops.encoding.converters import float64_to_pulse, pulse_to_float64

    mul = SpikeFP64Multiplier().to(device)

    # 测试值
    test_pairs = [
        (1.0, 1.0),
        (2.0, 3.0),
        (0.5, 0.5),
        (-1.0, 1.0),
        (1.5, -2.5),
        (1e10, 1e-10),
        (3.14159, 2.71828),
    ]

    # 添加随机值
    for _ in range(n_tests - len(test_pairs)):
        a = torch.randn(1).item() * 10
        b = torch.randn(1).item() * 10
        test_pairs.append((a, b))

    ulps = []
    for a, b in test_pairs[:n_tests]:
        a_t = torch.tensor([a], dtype=torch.float64, device=device)
        b_t = torch.tensor([b], dtype=torch.float64, device=device)

        ref = (a_t * b_t).item()

        a_pulse = float64_to_pulse(a_t)
        b_pulse = float64_to_pulse(b_t)

        with torch.no_grad():
            c_pulse = mul(a_pulse, b_pulse)

        c_snn = pulse_to_float64(c_pulse).item()

        # FP64 ULP
        if c_snn == ref:
            ulp = 0
        else:
            snn_bits = struct.unpack('>Q', struct.pack('>d', c_snn))[0]
            ref_bits = struct.unpack('>Q', struct.pack('>d', ref))[0]
            ulp = abs(snn_bits - ref_bits)

        ulps.append(ulp)

    print(f"  测试样本数: {len(ulps)}")
    print(f"  Max ULP: {max(ulps)}")
    print(f"  Mean ULP: {sum(ulps)/len(ulps):.4f}")
    print(f"  0-ULP Rate: {sum(1 for u in ulps if u == 0)/len(ulps)*100:.1f}%")

    return ulps


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 70)
    print("MofNeuroSim 中间精度对比测试")
    print("Baseline: PyTorch 默认精度计算结果")
    print("=" * 70)

    # 测试维度
    dims = [4, 8, 16, 32]

    # RMSNorm
    rmsnorm_results = test_rmsnorm_precision(dims)

    # LayerNorm
    layernorm_results = test_layernorm_precision(dims)

    # Softmax
    softmax_results = test_softmax_precision(dims)

    # FP64 Multiplier
    fp64_mul_ulps = test_fp64_mul_precision(20)

    # 汇总
    print("\n" + "=" * 70)
    print("汇总表格")
    print("=" * 70)

    print("\n【RMSNorm (FP64 中间精度)】")
    print(f"{'Dim':<8} {'Max ULP':<10} {'Mean ULP':<12} {'0-ULP%':<10} {'≤1-ULP%':<10}")
    print("-" * 50)
    for r in rmsnorm_results:
        print(f"{r['dim']:<8} {r['max']:<10} {r['mean']:<12.4f} {r['zero_rate']:<10.1f} {r['le1_rate']:<10.1f}")

    print("\n【LayerNorm (FP64 中间精度)】")
    print(f"{'Dim':<8} {'Max ULP':<10} {'Mean ULP':<12} {'0-ULP%':<10} {'≤1-ULP%':<10}")
    print("-" * 50)
    for r in layernorm_results:
        print(f"{r['dim']:<8} {r['max']:<10} {r['mean']:<12.4f} {r['zero_rate']:<10.1f} {r['le1_rate']:<10.1f}")

    print("\n【Softmax (FP32 中间精度)】")
    print(f"{'Dim':<8} {'Max ULP':<10} {'Mean ULP':<12} {'0-ULP%':<10} {'≤1-ULP%':<10}")
    print("-" * 50)
    for r in softmax_results:
        print(f"{r['dim']:<8} {r['max']:<10} {r['mean']:<12.4f} {r['zero_rate']:<10.1f} {r['le1_rate']:<10.1f}")

    print("\n【FP64 Multiplier】")
    print(f"  Max ULP: {max(fp64_mul_ulps)}, 0-ULP Rate: {sum(1 for u in fp64_mul_ulps if u == 0)/len(fp64_mul_ulps)*100:.1f}%")


if __name__ == "__main__":
    main()
