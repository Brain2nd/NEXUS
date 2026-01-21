"""
测试修改后的 SpikeFP32RMSNormFullFP64 (使用顺序累加器代替加法器树)
==================================================================

验证：
1. 小维度 (dim=4) - 快速验证基本正确性
2. 中等维度 (dim=64) - 验证累加精度
3. 大维度 (dim=1024) - Qwen3 hidden_size，验证初始化时间和正确性

Baseline: PyTorch RMSNorm 计算
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import struct
import time

# 使用 atomic_ops 的向量化转换器（GPU 加速）
from atomic_ops import float32_to_pulse, pulse_to_float32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"测试设备: {device}")


def compute_ulp_error_fp32(snn_result, ref_result):
    """计算 FP32 ULP 误差"""
    snn_bits = snn_result.cpu().view(torch.int32)
    ref_bits = ref_result.cpu().view(torch.int32)
    ulp_diff = (snn_bits - ref_bits).abs()
    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
    }


def pytorch_rmsnorm(x, weight, eps=1e-6):
    """PyTorch 参考实现 (float32)"""
    # x: [batch, dim]
    # RMS = sqrt(mean(x^2) + eps)
    mean_sq = (x ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return x / rms * weight


def test_rmsnorm_dim(dim, num_samples=1, verbose=True):
    """测试指定维度的 RMSNorm"""
    from atomic_ops.normalization.fp32.fp32_rmsnorm import SpikeFP32RMSNormFullFP64

    print(f"\n{'='*60}")
    print(f"测试 RMSNorm (dim={dim})")
    print(f"{'='*60}")

    # 1. 创建 SNN RMSNorm
    print(f"创建 SpikeFP32RMSNormFullFP64(dim={dim})...")
    t0 = time.time()
    rmsnorm = SpikeFP32RMSNormFullFP64(dim).to(device)
    init_time = time.time() - t0
    print(f"初始化耗时: {init_time:.2f}s")

    # 2. 生成测试输入 (Random + Boundary 混合)
    torch.manual_seed(42)

    # 随机值
    x_random = torch.randn(num_samples, dim, device=device)

    # 边界值混合 (将部分位置替换为边界值)
    boundary_values = [0.1, -0.1, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]
    for i, bv in enumerate(boundary_values[:min(len(boundary_values), dim)]):
        x_random[0, i] = bv

    x = x_random

    if verbose and dim <= 8:
        print(f"输入: {x[0].tolist()}")

    # 3. PyTorch 参考输出
    weight = rmsnorm.weight.data.to(device)
    ref_output = pytorch_rmsnorm(x, weight)

    if verbose and dim <= 8:
        print(f"PyTorch 参考输出: {ref_output[0].tolist()}")

    # 4. SNN 前向（使用 atomic_ops 的 GPU 加速转换器）
    print("SNN 前向传播...")
    x_pulse = float32_to_pulse(x, device=device)  # [batch, dim, 32] - GPU 向量化

    t0 = time.time()
    rmsnorm.reset()
    out_pulse = rmsnorm(x_pulse)
    forward_time = time.time() - t0
    print(f"前向传播耗时: {forward_time:.2f}s")

    # 5. 解码输出（使用 atomic_ops 的 GPU 加速转换器）
    snn_output = pulse_to_float32(out_pulse)  # GPU 向量化

    if verbose and dim <= 8:
        print(f"SNN 输出: {snn_output[0].tolist()}")

    # 6. 计算 ULP 误差
    ulp_stats = compute_ulp_error_fp32(snn_output, ref_output)

    print(f"\n结果统计:")
    print(f"  Max ULP: {ulp_stats['max_ulp']}")
    print(f"  Mean ULP: {ulp_stats['mean_ulp']:.4f}")
    print(f"  0-ULP Rate: {ulp_stats['zero_ulp_rate']:.1f}%")

    # 7. 逐元素对比 (仅小维度)
    if verbose and dim <= 8:
        print(f"\n逐元素对比:")
        snn_out_cpu = snn_output.cpu()
        ref_out_cpu = ref_output.cpu()
        for i in range(dim):
            snn_val = snn_out_cpu[0, i].item()
            ref_val = ref_out_cpu[0, i].item()
            snn_bits = struct.unpack('>I', struct.pack('>f', snn_val))[0]
            ref_bits = struct.unpack('>I', struct.pack('>f', ref_val))[0]
            ulp = abs(snn_bits - ref_bits)
            status = "✓" if ulp == 0 else f"✗ ULP={ulp}"
            print(f"  [{i}] SNN={snn_val:.6f}, Ref={ref_val:.6f} {status}")

    return {
        'dim': dim,
        'init_time': init_time,
        'forward_time': forward_time,
        **ulp_stats
    }


def test_initialization_only(dim):
    """仅测试初始化时间 (不做前向传播)"""
    from atomic_ops.normalization.fp32.fp32_rmsnorm import SpikeFP32RMSNormFullFP64

    print(f"\n{'='*60}")
    print(f"初始化测试 (dim={dim})")
    print(f"{'='*60}")

    t0 = time.time()
    rmsnorm = SpikeFP32RMSNormFullFP64(dim).to(device)
    init_time = time.time() - t0

    print(f"初始化耗时: {init_time:.2f}s")
    print(f"初始化速度: {'PASS' if init_time < 60 else 'SLOW'} (threshold: 60s)")

    return init_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RMSNorm 顺序累加器测试')
    parser.add_argument('--dim', type=int, default=None, help='仅测试指定维度')
    parser.add_argument('--init-only', action='store_true', help='仅测试初始化时间')
    parser.add_argument('--qwen3', action='store_true', help='测试 Qwen3 维度 (1024)')
    args = parser.parse_args()

    if args.init_only:
        if args.dim:
            test_initialization_only(args.dim)
        else:
            for dim in [4, 64, 256, 512, 1024]:
                test_initialization_only(dim)
    elif args.dim:
        test_rmsnorm_dim(args.dim)
    elif args.qwen3:
        # Qwen3-0.6B 配置: hidden_size=1024
        test_rmsnorm_dim(1024, verbose=False)
    else:
        # 默认: 测试多个维度
        results = []
        for dim in [4, 8, 16, 32, 64]:
            r = test_rmsnorm_dim(dim, verbose=(dim <= 8))
            results.append(r)

        # 汇总表格
        print(f"\n{'='*60}")
        print("汇总表格")
        print(f"{'='*60}")
        print(f"{'Dim':<8} {'Init(s)':<10} {'Fwd(s)':<10} {'Max ULP':<10} {'0-ULP%':<10}")
        print("-"*60)
        for r in results:
            print(f"{r['dim']:<8} {r['init_time']:<10.2f} {r['forward_time']:<10.2f} {r['max_ulp']:<10} {r['zero_ulp_rate']:<10.1f}")
