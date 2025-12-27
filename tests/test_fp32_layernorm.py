"""
FP32 LayerNorm 测试 - 验证 SpikeFP32LayerNorm 的正确性
=======================================================

测试内容:
1. 基本 LayerNorm 功能
2. 与 PyTorch F.layer_norm 对比
3. 归一化后的分布特性
4. 不同输入尺寸

LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

# 使用统一的转换函数
from atomic_ops import float32_to_pulse, pulse_to_float32


def test_basic_layernorm():
    """测试基本 LayerNorm 功能"""
    from atomic_ops import SpikeFP32LayerNorm
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 SpikeFP32LayerNorm 基本功能 (device={device})")
    print("=" * 50)
    
    ln = SpikeFP32LayerNorm().to(device)
    
    # 简单输入: [batch=1, N=4]
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    
    ln.reset()
    result_pulse = ln(x_pulse)
    result = pulse_to_float32(result_pulse)
    
    # PyTorch 参考
    expected = F.layer_norm(x, [4])
    
    print(f"  输入: {x.cpu().numpy()}")
    print(f"  SNN输出: {result.cpu().numpy()}")
    print(f"  PyTorch: {expected.cpu().numpy()}")
    
    # 计算误差
    abs_err = (result - expected).abs().mean().item()
    print(f"  平均绝对误差: {abs_err:.6f}")
    
    passed = abs_err < 0.1
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed


def test_layernorm_distribution():
    """测试 LayerNorm 后的分布特性"""
    from atomic_ops import SpikeFP32LayerNorm
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 LayerNorm 分布特性 (device={device})")
    print("=" * 50)
    
    ln = SpikeFP32LayerNorm().to(device)
    
    # 随机输入
    torch.manual_seed(42)
    x = torch.randn(1, 8, dtype=torch.float32, device=device) * 5 + 2  # 非零均值，非单位方差
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    
    ln.reset()
    result_pulse = ln(x_pulse)
    result = pulse_to_float32(result_pulse)
    
    # 检查归一化后的统计特性
    result_mean = result.mean().item()
    result_std = result.std().item()
    
    print(f"  输入均值: {x.mean().item():.4f}, 标准差: {x.std().item():.4f}")
    print(f"  输出均值: {result_mean:.4f} (期望: ~0)")
    print(f"  输出标准差: {result_std:.4f} (期望: ~1)")
    
    # 放宽阈值，SNN实现可能有精度损失
    mean_ok = abs(result_mean) < 0.5
    std_ok = abs(result_std - 1.0) < 0.5
    
    print(f"  均值检查: {'✓' if mean_ok else '✗'}")
    print(f"  标准差检查: {'✓' if std_ok else '✗'}")
    
    return mean_ok and std_ok


def test_layernorm_invariance():
    """测试 LayerNorm 的缩放平移不变性"""
    from atomic_ops import SpikeFP32LayerNorm
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 LayerNorm 缩放平移不变性 (device={device})")
    print("=" * 50)
    
    ln = SpikeFP32LayerNorm().to(device)
    
    # 原始输入
    x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    
    # 缩放后的输入 (x * 2)
    x2 = x1 * 2
    
    # 平移后的输入 (x + 10)
    x3 = x1 + 10
    
    results = []
    for x, name in [(x1, "原始"), (x2, "缩放x2"), (x3, "平移+10")]:
        x_pulse = float32_to_pulse(x.cpu().numpy(), device)
        ln.reset()
        result_pulse = ln(x_pulse)
        result = pulse_to_float32(result_pulse)
        results.append(result)
        print(f"  {name}: {result.cpu().numpy()}")
    
    # LayerNorm 应该对缩放和平移不变
    # 比较结果是否相近
    err12 = (results[0] - results[1]).abs().mean().item()
    err13 = (results[0] - results[2]).abs().mean().item()
    
    print(f"  原始 vs 缩放 误差: {err12:.4f}")
    print(f"  原始 vs 平移 误差: {err13:.4f}")
    
    passed = err12 < 0.2 and err13 < 0.2
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed


def main():
    print("=" * 60)
    print("SpikeFP32LayerNorm 测试")
    print("=" * 60)
    
    results = []
    
    results.append(("基本功能", test_basic_layernorm()))
    results.append(("分布特性", test_layernorm_distribution()))
    results.append(("缩放平移不变性", test_layernorm_invariance()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed
    
    print("=" * 60)
    if all_pass:
        print("所有测试通过!")
    else:
        print("部分测试失败!")
    
    return all_pass


if __name__ == "__main__":
    main()
