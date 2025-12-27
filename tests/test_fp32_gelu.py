"""
FP32 GELU 测试 - 验证 SpikeFP32GELU 的正确性
=============================================

测试内容:
1. 基本 GELU 功能
2. 与 PyTorch F.gelu 对比
3. 特殊值处理
4. 精度验证

GELU(x) ≈ x * sigmoid(1.702 * x)

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


def test_basic_gelu():
    """测试基本 GELU 功能"""
    from atomic_ops import SpikeFP32GELU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 SpikeFP32GELU 基本功能 (device={device})")
    print("=" * 50)
    
    gelu = SpikeFP32GELU().to(device)
    
    # 测试值
    test_cases = [
        (0.0, "零"),
        (1.0, "正数"),
        (-1.0, "负数"),
        (2.0, "较大正数"),
        (-2.0, "较大负数"),
        (0.5, "小正数"),
        (-0.5, "小负数"),
    ]
    
    all_pass = True
    for val, desc in test_cases:
        x = torch.tensor([val], dtype=torch.float32, device=device)
        x_pulse = float32_to_pulse(x.cpu().numpy(), device)
        
        gelu.reset()
        result_pulse = gelu(x_pulse)
        result = pulse_to_float32(result_pulse).item()
        
        # PyTorch 参考
        expected = F.gelu(x, approximate='tanh').item()
        
        # 计算相对误差
        if abs(expected) > 1e-6:
            rel_err = abs(result - expected) / abs(expected)
        else:
            rel_err = abs(result - expected)
        
        status = "✓" if rel_err < 0.01 else "✗"  # 1% 误差阈值
        print(f"  {status} {desc}: GELU({val}) = {result:.6f} (期望: {expected:.6f}, 误差: {rel_err:.2%})")
        
        if rel_err >= 0.01:
            all_pass = False
    
    return all_pass


def test_batch_gelu():
    """测试批量 GELU"""
    from atomic_ops import SpikeFP32GELU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 SpikeFP32GELU 批量处理 (device={device})")
    print("=" * 50)
    
    gelu = SpikeFP32GELU().to(device)
    
    # 随机批量输入
    torch.manual_seed(42)
    batch_size = 10
    x = torch.randn(batch_size, dtype=torch.float32, device=device) * 2
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    
    gelu.reset()
    result_pulse = gelu(x_pulse)
    result = pulse_to_float32(result_pulse)
    
    # PyTorch 参考
    expected = F.gelu(x, approximate='tanh')
    
    # 计算误差
    abs_err = (result - expected).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    
    print(f"  批量大小: {batch_size}")
    print(f"  最大绝对误差: {max_err:.6f}")
    print(f"  平均绝对误差: {mean_err:.6f}")
    
    # 计算相对误差
    rel_err = (abs_err / (expected.abs() + 1e-8)).mean().item()
    print(f"  平均相对误差: {rel_err:.2%}")
    
    passed = rel_err < 0.05  # 5% 阈值
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}")
    
    return passed


def test_gelu_properties():
    """测试 GELU 的数学性质"""
    from atomic_ops import SpikeFP32GELU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 GELU 数学性质 (device={device})")
    print("=" * 50)
    
    gelu = SpikeFP32GELU().to(device)
    
    all_pass = True
    
    # 性质1: GELU(0) ≈ 0
    x = torch.tensor([0.0], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    gelu.reset()
    result = pulse_to_float32(gelu(x_pulse)).item()
    prop1 = abs(result) < 0.01
    print(f"  {'✓' if prop1 else '✗'} GELU(0) ≈ 0: {result:.6f}")
    all_pass = all_pass and prop1
    
    # 性质2: GELU(x) ≈ x for large positive x
    x = torch.tensor([5.0], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    gelu.reset()
    result = pulse_to_float32(gelu(x_pulse)).item()
    prop2 = abs(result - 5.0) < 0.1
    print(f"  {'✓' if prop2 else '✗'} GELU(5) ≈ 5: {result:.6f}")
    all_pass = all_pass and prop2
    
    # 性质3: GELU(x) ≈ 0 for large negative x
    x = torch.tensor([-5.0], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    gelu.reset()
    result = pulse_to_float32(gelu(x_pulse)).item()
    prop3 = abs(result) < 0.1
    print(f"  {'✓' if prop3 else '✗'} GELU(-5) ≈ 0: {result:.6f}")
    all_pass = all_pass and prop3
    
    return all_pass


def main():
    print("=" * 60)
    print("SpikeFP32GELU 测试")
    print("=" * 60)
    
    results = []
    
    results.append(("基本功能", test_basic_gelu()))
    results.append(("批量处理", test_batch_gelu()))
    results.append(("数学性质", test_gelu_properties()))
    
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
