"""
FP32 Tanh 测试 - 验证 SpikeFP32Tanh 的正确性
=============================================

测试内容:
1. 基本 Tanh 功能
2. 与 PyTorch torch.tanh 对比
3. 特殊值处理
4. 精度验证

tanh(x) = (e^2x - 1) / (e^2x + 1)

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

# 使用统一的转换函数
from atomic_ops import float32_to_pulse, pulse_to_float32


def test_basic_tanh():
    """测试基本 Tanh 功能"""
    from atomic_ops import SpikeFP32Tanh
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 SpikeFP32Tanh 基本功能 (device={device})")
    print("=" * 50)
    
    tanh = SpikeFP32Tanh().to(device)
    
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
        
        tanh.reset()
        result_pulse = tanh(x_pulse)
        result = pulse_to_float32(result_pulse).item()
        
        # PyTorch 参考
        expected = torch.tanh(x).item()
        
        # 计算相对误差
        if abs(expected) > 1e-6:
            rel_err = abs(result - expected) / abs(expected)
        else:
            rel_err = abs(result - expected)
        
        status = "✓" if rel_err < 0.01 else "✗"  # 1% 误差阈值
        print(f"  {status} {desc}: tanh({val}) = {result:.6f} (期望: {expected:.6f}, 误差: {rel_err:.2%})")
        
        if rel_err >= 0.01:
            all_pass = False
    
    return all_pass


def test_tanh_bounds():
    """测试 Tanh 边界: -1 < tanh(x) < 1"""
    from atomic_ops import SpikeFP32Tanh
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 Tanh 边界 (device={device})")
    print("=" * 50)
    
    tanh = SpikeFP32Tanh().to(device)
    
    # 大正数 -> tanh 接近 1
    x = torch.tensor([10.0], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    tanh.reset()
    result = pulse_to_float32(tanh(x_pulse)).item()
    bound1 = abs(result - 1.0) < 0.01
    print(f"  {'✓' if bound1 else '✗'} tanh(10) ≈ 1: {result:.6f}")
    
    # 大负数 -> tanh 接近 -1
    x = torch.tensor([-10.0], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    tanh.reset()
    result = pulse_to_float32(tanh(x_pulse)).item()
    bound2 = abs(result + 1.0) < 0.01
    print(f"  {'✓' if bound2 else '✗'} tanh(-10) ≈ -1: {result:.6f}")
    
    # tanh(0) = 0
    x = torch.tensor([0.0], dtype=torch.float32, device=device)
    x_pulse = float32_to_pulse(x.cpu().numpy(), device)
    tanh.reset()
    result = pulse_to_float32(tanh(x_pulse)).item()
    bound3 = abs(result) < 0.01
    print(f"  {'✓' if bound3 else '✗'} tanh(0) = 0: {result:.6f}")
    
    return bound1 and bound2 and bound3


def test_tanh_symmetry():
    """测试 Tanh 对称性: tanh(-x) = -tanh(x)"""
    from atomic_ops import SpikeFP32Tanh
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n测试 Tanh 对称性 (device={device})")
    print("=" * 50)
    
    tanh = SpikeFP32Tanh().to(device)
    
    test_vals = [0.5, 1.0, 1.5, 2.0]
    all_pass = True
    
    for val in test_vals:
        # tanh(x)
        x_pos = torch.tensor([val], dtype=torch.float32, device=device)
        x_pos_pulse = float32_to_pulse(x_pos.cpu().numpy(), device)
        tanh.reset()
        result_pos = pulse_to_float32(tanh(x_pos_pulse)).item()
        
        # tanh(-x)
        x_neg = torch.tensor([-val], dtype=torch.float32, device=device)
        x_neg_pulse = float32_to_pulse(x_neg.cpu().numpy(), device)
        tanh.reset()
        result_neg = pulse_to_float32(tanh(x_neg_pulse)).item()
        
        # 检查对称性
        sym_err = abs(result_pos + result_neg)
        is_symmetric = sym_err < 0.01
        
        print(f"  {'✓' if is_symmetric else '✗'} tanh({val}) = {result_pos:.4f}, tanh({-val}) = {result_neg:.4f}, 差: {sym_err:.6f}")
        
        all_pass = all_pass and is_symmetric
    
    return all_pass


def main():
    print("=" * 60)
    print("SpikeFP32Tanh 测试")
    print("=" * 60)
    
    results = []
    
    results.append(("基本功能", test_basic_tanh()))
    results.append(("边界测试", test_tanh_bounds()))
    results.append(("对称性", test_tanh_symmetry()))
    
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
