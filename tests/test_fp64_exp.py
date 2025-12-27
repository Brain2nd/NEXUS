"""
FP64 Exp 测试 - 验证 SpikeFP32ExpHighPrecision 的 bit-exactness

目标：FP32 exp 达到 100% 0 ULP (与PyTorch完全一致)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import struct
from tqdm import tqdm

from atomic_ops import SpikeFP32ExpHighPrecision
from atomic_ops.fp32_exp import SpikeFP32Exp


def float32_to_bits(f):
    """FP32 -> 32位整数"""
    return struct.unpack('>I', struct.pack('>f', f))[0]


def bits_to_float32(b):
    """32位整数 -> FP32"""
    return struct.unpack('>f', struct.pack('>I', b))[0]


def float_to_pulse_batch(vals, device):
    """批量转换FP32值到脉冲"""
    vals = np.asarray(vals, dtype=np.float32)
    original_shape = vals.shape
    bits = vals.flatten().view(np.uint32)
    
    n = bits.size
    pulses = np.zeros((n, 32), dtype=np.float32)
    for i in range(32):
        shift = 31 - i
        pulses[:, i] = ((bits >> np.uint32(shift)) & np.uint32(1)).astype(np.float32)
    
    return torch.from_numpy(pulses.reshape(original_shape + (32,))).to(device)


def pulse_to_bits_batch(pulse):
    """批量转换脉冲到32位整数"""
    flat_pulse = pulse.reshape(-1, 32).cpu().numpy() > 0.5
    n = flat_pulse.shape[0]
    
    bits = np.zeros(n, dtype=np.uint32)
    for i in range(32):
        shift = np.uint32(31 - i)
        bits |= (flat_pulse[:, i].astype(np.uint32) << shift)
    
    return bits.reshape(pulse.shape[:-1])


def compute_ulp_error(result_bits, expected_bits, result_float, expected_float):
    """计算ULP误差"""
    is_nan_result = np.isnan(result_float)
    is_nan_expected = np.isnan(expected_float)
    
    is_inf_result = np.isinf(result_float)
    is_inf_expected = np.isinf(expected_float)
    
    nan_match = is_nan_result & is_nan_expected
    inf_match = is_inf_result & is_inf_expected & (np.sign(result_float) == np.sign(expected_float))
    zero_match = (result_float == 0) & (expected_float == 0)
    
    ulp = np.abs(result_bits.astype(np.int64) - expected_bits.astype(np.int64))
    ulp = np.where(nan_match | inf_match | zero_match, 0, ulp)
    
    nan_mismatch = is_nan_result != is_nan_expected
    inf_mismatch = is_inf_result != is_inf_expected
    ulp = np.where(nan_mismatch | inf_mismatch, 2**23, ulp)
    
    return ulp


def test_exp_high_precision(device='cuda', num_samples=100):
    """测试 SpikeFP32ExpHighPrecision"""
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP32ExpHighPrecision (FP64内部精度)")
    print(f"样本数: {num_samples}")
    print(f"{'='*60}")
    
    exp_module = SpikeFP32ExpHighPrecision().to(device)
    exp_module.eval()
    
    # 生成测试数据
    np.random.seed(42)
    test_vals = []
    
    # 关键范围: exp的有效输入区间 [-87, 88]
    test_vals.extend(np.random.uniform(-10, 10, num_samples // 4).tolist())
    test_vals.extend(np.random.uniform(-1, 1, num_samples // 4).tolist())
    test_vals.extend(np.random.uniform(-50, 50, num_samples // 4).tolist())
    
    # 边界值
    test_vals.extend([0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1])
    test_vals.extend([10.0, -10.0, 20.0, -20.0])
    
    # 补齐
    remaining = num_samples - len(test_vals)
    if remaining > 0:
        test_vals.extend(np.random.uniform(-80, 80, remaining).tolist())
    
    test_vals = np.array(test_vals[:num_samples], dtype=np.float32)
    
    # PyTorch参考
    expected = np.exp(test_vals)
    expected_bits = expected.view(np.uint32)
    
    # 转换为脉冲
    x_pulse = float_to_pulse_batch(test_vals, device)
    
    # SNN计算
    print("运行 SNN SpikeFP32ExpHighPrecision...")
    with torch.no_grad():
        result_pulse = exp_module(x_pulse)
    
    # 转换回数值
    result_bits = pulse_to_bits_batch(result_pulse)
    result_float = result_bits.view(np.float32)
    
    # 计算ULP误差
    ulp_errors = compute_ulp_error(result_bits, expected_bits, result_float, expected)
    
    # 统计
    exact_match = np.sum(ulp_errors == 0)
    within_1ulp = np.sum(ulp_errors <= 1)
    within_2ulp = np.sum(ulp_errors <= 2)
    max_ulp = np.max(ulp_errors)
    
    print(f"\n结果统计:")
    print(f"  精确匹配 (0 ULP): {exact_match}/{num_samples} ({100*exact_match/num_samples:.1f}%)")
    print(f"  ≤1 ULP: {within_1ulp}/{num_samples} ({100*within_1ulp/num_samples:.1f}%)")
    print(f"  ≤2 ULP: {within_2ulp}/{num_samples} ({100*within_2ulp/num_samples:.1f}%)")
    print(f"  最大 ULP: {max_ulp}")
    
    # 显示错误样本
    if max_ulp > 0:
        print(f"\n  错误样本 (前10个):")
        error_indices = np.where(ulp_errors > 0)[0][:10]
        for idx in error_indices:
            print(f"    [{idx}] exp({test_vals[idx]:.6f})")
            print(f"         期望: {expected[idx]:.10e} (bits: {expected_bits[idx]:08x})")
            print(f"         结果: {result_float[idx]:.10e} (bits: {result_bits[idx]:08x})")
            print(f"         ULP: {ulp_errors[idx]}")
    
    return exact_match, within_1ulp, within_2ulp, max_ulp


def test_exp_original(device='cuda', num_samples=100):
    """测试原版 SpikeFP32Exp (对比)"""
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP32Exp (原版FP32精度, 对比用)")
    print(f"样本数: {num_samples}")
    print(f"{'='*60}")
    
    exp_module = SpikeFP32Exp().to(device)
    exp_module.eval()
    
    np.random.seed(42)
    test_vals = []
    test_vals.extend(np.random.uniform(-10, 10, num_samples // 2).tolist())
    test_vals.extend(np.random.uniform(-1, 1, num_samples // 4).tolist())
    test_vals.extend([0.0, 1.0, -1.0, 0.5, -0.5])
    
    remaining = num_samples - len(test_vals)
    if remaining > 0:
        test_vals.extend(np.random.uniform(-50, 50, remaining).tolist())
    
    test_vals = np.array(test_vals[:num_samples], dtype=np.float32)
    expected = np.exp(test_vals)
    expected_bits = expected.view(np.uint32)
    
    x_pulse = float_to_pulse_batch(test_vals, device)
    
    print("运行 SNN SpikeFP32Exp...")
    with torch.no_grad():
        result_pulse = exp_module(x_pulse)
    
    result_bits = pulse_to_bits_batch(result_pulse)
    result_float = result_bits.view(np.float32)
    ulp_errors = compute_ulp_error(result_bits, expected_bits, result_float, expected)
    
    exact_match = np.sum(ulp_errors == 0)
    within_1ulp = np.sum(ulp_errors <= 1)
    within_2ulp = np.sum(ulp_errors <= 2)
    max_ulp = np.max(ulp_errors)
    
    print(f"\n结果统计:")
    print(f"  精确匹配 (0 ULP): {exact_match}/{num_samples} ({100*exact_match/num_samples:.1f}%)")
    print(f"  ≤1 ULP: {within_1ulp}/{num_samples} ({100*within_1ulp/num_samples:.1f}%)")
    print(f"  ≤2 ULP: {within_2ulp}/{num_samples} ({100*within_2ulp/num_samples:.1f}%)")
    print(f"  最大 ULP: {max_ulp}")
    
    return exact_match, within_1ulp, within_2ulp, max_ulp


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1000样本全面测试
    num_samples = 1000
    
    # 高精度版
    hp_exact, hp_1ulp, hp_2ulp, hp_max = test_exp_high_precision(device, num_samples)
    
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"高精度 SpikeFP32ExpHP:   {hp_exact}/{num_samples} 精确, max={hp_max}ULP")
    
    if hp_exact == num_samples:
        print("\n✅ SpikeFP32ExpHighPrecision 达到 100% bit-exact!")
    else:
        print(f"\n⚠️ 仍有 {num_samples - hp_exact} 个样本存在误差")


