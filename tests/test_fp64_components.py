"""
FP64组件测试 - 验证SpikeFP64Adder, SpikeFP64Multiplier, 转换器的bit-exactness

测试方法:
1. 随机生成1000个FP64测试样本
2. 转换为脉冲格式
3. 通过SNN组件计算
4. 转换回FP64并与PyTorch参考比较
5. 统计ULP误差分布
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import struct
from tqdm import tqdm

# 导入FP64组件和转换函数
from atomic_ops import (
    SpikeFP64Adder, SpikeFP64Multiplier,
    FP32ToFP64Converter, FP64ToFP32Converter,
    float64_to_pulse, pulse_to_float64
)


def float64_to_bits_batch(x):
    """将FP64张量转换为64位整数表示 (批量版本)"""
    x_np = x.detach().cpu().numpy().astype(np.float64)
    return x_np.view(np.uint64)


def bits_to_pulse_batch(bits, device):
    """将64位整数转换为[..., 64]脉冲张量 (批量版本)"""
    flat_bits = bits.ravel()
    n = flat_bits.size
    
    pulses = np.zeros((n, 64), dtype=np.float32)
    for i in range(64):
        shift = 63 - i
        pulses[:, i] = ((flat_bits >> np.uint64(shift)) & np.uint64(1)).astype(np.float32)
    
    return torch.from_numpy(pulses.reshape(bits.shape + (64,))).to(device)


def pulse_to_bits_batch(pulse):
    """将[..., 64]脉冲张量转换为64位整数 (批量版本)"""
    flat_pulse = pulse.reshape(-1, 64).cpu().numpy() > 0.5
    n = flat_pulse.shape[0]
    
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(64):
        shift = np.uint64(63 - i)
        bits |= (flat_pulse[:, i].astype(np.uint64) << shift)
    
    return bits.reshape(pulse.shape[:-1])


def bits_to_float64_batch(bits):
    """将64位整数转换回FP64"""
    return bits.view(np.float64)


def compute_ulp_error(result_bits, expected_bits, result_float, expected_float):
    """计算ULP误差"""
    # 处理特殊值
    is_nan_result = np.isnan(result_float)
    is_nan_expected = np.isnan(expected_float)
    
    is_inf_result = np.isinf(result_float)
    is_inf_expected = np.isinf(expected_float)
    
    # NaN匹配: 都是NaN则认为匹配
    nan_match = is_nan_result & is_nan_expected
    
    # Inf匹配: 都是Inf且符号相同
    inf_match = is_inf_result & is_inf_expected & (np.sign(result_float) == np.sign(expected_float))
    
    # 零匹配: 都是零 (忽略符号)
    zero_match = (result_float == 0) & (expected_float == 0)
    
    # 对于normal数值, 计算ULP差
    ulp = np.abs(result_bits.astype(np.int64) - expected_bits.astype(np.int64))
    
    # 特殊值设为0 ULP
    ulp = np.where(nan_match | inf_match | zero_match, 0, ulp)
    
    # 如果一个是NaN而另一个不是, 或者Inf不匹配, 设为最大值
    nan_mismatch = is_nan_result != is_nan_expected
    inf_mismatch = is_inf_result != is_inf_expected
    ulp = np.where(nan_mismatch | inf_mismatch, 2**52, ulp)
    
    return ulp


def test_fp64_adder(device='cuda', num_samples=1000):
    """测试SpikeFP64Adder"""
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP64Adder - {num_samples}样本")
    print(f"{'='*60}")
    
    adder = SpikeFP64Adder().to(device)
    adder.eval()
    
    # 生成测试数据
    np.random.seed(42)
    
    # 混合不同范围的数据
    a_vals = []
    b_vals = []
    
    # 普通范围
    a_vals.extend(np.random.uniform(-1e10, 1e10, num_samples // 4).tolist())
    b_vals.extend(np.random.uniform(-1e10, 1e10, num_samples // 4).tolist())
    
    # 小数
    a_vals.extend(np.random.uniform(-1e-10, 1e-10, num_samples // 4).tolist())
    b_vals.extend(np.random.uniform(-1e-10, 1e-10, num_samples // 4).tolist())
    
    # 相近的数 (减法测试)
    base = np.random.uniform(-1e5, 1e5, num_samples // 4)
    delta = np.random.uniform(-1e-5, 1e-5, num_samples // 4)
    a_vals.extend(base.tolist())
    b_vals.extend((base + delta).tolist())
    
    # 边界值
    a_vals.extend([0.0, 1.0, -1.0, np.finfo(np.float64).max, np.finfo(np.float64).min])
    b_vals.extend([0.0, 1.0, -1.0, np.finfo(np.float64).max, np.finfo(np.float64).min])
    
    # 补齐到num_samples
    remaining = num_samples - len(a_vals)
    if remaining > 0:
        a_vals.extend(np.random.uniform(-1e100, 1e100, remaining).tolist())
        b_vals.extend(np.random.uniform(-1e100, 1e100, remaining).tolist())
    
    a_vals = np.array(a_vals[:num_samples], dtype=np.float64)
    b_vals = np.array(b_vals[:num_samples], dtype=np.float64)
    
    # PyTorch参考结果
    expected = (a_vals + b_vals).astype(np.float64)
    
    # 转换为脉冲
    a_bits = float64_to_bits_batch(torch.from_numpy(a_vals))
    b_bits = float64_to_bits_batch(torch.from_numpy(b_vals))
    
    a_pulse = bits_to_pulse_batch(a_bits, device)
    b_pulse = bits_to_pulse_batch(b_bits, device)
    
    # SNN计算
    print("运行SNN FP64 Adder...")
    with torch.no_grad():
        result_pulse = adder(a_pulse, b_pulse)
    
    # 转换回数值
    result_bits = pulse_to_bits_batch(result_pulse)
    result_float = bits_to_float64_batch(result_bits)
    expected_bits = float64_to_bits_batch(torch.from_numpy(expected))
    
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
    
    # 显示一些错误样本
    if max_ulp > 0:
        print(f"\n  错误样本 (前5个):")
        error_indices = np.where(ulp_errors > 0)[0][:5]
        for idx in error_indices:
            print(f"    [{idx}] {a_vals[idx]} + {b_vals[idx]}")
            print(f"         期望: {expected[idx]} (bits: {expected_bits[idx]:016x})")
            print(f"         结果: {result_float[idx]} (bits: {result_bits[idx]:016x})")
            print(f"         ULP: {ulp_errors[idx]}")
    
    return exact_match, within_1ulp, within_2ulp, max_ulp


def test_fp64_multiplier(device='cuda', num_samples=1000):
    """测试SpikeFP64Multiplier"""
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP64Multiplier - {num_samples}样本")
    print(f"{'='*60}")
    
    multiplier = SpikeFP64Multiplier().to(device)
    multiplier.eval()
    
    # 生成测试数据
    np.random.seed(43)
    
    a_vals = []
    b_vals = []
    
    # 普通范围
    a_vals.extend(np.random.uniform(-1e5, 1e5, num_samples // 3).tolist())
    b_vals.extend(np.random.uniform(-1e5, 1e5, num_samples // 3).tolist())
    
    # 小数
    a_vals.extend(np.random.uniform(-1e-5, 1e-5, num_samples // 3).tolist())
    b_vals.extend(np.random.uniform(-1e-5, 1e-5, num_samples // 3).tolist())
    
    # 边界值
    a_vals.extend([0.0, 1.0, -1.0, 2.0, 0.5, -0.5])
    b_vals.extend([0.0, 1.0, -1.0, 2.0, 0.5, -0.5])
    
    # 补齐
    remaining = num_samples - len(a_vals)
    if remaining > 0:
        a_vals.extend(np.random.uniform(-1e50, 1e50, remaining).tolist())
        b_vals.extend(np.random.uniform(-1e50, 1e50, remaining).tolist())
    
    a_vals = np.array(a_vals[:num_samples], dtype=np.float64)
    b_vals = np.array(b_vals[:num_samples], dtype=np.float64)
    
    # PyTorch参考结果
    expected = (a_vals * b_vals).astype(np.float64)
    
    # 转换为脉冲
    a_bits = float64_to_bits_batch(torch.from_numpy(a_vals))
    b_bits = float64_to_bits_batch(torch.from_numpy(b_vals))
    
    a_pulse = bits_to_pulse_batch(a_bits, device)
    b_pulse = bits_to_pulse_batch(b_bits, device)
    
    # SNN计算
    print("运行SNN FP64 Multiplier...")
    with torch.no_grad():
        result_pulse = multiplier(a_pulse, b_pulse)
    
    # 转换回数值
    result_bits = pulse_to_bits_batch(result_pulse)
    result_float = bits_to_float64_batch(result_bits)
    expected_bits = float64_to_bits_batch(torch.from_numpy(expected))
    
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
    
    if max_ulp > 0:
        print(f"\n  错误样本 (前5个):")
        error_indices = np.where(ulp_errors > 0)[0][:5]
        for idx in error_indices:
            print(f"    [{idx}] {a_vals[idx]} * {b_vals[idx]}")
            print(f"         期望: {expected[idx]} (bits: {expected_bits[idx]:016x})")
            print(f"         结果: {result_float[idx]} (bits: {result_bits[idx]:016x})")
            print(f"         ULP: {ulp_errors[idx]}")
    
    return exact_match, within_1ulp, within_2ulp, max_ulp


def test_fp32_fp64_converters(device='cuda', num_samples=1000):
    """测试FP32<->FP64转换器"""
    print(f"\n{'='*60}")
    print(f"测试 FP32<->FP64 转换器 - {num_samples}样本")
    print(f"{'='*60}")
    
    fp32_to_fp64 = FP32ToFP64Converter().to(device)
    fp64_to_fp32 = FP64ToFP32Converter().to(device)
    
    # 生成FP32测试数据
    np.random.seed(44)
    fp32_vals = np.random.uniform(-1e10, 1e10, num_samples).astype(np.float32)
    
    # 添加边界值
    fp32_vals[:10] = [0.0, 1.0, -1.0, 0.5, -0.5, 
                      np.finfo(np.float32).max, np.finfo(np.float32).min,
                      np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny, 3.14159]
    
    # 转换为FP32脉冲 (32位)
    fp32_bits = fp32_vals.view(np.uint32)
    fp32_pulse = np.zeros((num_samples, 32), dtype=np.float32)
    for i in range(32):
        shift = 31 - i
        fp32_pulse[:, i] = ((fp32_bits >> np.uint32(shift)) & np.uint32(1)).astype(np.float32)
    fp32_pulse = torch.from_numpy(fp32_pulse).to(device)
    
    # FP32 -> FP64
    print("测试 FP32 -> FP64 转换...")
    with torch.no_grad():
        fp64_pulse = fp32_to_fp64(fp32_pulse)
    
    # 转换回数值验证
    fp64_bits = pulse_to_bits_batch(fp64_pulse)
    fp64_result = bits_to_float64_batch(fp64_bits)
    
    # FP32值转为FP64作为参考
    fp64_expected = fp32_vals.astype(np.float64)
    
    # 比较
    match_count = np.sum(fp64_result == fp64_expected)
    print(f"  FP32->FP64 精确匹配: {match_count}/{num_samples} ({100*match_count/num_samples:.1f}%)")
    
    # 显示一些不匹配的样本
    if match_count < num_samples:
        mismatch_indices = np.where(fp64_result != fp64_expected)[0][:5]
        print(f"  不匹配样本:")
        for idx in mismatch_indices:
            print(f"    [{idx}] FP32={fp32_vals[idx]}, 期望FP64={fp64_expected[idx]}, 结果={fp64_result[idx]}")
    
    # FP64 -> FP32 (生成新的FP64数据)
    print("\n测试 FP64 -> FP32 转换...")
    fp64_vals = np.random.uniform(-1e10, 1e10, num_samples).astype(np.float64)
    fp64_bits_in = float64_to_bits_batch(torch.from_numpy(fp64_vals))
    fp64_pulse_in = bits_to_pulse_batch(fp64_bits_in, device)
    
    with torch.no_grad():
        fp32_pulse_out = fp64_to_fp32(fp64_pulse_in)
    
    # 转换回FP32
    fp32_pulse_np = fp32_pulse_out.cpu().numpy() > 0.5
    fp32_bits_out = np.zeros(num_samples, dtype=np.uint32)
    for i in range(32):
        shift = np.uint32(31 - i)
        fp32_bits_out |= (fp32_pulse_np[:, i].astype(np.uint32) << shift)
    fp32_result = fp32_bits_out.view(np.float32)
    
    # 期望值: FP64转为FP32
    fp32_expected = fp64_vals.astype(np.float32)
    
    # 计算匹配
    match_count_32 = np.sum(fp32_result == fp32_expected)
    print(f"  FP64->FP32 精确匹配: {match_count_32}/{num_samples} ({100*match_count_32/num_samples:.1f}%)")
    
    return match_count, match_count_32


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试转换器
    conv_32to64, conv_64to32 = test_fp32_fp64_converters(device, 100)
    
    # 测试加法器 (先用少量样本)
    add_exact, add_1ulp, add_2ulp, add_max = test_fp64_adder(device, 100)
    
    # 测试乘法器 (先用少量样本)
    mul_exact, mul_1ulp, mul_2ulp, mul_max = test_fp64_multiplier(device, 100)
    
    print(f"\n{'='*60}")
    print("FP64组件测试总结")
    print(f"{'='*60}")
    print(f"FP32->FP64转换: {conv_32to64}/100 精确")
    print(f"FP64->FP32转换: {conv_64to32}/100 精确")
    print(f"FP64 Adder: {add_exact}/100 精确, {add_1ulp}/100 ≤1ULP, max={add_max}ULP")
    print(f"FP64 Multiplier: {mul_exact}/100 精确, {mul_1ulp}/100 ≤1ULP, max={mul_max}ULP")


