"""
高精度非线性组件测试 - 验证全链路FP64版本的bit-exactness（端到端浮点验证）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import struct

from atomic_ops import (SpikeFP32SigmoidFullFP64, SpikeFP32SiLUFullFP64,
                        SpikeFP32SoftmaxFullFP64)
from atomic_ops.pulse_decoder import PulseFP32Decoder


def float32_to_bits(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]


def bits_to_float32(b):
    return struct.unpack('>f', struct.pack('>I', b))[0]


def float_to_pulse_batch(vals, device):
    vals = np.asarray(vals, dtype=np.float32)
    original_shape = vals.shape
    bits = vals.flatten().view(np.uint32)
    
    n = bits.size
    pulses = np.zeros((n, 32), dtype=np.float32)
    for i in range(32):
        shift = 31 - i
        pulses[:, i] = ((bits >> np.uint32(shift)) & np.uint32(1)).astype(np.float32)
    
    return torch.from_numpy(pulses.reshape(original_shape + (32,))).to(device)


def compute_ulp_error(result_bits, expected_bits, result_float, expected_float):
    is_nan_result = np.isnan(result_float)
    is_nan_expected = np.isnan(expected_float)
    
    nan_match = is_nan_result & is_nan_expected
    zero_match = (result_float == 0) & (expected_float == 0)
    
    ulp = np.abs(result_bits.astype(np.int64) - expected_bits.astype(np.int64))
    ulp = np.where(nan_match | zero_match, 0, ulp)
    
    return ulp




# ==============================================================================
# 测试函数（端到端浮点验证）
# ==============================================================================
def test_sigmoid_hp(device='cuda', num_samples=100):
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP32SigmoidFullFP64 (全链路FP64)")
    print(f"{'='*60}")
    
    sigmoid_mod = SpikeFP32SigmoidFullFP64().to(device).eval()
    decoder = PulseFP32Decoder().to(device)
    
    np.random.seed(42)
    test_vals = np.random.uniform(-5, 5, num_samples).astype(np.float32)
    
    # PyTorch参考
    test_tensor = torch.from_numpy(test_vals).float()
    expected_tensor = torch.sigmoid(test_tensor)
    expected = expected_tensor.numpy()
    expected_bits = expected.view(np.uint32)
    
    x_pulse = float_to_pulse_batch(test_vals, device)
    
    print("运行 SNN Sigmoid...")
    with torch.no_grad():
        result_pulse = sigmoid_mod(x_pulse)
    
    decoder.reset()
    result_float = decoder(result_pulse).cpu().numpy()
    result_bits = result_float.view(np.uint32)
    
    ulp_errors = compute_ulp_error(result_bits, expected_bits, result_float, expected)
    
    exact = np.sum(ulp_errors == 0)
    within_1ulp = np.sum(ulp_errors <= 1)
    within_2ulp = np.sum(ulp_errors <= 2)
    max_ulp = np.max(ulp_errors)
    
    print(f"  精确 (0 ULP): {exact}/{num_samples} ({100*exact/num_samples:.1f}%)")
    print(f"  ≤1 ULP: {within_1ulp}/{num_samples}")
    print(f"  ≤2 ULP: {within_2ulp}/{num_samples}")
    print(f"  最大 ULP: {max_ulp}")
    
    if max_ulp > 2:
        print(f"  前5个误差样本:")
        errors = np.where(ulp_errors > 2)[0][:5]
        for idx in errors:
            print(f"    x={test_vals[idx]:.6f}: 结果={result_float[idx]:.10f}, 期望={expected[idx]:.10f}, ULP={ulp_errors[idx]}")
    
    return exact, max_ulp


def test_silu_hp(device='cuda', num_samples=100):
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP32SiLUFullFP64 (全链路FP64)")
    print(f"{'='*60}")
    
    silu_mod = SpikeFP32SiLUFullFP64().to(device).eval()
    decoder = PulseFP32Decoder().to(device)
    
    np.random.seed(123)
    test_vals = np.random.uniform(-5, 5, num_samples).astype(np.float32)
    
    # PyTorch参考: 使用torch.nn.functional.silu
    test_tensor = torch.from_numpy(test_vals).float()
    expected_tensor = torch.nn.functional.silu(test_tensor)
    expected = expected_tensor.numpy()
    expected_bits = expected.view(np.uint32)
    
    x_pulse = float_to_pulse_batch(test_vals, device)
    
    print("运行 SNN SiLU...")
    with torch.no_grad():
        result_pulse = silu_mod(x_pulse)
    
    decoder.reset()
    result_float = decoder(result_pulse).cpu().numpy()
    result_bits = result_float.view(np.uint32)
    
    ulp_errors = compute_ulp_error(result_bits, expected_bits, result_float, expected)
    
    exact = np.sum(ulp_errors == 0)
    within_1ulp = np.sum(ulp_errors <= 1)
    within_2ulp = np.sum(ulp_errors <= 2)
    max_ulp = np.max(ulp_errors)
    
    print(f"  精确 (0 ULP): {exact}/{num_samples} ({100*exact/num_samples:.1f}%)")
    print(f"  ≤1 ULP: {within_1ulp}/{num_samples}")
    print(f"  ≤2 ULP: {within_2ulp}/{num_samples}")
    print(f"  最大 ULP: {max_ulp}")
    
    return exact, max_ulp


def test_softmax_hp(device='cuda', num_samples=50, seq_len=4):
    print(f"\n{'='*60}")
    print(f"测试 SpikeFP32SoftmaxFullFP64 (N={seq_len}, 全链路FP64)")
    print(f"{'='*60}")
    
    softmax_mod = SpikeFP32SoftmaxFullFP64().to(device).eval()
    decoder = PulseFP32Decoder().to(device)
    
    np.random.seed(456)
    # [num_samples, seq_len] 输入
    test_vals = np.random.uniform(-3, 3, (num_samples, seq_len)).astype(np.float32)
    
    # PyTorch参考
    test_tensor = torch.from_numpy(test_vals).float()
    expected_tensor = torch.nn.functional.softmax(test_tensor, dim=-1)
    expected = expected_tensor.numpy()
    expected_bits = expected.view(np.uint32)
    
    x_pulse = float_to_pulse_batch(test_vals, device)  # [num_samples, seq_len, 32]
    
    print("运行 SNN Softmax...")
    with torch.no_grad():
        result_pulse = softmax_mod(x_pulse)
    
    decoder.reset()
    result_float = decoder(result_pulse).cpu().numpy()
    result_bits = result_float.view(np.uint32)
    
    ulp_errors = compute_ulp_error(result_bits.flatten(), expected_bits.flatten(), 
                                    result_float.flatten(), expected.flatten())
    
    total = num_samples * seq_len
    exact = np.sum(ulp_errors == 0)
    within_1ulp = np.sum(ulp_errors <= 1)
    within_2ulp = np.sum(ulp_errors <= 2)
    max_ulp = np.max(ulp_errors)
    
    print(f"  精确 (0 ULP): {exact}/{total} ({100*exact/total:.1f}%)")
    print(f"  ≤1 ULP: {within_1ulp}/{total}")
    print(f"  ≤2 ULP: {within_2ulp}/{total}")
    print(f"  最大 ULP: {max_ulp}")
    
    return exact, max_ulp, total


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    num_samples = 200
    
    sig_exact, sig_max = test_sigmoid_hp(device, num_samples)
    silu_exact, silu_max = test_silu_hp(device, num_samples)
    softmax_exact, softmax_max, softmax_total = test_softmax_hp(device, 50, 4)
    
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"Sigmoid: {sig_exact}/{num_samples} 精确, max={sig_max}ULP")
    print(f"SiLU:    {silu_exact}/{num_samples} 精确, max={silu_max}ULP")
    print(f"Softmax: {softmax_exact}/{softmax_total} 精确, max={softmax_max}ULP")


