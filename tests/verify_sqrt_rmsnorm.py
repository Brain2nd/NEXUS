"""
验证 SpikeFP64Sqrt 和 SpikeFP32RMSNormFullFP64
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.fp64_sqrt import SpikeFP64Sqrt
from atomic_ops.fp32_rmsnorm import SpikeFP32RMSNormFullFP64
from atomic_ops.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter

def float64_to_pulse(x):
    bits = struct.unpack('>Q', struct.pack('>d', x))[0]
    pulse = torch.zeros(64)
    for i in range(64):
        pulse[i] = float((bits >> (63 - i)) & 1)
    return pulse

def pulse_to_float64(pulse):
    bits = 0
    for i in range(64):
        if pulse[i] > 0.5:
            bits |= (1 << (63 - i))
    return struct.unpack('>d', struct.pack('>Q', bits))[0]

def float32_to_pulse(x):
    bits = struct.unpack('>I', struct.pack('>f', float(x)))[0]
    pulse = torch.zeros(32)
    for i in range(32):
        pulse[i] = float((bits >> (31 - i)) & 1)
    return pulse

def pulse_to_float32(pulse):
    bits = 0
    for i in range(32):
        if pulse[i] > 0.5:
            bits |= (1 << (31 - i))
    return struct.unpack('>f', struct.pack('>I', bits))[0]

def get_ulp_error_fp64(result, expected):
    if np.isnan(expected): return 0 if np.isnan(result) else float('inf')
    if np.isinf(expected): return 0 if np.isinf(result) and np.sign(result)==np.sign(expected) else float('inf')
    if expected == 0: return 0 if result == 0 else float('inf')
    bits_result = struct.unpack('>Q', struct.pack('>d', result))[0]
    bits_expected = struct.unpack('>Q', struct.pack('>d', expected))[0]
    return abs(int(bits_result) - int(bits_expected))

def test_sqrt():
    print("="*60)
    print("验证 SpikeFP64Sqrt (GPU)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sqrt_mod = SpikeFP64Sqrt(iterations=12).to(device)
    
    test_cases = [1.0, 2.0, 4.0, 0.5, 0.25, 100.0, 3.0, 0.1]
    
    exact_count = 0
    max_ulp = 0
    
    for x in test_cases:
        pulse_x = float64_to_pulse(x).unsqueeze(0).to(device)
        sqrt_mod.reset()
        result_pulse = sqrt_mod(pulse_x)
        result = pulse_to_float64(result_pulse[0].cpu())
        expected = np.sqrt(x)
        
        ulp = get_ulp_error_fp64(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        status = "✓" if ulp == 0 else f"✗ ULP={ulp}"
        print(f"  sqrt({x}) = {result} (期望: {expected}) {status}")
        if ulp == 0: exact_count += 1
            
    print(f"结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")

def test_rmsnorm():
    print("\n" + "="*60)
    print("验证 SpikeFP32RMSNormFullFP64 (GPU)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 4
    rmsnorm = SpikeFP32RMSNormFullFP64(dim).to(device)
    
    # Input: [1, 2, 3, 4]
    inputs = [1.0, 2.0, 3.0, 4.0]
    inputs_fp32 = torch.tensor(inputs, dtype=torch.float32)
    
    pulse_in = []
    for x in inputs:
        pulse_in.append(float32_to_pulse(x).unsqueeze(0))
    pulse_in = torch.stack(pulse_in, dim=1).to(device) # [1, 4, 32]
    
    rmsnorm.reset()
    result_pulse = rmsnorm(pulse_in) # [1, 4, 32]
    
    results = []
    for i in range(dim):
        results.append(pulse_to_float32(result_pulse[0, i].cpu()))
        
    # PyTorch Ref
    # RMS = sqrt(mean(x^2) + eps)
    # y = x / RMS * w (w=1)
    mean_sq = (inputs_fp32**2).mean()
    rms = torch.sqrt(mean_sq + 1e-6)
    expected = inputs_fp32 / rms
    
    print(f"输入: {inputs}")
    print(f"期望 RMS: {rms.item()}")
    
    max_ulp = 0
    exact_count = 0
    for i in range(dim):
        res = results[i]
        exp = expected[i].item()
        
        # FP32 ULP
        bits_res = struct.unpack('>I', struct.pack('>f', res))[0]
        bits_exp = struct.unpack('>I', struct.pack('>f', exp))[0]
        ulp = abs(int(bits_res) - int(bits_exp))
        max_ulp = max(max_ulp, ulp)
        
        status = "✓" if ulp == 0 else f"✗ ULP={ulp}"
        print(f"  y[{i}] = {res} (期望: {exp}) {status}")
        if ulp == 0: exact_count += 1
            
    print(f"结果: {exact_count}/{dim} 精确, max ULP = {max_ulp}")

if __name__ == "__main__":
    test_sqrt()
    test_rmsnorm()


