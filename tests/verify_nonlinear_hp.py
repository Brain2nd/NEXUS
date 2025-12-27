"""
验证高精度非线性函数 (Sigmoid, SiLU, Softmax) - 使用 GPU 加速
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.fp64_exp import (SpikeFP32SigmoidFullFP64, 
                                 SpikeFP32SiLUFullFP64, 
                                 SpikeFP32SoftmaxFullFP64)

def float32_to_pulse(x):
    """将float32转换为32位脉冲表示 (MSB first)"""
    bits = struct.unpack('>I', struct.pack('>f', float(x)))[0]
    pulse = torch.zeros(32)
    for i in range(32):
        pulse[i] = float((bits >> (31 - i)) & 1)
    return pulse

def pulse_to_float32(pulse):
    """将32位脉冲转换回float32"""
    bits = 0
    for i in range(32):
        if pulse[i] > 0.5:
            bits |= (1 << (31 - i))
    return struct.unpack('>f', struct.pack('>I', bits))[0]

def get_ulp_error_fp32(result, expected):
    """计算FP32的ULP误差"""
    if np.isnan(expected):
        return 0 if np.isnan(result) else float('inf')
    if np.isinf(expected):
        if np.isinf(result) and np.sign(result) == np.sign(expected):
            return 0
        return float('inf')
    if expected == 0:
        return 0 if result == 0 else float('inf')
    
    bits_result = struct.unpack('>I', struct.pack('>f', result))[0]
    bits_expected = struct.unpack('>I', struct.pack('>f', expected))[0]
    return abs(int(bits_result) - int(bits_expected))

def test_sigmoid():
    print("="*60)
    print("验证 Sigmoid (FP32 -> FP64 -> FP32)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigmoid = SpikeFP32SigmoidFullFP64().to(device)
    
    # 测试用例：包括 0, 正负数, 大数, 小数
    test_cases = [
        0.0, 1.0, -1.0, 0.5, -0.5, 
        2.0, -2.0, 5.0, -5.0, 
        10.0, -10.0, # 饱和区
        0.1, -0.1, 0.01, -0.01
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for x in test_cases:
        x_fp32 = np.float32(x)
        pulse_x = float32_to_pulse(x_fp32).unsqueeze(0).to(device)
        
        sigmoid.reset()
        result_pulse = sigmoid(pulse_x)
        result = pulse_to_float32(result_pulse[0].cpu())
        
        # PyTorch reference
        expected = torch.sigmoid(torch.tensor(x_fp32)).item()
        
        ulp = get_ulp_error_fp32(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  sigmoid({x}) = {result} (期望: {expected}) {status}")
        
    print(f"结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp

def test_silu():
    print("\n" + "="*60)
    print("验证 SiLU (FP32 -> FP64 -> FP32)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    silu = SpikeFP32SiLUFullFP64().to(device)
    
    test_cases = [
        0.0, 1.0, -1.0, 0.5, -0.5, 
        2.0, -2.0, 5.0, -5.0, 
        10.0, -10.0,
        0.1, -0.1
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for x in test_cases:
        x_fp32 = np.float32(x)
        pulse_x = float32_to_pulse(x_fp32).unsqueeze(0).to(device)
        
        silu.reset()
        result_pulse = silu(pulse_x)
        result = pulse_to_float32(result_pulse[0].cpu())
        
        # PyTorch reference
        expected = torch.nn.functional.silu(torch.tensor(x_fp32)).item()
        
        ulp = get_ulp_error_fp32(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  silu({x}) = {result} (期望: {expected}) {status}")
        
    print(f"结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp

def test_softmax():
    print("\n" + "="*60)
    print("验证 Softmax (FP32 -> FP64 -> FP32)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    softmax = SpikeFP32SoftmaxFullFP64().to(device)
    
    # [1.0, 2.0, 3.0]
    inputs = [1.0, 2.0, 3.0]
    inputs_fp32 = torch.tensor(inputs, dtype=torch.float32)
    
    pulse_in = []
    for x in inputs:
        pulse_in.append(float32_to_pulse(x).unsqueeze(0)) # [1, 32]
    pulse_in = torch.stack(pulse_in, dim=1).to(device) # [1, 3, 32]
    
    softmax.reset()
    result_pulse = softmax(pulse_in) # [1, 3, 32]
    
    results = []
    for i in range(3):
        results.append(pulse_to_float32(result_pulse[0, i].cpu()))
    
    expected = torch.softmax(inputs_fp32, dim=0)
    
    exact_count = 0
    max_ulp = 0
    
    for i in range(3):
        res = results[i]
        exp = expected[i].item()
        ulp = get_ulp_error_fp32(res, exp)
        max_ulp = max(max_ulp, ulp)
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        print(f"  softmax[{i}] = {res} (期望: {exp}) {status}")
        
    print(f"结果: {exact_count}/{3} 精确, max ULP = {max_ulp}")
    return exact_count, 3, max_ulp

if __name__ == "__main__":
    test_sigmoid()
    test_silu()
    test_softmax()


