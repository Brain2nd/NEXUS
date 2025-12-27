"""
验证所有FP64组件的精度 - 只读测试
"""
import torch
import numpy as np
import struct
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (SpikeFP64Adder, SpikeFP64Multiplier, SpikeFP64Exp,
                        SpikeFP64Divider)
from atomic_ops.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter


def float64_to_pulse(x):
    """将float64转换为64位脉冲表示 (MSB first)"""
    bits = struct.unpack('>Q', struct.pack('>d', x))[0]
    pulse = torch.zeros(64)
    for i in range(64):
        pulse[i] = float((bits >> (63 - i)) & 1)
    return pulse


def pulse_to_float64(pulse):
    """将64位脉冲转换回float64"""
    bits = 0
    for i in range(64):
        if pulse[i] > 0.5:
            bits |= (1 << (63 - i))
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


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


def get_ulp_error_fp64(result, expected):
    """计算FP64的ULP误差"""
    if np.isnan(expected):
        return 0 if np.isnan(result) else float('inf')
    if np.isinf(expected):
        if np.isinf(result) and np.sign(result) == np.sign(expected):
            return 0
        return float('inf')
    if expected == 0:
        return 0 if result == 0 else float('inf')
    
    bits_result = struct.unpack('>Q', struct.pack('>d', result))[0]
    bits_expected = struct.unpack('>Q', struct.pack('>d', expected))[0]
    return abs(int(bits_result) - int(bits_expected))


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


def test_fp64_adder():
    """测试 SpikeFP64Adder"""
    print("="*60)
    print("验证 SpikeFP64Adder")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    adder = SpikeFP64Adder().to(device)
    
    # 测试用例
    test_cases = [
        (1.0, 1.0),
        (1.0, 2.0),
        (1.5, 0.5),
        (10.0, 3.0),
        (1.0, -1.0),
        (0.5, 0.25),
        (1.234567890123456, 0.987654321098765),
        (1e10, 1e-10),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for a, b in test_cases:
        pulse_a = float64_to_pulse(a).unsqueeze(0).to(device)
        pulse_b = float64_to_pulse(b).unsqueeze(0).to(device)
        
        adder.reset()
        result_pulse = adder(pulse_a, pulse_b)
        result = pulse_to_float64(result_pulse[0].cpu())
        expected = a + b
        
        ulp = get_ulp_error_fp64(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  {a} + {b} = {result} (期望: {expected}) {status}")
    
    print(f"\n结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp


def test_fp64_multiplier():
    """测试 SpikeFP64Multiplier"""
    print("\n" + "="*60)
    print("验证 SpikeFP64Multiplier")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP64Multiplier().to(device)
    
    test_cases = [
        (1.0, 1.0),
        (2.0, 3.0),
        (1.5, 2.0),
        (0.5, 0.5),
        (1.234567890123456, 0.987654321098765),
        (2.0, -3.0),
        (-2.0, -3.0),
        (1e10, 1e-10),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for a, b in test_cases:
        pulse_a = float64_to_pulse(a).unsqueeze(0).to(device)
        pulse_b = float64_to_pulse(b).unsqueeze(0).to(device)
        
        mul.reset()
        result_pulse = mul(pulse_a, pulse_b)
        result = pulse_to_float64(result_pulse[0].cpu())
        expected = a * b
        
        ulp = get_ulp_error_fp64(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  {a} * {b} = {result} (期望: {expected}) {status}")
    
    print(f"\n结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp


def test_fp64_exp():
    """测试 SpikeFP64Exp"""
    print("\n" + "="*60)
    print("验证 SpikeFP64Exp")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_mod = SpikeFP64Exp().to(device)
    
    test_cases = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        0.1,
        -0.1,
        3.0,
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for x in test_cases:
        pulse_x = float64_to_pulse(x).unsqueeze(0).to(device)
        
        exp_mod.reset()
        result_pulse = exp_mod(pulse_x)
        result = pulse_to_float64(result_pulse[0].cpu())
        expected = math.exp(x)
        
        ulp = get_ulp_error_fp64(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  exp({x}) = {result:.15f} (期望: {expected:.15f}) {status}")
    
    print(f"\n结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp


def test_fp64_divider():
    """测试 SpikeFP64Divider (已知应该100% bit-exact)"""
    print("\n" + "="*60)
    print("验证 SpikeFP64Divider (参照)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    div = SpikeFP64Divider().to(device)
    
    test_cases = [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (10.0, 3.0),
        (1.0, 3.0),
        (3.0, 2.0),
        (1.234567890123456, 0.987654321098765),
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for a, b in test_cases:
        pulse_a = float64_to_pulse(a).unsqueeze(0).to(device)
        pulse_b = float64_to_pulse(b).unsqueeze(0).to(device)
        
        div.reset()
        result_pulse = div(pulse_a, pulse_b)
        result = pulse_to_float64(result_pulse[0].cpu())
        expected = a / b
        
        ulp = get_ulp_error_fp64(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  {a} / {b} = {result} (期望: {expected}) {status}")
    
    print(f"\n结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp


def test_fp32_to_fp64_converter():
    """测试 FP32ToFP64Converter"""
    print("\n" + "="*60)
    print("验证 FP32ToFP64Converter")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conv = FP32ToFP64Converter().to(device)
    
    test_cases = [
        1.0,
        0.5,
        0.25,
        3.14159,
        -1.0,
        1e10,
        1e-10,
        0.0,
    ]
    
    exact_count = 0
    
    for x in test_cases:
        x_fp32 = np.float32(x)
        pulse32 = float32_to_pulse(x_fp32).unsqueeze(0).to(device)
        
        conv.reset()
        pulse64 = conv(pulse32)
        result = pulse_to_float64(pulse64[0].cpu())
        expected = float(x_fp32)  # FP32转FP64应该精确
        
        if result == expected:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ({result} vs {expected})"
        
        print(f"  FP32({x_fp32}) -> FP64 = {result} {status}")
    
    print(f"\n结果: {exact_count}/{len(test_cases)} 精确")
    return exact_count, len(test_cases)


def test_fp64_to_fp32_converter():
    """测试 FP64ToFP32Converter"""
    print("\n" + "="*60)
    print("验证 FP64ToFP32Converter")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conv = FP64ToFP32Converter().to(device)
    
    test_cases = [
        1.0,
        0.5,
        0.25,
        3.14159265358979,
        -1.0,
        1e10,
        1e-10,
        0.0,
        1.0/3.0,  # 需要舍入
    ]
    
    exact_count = 0
    max_ulp = 0
    
    for x in test_cases:
        pulse64 = float64_to_pulse(x).unsqueeze(0).to(device)
        
        conv.reset()
        pulse32 = conv(pulse64)
        result = pulse_to_float32(pulse32[0].cpu())
        expected = np.float32(x)  # Python/NumPy的舍入
        
        ulp = get_ulp_error_fp32(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  FP64({x}) -> FP32 = {result} (期望: {expected}) {status}")
    
    print(f"\n结果: {exact_count}/{len(test_cases)} 精确, max ULP = {max_ulp}")
    return exact_count, len(test_cases), max_ulp


if __name__ == "__main__":
    print("FP64 组件精度验证")
    print("="*60)
    
    results = {}
    
    # 测试除法器 (参照，应该100%)
    div_exact, div_total, div_max = test_fp64_divider()
    results['Divider'] = (div_exact, div_total, div_max)
    
    # 测试加法器
    add_exact, add_total, add_max = test_fp64_adder()
    results['Adder'] = (add_exact, add_total, add_max)
    
    # 测试乘法器
    mul_exact, mul_total, mul_max = test_fp64_multiplier()
    results['Multiplier'] = (mul_exact, mul_total, mul_max)
    
    # 测试Exp
    exp_exact, exp_total, exp_max = test_fp64_exp()
    results['Exp'] = (exp_exact, exp_total, exp_max)
    
    # 测试转换器
    conv32to64_exact, conv32to64_total = test_fp32_to_fp64_converter()
    results['FP32ToFP64'] = (conv32to64_exact, conv32to64_total, 0)
    
    conv64to32_exact, conv64to32_total, conv64to32_max = test_fp64_to_fp32_converter()
    results['FP64ToFP32'] = (conv64to32_exact, conv64to32_total, conv64to32_max)
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    for name, (exact, total, max_ulp) in results.items():
        pct = 100 * exact / total
        status = "✅" if exact == total else "❌"
        print(f"  {name}: {exact}/{total} ({pct:.0f}%) 精确, max ULP = {max_ulp} {status}")


