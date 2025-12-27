"""
FP64 除法器测试 - 验证 SpikeFP64Divider 的 bit-exactness
"""
import torch
import numpy as np
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import SpikeFP64Divider, FP32ToFP64Converter, FP64ToFP32Converter


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


def get_ulp_error(result, expected):
    """计算ULP误差"""
    if np.isnan(expected):
        return 0 if np.isnan(result) else float('inf')
    if np.isinf(expected):
        if np.isinf(result) and np.sign(result) == np.sign(expected):
            return 0
        return float('inf')
    if expected == 0:
        return 0 if result == 0 else float('inf')
    
    # 计算ULP
    bits_result = struct.unpack('>Q', struct.pack('>d', result))[0]
    bits_expected = struct.unpack('>Q', struct.pack('>d', expected))[0]
    return abs(int(bits_result) - int(bits_expected))


def test_fp64_divider():
    """测试FP64除法器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    divider = SpikeFP64Divider().to(device)
    
    # 测试用例
    test_cases = [
        # (被除数, 除数, 预期结果)
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (1.0, 2.0, 0.5),
        (10.0, 3.0, 10.0/3.0),
        (1.0, 3.0, 1.0/3.0),
        (100.0, 7.0, 100.0/7.0),
        (-1.0, 2.0, -0.5),
        (1.0, -2.0, -0.5),
        (-1.0, -2.0, 0.5),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, float('inf')),
        (-1.0, 0.0, float('-inf')),
        (0.0, 0.0, float('nan')),
        (float('inf'), 1.0, float('inf')),
        (1.0, float('inf'), 0.0),
        (float('inf'), float('inf'), float('nan')),
    ]
    
    print(f"\n测试 {len(test_cases)} 个基础用例...")
    
    exact_count = 0
    max_ulp = 0
    
    for a, b, expected in test_cases:
        # 转换为脉冲
        pulse_a = float64_to_pulse(a).unsqueeze(0).to(device)
        pulse_b = float64_to_pulse(b).unsqueeze(0).to(device)
        
        # 计算
        divider.reset()
        result_pulse = divider(pulse_a, pulse_b)
        
        # 转换回float64
        result = pulse_to_float64(result_pulse[0].cpu())
        
        # 计算误差
        ulp = get_ulp_error(result, expected)
        max_ulp = max(max_ulp, ulp) if ulp != float('inf') else max_ulp
        
        if ulp == 0:
            exact_count += 1
            status = "✓"
        else:
            status = f"✗ ULP={ulp}"
        
        print(f"  {a} / {b} = {result} (期望: {expected}) {status}")
    
    print(f"\n基础用例: {exact_count}/{len(test_cases)} 精确")
    
    # 随机测试
    print("\n随机测试 100 个样本...")
    np.random.seed(42)
    
    random_exact = 0
    random_max_ulp = 0
    
    for _ in range(100):
        # 生成随机数
        exp_a = np.random.randint(-100, 100)
        exp_b = np.random.randint(-100, 100)
        mant_a = np.random.random() + 1.0
        mant_b = np.random.random() + 1.0
        sign_a = np.random.choice([-1, 1])
        sign_b = np.random.choice([-1, 1])
        
        a = sign_a * mant_a * (2.0 ** exp_a)
        b = sign_b * mant_b * (2.0 ** exp_b)
        expected = a / b
        
        pulse_a = float64_to_pulse(a).unsqueeze(0).to(device)
        pulse_b = float64_to_pulse(b).unsqueeze(0).to(device)
        
        divider.reset()
        result_pulse = divider(pulse_a, pulse_b)
        result = pulse_to_float64(result_pulse[0].cpu())
        
        ulp = get_ulp_error(result, expected)
        if ulp == 0:
            random_exact += 1
        elif ulp != float('inf'):
            random_max_ulp = max(random_max_ulp, ulp)
    
    print(f"随机测试: {random_exact}/100 精确, max ULP = {random_max_ulp}")
    
    # 批量测试
    print("\n批量处理测试...")
    batch_size = 10
    
    batch_a = torch.stack([float64_to_pulse(float(i+1)) for i in range(batch_size)]).to(device)
    batch_b = torch.stack([float64_to_pulse(float(i+2)) for i in range(batch_size)]).to(device)
    
    divider.reset()
    batch_result = divider(batch_a, batch_b)
    
    batch_exact = 0
    for i in range(batch_size):
        result = pulse_to_float64(batch_result[i].cpu())
        expected = float(i+1) / float(i+2)
        ulp = get_ulp_error(result, expected)
        if ulp == 0:
            batch_exact += 1
        print(f"  {i+1}/{i+2} = {result} (期望: {expected}) ULP={ulp}")
    
    print(f"批量测试: {batch_exact}/{batch_size} 精确")
    
    # 总结
    total_exact = exact_count + random_exact + batch_exact
    total_tests = len(test_cases) + 100 + batch_size
    
    print(f"\n{'='*50}")
    print(f"总计: {total_exact}/{total_tests} 精确")
    if total_exact == total_tests:
        print("✅ SpikeFP64Divider 达到 100% bit-exact!")
    else:
        print(f"⚠️ 仍有误差，需要调试")


if __name__ == "__main__":
    test_fp64_divider()


