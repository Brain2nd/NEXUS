"""
测试 FP32 组件的正确性（纯SNN门电路实现）

端到端浮点验证：
1. 生成随机浮点数
2. 编码成 SNN 脉冲
3. SNN 运算
4. 解码回浮点数
5. 直接与 PyTorch 结果比较
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import PulseFloatingPointEncoder
from SNNTorch.atomic_ops.fp32_components import FP8ToFP32Converter, FP32ToFP8Converter
from SNNTorch.atomic_ops.fp32_adder import SpikeFP32Adder
from SNNTorch.atomic_ops.pulse_decoder import PulseFP32Decoder, PulseFloatingPointDecoder


def float32_to_pulses(x: torch.Tensor, device) -> torch.Tensor:
    """将 FP32 浮点数转换为 32 位脉冲序列（边界组件）
    
    使用 numpy uint32 避免溢出问题。
    
    Args:
        x: FP32 张量
        device: 目标设备
    Returns:
        [..., 32] 脉冲张量
    """
    x_np = x.cpu().numpy().astype(np.float32)
    bits_np = x_np.view(np.uint32)
    
    output_shape = x.shape + (32,)
    pulses = torch.zeros(output_shape, device=device, dtype=torch.float32)
    
    for i in range(32):
        bit_val = (bits_np >> (31 - i)) & 1
        pulses[..., i] = torch.tensor(bit_val, dtype=torch.float32, device=device)
    
    return pulses


def test_fp8_to_fp32():
    """测试 FP8 -> FP32 转换（端到端浮点验证）"""
    print("="*60)
    print("FP8 -> FP32 转换器测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    converter = FP8ToFP32Converter().to(device)
    fp32_decoder = PulseFP32Decoder().to(device)
    
    test_values = [1.0, 0.5, 0.25, 2.0, 4.0, -1.0, -0.5, 0.0, 0.125, 3.5]
    
    print("| 原值    | FP8值   | SNN FP32  | 误差      | 正确? |")
    print("|---------|---------|-----------|-----------|-------|")
    
    all_correct = True
    for val in test_values:
        x = torch.tensor([[val]], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        expected = x_fp8.float().item()
        
        # 编码为 FP8 脉冲
        fp8_pulse = encoder(x_fp8.float()).squeeze(1)
        
        # 转换为 FP32
        converter.reset()
        fp32_pulse = converter(fp8_pulse)
        
        # 解码回浮点数
        fp32_decoder.reset()
        result = fp32_decoder(fp32_pulse).item()
        
        error = abs(result - expected)
        correct = error < 1e-6
        if not correct:
            all_correct = False
        
        status = "✓" if correct else "✗"
        print(f"| {val:7.4f} | {expected:7.4f} | {result:9.6f} | {error:9.7f} | {status:5s} |")
    
    print(f"\n总体结果: {'全部正确 ✓' if all_correct else '有错误 ✗'}")
    return all_correct


def test_fp32_to_fp8():
    """测试 FP32 -> FP8 转换（端到端浮点验证）"""
    print("\n" + "="*60)
    print("FP32 -> FP8 转换器测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    converter = FP32ToFP8Converter().to(device)
    fp8_decoder = PulseFloatingPointDecoder().to(device)
    
    test_values = [1.0, 0.5, 0.25, 2.0, 4.0, -1.0, -0.5, 0.0, 0.125, 3.5]
    
    print("| 原值    | PyTorch FP8 | SNN FP8   | 误差      | 正确? |")
    print("|---------|-------------|-----------|-----------|-------|")
    
    all_correct = True
    for val in test_values:
        x = torch.tensor([[val]], device=device, dtype=torch.float32)
        expected_fp8 = x.to(torch.float8_e4m3fn).float().item()
        
        # 转为 FP32 脉冲
        fp32_pulse = float32_to_pulses(x, device)
        
        # 转换为 FP8
        converter.reset()
        fp8_pulse = converter(fp32_pulse)
        
        # 解码回浮点数
        fp8_decoder.reset()
        result = fp8_decoder(fp8_pulse).item()
        
        error = abs(result - expected_fp8)
        correct = error < 1e-6
        if not correct:
            all_correct = False
        
        status = "✓" if correct else "✗"
        print(f"| {val:7.4f} | {expected_fp8:11.6f} | {result:9.6f} | {error:9.7f} | {status:5s} |")
    
    print(f"\n总体结果: {'全部正确 ✓' if all_correct else '有错误 ✗'}")
    return all_correct


def test_fp32_adder():
    """测试 FP32 加法器（端到端浮点验证）"""
    print("\n" + "="*60)
    print("FP32 加法器测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    adder = SpikeFP32Adder().to(device)
    decoder = PulseFP32Decoder().to(device)
    
    test_cases = [
        (1.0, 1.0),
        (1.0, 2.0),
        (0.5, 0.5),
        (1.0, -1.0),
        (2.0, -1.0),
        (0.25, 0.75),
        (3.0, 4.0),
        (-1.0, -1.0),
        (1.5, 0.5),
        (0.125, 0.125),
    ]
    
    print("| A       | B       | 期望    | SNN结果  | 误差     | 状态 |")
    print("|---------|---------|---------|----------|----------|------|")
    
    all_correct = True
    for a_val, b_val in test_cases:
        a = torch.tensor([[a_val]], device=device, dtype=torch.float32)
        b = torch.tensor([[b_val]], device=device, dtype=torch.float32)
        expected = (a + b).item()
        
        # 转为脉冲
        a_pulse = float32_to_pulses(a, device)
        b_pulse = float32_to_pulses(b, device)
        
        # SNN 加法
        adder.reset()
        sum_pulse = adder(a_pulse, b_pulse)
        
        # 解码回浮点数
        decoder.reset()
        result = decoder(sum_pulse).item()
        
        error = abs(result - expected)
        correct = error < 1e-6
        if not correct:
            all_correct = False
        
        status = "✓" if correct else "✗"
        print(f"| {a_val:7.4f} | {b_val:7.4f} | {expected:7.4f} | {result:8.5f} | {error:8.6f} | {status:4s} |")
    
    print(f"\n总体结果: {'全部正确 ✓' if all_correct else '有错误 ✗'}")
    return all_correct


def test_roundtrip():
    """测试 FP8 -> FP32 -> FP8 往返（端到端浮点验证）"""
    print("\n" + "="*60)
    print("FP8 -> FP32 -> FP8 往返测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    fp8_decoder = PulseFloatingPointDecoder().to(device)
    
    fp8_to_fp32 = FP8ToFP32Converter().to(device)
    fp32_to_fp8 = FP32ToFP8Converter().to(device)
    
    test_values = [1.0, 0.5, 0.25, 2.0, -1.0, -0.5, 0.0, 0.125]
    
    print("| 原值    | FP8 orig | FP8 back | 匹配? |")
    print("|---------|----------|----------|-------|")
    
    all_correct = True
    for val in test_values:
        x = torch.tensor([[val]], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        orig_float = x_fp8.float().item()
        
        # FP8 脉冲
        fp8_pulse = encoder(x_fp8.float()).squeeze(1)
        
        # FP8 -> FP32 -> FP8
        fp8_to_fp32.reset()
        fp32_pulse = fp8_to_fp32(fp8_pulse)
        
        fp32_to_fp8.reset()
        fp8_back_pulse = fp32_to_fp8(fp32_pulse)
        
        # 解码
        fp8_decoder.reset()
        back_float = fp8_decoder(fp8_back_pulse).item()
        
        match = abs(orig_float - back_float) < 1e-6
        if not match:
            all_correct = False
        
        status = "✓" if match else "✗"
        print(f"| {val:7.4f} | {orig_float:8.5f} | {back_float:8.5f} | {status:5s} |")
    
    print(f"\n总体结果: {'全部正确 ✓' if all_correct else '有错误 ✗'}")
    return all_correct


if __name__ == "__main__":
    test_fp8_to_fp32()
    test_fp32_to_fp8()
    test_fp32_adder()
    test_roundtrip()
