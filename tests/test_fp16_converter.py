"""
测试 FP8 <-> FP16 转换器的正确性（纯SNN门电路实现）

端到端浮点验证：
1. 生成随机浮点数
2. 编码/转换
3. 解码回浮点数
4. 直接与参考值比较
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import PulseFloatingPointEncoder
from SNNTorch.atomic_ops.fp16_components import FP8ToFP16Converter, FP16ToFP8Converter
from SNNTorch.atomic_ops.pulse_decoder import PulseFP16Decoder, PulseFloatingPointDecoder


def test_fp8_to_fp16():
    """测试 FP8 -> FP16 转换（端到端浮点验证）"""
    print("="*60)
    print("FP8 -> FP16 转换器测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    converter = FP8ToFP16Converter().to(device)
    fp16_decoder = PulseFP16Decoder().to(device)
    
    # 测试值
    test_values = [
        1.0, 0.5, 0.25, 2.0, 4.0, 
        -1.0, -0.5, 0.0,
        0.125, 3.5, 7.0, 0.0625
    ]
    
    print("| 原值    | FP8值   | SNN FP16  | 误差      | 正确? |")
    print("|---------|---------|-----------|-----------|-------|")
    
    all_correct = True
    for val in test_values:
        x = torch.tensor([[val]], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        expected = x_fp8.float().item()  # FP8 的精确值
        
        # 编码为 FP8 脉冲
        fp8_pulse = encoder(x_fp8.float()).squeeze(1)
        
        # 转换为 FP16
        converter.reset()
        fp16_pulse = converter(fp8_pulse)
        
        # 解码回浮点数
        fp16_decoder.reset()
        result = fp16_decoder(fp16_pulse).item()
        
        error = abs(result - expected)
        correct = error < 1e-6
        if not correct:
            all_correct = False
        
        status = "✓" if correct else "✗"
        print(f"| {val:7.4f} | {expected:7.4f} | {result:9.6f} | {error:9.7f} | {status:5s} |")
    
    print(f"\n总体结果: {'全部正确 ✓' if all_correct else '有错误 ✗'}")
    return all_correct


def test_fp16_to_fp8():
    """测试 FP8 -> FP16 -> FP8 往返（端到端浮点验证）"""
    print("\n" + "="*60)
    print("FP8 -> FP16 -> FP8 往返测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    fp8_to_fp16 = FP8ToFP16Converter().to(device)
    fp16_to_fp8 = FP16ToFP8Converter().to(device)
    fp8_decoder = PulseFloatingPointDecoder().to(device)
    
    # 测试值
    test_values = [1.0, 0.5, 0.25, 2.0, 4.0, -1.0, -0.5, 0.0, 0.125, 3.5]
    
    print("| 原值    | FP8 orig | FP8 back | 匹配? |")
    print("|---------|----------|----------|-------|")
    
    all_correct = True
    for val in test_values:
        x = torch.tensor([[val]], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        orig_float = x_fp8.float().item()
        
        # 编码为 FP8 脉冲
        fp8_pulse = encoder(x_fp8.float()).squeeze(1)
        
        # FP8 -> FP16 -> FP8
        fp8_to_fp16.reset()
        fp16_pulse = fp8_to_fp16(fp8_pulse)
        
        fp16_to_fp8.reset()
        fp8_back_pulse = fp16_to_fp8(fp16_pulse)
        
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


def test_random():
    """随机数往返测试（端到端浮点验证）"""
    print("\n" + "="*60)
    print("随机数往返测试（100个样本）端到端浮点验证")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    fp8_to_fp16 = FP8ToFP16Converter().to(device)
    fp16_to_fp8 = FP16ToFP8Converter().to(device)
    fp8_decoder = PulseFloatingPointDecoder().to(device)
    
    torch.manual_seed(42)
    
    n_tests = 100
    correct = 0
    
    for _ in range(n_tests):
        # 生成随机浮点数
        val = torch.randn(1).item() * 2.0
        
        x = torch.tensor([[val]], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        orig_float = x_fp8.float().item()
        
        # 编码
        fp8_pulse = encoder(x_fp8.float()).squeeze(1)
        
        # FP8 -> FP16 -> FP8
        fp8_to_fp16.reset()
        fp16_pulse = fp8_to_fp16(fp8_pulse)
        
        fp16_to_fp8.reset()
        fp8_back_pulse = fp16_to_fp8(fp16_pulse)
        
        # 解码
        fp8_decoder.reset()
        back_float = fp8_decoder(fp8_back_pulse).item()
        
        if abs(orig_float - back_float) < 1e-6:
            correct += 1
    
    rate = correct / n_tests * 100
    print(f"匹配率: {rate:.1f}% ({correct}/{n_tests})")
    
    return rate


if __name__ == "__main__":
    test_fp8_to_fp16()
    test_fp16_to_fp8()
    test_random()
