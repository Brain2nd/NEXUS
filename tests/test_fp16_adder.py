"""
测试 FP16 加法器的正确性（纯SNN门电路实现）

端到端浮点验证：
1. 生成随机浮点数
2. 编码成 SNN 脉冲
3. SNN 加法运算
4. 解码回浮点数
5. 直接与 PyTorch FP16 加法结果比较
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import PulseFloatingPointEncoder
from SNNTorch.atomic_ops.fp16_components import FP8ToFP16Converter, FP16ToFP8Converter
from SNNTorch.atomic_ops.fp16_adder import SpikeFP16Adder
from SNNTorch.atomic_ops.pulse_decoder import PulseFP16Decoder, PulseFloatingPointDecoder


def float16_to_pulses(x: torch.Tensor, device) -> torch.Tensor:
    """将 FP16 浮点数转换为 16 位脉冲序列（边界组件）
    
    使用 numpy uint16 避免溢出问题。
    
    Args:
        x: FP16 张量
        device: 目标设备
    Returns:
        [..., 16] 脉冲张量
    """
    x_np = x.cpu().numpy().astype(np.float16)
    bits_np = x_np.view(np.uint16)
    
    # 构造输出脉冲
    output_shape = x.shape + (16,)
    pulses = torch.zeros(output_shape, device=device, dtype=torch.float32)
    
    for i in range(16):
        bit_val = (bits_np >> (15 - i)) & 1
        pulses[..., i] = torch.tensor(bit_val, dtype=torch.float32, device=device)
    
    return pulses


def test_fp16_adder_basic():
    """测试 FP16 加法器基本功能（端到端浮点验证）"""
    print("="*60)
    print("FP16 加法器基本测试（端到端浮点验证）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    adder = SpikeFP16Adder().to(device)
    decoder = PulseFP16Decoder().to(device)
    
    # 测试用例
    test_cases = [
        (1.0, 1.0),
        (1.0, 2.0),
        (0.5, 0.5),
        (1.0, -1.0),
        (2.0, -1.0),
        (0.0, 1.0),
        (0.25, 0.75),
        (-0.5, -0.5),
        (3.0, 4.0),
        (1.5, 2.5),
    ]
    
    print("| A      | B      | 期望    | SNN结果  | 误差     | 状态 |")
    print("|--------|--------|---------|----------|----------|------|")
    
    all_correct = True
    for a_val, b_val in test_cases:
        # 1. 量化到 FP16
        a_fp16 = torch.tensor([a_val], dtype=torch.float16)
        b_fp16 = torch.tensor([b_val], dtype=torch.float16)
        expected = (a_fp16 + b_fp16).float().item()
        
        # 2. 编码成脉冲
        a_pulses = float16_to_pulses(a_fp16, device).squeeze(0)  # [16]
        b_pulses = float16_to_pulses(b_fp16, device).squeeze(0)  # [16]
        
        # 3. SNN 加法
        adder.reset()
        sum_pulses = adder(a_pulses, b_pulses)
        
        # 4. 解码回浮点
        decoder.reset()
        snn_result = decoder(sum_pulses.unsqueeze(0)).item()
        
        # 5. 100%位精确比较
        # 将期望结果编码为脉冲，然后比较位
        expected_fp16 = torch.tensor([expected], dtype=torch.float16)
        expected_pulses = float16_to_pulses(expected_fp16, device).squeeze(0)
        
        # 位精确比较
        bits_match = (sum_pulses == expected_pulses).all().item()
        if not bits_match:
            all_correct = False
        
        error = abs(snn_result - expected)
        status = "✓" if bits_match else "✗"
        print(f"| {a_val:6.2f} | {b_val:6.2f} | {expected:7.3f} | {snn_result:8.4f} | {error:8.5f} | {status:4s} [位精确]|")
    
    print(f"\n总体结果: {'全部正确 ✓' if all_correct else '有错误 ✗'}")
    return all_correct


def test_fp16_accumulation():
    """测试 FP16 累加（多个 FP8 值转 FP16 后累加）- 端到端浮点验证"""
    print("\n" + "="*60)
    print("FP16 累加测试（FP8 -> FP16 累加）端到端浮点验证")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    fp8_decoder = PulseFloatingPointDecoder().to(device)
    fp16_decoder = PulseFP16Decoder().to(device)
    
    fp8_to_fp16 = FP8ToFP16Converter().to(device)
    fp16_adder = SpikeFP16Adder().to(device)
    fp16_to_fp8 = FP16ToFP8Converter().to(device)
    
    # 测试累加
    values = [0.5, 0.25, 0.125, 0.0625]
    
    print(f"\n累加序列: {values}")
    
    # PyTorch 参考：FP16 累加
    fp16_values = [torch.tensor(v, dtype=torch.float16) for v in values]
    expected_sum = fp16_values[0]
    for v in fp16_values[1:]:
        expected_sum = expected_sum + v
    expected_sum_float = expected_sum.float().item()
    print(f"PyTorch FP16 累加结果: {expected_sum_float}")
    
    # SNN 累加
    fp8_pulses = []
    for val in values:
        x = torch.tensor([[val]], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        x_pulse = encoder(x_fp8.float()).squeeze(1)
        fp8_pulses.append(x_pulse)
    
    # 转 FP16 并累加
    fp16_pulses = []
    for fp8_p in fp8_pulses:
        fp8_to_fp16.reset()
        fp16_p = fp8_to_fp16(fp8_p)
        fp16_pulses.append(fp16_p)
    
    # 累加
    acc = fp16_pulses[0]
    for i in range(1, len(fp16_pulses)):
        fp16_adder.reset()
        acc = fp16_adder(acc, fp16_pulses[i])
    
    # 解码结果
    fp16_decoder.reset()
    snn_sum_fp16 = fp16_decoder(acc).item()
    print(f"SNN FP16 累加结果: {snn_sum_fp16}")
    
    # 直接比较浮点数
    error = abs(snn_sum_fp16 - expected_sum_float)
    match = error < 1e-3 or abs(error / expected_sum_float) < 0.01
    print(f"误差: {error:.6f}")
    print(f"匹配: {'✓' if match else '✗'}")
    
    return match


def test_random_additions():
    """随机数加法测试（端到端浮点验证）"""
    print("\n" + "="*60)
    print("随机数加法测试（100对）端到端浮点验证")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    adder = SpikeFP16Adder().to(device)
    decoder = PulseFP16Decoder().to(device)
    
    torch.manual_seed(42)
    
    n_tests = 100
    correct = 0
    
    for _ in range(n_tests):
        # 1. 生成随机浮点数
        a_val = torch.randn(1).item() * 2
        b_val = torch.randn(1).item() * 2
        
        # 2. 量化到 FP16
        a_fp16 = torch.tensor([a_val], dtype=torch.float16)
        b_fp16 = torch.tensor([b_val], dtype=torch.float16)
        expected = (a_fp16 + b_fp16).float().item()
        
        # 3. 编码成脉冲
        a_pulses = float16_to_pulses(a_fp16, device).squeeze(0)
        b_pulses = float16_to_pulses(b_fp16, device).squeeze(0)
        
        # 4. SNN 加法
        adder.reset()
        sum_pulses = adder(a_pulses, b_pulses)
        
        # 5. 解码回浮点
        decoder.reset()
        snn_result = decoder(sum_pulses.unsqueeze(0)).item()
        
        # 6. 直接比较浮点数
        error = abs(snn_result - expected)
        rel_error = error / max(abs(expected), 1e-10)
        
        if rel_error < 0.05 or error < 1e-3:
            correct += 1
    
    rate = correct / n_tests * 100
    print(f"正确率: {rate:.1f}% ({correct}/{n_tests})")
    
    return rate


if __name__ == "__main__":
    test_fp16_adder_basic()
    test_fp16_accumulation()
    test_random_additions()
