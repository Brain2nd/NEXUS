"""
FP8 乘法器测试（端到端浮点验证）
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import PulseFloatingPointEncoder, SpikeFP8Multiplier
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder


def test_fp8_multiplier_basic():
    """测试基本乘法正确性（端到端浮点验证）"""
    print("=== FP8 乘法器基本测试（端到端浮点验证）===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    mul = SpikeFP8Multiplier().to(device)
    
    # 测试用例
    test_cases = [
        (1.5, 2.5),
        (1.0, 1.0),
        (2.0, 3.0),
        (-1.0, 2.0),
        (-2.0, -3.0),
        (0.5, 2.0),
        (0.5, 0.5),
    ]
    
    print("\n| A     | B     | 期望    | SNN结果 | 状态 |")
    print("|-------|-------|---------|---------|------|")
    
    all_pass = True
    for a_val, b_val in test_cases:
        # 量化到 FP8
        a_fp8 = torch.tensor([a_val], device=device).to(torch.float8_e4m3fn)
        b_fp8 = torch.tensor([b_val], device=device).to(torch.float8_e4m3fn)
        
        # PyTorch 参考
        expected = (a_fp8.float() * b_fp8.float()).to(torch.float8_e4m3fn).float().item()
        
        # SNN 计算
        encoder.reset()
        a_pulse = encoder(a_fp8.float())
        encoder.reset()
        b_pulse = encoder(b_fp8.float())
        
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)
        
        decoder.reset()
        snn_result = decoder(result_pulse).item()
        
        # 直接比较浮点数
        match = abs(snn_result - expected) < 1e-6 or snn_result == expected
        if not match:
            all_pass = False
        
        status = "✓" if match else "✗"
        print(f"| {a_fp8.float().item():5.2f} | {b_fp8.float().item():5.2f} | {expected:7.3f} | {snn_result:7.3f} | {status:4s} |")
    
    return all_pass


def test_fp8_mul_comprehensive():
    """乘法器全面测试（端到端浮点验证）"""
    print("\n=== FP8 乘法器全面测试 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3, 
                                         scan_integer_bits=10, scan_decimal_bits=10).to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    mul = SpikeFP8Multiplier().to(device)
    
    test_cases = [
        # Normal * Normal
        (1.0, 1.0, "1.0 * 1.0"),
        (2.0, 3.0, "2.0 * 3.0"),
        (1.5, 2.5, "1.5 * 2.5"),
        (-1.0, 2.0, "-1.0 * 2.0"),
        (-2.0, -3.0, "-2.0 * -3.0"),
        # 零
        (0.0, 5.0, "0.0 * 5.0"),
        (3.0, 0.0, "3.0 * 0.0"),
    ]
    
    print("\n| 描述         | 期望    | SNN结果 | 状态 |")
    print("|--------------|---------|---------|------|")
    
    all_pass = True
    for a, b, desc in test_cases:
        a_fp8 = torch.tensor([[a]], device=device).to(torch.float8_e4m3fn)
        b_fp8 = torch.tensor([[b]], device=device).to(torch.float8_e4m3fn)
        expected = (a_fp8.float() * b_fp8.float()).to(torch.float8_e4m3fn).float().item()
        
        encoder.reset()
        a_pulse = encoder(a_fp8.float())
        encoder.reset()
        b_pulse = encoder(b_fp8.float())
        
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)
        
        decoder.reset()
        snn_result = decoder(result_pulse).item()
        
        match = abs(snn_result - expected) < 1e-6 or snn_result == expected
        if not match:
            all_pass = False
        
        status = "✓" if match else "✗"
        print(f"| {desc:12s} | {expected:7.3f} | {snn_result:7.3f} | {status:4s} |")
    
    return all_pass


def test_fp8_mul_random():
    """随机测试（端到端浮点验证）"""
    print("\n=== FP8 乘法器随机测试（1000对）===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3, 
                                         scan_integer_bits=10, scan_decimal_bits=10).to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    mul = SpikeFP8Multiplier().to(device)
    
    N = 1000
    torch.manual_seed(42)
    
    # 生成随机浮点数
    a_raw = torch.randn(N, device=device) * 5
    b_raw = torch.randn(N, device=device) * 5
    a_fp8 = a_raw.to(torch.float8_e4m3fn)
    b_fp8 = b_raw.to(torch.float8_e4m3fn)
    
    # PyTorch 参考
    ref = (a_fp8.float() * b_fp8.float()).to(torch.float8_e4m3fn).float()
    
    # SNN 计算
    encoder.reset()
    p_a = encoder(a_fp8.float())
    encoder.reset()
    p_b = encoder(b_fp8.float())
    mul.reset()
    p_out = mul(p_a, p_b)
    
    decoder.reset()
    snn_result = decoder(p_out)
    
    # 直接比较浮点数
    match = torch.isclose(snn_result, ref, rtol=1e-5, atol=1e-6) | (snn_result == ref)
    match_count = match.sum().item()
    
    print(f"匹配率: {match_count}/{N} = {match_count/N*100:.1f}%")
    
    if match_count < N:
        mismatch_idx = (~match).nonzero(as_tuple=True)[0][:5]
        print("不匹配样例（前5个）:")
        for idx in mismatch_idx:
            i = idx.item()
            print(f"  {a_fp8[i].float().item():.4f} * {b_fp8[i].float().item():.4f} = Ref:{ref[i].item():.4f}, SNN:{snn_result[i].item():.4f}")
    
    return match_count / N * 100


if __name__ == "__main__":
    basic_pass = test_fp8_multiplier_basic()
    comp_pass = test_fp8_mul_comprehensive()
    random_rate = test_fp8_mul_random()
    
    print("\n=== 总结 ===")
    print(f"基本测试: {'✓' if basic_pass else '✗'}")
    print(f"全面测试: {'✓' if comp_pass else '✗'}")
    print(f"随机测试: {random_rate:.1f}%")
