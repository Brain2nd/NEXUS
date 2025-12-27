import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import PulseFloatingPointEncoder

def get_fp8_e4m3_bits(x_float):
    """
    Software reference to extract S, E, M bits from a float value according to E4M3 standard.
    This is used to verify the SNN output.
    Standard: E4M3FN (1 sign, 4 exp, 3 mantissa, bias 7)
    """
    # Convert to FP8 tensor to handle rounding/clamping correctly
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    # Convert back to uint8 representation to read bits
    x_uint8 = x_fp8.view(torch.uint8)
    
    val = x_uint8.item()
    
    # Extract bits
    # Bit 7: Sign
    s = (val >> 7) & 1
    # Bit 6-3: Exponent
    e_val = (val >> 3) & 0xF
    # Bit 2-0: Mantissa
    m_val = val & 0x7
    
    # Convert to lists
    s_bits = [s]
    e_bits = [(e_val >> (3-i)) & 1 for i in range(4)]
    m_bits = [(m_val >> (2-i)) & 1 for i in range(3)]
    
    return s_bits + e_bits + m_bits

def test_fp8_random():
    print("=== Testing PulseFloatingPointEncoder with Random FP8 Data ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 配置编码器
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, 
        mantissa_bits=3,
        scan_integer_bits=10, 
        scan_decimal_bits=10
    ).to(device)
    
    # 生成随机数据 (float32 -> fp8 -> float32 确保值是合法的 FP8 离散点)
    # 使用 randn 生成正态分布数据，覆盖不同量级
    N_SAMPLES = 20
    raw_data = torch.randn(N_SAMPLES, device=device) * 5.0 # 扩大方差以覆盖更大范围
    
    # Round trip to FP8 to get valid discrete values
    input_fp8 = raw_data.to(torch.float8_e4m3fn)
    input_float = input_fp8.float().unsqueeze(1) # [Batch, 1]
    
    # SNN Encoding
    output = encoder(input_float) # [Batch, 1, 8]
    
    print(f"Testing {N_SAMPLES} random samples...\n")
    print(f"{'Value':<12} | {'SNN Output (S E M)':<25} | {'Reference (True FP8)':<25} | Result")
    print("-" * 80)
    
    passed = 0
    for i in range(N_SAMPLES):
        val = input_float[i].item()
        
        # SNN Result
        snn_bits = output[i].squeeze().int().tolist()
        
        # Reference Result
        ref_bits = get_fp8_e4m3_bits(input_float[i])
        
        # Compare
        is_match = (snn_bits == ref_bits)
        if is_match: passed += 1
        
        # Format for display
        snn_str = f"{snn_bits[0]} {''.join(map(str, snn_bits[1:5]))} {''.join(map(str, snn_bits[5:]))}"
        ref_str = f"{ref_bits[0]} {''.join(map(str, ref_bits[1:5]))} {''.join(map(str, ref_bits[5:]))}"
        res_str = "PASS" if is_match else "FAIL"
        
        print(f"{val:<12.6f} | {snn_str:<25} | {ref_str:<25} | {res_str}")

    print("-" * 80)
    print(f"Total Passed: {passed}/{N_SAMPLES}")
    
    if passed == N_SAMPLES:
        print("\nSUCCESS: SNN Encoder matches PyTorch FP8 binary representation perfectly.")
    else:
        print("\nWARNING: Some mismatches found. This might be due to Subnormal numbers or NaN/Inf handling differences.")

def test_fp8_subnormal():
    """专门测试 Subnormal 数的编码 (E=0, M!=0)"""
    print("\n=== Testing Subnormal Numbers ===")
    print("FP8 E4M3 Subnormal range: [2^-9, 2^-6) = [0.001953, 0.015625)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, 
        mantissa_bits=3,
        scan_integer_bits=10, 
        scan_decimal_bits=10
    ).to(device)
    
    # 所有 7 个 subnormal 值 (E=0, M=1~7)
    # Value = M/8 * 2^-6 = M * 2^-9
    subnormal_values = [
        (1 * 2**-9, "M=001"),  # 0.001953
        (2 * 2**-9, "M=010"),  # 0.003906
        (3 * 2**-9, "M=011"),  # 0.005859
        (4 * 2**-9, "M=100"),  # 0.007812
        (5 * 2**-9, "M=101"),  # 0.009766
        (6 * 2**-9, "M=110"),  # 0.011719
        (7 * 2**-9, "M=111"),  # 0.013672
    ]
    
    # 加上负数版本
    all_subnormals = subnormal_values + [(-v, f"-{d}") for v, d in subnormal_values]
    # 加上零
    all_subnormals.append((0.0, "Zero"))
    
    print(f"\n{'Value':<12} | {'Desc':<8} | {'SNN (S E M)':<15} | {'Ref (S E M)':<15} | Result")
    print("-" * 70)
    
    passed = 0
    for val, desc in all_subnormals:
        x = torch.tensor([val], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        x_fp8_f32 = x_fp8.float()
        
        # SNN
        pulse = encoder(x_fp8_f32.unsqueeze(0))
        snn_bits = pulse.squeeze().int().tolist()
        
        # Reference
        ref_bits = get_fp8_e4m3_bits(x_fp8_f32)
        
        is_match = (snn_bits == ref_bits)
        if is_match: passed += 1
        
        snn_str = f"{snn_bits[0]} {''.join(map(str, snn_bits[1:5]))} {''.join(map(str, snn_bits[5:]))}"
        ref_str = f"{ref_bits[0]} {''.join(map(str, ref_bits[1:5]))} {''.join(map(str, ref_bits[5:]))}"
        res = "PASS" if is_match else "FAIL"
        
        print(f"{val:<12.6f} | {desc:<8} | {snn_str:<15} | {ref_str:<15} | {res}")
    
    total = len(all_subnormals)
    print("-" * 70)
    print(f"Subnormal Test: {passed}/{total}")
    
    return passed == total


def test_fp8_boundary():
    """测试边界值: 最大值、最小Normal、零"""
    print("\n=== Testing Boundary Values ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, 
        mantissa_bits=3,
        scan_integer_bits=10, 
        scan_decimal_bits=10
    ).to(device)
    
    boundary_values = [
        (448.0, "Max (E=14,M=7)"),
        (2**-6, "Min Normal"),
        (2**-6 * 1.875, "Max Normal E=1"),
        (0.0, "Zero"),
        (-448.0, "Min (negative)"),
    ]
    
    print(f"\n{'Value':<12} | {'Desc':<18} | {'SNN (S E M)':<15} | {'Ref (S E M)':<15} | Result")
    print("-" * 80)
    
    passed = 0
    for val, desc in boundary_values:
        x = torch.tensor([val], device=device)
        x_fp8 = x.to(torch.float8_e4m3fn)
        x_fp8_f32 = x_fp8.float()
        
        pulse = encoder(x_fp8_f32.unsqueeze(0))
        snn_bits = pulse.squeeze().int().tolist()
        ref_bits = get_fp8_e4m3_bits(x_fp8_f32)
        
        is_match = (snn_bits == ref_bits)
        if is_match: passed += 1
        
        snn_str = f"{snn_bits[0]} {''.join(map(str, snn_bits[1:5]))} {''.join(map(str, snn_bits[5:]))}"
        ref_str = f"{ref_bits[0]} {''.join(map(str, ref_bits[1:5]))} {''.join(map(str, ref_bits[5:]))}"
        res = "PASS" if is_match else "FAIL"
        
        print(f"{val:<12.6f} | {desc:<18} | {snn_str:<15} | {ref_str:<15} | {res}")
    
    total = len(boundary_values)
    print("-" * 80)
    print(f"Boundary Test: {passed}/{total}")
    
    return passed == total


if __name__ == "__main__":
    test_fp8_random()
    sub_ok = test_fp8_subnormal()
    bound_ok = test_fp8_boundary()
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Subnormal: {'PASS' if sub_ok else 'FAIL'}")
    print(f"Boundary:  {'PASS' if bound_ok else 'FAIL'}")
