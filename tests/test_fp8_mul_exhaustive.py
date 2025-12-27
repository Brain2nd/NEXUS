import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import SpikeFP8Multiplier, PulseFloatingPointEncoder

def decode_snn_result(s_bits, e_bits, m_prod_bits):
    """
    解码 SNN 乘法器的原始输出 (未归一化)
    s_bits: [Batch, 1]
    e_bits: [Batch, 4] (Little Endian from output?) No, output logic says Big Endian.
            Wait, my SpikeFP8Multiplier output E_out is final_e.flip(1) -> Big Endian.
    m_prod_bits: [Batch, 8] (Big Endian from output)
    """
    # 1. Sign
    sign = (-1) ** s_bits.float()
    
    # 2. Exponent
    # E_out = Ea + Eb - 7
    # bits to int (Big Endian)
    e_val = torch.zeros_like(s_bits.float())
    for i in range(4):
        e_val += e_bits[:, i:i+1] * (2 ** (3-i))
        
    # 3. Mantissa Product
    # Input M was 1.xxx (treated as integer X / 8)
    # Product is (Xa/8) * (Xb/8) = Prod / 64
    m_int = torch.zeros_like(s_bits.float())
    for i in range(8):
        m_int += m_prod_bits[:, i:i+1] * (2 ** (7-i))
        
    m_val = m_int / 64.0
    
    # Result
    # Val = Sign * 2^(E - Bias) * M_val
    # But wait, E_out already subtracted Bias? 
    # In code: E_out = (Ea + Eb) + (-7). 
    # Normal float val = 2^(E_stored - Bias) * 1.M
    # Input A: 2^(Ea - 7) * Ma
    # Input B: 2^(Eb - 7) * Mb
    # A*B = 2^(Ea+Eb - 14) * (Ma*Mb)
    # My Adder Logic: E_out = Ea + Eb - 7.
    # So A*B should be: 2^(E_out - 7) * (Ma*Mb)
    # Yes, we need to subtract Bias (7) again during decoding.
    
    res = sign * (2 ** (e_val - 7)) * m_val
    return res

def test_exhaustive():
    print("=== Comprehensive Random Testing for SpikeFP8Multiplier ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare Models
    # Encoder: E4M3 (E=4, M=3)
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    multiplier = SpikeFP8Multiplier().to(device)
    
    # 2. Generate Random Data
    N = 100
    # Avoid subnormal/zero for now to test core logic (E=0 cases might fail due to simplified adder logic)
    # Generate values in range [0.1, 10] to keep E in valid range
    a_float = (torch.rand(N, 1, device=device) + 0.1) * (torch.randint(0, 2, (N, 1), device=device) * 2 - 1) * 4
    b_float = (torch.rand(N, 1, device=device) + 0.1) * (torch.randint(0, 2, (N, 1), device=device) * 2 - 1) * 4
    
    # 3. Encode Inputs
    # Encoder output: [N, 8] -> S(1) E(4) M(3)
    # But SpikeFP8Multiplier expects separate S, E, M logic inside?
    # No, forward(A, B) expects [Batch, 8] tensor.
    # Note: PulseFloatingPointEncoder returns [N, 1, 8]. Need squeeze.
    a_enc = encoder(a_float).squeeze(1)
    b_enc = encoder(b_float).squeeze(1)
    
    # 4. Run SNN Multiplier
    multiplier.reset()
    s_out, e_out, m_prod = multiplier(a_enc, b_enc)
    
    # 5. Verify Results - 100%位精确比较
    # Decode SNN result
    snn_res = decode_snn_result(s_out, e_out, m_prod)
    
    # 使用量化后的输入计算参考结果 (FP8量化后的精确乘积)
    expected_res = a_float * b_float  # a_float和b_float已经是量化后的值
    
    print(f"\nTesting {N} samples (100%位精确要求)...")
    print(f"{'A':<10} * {'B':<10} = {'Expected':<10} | {'SNN':<10} | {'Match':<10}")
    print("-" * 60)
    
    # 位精确比较
    import struct
    bit_match_count = 0
    for i in range(N):
        snn_bits = struct.unpack('>I', struct.pack('>f', snn_res[i].item()))[0]
        exp_bits = struct.unpack('>I', struct.pack('>f', expected_res[i].item()))[0]
        match = (snn_bits == exp_bits)
        if match:
            bit_match_count += 1
        if i < 10:  # Print first 10
            status = "✓" if match else "✗"
            print(f"{a_float[i].item():<10.4f} * {b_float[i].item():<10.4f} = {expected_res[i].item():<10.4f} | {snn_res[i].item():<10.4f} | {status}")
    
    accuracy = bit_match_count / N * 100
    print(f"\n位精确匹配: {bit_match_count}/{N} ({accuracy:.1f}%)")
    print(f"测试结果: {'通过 ✓' if accuracy == 100 else '失败 ✗ (要求100%位精确)'}")

if __name__ == "__main__":
    test_exhaustive()

