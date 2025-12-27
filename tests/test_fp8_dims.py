import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import PulseFloatingPointEncoder, SpikeFP8Multiplier, SpikeFP8Adder_Spatial

def get_fp8_bits(x_float):
    """获取FP8的位表示"""
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    x_uint8 = x_fp8.view(torch.uint8)
    val = x_uint8.item()
    s = (val >> 7) & 1
    e_val = (val >> 3) & 0xF
    m_val = val & 0x7
    s_bits = [s]
    e_bits = [(e_val >> (3-i)) & 1 for i in range(4)]
    m_bits = [(m_val >> (2-i)) & 1 for i in range(3)]
    return s_bits + e_bits + m_bits

def bits_to_str(bits):
    return f"{bits[0]} {''.join(map(str, bits[1:5]))} {''.join(map(str, bits[5:]))}"

def reset_all(encoder, mul, add):
    encoder.sign_node.reset()
    encoder.binary_encoder.reset()
    mul.reset()

def test_shape(encoder, mul, add, shape, device, test_name):
    """测试 - 包含正负数，位级精确验证"""
    print(f"\n{'='*70}")
    print(f"测试: {test_name} | Shape: {shape}")
    print('='*70)
    
    # 生成正负数混合，round trip到FP8
    # 使用FP8完整范围
    sign_a = (torch.rand(shape, device=device) > 0.5).float() * 2 - 1  # -1 or 1
    sign_b = (torch.rand(shape, device=device) > 0.5).float() * 2 - 1
    raw_a = sign_a * (torch.rand(shape, device=device) * 4.0 + 0.1)  # ±[0.1, 4.1]
    raw_b = sign_b * (torch.rand(shape, device=device) * 2.5 + 0.1)  # ±[0.1, 2.6]
    
    fp8_a = raw_a.to(torch.float8_e4m3fn).float()
    fp8_b = raw_b.to(torch.float8_e4m3fn).float()
    
    expected_mul = (fp8_a * fp8_b).to(torch.float8_e4m3fn).float()
    expected_add = (fp8_a + fp8_b).to(torch.float8_e4m3fn).float()
    
    print(f"Input A (sample): {fp8_a.flatten()[:4].tolist()}")
    print(f"Input B (sample): {fp8_b.flatten()[:4].tolist()}")
    
    # 编码
    reset_all(encoder, mul, add)
    enc_a = encoder(fp8_a.unsqueeze(-1))
    
    reset_all(encoder, mul, add)
    enc_b = encoder(fp8_b.unsqueeze(-1))
    
    print(f"Encoded shape: {enc_a.shape}")
    
    # === 乘法测试 ===
    reset_all(encoder, mul, add)
    out_mul = mul(enc_a, enc_b)
    
    flat_a = fp8_a.flatten()
    flat_b = fp8_b.flatten()
    flat_mul_expected = expected_mul.flatten()
    flat_out_mul = out_mul.reshape(-1, 8)
    
    mul_passed = 0
    mul_total = min(flat_a.numel(), 50)
    mul_failures = []
    
    for i in range(mul_total):
        snn_bits = flat_out_mul[i].int().tolist()
        ref_bits = get_fp8_bits(torch.tensor(flat_mul_expected[i].item()))
        
        if snn_bits == ref_bits:
            mul_passed += 1
        else:
            if len(mul_failures) < 5:
                mul_failures.append({
                    'a': flat_a[i].item(),
                    'b': flat_b[i].item(),
                    'expected': flat_mul_expected[i].item(),
                    'snn': bits_to_str(snn_bits),
                    'ref': bits_to_str(ref_bits)
                })
    
    mul_pass = (mul_passed == mul_total)
    print(f"\n[乘法] 位级精确匹配: {mul_passed}/{mul_total}")
    if mul_failures:
        print(f"  失败样例:")
        for f in mul_failures:
            print(f"    {f['a']:.4f} × {f['b']:.4f} = {f['expected']:.4f}")
            print(f"      SNN: {f['snn']} | REF: {f['ref']}")
    print(f"  结果: {'PASS' if mul_pass else 'FAIL'}")
    
    # === 加法测试 ===
    reset_all(encoder, mul, add)
    out_add = add(enc_a, enc_b)
    
    flat_add_expected = expected_add.flatten()
    flat_out_add = out_add.reshape(-1, 8)
    
    add_passed = 0
    add_total = min(flat_a.numel(), 50)
    add_failures = []
    
    for i in range(add_total):
        snn_bits = flat_out_add[i].int().tolist()
        ref_bits = get_fp8_bits(torch.tensor(flat_add_expected[i].item()))
        
        if snn_bits == ref_bits:
            add_passed += 1
        else:
            if len(add_failures) < 5:
                add_failures.append({
                    'a': flat_a[i].item(),
                    'b': flat_b[i].item(),
                    'expected': flat_add_expected[i].item(),
                    'snn': bits_to_str(snn_bits),
                    'ref': bits_to_str(ref_bits)
                })
    
    add_pass = (add_passed == add_total)
    print(f"\n[加法] 位级精确匹配: {add_passed}/{add_total}")
    if add_failures:
        print(f"  失败样例:")
        for f in add_failures:
            print(f"    {f['a']:.4f} + {f['b']:.4f} = {f['expected']:.4f}")
            print(f"      SNN: {f['snn']} | REF: {f['ref']}")
    print(f"  结果: {'PASS' if add_pass else 'FAIL'}")
    
    return mul_pass and add_pass

def main():
    print("="*70)
    print("FP8 SNN 组件测试 (正负数混合，位级精确验证)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # scan_decimal_bits=16 以支持FP8的完整范围（最小正规数 2^(-6)）
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=8, scan_decimal_bits=16
    ).to(device)
    mul = SpikeFP8Multiplier().to(device)
    add = SpikeFP8Adder_Spatial().to(device)
    
    test_cases = [
        ((4,), "1D: [4]"),
        ((2, 3), "2D: [2, 3]"),
        ((2, 3, 4), "3D: [2, 3, 4]"),
        ((2, 2, 2, 2), "4D: [2, 2, 2, 2]"),
    ]
    
    results = []
    for shape, name in test_cases:
        passed = test_shape(encoder, mul, add, shape, device, name)
        results.append((name, passed))
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print("="*70)
    if all_pass:
        print("所有测试通过！(位级精确)")
    else:
        print("存在失败的测试！需要修复乘法/加法组件")
    print("="*70)

if __name__ == "__main__":
    main()
