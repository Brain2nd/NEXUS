"""
实验一：极端边界情况压力测试 (Corner Case Stress Test)

目的：验证SNN FP8加法器的正确性，覆盖以下边界情况：
1. Zero Handling: 0+0, 0+(-0), (-0)+(-0)
2. Subnormal Numbers: 极小非规格化数相加
3. Boundary Crossing: 最大subnormal + 最小normal
4. Cancellation: x + (-x) = +0
5. Overflow: 最大正数 + 最大正数

端到端浮点验证：直接比较浮点数结果
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
from SNNTorch.atomic_ops import SpikeFP8Adder_Spatial, PulseFloatingPointEncoder
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder


def test_single_case(encoder, decoder, adder, name, a_float, b_float, device):
    """测试单个用例（端到端浮点验证）"""
    # 量化到 FP8
    a_fp8 = torch.tensor([a_float], device=device).to(torch.float8_e4m3fn)
    b_fp8 = torch.tensor([b_float], device=device).to(torch.float8_e4m3fn)
    
    # PyTorch 参考
    ref_result = (a_fp8.float() + b_fp8.float()).to(torch.float8_e4m3fn).float().item()
    
    # SNN 计算
    encoder.reset()
    a_pulse = encoder(a_fp8.float())
    encoder.reset()
    b_pulse = encoder(b_fp8.float())
    
    adder.reset()
    result_pulse = adder(a_pulse.squeeze(0), b_pulse.squeeze(0))
    
    decoder.reset()
    snn_result = decoder(result_pulse.unsqueeze(0)).item()
    
    # 直接比较浮点数
    match = abs(snn_result - ref_result) < 1e-6 or snn_result == ref_result
    
    return {
        'name': name,
        'a_float': a_fp8.float().item(),
        'b_float': b_fp8.float().item(),
        'snn_result': snn_result,
        'ref_result': ref_result,
        'match': match
    }


def run_corner_case_tests(encoder, decoder, adder, device):
    """运行所有边界测试用例"""
    print(f"\n{'='*70}")
    print(f"边界测试: SpikeFP8Adder_Spatial (端到端浮点验证)")
    print(f"{'='*70}")
    
    results = []
    
    # ========== 1. Zero Handling ==========
    print("\n--- 1. Zero Handling ---")
    zero_cases = [
        ("0 + 0", 0.0, 0.0),
        ("(+0) + (-0)", 0.0, -0.0),
        ("(-0) + (-0)", -0.0, -0.0),
        ("(-0) + (+0)", -0.0, 0.0),
    ]
    for name, a, b in zero_cases:
        r = test_single_case(encoder, decoder, adder, name, a, b, device)
        results.append(r)
        status = "✓" if r['match'] else "✗"
        print(f"  {status} {name}: A={r['a_float']:.4f}, B={r['b_float']:.4f}")
        print(f"      SNN={r['snn_result']:.6f}, Ref={r['ref_result']:.6f}")
    
    # ========== 2. Subnormal Numbers ==========
    print("\n--- 2. Subnormal Numbers ---")
    # FP8 E4M3 subnormal: 2^-6 * (M/8), M in [1, 7]
    min_sub = 2**-6 * (1/8)  # 0.001953125
    subnormal_cases = [
        ("min_sub + min_sub", min_sub, min_sub),
        ("sub + sub", min_sub * 2, min_sub * 3),
        ("max_sub + min_sub", min_sub * 7, min_sub),
        ("(-sub) + sub", -min_sub, min_sub),
        ("(-sub) + (-sub)", -min_sub * 2, -min_sub * 3),
    ]
    for name, a, b in subnormal_cases:
        r = test_single_case(encoder, decoder, adder, name, a, b, device)
        results.append(r)
        status = "✓" if r['match'] else "✗"
        print(f"  {status} {name}: A={r['a_float']:.6f}, B={r['b_float']:.6f}")
        print(f"      SNN={r['snn_result']:.6f}, Ref={r['ref_result']:.6f}")
    
    # ========== 3. Boundary Crossing ==========
    print("\n--- 3. Boundary Crossing (Subnormal ↔ Normal) ---")
    max_sub = min_sub * 7  # 最大 subnormal
    min_norm = 2**-6  # 最小 normal: 2^-6 * 1.0 = 0.015625
    boundary_cases = [
        ("max_sub + min_norm", max_sub, min_norm),
        ("max_sub + max_sub", max_sub, max_sub),
        ("min_norm + min_norm", min_norm, min_norm),
        ("min_norm - max_sub", min_norm, -max_sub),
    ]
    for name, a, b in boundary_cases:
        r = test_single_case(encoder, decoder, adder, name, a, b, device)
        results.append(r)
        status = "✓" if r['match'] else "✗"
        print(f"  {status} {name}: A={r['a_float']:.6f}, B={r['b_float']:.6f}")
        print(f"      SNN={r['snn_result']:.6f}, Ref={r['ref_result']:.6f}")
    
    # ========== 4. Exact Cancellation ==========
    print("\n--- 4. Exact Cancellation (x + (-x)) ---")
    cancel_values = [0.5, 1.0, 2.0, 4.0, 8.0, 0.25, 0.125]
    for val in cancel_values:
        name = f"{val} + (-{val})"
        r = test_single_case(encoder, decoder, adder, name, val, -val, device)
        results.append(r)
        status = "✓" if r['match'] else "✗"
        is_zero = abs(r['snn_result']) < 1e-10
        print(f"  {status} {name}: Result={r['snn_result']:.6f}, IsZero={is_zero}")
    
    # 随机抵消测试
    print("\n  --- 随机抵消测试 (100组) ---")
    torch.manual_seed(42)
    cancel_pass = 0
    
    for _ in range(100):
        val = torch.randn(1).item() * 5
        val_fp8 = torch.tensor([val]).to(torch.float8_e4m3fn).float().item()
        
        r = test_single_case(encoder, decoder, adder, "cancel", val_fp8, -val_fp8, device)
        if r['match']:
            cancel_pass += 1
    
    print(f"  随机抵消: {cancel_pass}/100 通过")
    
    # ========== 5. Overflow ==========
    print("\n--- 5. Overflow / Saturation ---")
    max_fp8 = 448.0  # FP8 E4M3 最大值
    overflow_cases = [
        ("max + max", max_fp8, max_fp8),
        ("max + half_max", max_fp8, max_fp8 / 2),
        ("(-max) + (-max)", -max_fp8, -max_fp8),
        ("large + large", 128.0, 128.0),
    ]
    for name, a, b in overflow_cases:
        r = test_single_case(encoder, decoder, adder, name, a, b, device)
        results.append(r)
        status = "✓" if r['match'] else "✗"
        print(f"  {status} {name}: A={r['a_float']:.2f}, B={r['b_float']:.2f}")
        print(f"      SNN={r['snn_result']:.2f}, Ref={r['ref_result']:.2f}")
    
    # ========== 统计结果 ==========
    print(f"\n{'='*70}")
    passed = sum(1 for r in results if r['match'])
    total = len(results)
    print(f"总计: {passed}/{total} 通过")
    
    if passed < total:
        print("\n失败用例详情:")
        for r in results:
            if not r['match']:
                print(f"  {r['name']}: SNN={r['snn_result']:.6f}, Ref={r['ref_result']:.6f}")
    
    return passed, total, cancel_pass


def main():
    print("="*70)
    print("实验一：极端边界情况压力测试 (端到端浮点验证)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 初始化组件
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    adder = SpikeFP8Adder_Spatial().to(device)
    
    # 运行测试
    passed, total, cancel_pass = run_corner_case_tests(encoder, decoder, adder, device)
    
    # ========== 最终汇总 ==========
    print("\n" + "="*70)
    print("实验一 最终结果汇总")
    print("="*70)
    
    print(f"\n| 测试类别 | 结果 |")
    print(f"|----------|------|")
    print(f"| 边界用例 | {passed}/{total} |")
    print(f"| 随机抵消 | {cancel_pass}/100 |")
    
    all_pass = (passed == total and cancel_pass == 100)
    
    print("\n结论:")
    if all_pass:
        print("  ✓ 所有边界测试通过！")
    else:
        print("  ✗ 存在测试失败，需要检查相关实现")
    
    print("="*70)
    
    return all_pass


if __name__ == "__main__":
    main()
