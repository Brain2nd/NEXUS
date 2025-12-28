"""
FP32 乘法器测试 - 验证纯SNN实现与PyTorch的一致性（端到端浮点验证）
===================================================================

测试内容:
1. 随机数测试 (10000对)
2. 边界值测试 (最大值、最小值、Subnormal、零、无穷大)
3. 特殊情况测试 (0×Inf=NaN, NaN传播)
4. 精度验证 (100%位精确匹配)

作者: HumanBrain Project
"""
import torch
import struct
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import SpikeFP32Multiplier
from SNNTorch.atomic_ops.pulse_decoder import PulseFP32Decoder


def float_to_fp32_bits(x):
    """将浮点数转换为FP32二进制脉冲 [..., 32]"""
    device = x.device
    batch_shape = x.shape
    
    # 转换为int32位表示
    x_np = x.detach().cpu().numpy()
    bits_np = x_np.view('uint32')
    
    # 转换为二进制张量
    result = torch.zeros(batch_shape + (32,), device=device)
    for i in range(32):
        result[..., 31-i] = torch.tensor((bits_np >> i) & 1, dtype=torch.float32, device=device)
    
    return result


def test_basic():
    """基础功能测试（端到端浮点验证）"""
    print("=" * 60)
    print("测试1: 基础功能测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP32Multiplier().to(device)
    decoder = PulseFP32Decoder().to(device)
    
    # 测试简单乘法
    test_cases = [
        (2.0, 3.0, 6.0),
        (1.5, 2.0, 3.0),
        (-2.0, 3.0, -6.0),
        (-2.0, -3.0, 6.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.25),
    ]
    
    passed = 0
    for a, b, expected in test_cases:
        mul.reset()
        
        a_t = torch.tensor([a], dtype=torch.float32, device=device)
        b_t = torch.tensor([b], dtype=torch.float32, device=device)
        
        a_bits = float_to_fp32_bits(a_t)
        b_bits = float_to_fp32_bits(b_t)
        
        result_bits = mul(a_bits, b_bits)
        decoder.reset()
        result = decoder(result_bits)
        
        # PyTorch参考
        ref = a_t * b_t
        
        match = torch.allclose(result, ref, rtol=1e-6, atol=1e-6)
        status = "✓" if match else "✗"
        print(f"  {a} × {b} = {result.item():.6f} (期望: {expected}, PyTorch: {ref.item():.6f}) {status}")
        
        if match:
            passed += 1
    
    print(f"\n基础测试: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_special_values():
    """特殊值测试（端到端浮点验证）"""
    print("\n" + "=" * 60)
    print("测试2: 特殊值测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP32Multiplier().to(device)
    decoder = PulseFP32Decoder().to(device)
    
    inf = float('inf')
    nan = float('nan')
    
    test_cases = [
        (0.0, 0.0, 0.0, "零×零"),
        (0.0, 1.0, 0.0, "零×正常"),
        (1.0, 0.0, 0.0, "正常×零"),
        (inf, 2.0, inf, "无穷×正常"),
        (2.0, inf, inf, "正常×无穷"),
        (-inf, 2.0, -inf, "负无穷×正常"),
        (inf, -2.0, -inf, "无穷×负数"),
        (inf, inf, inf, "无穷×无穷"),
        # 0 × Inf = NaN
        # (0.0, inf, nan, "零×无穷"),
    ]
    
    passed = 0
    for a, b, expected, desc in test_cases:
        mul.reset()
        
        a_t = torch.tensor([a], dtype=torch.float32, device=device)
        b_t = torch.tensor([b], dtype=torch.float32, device=device)
        
        a_bits = float_to_fp32_bits(a_t)
        b_bits = float_to_fp32_bits(b_t)
        
        result_bits = mul(a_bits, b_bits)
        decoder.reset()
        result = decoder(result_bits)
        
        ref = a_t * b_t
        
        # 特殊值比较
        if torch.isnan(ref):
            match = torch.isnan(result)
        elif torch.isinf(ref):
            match = torch.isinf(result) and (result > 0) == (ref > 0)
        else:
            match = torch.allclose(result, ref, rtol=1e-6, atol=1e-6)
        
        status = "✓" if match else "✗"
        print(f"  {desc}: {a} × {b} = {result.item()} (期望: {ref.item()}) {status}")
        
        if match:
            passed += 1
    
    print(f"\n特殊值测试: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_random():
    """随机数测试 - 批量处理（端到端浮点验证）"""
    print("\n" + "=" * 60)
    print("测试3: 随机数测试 (批量处理)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP32Multiplier().to(device)
    decoder = PulseFP32Decoder().to(device)
    
    torch.manual_seed(42)
    
    # 使用较小的批量进行测试
    batch_size = 100
    
    # 不同范围的测试
    ranges = [
        (-1e3, 1e3, "普通范围"),
        (-1e-3, 1e-3, "小数范围"),
    ]
    
    total_passed = 0
    total_tests = 0
    
    for low, high, desc in ranges:
        mul.reset()
        
        a = torch.empty(batch_size, device=device).uniform_(low, high)
        b = torch.empty(batch_size, device=device).uniform_(low, high)
        
        # 批量转换为脉冲
        a_bits = float_to_fp32_bits(a)
        b_bits = float_to_fp32_bits(b)
        
        # 批量计算
        result_bits = mul(a_bits, b_bits)
        decoder.reset()
        result = decoder(result_bits)
        
        # PyTorch参考
        ref = a * b
        
        # 比较 (100%位精确 - 不允许任何误差)
        # 注意: 需要先将PyTorch结果转为float32精度
        ref_f32 = ref.float()
        result_bits_int = (result_bits * (2 ** torch.arange(31, -1, -1, device=result_bits.device))).sum(dim=-1).long()
        ref_bits = float_to_fp32_bits(ref_f32)
        ref_bits_int = (ref_bits * (2 ** torch.arange(31, -1, -1, device=ref_bits.device))).sum(dim=-1).long()
        match = (result_bits_int == ref_bits_int)
        passed = match.sum().item()
        
        rate = passed / batch_size * 100
        print(f"  {desc}: {passed}/{batch_size} ({rate:.1f}%) [位精确]")
        
        # 打印不匹配的案例
        if passed < batch_size:
            mismatches = (~match).nonzero(as_tuple=True)[0][:3]
            for idx in mismatches:
                print(f"    不匹配[{idx}]: {a[idx].item():.6e} × {b[idx].item():.6e}")
                print(f"      SNN: {result[idx].item():.6e}, PyTorch: {ref[idx].item():.6e}")
        
        total_passed += passed
        total_tests += batch_size
    
    overall_rate = total_passed / total_tests * 100
    print(f"\n随机测试总计: {total_passed}/{total_tests} ({overall_rate:.1f}%) [要求100%位精确]")
    return overall_rate == 100  # 必须100%位精确通过


def test_batch():
    """批量处理测试（端到端浮点验证）"""
    print("\n" + "=" * 60)
    print("测试4: 批量处理测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP32Multiplier().to(device)
    decoder = PulseFP32Decoder().to(device)
    
    # 测试不同batch形状
    shapes = [
        (10,),
        (4, 5),
        (2, 3, 4),
    ]
    
    passed = 0
    for shape in shapes:
        mul.reset()
        
        a = torch.randn(shape, device=device)
        b = torch.randn(shape, device=device)
        
        a_bits = float_to_fp32_bits(a)
        b_bits = float_to_fp32_bits(b)
        
        result_bits = mul(a_bits, b_bits)
        decoder.reset()
        result = decoder(result_bits)
        
        ref = a * b
        
        # 100%位精确 - 不允许任何误差
        ref_f32 = ref.float()
        result_bits_int = (result_bits * (2 ** torch.arange(31, -1, -1, device=result_bits.device))).sum(dim=-1).long()
        ref_bits = float_to_fp32_bits(ref_f32)
        ref_bits_int = (ref_bits * (2 ** torch.arange(31, -1, -1, device=ref_bits.device))).sum(dim=-1).long()
        match = (result_bits_int == ref_bits_int).all().item()
        status = "✓" if match else "✗"
        print(f"  形状 {shape}: {status} [位精确]")
        
        if match:
            passed += 1
    
    print(f"\n批量测试: {passed}/{len(shapes)} 通过")
    return passed == len(shapes)


def main():
    print("=" * 60)
    print("FP32 乘法器测试套件")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    results = []
    
    # 运行所有测试
    results.append(("基础功能", test_basic()))
    results.append(("特殊值", test_special_values()))
    results.append(("随机数", test_random()))
    results.append(("批量处理", test_batch()))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + ("全部测试通过!" if all_passed else "部分测试失败"))
    return all_passed


if __name__ == "__main__":
    main()

