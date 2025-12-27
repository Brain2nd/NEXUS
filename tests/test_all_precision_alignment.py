"""
多精度对齐测试 (Multi-Precision Alignment Test)
==============================================

验证 SNN FP8 Linear 层与 PyTorch 参考的对齐。

端到端浮点验证：使用解码器解码后直接比较浮点数

运行方式
--------
```bash
python SNNTorch/tests/test_all_precision_alignment.py
```
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_MultiPrecision
)
from SNNTorch.atomic_ops.pulse_decoder import (
    PulseFloatingPointDecoder, PulseFP16Decoder, PulseFP32Decoder
)


def pytorch_fp8_accumulate_reference(x_fp8_f32, w_fp8_f32):
    """PyTorch FP8 Linear 参考实现（输出FP8）
    
    注意：这是FP8累加模式，每次加法都舍入到FP8
    """
    batch_size = x_fp8_f32.shape[0]
    out_features = w_fp8_f32.shape[0]
    in_features = w_fp8_f32.shape[1]
    device = x_fp8_f32.device
    
    results = torch.zeros(batch_size, out_features, device=device, dtype=torch.float32)
    
    for b in range(batch_size):
        for o in range(out_features):
            products = x_fp8_f32[b] * w_fp8_f32[o]
            products_fp8 = products.to(torch.float8_e4m3fn)
            
            acc = products_fp8[0]
            for i in range(1, in_features):
                sum_tmp = acc.float() + products_fp8[i].float()
                acc = sum_tmp.to(torch.float8_e4m3fn)
            
            results[b, o] = acc.float()
    
    return results


def pytorch_fp16_accumulate_reference(x_fp8_f32, w_fp8_f32):
    """PyTorch FP16 Linear 参考实现（输出FP16）"""
    # PyTorch FP16 Linear: FP32 matmul -> FP16
    y_fp32 = x_fp8_f32 @ w_fp8_f32.T
    y_fp16 = y_fp32.to(torch.float16)
    return y_fp16.float()  # 转回float32用于比较


def pytorch_fp32_accumulate_reference(x_fp8_f32, w_fp8_f32):
    """PyTorch FP32 Linear 参考实现（输出FP32）"""
    # PyTorch FP32 Linear: FP32 matmul -> FP32
    y_fp32 = x_fp8_f32 @ w_fp8_f32.T
    return y_fp32


def test_fp8_accumulate_alignment():
    """测试 FP8 累加模式（端到端浮点验证）"""
    print("\n" + "="*70)
    print("测试 FP8 累加模式（端到端浮点验证）")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFloatingPointDecoder().to(device)
    
    total_match = 0
    total_count = 0
    
    for (in_f, out_f, batch, seed, scale) in [
        (4, 2, 20, 42, 0.5),
        (8, 4, 50, 123, 0.3),
        (16, 8, 100, 456, 0.4)
    ]:
        torch.manual_seed(seed)
        
        x_float = torch.randn(batch, in_f, device=device) * scale
        w_float = torch.randn(out_f, in_f, device=device) * scale
        
        x_fp8 = x_float.to(torch.float8_e4m3fn)
        w_fp8 = w_float.to(torch.float8_e4m3fn)
        x_fp8_f32 = x_fp8.float()
        w_fp8_f32 = w_fp8.float()
        
        # PyTorch FP8 累加参考
        y_ref = pytorch_fp8_accumulate_reference(x_fp8_f32, w_fp8_f32)
        
        # SNN FP8 Linear（输出FP8脉冲）
        encoder.reset()
        x_pulse = encoder(x_fp8_f32)
        snn = SpikeFP8Linear_MultiPrecision(
            in_f, out_f, 
            accum_precision='fp8', 
            mode='sequential'
        ).to(device)
        snn.set_weight_from_float(w_fp8_f32, encoder)
        snn.reset()
        y_snn_pulse = snn(x_pulse)  # 输出FP8脉冲 [..., 8]
        
        # 验证输出维度
        assert y_snn_pulse.shape[-1] == 8, f"FP8模式应输出8位脉冲，实际为{y_snn_pulse.shape[-1]}"
        
        decoder.reset()
        y_snn = decoder(y_snn_pulse)  # 解码回FP8浮点数
        
        # 直接比较浮点数
        match = torch.isclose(y_snn, y_ref, rtol=1e-5, atol=1e-6) | (y_snn == y_ref)
        match_count = match.sum().item()
        total_count_local = match.numel()
        
        total_match += match_count
        total_count += total_count_local
        
        rate = match_count / total_count_local * 100
        print(f"  in={in_f}, out={out_f}, batch={batch}: {rate:.1f}% ({match_count}/{total_count_local})")
    
    overall_rate = total_match / total_count * 100
    print(f"\nFP8 累加总体对齐率: {overall_rate:.2f}% ({total_match}/{total_count})")
    return overall_rate >= 99


def test_fp16_accumulate_alignment():
    """测试 FP16 累加模式（端到端浮点验证）"""
    print("\n" + "="*70)
    print("测试 FP16 累加模式（端到端浮点验证）")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFP16Decoder().to(device)  # 使用FP16解码器
    
    total_match = 0
    total_count = 0
    
    for (in_f, out_f, batch, seed, scale) in [
        (4, 2, 20, 42, 0.5),
        (8, 4, 50, 123, 0.3),
        (16, 8, 100, 456, 0.4)
    ]:
        torch.manual_seed(seed)
        
        x_float = torch.randn(batch, in_f, device=device) * scale
        w_float = torch.randn(out_f, in_f, device=device) * scale
        
        x_fp8 = x_float.to(torch.float8_e4m3fn)
        w_fp8 = w_float.to(torch.float8_e4m3fn)
        x_fp8_f32 = x_fp8.float()
        w_fp8_f32 = w_fp8.float()
        
        # PyTorch FP16 Linear 参考（输出FP16）
        y_ref = pytorch_fp16_accumulate_reference(x_fp8_f32, w_fp8_f32)
        
        # SNN FP16 Linear（输出FP16脉冲）
        encoder.reset()
        x_pulse = encoder(x_fp8_f32)
        snn = SpikeFP8Linear_MultiPrecision(
            in_f, out_f, 
            accum_precision='fp16', 
            mode='sequential'
        ).to(device)
        snn.set_weight_from_float(w_fp8_f32, encoder)
        snn.reset()
        y_snn_pulse = snn(x_pulse)  # 输出FP16脉冲 [..., 16]
        
        # 验证输出维度
        assert y_snn_pulse.shape[-1] == 16, f"FP16模式应输出16位脉冲，实际为{y_snn_pulse.shape[-1]}"
        
        decoder.reset()
        y_snn = decoder(y_snn_pulse)  # 解码回FP16浮点数
        
        match = torch.isclose(y_snn, y_ref, rtol=1e-5, atol=1e-6) | (y_snn == y_ref)
        match_count = match.sum().item()
        total_count_local = match.numel()
        
        total_match += match_count
        total_count += total_count_local
        
        rate = match_count / total_count_local * 100
        print(f"  in={in_f}, out={out_f}, batch={batch}: {rate:.1f}% ({match_count}/{total_count_local})")
    
    overall_rate = total_match / total_count * 100
    print(f"\nFP16 累加总体对齐率: {overall_rate:.2f}% ({total_match}/{total_count})")
    return overall_rate >= 99


def test_fp32_accumulate_alignment():
    """测试 FP32 累加模式（端到端浮点验证）"""
    print("\n" + "="*70)
    print("测试 FP32 累加模式（端到端浮点验证）")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFP32Decoder().to(device)  # 使用FP32解码器
    
    total_match = 0
    total_count = 0
    
    for (in_f, out_f, batch, seed, scale) in [
        (4, 2, 20, 42, 0.5),
        (8, 4, 50, 123, 0.3),
        (16, 8, 100, 456, 0.4)
    ]:
        torch.manual_seed(seed)
        
        x_float = torch.randn(batch, in_f, device=device) * scale
        w_float = torch.randn(out_f, in_f, device=device) * scale
        
        x_fp8 = x_float.to(torch.float8_e4m3fn)
        w_fp8 = w_float.to(torch.float8_e4m3fn)
        x_fp8_f32 = x_fp8.float()
        w_fp8_f32 = w_fp8.float()
        
        # PyTorch FP32 Linear 参考（输出FP32）
        y_ref = pytorch_fp32_accumulate_reference(x_fp8_f32, w_fp8_f32)
        
        # SNN FP32 Linear（输出FP32脉冲）
        encoder.reset()
        x_pulse = encoder(x_fp8_f32)
        snn = SpikeFP8Linear_MultiPrecision(
            in_f, out_f, 
            accum_precision='fp32', 
            mode='sequential'
        ).to(device)
        snn.set_weight_from_float(w_fp8_f32, encoder)
        snn.reset()
        y_snn_pulse = snn(x_pulse)  # 输出FP32脉冲 [..., 32]
        
        # 验证输出维度
        assert y_snn_pulse.shape[-1] == 32, f"FP32模式应输出32位脉冲，实际为{y_snn_pulse.shape[-1]}"
        
        decoder.reset()
        y_snn = decoder(y_snn_pulse)  # 解码回FP32浮点数
        
        match = torch.isclose(y_snn, y_ref, rtol=1e-5, atol=1e-6) | (y_snn == y_ref)
        match_count = match.sum().item()
        total_count_local = match.numel()
        
        total_match += match_count
        total_count += total_count_local
        
        rate = match_count / total_count_local * 100
        print(f"  in={in_f}, out={out_f}, batch={batch}: {rate:.1f}% ({match_count}/{total_count_local})")
    
    overall_rate = total_match / total_count * 100
    print(f"\nFP32 累加总体对齐率: {overall_rate:.2f}% ({total_match}/{total_count})")
    return overall_rate >= 99


def main():
    print("="*70)
    print("多精度对齐测试（端到端浮点验证）")
    print("="*70)
    
    fp8_pass = test_fp8_accumulate_alignment()
    fp16_pass = test_fp16_accumulate_alignment()
    fp32_pass = test_fp32_accumulate_alignment()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    all_pass = fp8_pass and fp16_pass and fp32_pass
    
    if all_pass:
        print("\n✓ 成功！所有累加精度均实现高对齐")
    else:
        print("\n⚠ 部分测试未达到预期")
        if not fp8_pass:
            print("  - FP8 累加未通过")
        if not fp16_pass:
            print("  - FP16 累加未通过")
        if not fp32_pass:
            print("  - FP32 累加未通过")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
