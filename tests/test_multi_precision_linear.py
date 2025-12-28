"""
测试多精度累加 Linear 层与朴素 PyTorch linear 的对齐情况

比较目标：PyTorch 的 nn.Linear 使用 FP32 累加后转回 FP8

端到端浮点验证：使用框架解码器解码后比较浮点数
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast,
    SpikeFP8Linear_MultiPrecision
)
from SNNTorch.atomic_ops.pulse_decoder import (
    PulseFloatingPointDecoder, PulseFP16Decoder, PulseFP32Decoder
)


def test_precision_comparison():
    """比较不同累加精度与 PyTorch 的对齐情况（端到端浮点验证）"""
    print("="*70)
    print("多精度累加 Linear 层测试（端到端浮点验证）")
    print("比较目标: PyTorch FP8 matmul (FP32累加)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 编码器和解码器
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder_fp8 = PulseFloatingPointDecoder().to(device)
    decoder_fp16 = PulseFP16Decoder().to(device)
    decoder_fp32 = PulseFP32Decoder().to(device)
    
    # 测试配置
    in_features = 4
    out_features = 2
    batch_size = 20
    
    torch.manual_seed(42)
    
    # 生成 FP8 数据
    x_float = torch.randn(batch_size, in_features, device=device) * 0.5
    w_float = torch.randn(out_features, in_features, device=device) * 0.5
    
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    print(f"\n配置: in={in_features}, out={out_features}, batch={batch_size}")
    
    # PyTorch 参考: FP32 matmul
    y_pytorch_fp32 = x_fp8_f32 @ w_fp8_f32.T
    
    print(f"\n--- PyTorch 参考 (FP32累加) ---")
    print(f"y_pytorch 部分输出: {y_pytorch_fp32[:3, :].tolist()}")
    
    # 编码输入
    x_pulse = encoder(x_fp8_f32)
    
    # 测试结果字典
    results = {}
    
    # 1. FP8 Sequential 累加
    print(f"\n--- SNN FP8 Sequential 累加 ---")
    try:
        snn_fp8_seq = SpikeFP8Linear_MultiPrecision(
            in_features, out_features, 
            accum_precision='fp8', 
            mode='sequential'
        ).to(device)
        snn_fp8_seq.set_weight_from_float(w_fp8_f32, encoder)
        snn_fp8_seq.reset()
        y_fp8_seq_pulse = snn_fp8_seq(x_pulse)
        
        decoder_fp8.reset()
        y_fp8_seq = decoder_fp8(y_fp8_seq_pulse)
        
        # 比较：PyTorch FP32 结果转 FP8 后比较
        y_ref = y_pytorch_fp32.to(torch.float8_e4m3fn).float()
        match = torch.isclose(y_fp8_seq, y_ref, rtol=1e-5, atol=1e-6) | (y_fp8_seq == y_ref)
        rate_fp8_seq = match.sum().item() / match.numel() * 100
        print(f"与 PyTorch 匹配率: {rate_fp8_seq:.1f}%")
        results['fp8_seq'] = rate_fp8_seq
    except Exception as e:
        print(f"失败: {e}")
        results['fp8_seq'] = 0
    
    # 2. FP8 Tree 累加
    print(f"\n--- SNN FP8 Tree 累加 ---")
    try:
        snn_fp8_tree = SpikeFP8Linear_MultiPrecision(
            in_features, out_features, 
            accum_precision='fp8', 
            mode='tree'
        ).to(device)
        snn_fp8_tree.set_weight_from_float(w_fp8_f32, encoder)
        snn_fp8_tree.reset()
        y_fp8_tree_pulse = snn_fp8_tree(x_pulse)
        
        decoder_fp8.reset()
        y_fp8_tree = decoder_fp8(y_fp8_tree_pulse)
        
        y_ref = y_pytorch_fp32.to(torch.float8_e4m3fn).float()
        match = torch.isclose(y_fp8_tree, y_ref, rtol=1e-5, atol=1e-6) | (y_fp8_tree == y_ref)
        rate_fp8_tree = match.sum().item() / match.numel() * 100
        print(f"与 PyTorch 匹配率: {rate_fp8_tree:.1f}%")
        results['fp8_tree'] = rate_fp8_tree
    except Exception as e:
        print(f"失败: {e}")
        results['fp8_tree'] = 0
    
    # 3. FP16 累加
    print(f"\n--- SNN FP16 累加 (更高精度) ---")
    try:
        snn_fp16 = SpikeFP8Linear_MultiPrecision(
            in_features, out_features, 
            accum_precision='fp16', 
            mode='sequential'
        ).to(device)
        snn_fp16.set_weight_from_float(w_fp8_f32, encoder)
        snn_fp16.reset()
        y_fp16_pulse = snn_fp16(x_pulse)
        
        decoder_fp16.reset()
        y_fp16 = decoder_fp16(y_fp16_pulse)
        
        # 比较：PyTorch FP32 结果转 FP16 后比较
        y_ref = y_pytorch_fp32.to(torch.float16).float()
        match = torch.isclose(y_fp16, y_ref, rtol=1e-5, atol=1e-6) | (y_fp16 == y_ref)
        rate_fp16 = match.sum().item() / match.numel() * 100
        print(f"与 PyTorch 匹配率: {rate_fp16:.1f}%")
        results['fp16'] = rate_fp16
    except Exception as e:
        print(f"失败: {e}")
        import traceback
        traceback.print_exc()
        results['fp16'] = 0
    
    # 4. FP32 累加（应该 100% 对齐）
    print(f"\n--- SNN FP32 累加 (最高精度) ---")
    try:
        snn_fp32 = SpikeFP8Linear_MultiPrecision(
            in_features, out_features, 
            accum_precision='fp32', 
            mode='sequential'
        ).to(device)
        snn_fp32.set_weight_from_float(w_fp8_f32, encoder)
        snn_fp32.reset()
        y_fp32_pulse = snn_fp32(x_pulse)
        
        decoder_fp32.reset()
        y_fp32 = decoder_fp32(y_fp32_pulse)
        
        # 直接比较 FP32
        match = torch.isclose(y_fp32, y_pytorch_fp32, rtol=1e-5, atol=1e-6) | (y_fp32 == y_pytorch_fp32)
        rate_fp32 = match.sum().item() / match.numel() * 100
        print(f"与 PyTorch 匹配率: {rate_fp32:.1f}%")
        results['fp32'] = rate_fp32
    except Exception as e:
        print(f"失败: {e}")
        import traceback
        traceback.print_exc()
        results['fp32'] = 0
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"| 累加模式         | 与 PyTorch (FP32累加) 匹配率 |")
    print(f"|------------------|------------------------------|")
    for mode, rate in results.items():
        print(f"| {mode:16s} | {rate:28.1f}% |")
    
    print("\n注: FP32 累加应该达到 ~100% 对齐")
    print("    FP16/FP8 累加由于精度损失，对齐率会较低")
    
    return results


def test_larger_scale():
    """更大规模测试（端到端浮点验证）"""
    print("\n" + "="*70)
    print("大规模测试 (in=16, out=8, batch=50) 端到端浮点验证")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder_fp8 = PulseFloatingPointDecoder().to(device)
    decoder_fp16 = PulseFP16Decoder().to(device)
    decoder_fp32 = PulseFP32Decoder().to(device)
    
    in_features = 16
    out_features = 8
    batch_size = 50
    
    torch.manual_seed(123)
    
    x_float = torch.randn(batch_size, in_features, device=device) * 0.3
    w_float = torch.randn(out_features, in_features, device=device) * 0.3
    
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    # PyTorch 参考
    y_pytorch_fp32 = x_fp8_f32 @ w_fp8_f32.T
    
    x_pulse = encoder(x_fp8_f32)
    
    results = {}
    
    # FP8 Sequential
    print("测试 FP8 Sequential...")
    snn_fp8 = SpikeFP8Linear_MultiPrecision(
        in_features, out_features, 
        accum_precision='fp8', 
        mode='sequential'
    ).to(device)
    snn_fp8.set_weight_from_float(w_fp8_f32, encoder)
    snn_fp8.reset()
    y_fp8_pulse = snn_fp8(x_pulse)
    
    decoder_fp8.reset()
    y_fp8 = decoder_fp8(y_fp8_pulse)
    y_ref = y_pytorch_fp32.to(torch.float8_e4m3fn).float()
    match = torch.isclose(y_fp8, y_ref, rtol=1e-5, atol=1e-6) | (y_fp8 == y_ref)
    results['fp8_seq'] = match.sum().item() / match.numel() * 100
    
    # FP16
    print("测试 FP16...")
    snn_fp16 = SpikeFP8Linear_MultiPrecision(
        in_features, out_features, 
        accum_precision='fp16', 
        mode='sequential'
    ).to(device)
    snn_fp16.set_weight_from_float(w_fp8_f32, encoder)
    snn_fp16.reset()
    y_fp16_pulse = snn_fp16(x_pulse)
    
    decoder_fp16.reset()
    y_fp16 = decoder_fp16(y_fp16_pulse)
    y_ref = y_pytorch_fp32.to(torch.float16).float()
    match = torch.isclose(y_fp16, y_ref, rtol=1e-5, atol=1e-6) | (y_fp16 == y_ref)
    results['fp16'] = match.sum().item() / match.numel() * 100
    
    # FP32
    print("测试 FP32...")
    snn_fp32 = SpikeFP8Linear_MultiPrecision(
        in_features, out_features, 
        accum_precision='fp32', 
        mode='sequential'
    ).to(device)
    snn_fp32.set_weight_from_float(w_fp8_f32, encoder)
    snn_fp32.reset()
    y_fp32_pulse = snn_fp32(x_pulse)
    
    decoder_fp32.reset()
    y_fp32 = decoder_fp32(y_fp32_pulse)
    match = torch.isclose(y_fp32, y_pytorch_fp32, rtol=1e-5, atol=1e-6) | (y_fp32 == y_pytorch_fp32)
    results['fp32'] = match.sum().item() / match.numel() * 100
    
    print(f"\n结果:")
    print(f"  FP8 Sequential:  {results['fp8_seq']:.1f}% 匹配")
    print(f"  FP16 Sequential: {results['fp16']:.1f}% 匹配")
    print(f"  FP32 Sequential: {results['fp32']:.1f}% 匹配")
    
    # FP32 应该最高
    if results['fp32'] >= results['fp16'] >= results['fp8_seq']:
        print("\n✓ 精度递增：FP8 < FP16 < FP32")
    else:
        print("\n⚠ 精度顺序异常")
    
    return results


if __name__ == "__main__":
    test_precision_comparison()
    test_larger_scale()

