"""
测试多精度累加 Linear 层与朴素 PyTorch linear 的对齐情况

比较目标：PyTorch 的 nn.Linear 使用 FP32 累加后转回 FP8
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast,
    SpikeFP8Linear_MultiPrecision
)


def pulse_to_fp8_bytes(pulse):
    """脉冲 -> FP8 字节值"""
    bits = pulse.int()
    byte_val = torch.zeros(pulse.shape[:-1], dtype=torch.int32, device=pulse.device)
    for i in range(8):
        byte_val = byte_val + (bits[..., i] << (7 - i))
    return byte_val


def fp8_bytes_to_float(bytes_val):
    """FP8 bytes -> float"""
    return bytes_val.view(torch.uint8).view(torch.float8_e4m3fn).float()


def test_precision_comparison():
    """比较不同累加精度与 PyTorch 的对齐情况"""
    print("="*70)
    print("多精度累加 Linear 层测试")
    print("比较目标: PyTorch FP8 matmul (FP32累加 -> FP8)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 编码器
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
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
    
    # PyTorch 参考: FP32 matmul 后转 FP8
    y_float = x_fp8_f32 @ w_fp8_f32.T
    y_pytorch = y_float.to(torch.float8_e4m3fn)
    y_pytorch_bytes = y_pytorch.view(torch.uint8).int()
    
    print(f"\n--- PyTorch 参考 (FP32累加 -> FP8) ---")
    print(f"y_pytorch 部分输出: {y_pytorch_bytes[:3].tolist()}")
    
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
        y_fp8_seq_bytes = pulse_to_fp8_bytes(y_fp8_seq_pulse)
        
        match_fp8_seq = (y_pytorch_bytes == y_fp8_seq_bytes)
        rate_fp8_seq = match_fp8_seq.sum().item() / match_fp8_seq.numel() * 100
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
        y_fp8_tree_bytes = pulse_to_fp8_bytes(y_fp8_tree_pulse)
        
        match_fp8_tree = (y_pytorch_bytes == y_fp8_tree_bytes)
        rate_fp8_tree = match_fp8_tree.sum().item() / match_fp8_tree.numel() * 100
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
        y_fp16_bytes = pulse_to_fp8_bytes(y_fp16_pulse)
        
        match_fp16 = (y_pytorch_bytes == y_fp16_bytes)
        rate_fp16 = match_fp16.sum().item() / match_fp16.numel() * 100
        print(f"与 PyTorch 匹配率: {rate_fp16:.1f}%")
        results['fp16'] = rate_fp16
    except Exception as e:
        print(f"失败: {e}")
        import traceback
        traceback.print_exc()
        results['fp16'] = 0
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"| 累加模式         | 与 PyTorch (FP32累加) 匹配率 |")
    print(f"|------------------|------------------------------|")
    for mode, rate in results.items():
        print(f"| {mode:16s} | {rate:28.1f}% |")
    
    print("\n注: PyTorch FP8 matmul 使用 FP32 内部累加，")
    print("    而 SNN 使用 FP8/FP16 累加，因此存在差异是预期的。")
    print("    FP16 累加应该比 FP8 累加更接近 PyTorch 行为。")
    
    return results


def test_larger_scale():
    """更大规模测试"""
    print("\n" + "="*70)
    print("大规模测试 (in=16, out=8, batch=50)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
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
    y_float = x_fp8_f32 @ w_fp8_f32.T
    y_pytorch = y_float.to(torch.float8_e4m3fn)
    y_pytorch_bytes = y_pytorch.view(torch.uint8).int()
    
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
    y_fp8_bytes = pulse_to_fp8_bytes(y_fp8_pulse)
    match_fp8 = (y_pytorch_bytes == y_fp8_bytes)
    results['fp8_seq'] = match_fp8.sum().item() / match_fp8.numel() * 100
    
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
    y_fp16_bytes = pulse_to_fp8_bytes(y_fp16_pulse)
    match_fp16 = (y_pytorch_bytes == y_fp16_bytes)
    results['fp16'] = match_fp16.sum().item() / match_fp16.numel() * 100
    
    print(f"\n结果:")
    print(f"  FP8 Sequential:  {results['fp8_seq']:.1f}% 匹配")
    print(f"  FP16 Sequential: {results['fp16']:.1f}% 匹配")
    
    # FP16 应该更高
    if results['fp16'] >= results['fp8_seq']:
        print("\n✓ FP16 累加确实比 FP8 累加更接近 PyTorch")
    else:
        print("\n✗ 意外：FP16 累加未能改善对齐")
    
    return results


if __name__ == "__main__":
    test_precision_comparison()
    test_larger_scale()

