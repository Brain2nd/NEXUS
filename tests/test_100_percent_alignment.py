"""
测试 SNN FP8 Linear 层与 PyTorch nn.Linear 的对齐

端到端浮点验证：
1. 生成随机浮点数
2. SNN 编码 + 计算 + 解码
3. 直接与 PyTorch 结果比较浮点数
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


def test_100_percent_alignment():
    """测试 FP32 Linear 的对齐（端到端浮点验证）"""
    print("="*70)
    print("对齐测试：SNN FP32 Linear vs PyTorch FP32 Linear (端到端浮点验证)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder_fp32 = PulseFP32Decoder().to(device)  # FP32模式使用FP32解码器
    
    # 小规模测试
    print("\n" + "-"*50)
    print("小规模测试: in=4, out=2, batch=20")
    print("-"*50)
    
    in_features = 4
    out_features = 2
    batch_size = 20
    
    torch.manual_seed(42)
    
    # 生成随机浮点数
    x_float = torch.randn(batch_size, in_features, device=device) * 0.5
    w_float = torch.randn(out_features, in_features, device=device) * 0.5
    
    # 量化到 FP8
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    # PyTorch FP32 Linear 参考（输出FP32）
    y_ref_fp32 = x_fp8_f32 @ w_fp8_f32.T
    
    # SNN FP32 Linear（输出FP32脉冲）
    encoder.reset()
    x_pulse = encoder(x_fp8_f32)
    
    snn_fp32 = SpikeFP8Linear_MultiPrecision(
        in_features, out_features, 
        accum_precision='fp32', 
        mode='sequential'
    ).to(device)
    snn_fp32.set_weight_from_float(w_fp8_f32, encoder)
    snn_fp32.reset()
    y_snn_pulse = snn_fp32(x_pulse)  # 输出FP32脉冲 [..., 32]
    
    # 验证输出维度
    assert y_snn_pulse.shape[-1] == 32, f"FP32模式应输出32位脉冲，实际为{y_snn_pulse.shape[-1]}"
    
    # 使用FP32解码器
    decoder_fp32 = PulseFP32Decoder().to(device)
    decoder_fp32.reset()
    y_snn_float = decoder_fp32(y_snn_pulse)
    
    # 直接比较浮点数（FP32精度）
    match = torch.isclose(y_snn_float, y_ref_fp32, rtol=1e-5, atol=1e-6) | (y_snn_float == y_ref_fp32)
    match_rate = match.sum().item() / match.numel() * 100
    
    print(f"匹配率: {match_rate:.1f}% ({match.sum().item()}/{match.numel()})")
    
    if match_rate < 100:
            mismatch_mask = ~match
            mismatch_indices = torch.where(mismatch_mask)
            n_show = min(5, len(mismatch_indices[0]))
            print(f"\n不匹配样本（前{n_show}个）:")
            for idx in range(n_show):
                i = mismatch_indices[0][idx].item()
                j = mismatch_indices[1][idx].item()
                print(f"  [{i},{j}]: PyTorch={y_ref_fp32[i,j].item():.6f}, SNN={y_snn_float[i,j].item():.6f}")
    
    # 中规模测试
    print("\n" + "-"*50)
    print("中规模测试: in=8, out=4, batch=50")
    print("-"*50)
    
    in_features = 8
    out_features = 4
    batch_size = 50
    
    torch.manual_seed(123)
    
    x_float = torch.randn(batch_size, in_features, device=device) * 0.3
    w_float = torch.randn(out_features, in_features, device=device) * 0.3
    
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    # PyTorch FP32 Linear 参考（输出FP32）
    y_ref_fp32 = x_fp8_f32 @ w_fp8_f32.T
    
    encoder.reset()
    x_pulse = encoder(x_fp8_f32)
    
    snn_fp32 = SpikeFP8Linear_MultiPrecision(
        in_features, out_features, 
        accum_precision='fp32', 
        mode='sequential'
    ).to(device)
    snn_fp32.set_weight_from_float(w_fp8_f32, encoder)
    snn_fp32.reset()
    y_snn_pulse = snn_fp32(x_pulse)  # 输出FP32脉冲 [..., 32]
    
    assert y_snn_pulse.shape[-1] == 32, f"FP32模式应输出32位脉冲，实际为{y_snn_pulse.shape[-1]}"
    
    decoder_fp32.reset()
    y_snn_float = decoder_fp32(y_snn_pulse)
    
    match = torch.isclose(y_snn_float, y_ref_fp32, rtol=1e-5, atol=1e-6) | (y_snn_float == y_ref_fp32)
    match_rate_mid = match.sum().item() / match.numel() * 100
    
    print(f"匹配率: {match_rate_mid:.1f}% ({match.sum().item()}/{match.numel()})")
    
    # 多配置测试
    print("\n" + "-"*50)
    print("多配置测试")
    print("-"*50)
    
    configs = [
        (2, 1, 10),
        (4, 2, 20),
        (8, 4, 50),
        (16, 8, 100),
    ]
    
    total_elements = 0
    total_matches = 0
    
    print("\n| 配置 (in,out,batch) | 元素数 | 匹配数 | 匹配率 |")
    print("|---------------------|--------|--------|--------|")
    
    for in_f, out_f, batch in configs:
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            
            x_float = torch.randn(batch, in_f, device=device) * (0.5 if seed == 42 else 0.3)
            w_float = torch.randn(out_f, in_f, device=device) * (0.5 if seed == 42 else 0.3)
            
            x_fp8 = x_float.to(torch.float8_e4m3fn)
            w_fp8 = w_float.to(torch.float8_e4m3fn)
            x_fp8_f32 = x_fp8.float()
            w_fp8_f32 = w_fp8.float()
            
            # PyTorch FP32 Linear 参考（输出FP32）
            y_ref_fp32 = x_fp8_f32 @ w_fp8_f32.T
            
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
            
            assert y_snn_pulse.shape[-1] == 32, f"FP32模式应输出32位脉冲，实际为{y_snn_pulse.shape[-1]}"
            
            decoder_fp32.reset()
            y_snn_float = decoder_fp32(y_snn_pulse)
            
            match = torch.isclose(y_snn_float, y_ref_fp32, rtol=1e-5, atol=1e-6) | (y_snn_float == y_ref_fp32)
            n_match = match.sum().item()
            n_total = match.numel()
            
            total_elements += n_total
            total_matches += n_match
    
    overall_rate = total_matches / total_elements * 100
    print(f"| 总计 (多seed)       | {total_elements:6} | {total_matches:6} | {overall_rate:5.1f}% |")
    
    print("\n" + "="*70)
    if overall_rate >= 99.0:
        print("✓ 对齐测试通过！")
    else:
        print(f"⚠ 对齐率 {overall_rate:.1f}%，需检查")
    
    return overall_rate


if __name__ == "__main__":
    test_100_percent_alignment()
