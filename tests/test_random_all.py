"""
大规模随机数据测试 - 验证所有原子组件

端到端浮点验证：使用解码器解码后直接比较浮点数
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Adder_Spatial,
    SpikeFP8Multiplier,
    SpikeFP8Linear_MultiPrecision
)
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder, PulseFP32Decoder


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("="*60)
    print("大规模随机数据测试（端到端浮点验证）")
    print("="*60)
    
    encoder = PulseFloatingPointEncoder(4, 3, 10, 10).to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    adder = SpikeFP8Adder_Spatial().to(device)
    mul = SpikeFP8Multiplier().to(device)
    
    N = 1000  # 测试样本数
    
    # ========== 1. 编码器测试 ==========
    print("\n[1] 编码器 (Encoder) - 1000个随机FP8值")
    
    # 生成随机浮点数
    raw = torch.randn(N, device=device) * 10
    fp8_vals = raw.to(torch.float8_e4m3fn)
    fp8_f32 = fp8_vals.float()
    
    # SNN编码
    pulse = encoder(fp8_f32)
    
    # 解码回浮点数
    decoder.reset()
    decoded = decoder(pulse)
    
    # 直接比较浮点数
    enc_match = torch.isclose(decoded, fp8_f32, rtol=1e-5, atol=1e-6) | (decoded == fp8_f32)
    enc_match_count = enc_match.sum().item()
    print(f"    匹配: {enc_match_count}/{N} = {enc_match_count/N*100:.1f}%")
    
    # ========== 2. 加法器测试 ==========
    print("\n[2] 加法器 (Adder) - 1000对随机FP8加法")
    
    a_raw = torch.randn(N, device=device) * 5
    b_raw = torch.randn(N, device=device) * 5
    a_fp8 = a_raw.to(torch.float8_e4m3fn)
    b_fp8 = b_raw.to(torch.float8_e4m3fn)
    
    # PyTorch 参考: FP8 + FP8 -> FP8
    ref_sum = (a_fp8.float() + b_fp8.float()).to(torch.float8_e4m3fn).float()
    
    # SNN
    encoder.reset()
    p_a = encoder(a_fp8.float())
    encoder.reset()
    p_b = encoder(b_fp8.float())
    adder.reset()
    p_sum = adder(p_a, p_b)
    
    decoder.reset()
    snn_sum = decoder(p_sum)
    
    # 直接比较浮点数
    add_match = torch.isclose(snn_sum, ref_sum, rtol=1e-5, atol=1e-6) | (snn_sum == ref_sum)
    add_match_count = add_match.sum().item()
    print(f"    匹配: {add_match_count}/{N} = {add_match_count/N*100:.1f}%")
    
    # ========== 3. 乘法器测试 ==========
    print("\n[3] 乘法器 (Multiplier) - 1000对随机FP8乘法")
    
    a_raw = torch.randn(N, device=device) * 3
    b_raw = torch.randn(N, device=device) * 3
    a_fp8 = a_raw.to(torch.float8_e4m3fn)
    b_fp8 = b_raw.to(torch.float8_e4m3fn)
    
    # PyTorch 参考: FP8 * FP8 -> FP8
    ref_prod = (a_fp8.float() * b_fp8.float()).to(torch.float8_e4m3fn).float()
    
    # SNN
    encoder.reset()
    p_a = encoder(a_fp8.float())
    encoder.reset()
    p_b = encoder(b_fp8.float())
    mul.reset()
    p_prod = mul(p_a, p_b)
    
    decoder.reset()
    snn_prod = decoder(p_prod)
    
    # 直接比较浮点数
    mul_match = torch.isclose(snn_prod, ref_prod, rtol=1e-5, atol=1e-6) | (snn_prod == ref_prod)
    mul_match_count = mul_match.sum().item()
    print(f"    匹配: {mul_match_count}/{N} = {mul_match_count/N*100:.1f}%")
    
    # 打印不匹配样例
    if mul_match_count < N:
        mismatch_idx = (~mul_match).nonzero(as_tuple=True)[0][:5]
        print(f"    不匹配样例 (前5个):")
        for idx in mismatch_idx:
            i = idx.item()
            print(f"      {a_fp8[i].float().item():.4f} * {b_fp8[i].float().item():.4f} = Ref:{ref_prod[i].item():.4f}, SNN:{snn_prod[i].item():.4f}")
    
    # ========== 4. Linear层测试 (小规模) ==========
    print("\n[4] Linear层 (FP32累加) - 100个样本, in=8, out=4")
    
    in_f, out_f, batch = 8, 4, 100
    x_raw = torch.randn(batch, in_f, device=device) * 0.5
    w_raw = torch.randn(out_f, in_f, device=device) * 0.5
    
    x_fp8 = x_raw.to(torch.float8_e4m3fn)
    w_fp8 = w_raw.to(torch.float8_e4m3fn)
    
    # PyTorch FP32 Linear 参考（输出FP32）
    ref_result = x_fp8.float() @ w_fp8.float().T
    
    # SNN FP32 Linear（输出FP32脉冲）
    encoder.reset()
    x_pulse = encoder(x_fp8.float())
    linear = SpikeFP8Linear_MultiPrecision(in_f, out_f, accum_precision='fp32', mode='sequential').to(device)
    linear.set_weight_from_float(w_fp8.float(), encoder)
    linear.reset()
    y_pulse = linear(x_pulse)  # 输出FP32脉冲 [..., 32]
    
    # 使用FP32解码器
    decoder_fp32 = PulseFP32Decoder().to(device)
    decoder_fp32.reset()
    snn_result = decoder_fp32(y_pulse)
    
    # 直接比较浮点数
    lin_match = torch.isclose(snn_result, ref_result, rtol=1e-5, atol=1e-6) | (snn_result == ref_result)
    lin_match_count = lin_match.sum().item()
    total = batch * out_f
    print(f"    匹配: {lin_match_count}/{total} = {lin_match_count/total*100:.1f}%")
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"| 组件     | 匹配率        |")
    print(f"|----------|---------------|")
    print(f"| 编码器   | {enc_match_count/N*100:5.1f}% ({enc_match_count}/{N}) |")
    print(f"| 加法器   | {add_match_count/N*100:5.1f}% ({add_match_count}/{N}) |")
    print(f"| 乘法器   | {mul_match_count/N*100:5.1f}% ({mul_match_count}/{N}) |")
    print(f"| Linear   | {lin_match_count/total*100:5.1f}% ({lin_match_count}/{total}) |")


if __name__ == "__main__":
    main()
