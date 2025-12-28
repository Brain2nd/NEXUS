"""
正确的FP8 Linear测试：直接对比PyTorch FP8 matmul（端到端浮点验证）
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast
)
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder


def main():
    """端到端浮点验证"""
    print("="*70)
    print("正确的FP8 Linear测试（端到端浮点验证）")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建FP8输入和权重
    in_features = 4
    out_features = 2
    batch_size = 10
    
    torch.manual_seed(42)
    
    # 直接创建FP8张量
    x_float = torch.randn(batch_size, in_features, device=device) * 0.5
    w_float = torch.randn(out_features, in_features, device=device) * 0.5
    
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    
    print(f"\nx_fp8 dtype: {x_fp8.dtype}")
    print(f"w_fp8 dtype: {w_fp8.dtype}")
    print(f"x_fp8 shape: {x_fp8.shape}")
    print(f"w_fp8 shape: {w_fp8.shape}")
    
    # PyTorch 参考计算
    print("\n--- PyTorch 参考计算 ---")
    y_pytorch = (x_fp8.float() @ w_fp8.float().T).to(torch.float8_e4m3fn).float()
    print(f"y_pytorch:\n{y_pytorch}")
    
    # SNN计算
    print("\n--- SNN FP8 计算 ---")
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    x_pulse = encoder(x_fp8_f32)
    
    # 测试两种模式
    for mode in ['sequential', 'tree']:
        print(f"\n模式: {mode}")
        snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode=mode).to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        
        y_snn_pulse = snn_linear(x_pulse)
        
        decoder.reset()
        y_snn = decoder(y_snn_pulse)
        
        print(f"y_snn:\n{y_snn}")
        
        # 直接比较浮点数
        match = torch.isclose(y_snn, y_pytorch, rtol=1e-5, atol=1e-6) | (y_snn == y_pytorch)
        n_match = match.sum().item()
        total = y_snn.numel()
        rate = n_match / total * 100
        
        print(f"匹配: {n_match}/{total} = {rate:.1f}%")
        
        if n_match < total:
            # 显示不匹配的位置
            diff_idx = torch.where(~match)
            print(f"不匹配位置 (前5个):")
            for i in range(min(5, len(diff_idx[0]))):
                b, o = diff_idx[0][i].item(), diff_idx[1][i].item()
                print(f"  [{b},{o}]: PyTorch={y_pytorch[b,o].item():.6f}, SNN={y_snn[b,o].item():.6f}")


if __name__ == "__main__":
    main()

