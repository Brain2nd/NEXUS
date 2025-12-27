"""
正确的FP8 Linear测试：直接对比PyTorch FP8 matmul
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast
)


def pulse_to_fp8_bytes(pulse):
    bits = pulse.int()
    byte_val = torch.zeros(pulse.shape[:-1], dtype=torch.int32, device=pulse.device)
    for i in range(8):
        byte_val = byte_val + (bits[..., i] << (7 - i))
    return byte_val


def main():
    print("="*70)
    print("正确的FP8 Linear测试")
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
    
    # PyTorch FP8 matmul
    print("\n--- PyTorch FP8 计算 ---")
    try:
        # 尝试直接FP8 matmul
        y_pytorch = torch._scaled_mm(
            x_fp8, w_fp8.T,
            out_dtype=torch.float8_e4m3fn
        )
        print(f"使用 torch._scaled_mm")
        y_pytorch_bytes = y_pytorch.view(torch.uint8).int()
    except Exception as e:
        print(f"torch._scaled_mm 不支持: {e}")
        # 退回到float计算
        y_float = x_fp8.float() @ w_fp8.float().T
        y_pytorch = y_float.to(torch.float8_e4m3fn)
        y_pytorch_bytes = y_pytorch.view(torch.uint8).int()
        print(f"使用 float32 matmul 后转 FP8")
    
    print(f"y_pytorch dtype: {y_pytorch.dtype}")
    print(f"y_pytorch_bytes:\n{y_pytorch_bytes}")
    
    # SNN计算
    print("\n--- SNN FP8 计算 ---")
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
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
        y_snn_bytes = pulse_to_fp8_bytes(y_snn_pulse)
        
        print(f"y_snn_bytes:\n{y_snn_bytes}")
        
        # 比较
        match = (y_pytorch_bytes == y_snn_bytes)
        n_match = match.sum().item()
        total = y_snn_bytes.numel()
        rate = n_match / total * 100
        
        print(f"匹配: {n_match}/{total} = {rate:.1f}%")
        
        if n_match < total:
            # 显示不匹配的位置
            diff_idx = torch.where(~match)
            print(f"不匹配位置 (前5个):")
            for i in range(min(5, len(diff_idx[0]))):
                b, o = diff_idx[0][i].item(), diff_idx[1][i].item()
                print(f"  [{b},{o}]: PyTorch={y_pytorch_bytes[b,o].item()}, SNN={y_snn_bytes[b,o].item()}")


if __name__ == "__main__":
    main()

