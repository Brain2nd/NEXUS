"""
调试SpikeFP8Linear_Fast实现
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Multiplier,
    SpikeFP8Adder_Spatial,
    SpikeFP8Linear_Fast
)


def pulse_to_fp8_bytes(pulse):
    bits = pulse.int()
    byte_val = torch.zeros(pulse.shape[:-1], dtype=torch.int32, device=pulse.device)
    for i in range(8):
        byte_val = byte_val + (bits[..., i] << (7 - i))
    return byte_val


def float_to_fp8_tensor(x):
    return x.to(torch.float8_e4m3fn)


def fp8_tensor_to_bytes(x_fp8):
    return x_fp8.view(torch.uint8).int()


def test_linear_vs_manual():
    """比较Linear层实现和手动计算"""
    print("\n" + "="*70)
    print("Linear层实现 vs 手动计算")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    mul = SpikeFP8Multiplier().to(device)
    adder = SpikeFP8Adder_Spatial().to(device)
    
    in_features = 4
    out_features = 2
    batch_size = 3
    
    torch.manual_seed(42)
    
    w_float = torch.randn(out_features, in_features, device=device) * 0.3
    x_float = torch.randn(batch_size, in_features, device=device) * 0.3
    
    w_fp8 = float_to_fp8_tensor(w_float)
    x_fp8 = float_to_fp8_tensor(x_float)
    w_fp8_f32 = w_fp8.float()
    x_fp8_f32 = x_fp8.float()
    
    x_pulse = encoder(x_fp8_f32)  # [batch, in_features, 8]
    w_pulse = encoder(w_fp8_f32)  # [out_features, in_features, 8]
    
    print(f"\nx_pulse shape: {x_pulse.shape}")
    print(f"w_pulse shape: {w_pulse.shape}")
    
    # ===== 手动计算 =====
    print(f"\n手动计算:")
    manual_result = torch.zeros(batch_size, out_features, 8, device=device)
    
    for b in range(batch_size):
        for o in range(out_features):
            # 乘积
            products = []
            for i in range(in_features):
                mul.reset()
                p = mul(x_pulse[b, i], w_pulse[o, i])
                products.append(p)
            
            # 顺序累加
            acc = products[0]
            for i in range(1, in_features):
                adder.reset()
                acc = adder(acc, products[i])
            
            manual_result[b, o] = acc
    
    manual_bytes = pulse_to_fp8_bytes(manual_result)
    print(f"manual_result shape: {manual_result.shape}")
    print(f"manual_bytes:\n{manual_bytes}")
    
    # ===== Linear层计算 =====
    print(f"\nLinear层计算:")
    snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode='sequential').to(device)
    snn_linear.set_weight_from_float(w_fp8_f32, encoder)
    
    print(f"weight_pulse shape: {snn_linear.weight_pulse.shape}")
    
    # 检查权重编码
    w_linear_bytes = pulse_to_fp8_bytes(snn_linear.weight_pulse)
    w_manual_bytes = pulse_to_fp8_bytes(w_pulse)
    print(f"权重编码匹配: {(w_linear_bytes == w_manual_bytes).all().item()}")
    
    snn_linear.reset()
    linear_result = snn_linear(x_pulse)
    
    print(f"linear_result shape: {linear_result.shape}")
    linear_bytes = pulse_to_fp8_bytes(linear_result)
    print(f"linear_bytes:\n{linear_bytes}")
    
    # ===== 比较 =====
    print(f"\n比较:")
    match = (manual_bytes == linear_bytes)
    print(f"匹配矩阵:\n{match.int()}")
    print(f"匹配率: {match.float().mean().item() * 100:.1f}%")
    
    if not match.all():
        diff_idx = torch.where(~match)
        print(f"\n不匹配位置:")
        for idx in range(min(5, len(diff_idx[0]))):
            b, o = diff_idx[0][idx].item(), diff_idx[1][idx].item()
            print(f"  (b={b}, o={o}): manual={manual_bytes[b,o].item()}, linear={linear_bytes[b,o].item()}")


def test_linear_forward_trace():
    """追踪Linear层forward过程"""
    print("\n" + "="*70)
    print("Linear层forward追踪")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    in_features = 2
    out_features = 1
    batch_size = 2
    
    x_float = torch.tensor([[0.5, 0.25], [0.25, 0.5]], device=device)
    w_float = torch.tensor([[0.25, 0.5]], device=device)
    
    x_fp8 = float_to_fp8_tensor(x_float)
    w_fp8 = float_to_fp8_tensor(w_float)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    x_pulse = encoder(x_fp8_f32)  # [2, 2, 8]
    
    snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode='sequential').to(device)
    snn_linear.set_weight_from_float(w_fp8_f32, encoder)
    
    print(f"\nx_pulse shape: {x_pulse.shape}")
    print(f"weight_pulse shape: {snn_linear.weight_pulse.shape}")
    
    # 手动执行forward
    print(f"\n手动执行forward:")
    
    # x_expanded = x.unsqueeze(-3)
    x_expanded = x_pulse.unsqueeze(-3)
    print(f"x_expanded shape: {x_expanded.shape}")  # [2, 1, 2, 8]
    
    # products = self.mul(x_expanded, self.weight_pulse)
    snn_linear.mul.reset()
    products = snn_linear.mul(x_expanded, snn_linear.weight_pulse)
    print(f"products shape: {products.shape}")  # [2, 1, 2, 8]
    print(f"products bytes:\n{pulse_to_fp8_bytes(products)}")
    
    # Sequential accumulate
    print(f"\n顺序累加:")
    acc = products[..., 0, :]
    print(f"acc[0] shape: {acc.shape}, bytes: {pulse_to_fp8_bytes(acc)}")
    
    for i in range(1, in_features):
        snn_linear.adders[i-1].reset()
        acc = snn_linear.adders[i-1](acc, products[..., i, :])
        print(f"acc[{i}] shape: {acc.shape}, bytes: {pulse_to_fp8_bytes(acc)}")
    
    print(f"\n最终结果 shape: {acc.shape}")
    print(f"最终结果 bytes: {pulse_to_fp8_bytes(acc)}")


if __name__ == "__main__":
    test_linear_forward_trace()
    test_linear_vs_manual()

