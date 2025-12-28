"""
实验四：正确的端到端验证（端到端浮点验证）
使用手动FP8顺序累加作为参考（而非PyTorch matmul）
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast
)
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder


def float_to_fp8_tensor(x):
    return x.to(torch.float8_e4m3fn)


def fp8_tensor_to_bytes(x_fp8):
    return x_fp8.view(torch.uint8).int()


def pulse_to_fp8_bytes(pulse):
    """脉冲转FP8字节值（仅用于调试比特比较）"""
    bits = pulse.int()
    byte_val = torch.zeros(pulse.shape[:-1], dtype=torch.int32, device=pulse.device)
    for i in range(8):
        byte_val = byte_val + (bits[..., i] << (7 - i))
    return byte_val


def manual_fp8_matmul_sequential(x_fp8, w_fp8):
    """手动FP8矩阵乘法（顺序累加）
    
    这是SNN应该匹配的参考实现
    """
    x_f32 = x_fp8.float()
    w_f32 = w_fp8.float()
    
    batch_size = x_f32.shape[0]
    in_features = x_f32.shape[1]
    out_features = w_f32.shape[0]
    
    result = torch.zeros(batch_size, out_features, dtype=torch.float32, device=x_f32.device)
    
    for b in range(batch_size):
        for o in range(out_features):
            # 计算所有乘积
            products = []
            for i in range(in_features):
                p = x_f32[b, i] * w_f32[o, i]
                p_fp8 = float_to_fp8_tensor(p.unsqueeze(0))
                products.append(p_fp8.float().squeeze())
            
            # 顺序累加
            acc = products[0]
            for i in range(1, in_features):
                acc_sum = acc + products[i]
                acc_fp8 = float_to_fp8_tensor(acc_sum.unsqueeze(0))
                acc = acc_fp8.float().squeeze()
            
            result[b, o] = acc
    
    return float_to_fp8_tensor(result)


def test_vs_manual_sequential():
    """测试SNN vs 手动顺序FP8累加"""
    print("\n" + "="*70)
    print("测试: SNN vs 手动顺序FP8累加 (正确的参考)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    configs = [
        (2, 1, 10),
        (4, 2, 20),
        (8, 4, 50),
        (16, 8, 100),
    ]
    
    total_elements = 0
    total_matches = 0
    
    print("\n| 配置 (Din,Dout,Batch) | 元素数 | 匹配数 | 匹配率 |")
    print("|----------------------|--------|--------|--------|")
    
    for in_features, out_features, batch_size in configs:
        torch.manual_seed(in_features * out_features + batch_size)
        
        # 生成FP8数据
        w_float = torch.randn(out_features, in_features, device=device) * 0.3
        x_float = torch.randn(batch_size, in_features, device=device) * 0.3
        
        w_fp8 = float_to_fp8_tensor(w_float)
        x_fp8 = float_to_fp8_tensor(x_float)
        w_fp8_f32 = w_fp8.float()
        x_fp8_f32 = x_fp8.float()
        
        # 参考: 手动顺序累加
        y_ref_fp8 = manual_fp8_matmul_sequential(x_fp8, w_fp8)
        y_ref_bytes = fp8_tensor_to_bytes(y_ref_fp8)
        
        # SNN计算
        x_pulse = encoder(x_fp8_f32)
        snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode='sequential').to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        y_snn_pulse = snn_linear(x_pulse)
        y_snn_bytes = pulse_to_fp8_bytes(y_snn_pulse)
        
        # 统计
        match = (y_ref_bytes == y_snn_bytes)
        n_elements = match.numel()
        n_matches = match.sum().item()
        match_rate = n_matches / n_elements * 100
        
        total_elements += n_elements
        total_matches += n_matches
        
        print(f"| ({in_features:2},{out_features:2},{batch_size:3})             | {n_elements:6} | {n_matches:6} | {match_rate:5.1f}% |")
        
        # 显示不匹配样本
        if n_matches < n_elements and in_features <= 4:
            diff_idx = torch.where(~match)
            for idx in range(min(3, len(diff_idx[0]))):
                b, o = diff_idx[0][idx].item(), diff_idx[1][idx].item()
                print(f"    样本[{b},{o}]: ref={y_ref_bytes[b,o].item()}, snn={y_snn_bytes[b,o].item()}")
    
    overall_rate = total_matches / total_elements * 100
    print(f"|----------------------|--------|--------|--------|")
    print(f"| 总计                 | {total_elements:6} | {total_matches:6} | {overall_rate:5.1f}% |")
    
    return overall_rate


def test_vs_pytorch_matmul():
    """测试SNN vs PyTorch matmul (预期有差异)"""
    print("\n" + "="*70)
    print("参考: SNN vs PyTorch matmul (预期有差异)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    configs = [(4, 2, 20), (8, 4, 50)]
    
    total_elements = 0
    total_matches = 0
    
    print("\n| 配置 | SNN vs Manual | SNN vs Matmul | Manual vs Matmul |")
    print("|------|---------------|---------------|------------------|")
    
    for in_features, out_features, batch_size in configs:
        torch.manual_seed(in_features * out_features + batch_size)
        
        w_float = torch.randn(out_features, in_features, device=device) * 0.3
        x_float = torch.randn(batch_size, in_features, device=device) * 0.3
        
        w_fp8 = float_to_fp8_tensor(w_float)
        x_fp8 = float_to_fp8_tensor(x_float)
        w_fp8_f32 = w_fp8.float()
        x_fp8_f32 = x_fp8.float()
        
        # 手动顺序
        y_manual = manual_fp8_matmul_sequential(x_fp8, w_fp8)
        y_manual_bytes = fp8_tensor_to_bytes(y_manual)
        
        # PyTorch matmul
        y_matmul = x_fp8_f32 @ w_fp8_f32.T
        y_matmul_fp8 = float_to_fp8_tensor(y_matmul)
        y_matmul_bytes = fp8_tensor_to_bytes(y_matmul_fp8)
        
        # SNN
        x_pulse = encoder(x_fp8_f32)
        snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode='sequential').to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        y_snn_pulse = snn_linear(x_pulse)
        y_snn_bytes = pulse_to_fp8_bytes(y_snn_pulse)
        
        # 统计
        snn_vs_manual = (y_snn_bytes == y_manual_bytes).float().mean().item() * 100
        snn_vs_matmul = (y_snn_bytes == y_matmul_bytes).float().mean().item() * 100
        manual_vs_matmul = (y_manual_bytes == y_matmul_bytes).float().mean().item() * 100
        
        print(f"| ({in_features},{out_features},{batch_size}) | {snn_vs_manual:11.1f}% | {snn_vs_matmul:11.1f}% | {manual_vs_matmul:14.1f}% |")


def main():
    print("="*70)
    print("实验四：端到端验证 - SNN vs 正确的顺序FP8参考")
    print("="*70)
    
    # 核心测试: SNN vs 手动顺序累加
    seq_rate = test_vs_manual_sequential()
    
    # 参考对比
    test_vs_pytorch_matmul()
    
    # 总结
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    if seq_rate == 100.0:
        print("\n✓ SNN FP8 与 手动顺序FP8累加 100% bit-exact一致！")
        print("  证明了SNN脉冲域计算的数学正确性")
        print("\n注意: PyTorch matmul使用优化算法，与简单顺序累加有细微差异")
        print("      这是FP8/浮点数的固有特性，不是SNN的问题")
    else:
        print(f"\n✗ 匹配率 {seq_rate:.1f}%，仍有问题需要调试")


if __name__ == "__main__":
    main()
