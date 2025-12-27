"""
实验四：端到端验证 - SNN FP8 vs ANN FP8 Linear层

端到端浮点验证：
1. 生成随机浮点数
2. SNN 编码 + 计算 + 解码
3. 直接与 PyTorch 结果比较浮点数
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast
)
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder


def test_sequential_mode():
    """测试sequential模式与PyTorch的一致性（端到端浮点验证）"""
    print("\n" + "="*70)
    print("测试1: Sequential模式 vs PyTorch (端到端浮点验证)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFloatingPointDecoder().to(device)
    
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
        
        # 生成随机浮点数
        w_float = torch.randn(out_features, in_features, device=device) * 0.3
        x_float = torch.randn(batch_size, in_features, device=device) * 0.3
        
        # 量化到 FP8
        w_fp8 = w_float.to(torch.float8_e4m3fn)
        x_fp8 = x_float.to(torch.float8_e4m3fn)
        w_fp8_f32 = w_fp8.float()
        x_fp8_f32 = x_fp8.float()
        
        # PyTorch 参考 (FP32 matmul -> FP8)
        y_ref_f32 = x_fp8_f32 @ w_fp8_f32.T
        y_ref_fp8 = y_ref_f32.to(torch.float8_e4m3fn).float()
        
        # SNN计算 (sequential模式)
        x_pulse = encoder(x_fp8_f32)
        snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode='sequential').to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        y_snn_pulse = snn_linear(x_pulse)
        
        # 解码回浮点数
        decoder.reset()
        y_snn_float = decoder(y_snn_pulse)
        
        # 直接比较浮点数
        match = torch.isclose(y_snn_float, y_ref_fp8, rtol=1e-5, atol=1e-6) | (y_snn_float == y_ref_fp8)
        n_elements = match.numel()
        n_matches = match.sum().item()
        match_rate = n_matches / n_elements * 100
        
        total_elements += n_elements
        total_matches += n_matches
        
        print(f"| ({in_features:2},{out_features:2},{batch_size:3})             | {n_elements:6} | {n_matches:6} | {match_rate:5.1f}% |")
    
    overall_rate = total_matches / total_elements * 100
    print(f"|----------------------|--------|--------|--------|")
    print(f"| 总计                 | {total_elements:6} | {total_matches:6} | {overall_rate:5.1f}% |")
    
    return overall_rate


def test_tree_mode():
    """测试tree模式（端到端浮点验证）"""
    print("\n" + "="*70)
    print("测试2: Tree模式 vs PyTorch (端到端浮点验证)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFloatingPointDecoder().to(device)
    
    configs = [
        (4, 2, 20),
        (8, 4, 50),
    ]
    
    total_elements = 0
    total_matches = 0
    
    print("\n| 配置 (Din,Dout,Batch) | 元素数 | 匹配数 | 匹配率 |")
    print("|----------------------|--------|--------|--------|")
    
    for in_features, out_features, batch_size in configs:
        torch.manual_seed(in_features * out_features + batch_size)
        
        w_float = torch.randn(out_features, in_features, device=device) * 0.3
        x_float = torch.randn(batch_size, in_features, device=device) * 0.3
        
        w_fp8 = w_float.to(torch.float8_e4m3fn)
        x_fp8 = x_float.to(torch.float8_e4m3fn)
        w_fp8_f32 = w_fp8.float()
        x_fp8_f32 = x_fp8.float()
        
        # PyTorch 参考
        y_ref_f32 = x_fp8_f32 @ w_fp8_f32.T
        y_ref_fp8 = y_ref_f32.to(torch.float8_e4m3fn).float()
        
        # SNN (tree模式)
        x_pulse = encoder(x_fp8_f32)
        snn_linear = SpikeFP8Linear_Fast(in_features, out_features, mode='tree').to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        y_snn_pulse = snn_linear(x_pulse)
        
        # 解码
        decoder.reset()
        y_snn_float = decoder(y_snn_pulse)
        
        # 比较
        match = torch.isclose(y_snn_float, y_ref_fp8, rtol=1e-5, atol=1e-6) | (y_snn_float == y_ref_fp8)
        n_elements = match.numel()
        n_matches = match.sum().item()
        match_rate = n_matches / n_elements * 100
        
        total_elements += n_elements
        total_matches += n_matches
        
        print(f"| ({in_features:2},{out_features:2},{batch_size:3})             | {n_elements:6} | {n_matches:6} | {match_rate:5.1f}% |")
    
    overall_rate = total_matches / total_elements * 100
    print(f"|----------------------|--------|--------|--------|")
    print(f"| 总计                 | {total_elements:6} | {total_matches:6} | {overall_rate:5.1f}% |")
    
    return overall_rate


def test_latency_comparison():
    """比较两种模式的延迟"""
    print("\n" + "="*70)
    print("测试3: 延迟对比")
    print("="*70)
    
    print("\n| in_features | Sequential延迟 | Tree延迟 | 加速比 |")
    print("|-------------|---------------|----------|--------|")
    
    for in_features in [4, 8, 16, 32, 64, 128]:
        seq_linear = SpikeFP8Linear_Fast(in_features, 1, mode='sequential')
        tree_linear = SpikeFP8Linear_Fast(in_features, 1, mode='tree')
        
        seq_latency = seq_linear.get_latency()
        tree_latency = tree_linear.get_latency()
        speedup = seq_latency / tree_latency
        
        print(f"| {in_features:11} | {seq_latency:13} | {tree_latency:8} | {speedup:6.1f}x |")


def test_detailed_single_case():
    """详细测试单个案例（端到端浮点验证）"""
    print("\n" + "="*70)
    print("测试4: 单案例详细对比（端到端浮点验证）")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFloatingPointDecoder().to(device)
    
    # 简单案例
    x_float = torch.tensor([[0.5, 0.25, -0.125, 0.375]], device=device)
    w_float = torch.tensor([[0.25, -0.5, 0.125, 0.25]], device=device)
    
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    print(f"\nx (FP8): {x_fp8_f32[0].tolist()}")
    print(f"w (FP8): {w_fp8_f32[0].tolist()}")
    
    # PyTorch 参考
    y_ref = x_fp8_f32 @ w_fp8_f32.T
    y_ref_fp8 = y_ref.to(torch.float8_e4m3fn).float()
    
    print(f"\nPyTorch 输出: {y_ref.item():.6f}")
    print(f"PyTorch FP8 输出: {y_ref_fp8.item():.6f}")
    
    # SNN Sequential
    x_pulse = encoder(x_fp8_f32)
    snn_seq = SpikeFP8Linear_Fast(4, 1, mode='sequential').to(device)
    snn_seq.set_weight_from_float(w_fp8_f32, encoder)
    snn_seq.reset()
    y_seq_pulse = snn_seq(x_pulse)
    
    decoder.reset()
    y_seq_float = decoder(y_seq_pulse)
    
    print(f"\nSNN Sequential 输出: {y_seq_float.item():.6f}")
    match_seq = abs(y_seq_float.item() - y_ref_fp8.item()) < 1e-6
    print(f"与PyTorch匹配: {'✓' if match_seq else '✗'}")
    
    # SNN Tree
    snn_tree = SpikeFP8Linear_Fast(4, 1, mode='tree').to(device)
    snn_tree.set_weight_from_float(w_fp8_f32, encoder)
    snn_tree.reset()
    y_tree_pulse = snn_tree(x_pulse)
    
    decoder.reset()
    y_tree_float = decoder(y_tree_pulse)
    
    print(f"\nSNN Tree 输出: {y_tree_float.item():.6f}")
    match_tree = abs(y_tree_float.item() - y_ref_fp8.item()) < 1e-6
    print(f"与PyTorch匹配: {'✓' if match_tree else '✗'}")


def main():
    print("="*70)
    print("实验四：端到端验证 - SNN FP8 vs ANN FP8 (端到端浮点验证)")
    print("="*70)
    
    # 测试1: Sequential模式
    seq_rate = test_sequential_mode()
    
    # 测试2: Tree模式
    tree_rate = test_tree_mode()
    
    # 测试3: 延迟对比
    test_latency_comparison()
    
    # 测试4: 详细案例
    test_detailed_single_case()
    
    # 总结
    print("\n" + "="*70)
    print("实验四总结")
    print("="*70)
    
    print(f"\n| 模式 | 与PyTorch匹配率 | 延迟 | 适用场景 |")
    print(f"|------|----------------|------|---------|")
    print(f"| Sequential | {seq_rate:.1f}% | O(N) | 需要高精度 |")
    print(f"| Tree | {tree_rate:.1f}% | O(log N) | 追求低延迟 |")
    
    if seq_rate >= 99.0:
        print("\n✓ Sequential模式与PyTorch高度一致！")
    else:
        print(f"\n⚠ Sequential模式匹配率 {seq_rate:.1f}%")


if __name__ == "__main__":
    main()
