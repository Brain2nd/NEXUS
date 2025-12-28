"""
FP32组件全面验证测试 - 真正并行版（端到端浮点验证）
===================================================

使用GPU + 向量化编码/解码实现真正并行。

作者: HumanBrain Project
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from SNNTorch.atomic_ops.pulse_decoder import PulseFP32Decoder


def float_to_pulse_vectorized(x: torch.Tensor) -> torch.Tensor:
    """向量化：浮点 -> FP32脉冲 [...] -> [..., 32]"""
    # 使用view将float32转为int32位模式
    x_int = x.view(torch.int32)
    # 提取每一位
    bits = []
    for i in range(31, -1, -1):
        bit = ((x_int >> i) & 1).float()
        bits.append(bit)
    return torch.stack(bits, dim=-1)


def test_exp(device, n_samples=100):
    """测试SpikeFP32Exp - 并行（端到端浮点验证）"""
    from SNNTorch.atomic_ops import SpikeFP32Exp
    
    exp_mod = SpikeFP32Exp().to(device)
    decoder = PulseFP32Decoder().to(device)
    torch.manual_seed(42)
    
    x = torch.randn(n_samples, device=device) * 2
    x_pulse = float_to_pulse_vectorized(x)  # [N, 32] - 向量化
    
    exp_mod.reset()
    result_pulse = exp_mod(x_pulse)  # [N, 32] - GPU并行
    
    decoder.reset()
    snn_result = decoder(result_pulse)
    snn_bits = snn_result.view(torch.int32)
    pytorch_bits = torch.exp(x).view(torch.int32)  # 直接view
    
    match = (snn_bits == pytorch_bits).sum().item()
    return match, n_samples


def test_sqrt(device, n_samples=100):
    """测试SpikeFP32Sqrt - 并行（端到端浮点验证）"""
    from SNNTorch.atomic_ops import SpikeFP32Sqrt
    
    sqrt_mod = SpikeFP32Sqrt().to(device)
    decoder = PulseFP32Decoder().to(device)
    torch.manual_seed(42)
    
    x = torch.abs(torch.randn(n_samples, device=device)) * 10 + 0.1
    x_pulse = float_to_pulse_vectorized(x)
    
    sqrt_mod.reset()
    result_pulse = sqrt_mod(x_pulse)
    
    decoder.reset()
    snn_result = decoder(result_pulse)
    snn_bits = snn_result.view(torch.int32)
    pytorch_bits = torch.sqrt(x).view(torch.int32)
    
    match = (snn_bits == pytorch_bits).sum().item()
    return match, n_samples


def test_sigmoid(device, n_samples=100):
    """测试SpikeFP32Sigmoid - 并行（端到端浮点验证）"""
    from SNNTorch.atomic_ops import SpikeFP32Sigmoid
    
    sig_mod = SpikeFP32Sigmoid().to(device)
    decoder = PulseFP32Decoder().to(device)
    torch.manual_seed(42)
    
    x = torch.randn(n_samples, device=device) * 3
    x_pulse = float_to_pulse_vectorized(x)
    
    sig_mod.reset()
    result_pulse = sig_mod(x_pulse)
    
    decoder.reset()
    snn_result = decoder(result_pulse)
    snn_bits = snn_result.view(torch.int32)
    pytorch_bits = torch.sigmoid(x).view(torch.int32)
    
    match = (snn_bits == pytorch_bits).sum().item()
    return match, n_samples


def test_silu(device, n_samples=100):
    """测试SpikeFP32SiLU - 并行（端到端浮点验证）"""
    from SNNTorch.atomic_ops import SpikeFP32SiLU
    
    silu_mod = SpikeFP32SiLU().to(device)
    decoder = PulseFP32Decoder().to(device)
    torch.manual_seed(42)
    
    x = torch.randn(n_samples, device=device) * 3
    x_pulse = float_to_pulse_vectorized(x)
    
    silu_mod.reset()
    result_pulse = silu_mod(x_pulse)
    
    decoder.reset()
    snn_result = decoder(result_pulse)
    snn_bits = snn_result.view(torch.int32)
    pytorch_bits = F.silu(x).view(torch.int32)
    
    match = (snn_bits == pytorch_bits).sum().item()
    return match, n_samples


def test_embedding(device):
    """测试SpikeFP32Embedding（端到端浮点验证）"""
    from SNNTorch.atomic_ops import SpikeFP32Embedding
    
    vocab_size = 16
    embed_dim = 8
    
    decoder = PulseFP32Decoder().to(device)
    
    torch.manual_seed(42)
    ann_emb = nn.Embedding(vocab_size, embed_dim).to(device)
    snn_emb = SpikeFP32Embedding(vocab_size, embed_dim).to(device)
    snn_emb.from_nn_embedding(ann_emb)
    
    # 批量测试所有token
    tokens = torch.arange(vocab_size, device=device)
    
    with torch.no_grad():
        ann_out = ann_emb(tokens)  # [V, D]
    snn_pulse = snn_emb(tokens)  # [V, D, 32]
    
    decoder.reset()
    snn_result = decoder(snn_pulse)
    
    ann_bits = ann_out.view(torch.int32)  # [V, D]
    snn_bits = snn_result.view(torch.int32)  # [V, D]
    
    match = (ann_bits == snn_bits).sum().item()
    total = vocab_size * embed_dim
    return match, total


def main():
    print("="*60)
    print("FP32组件验证测试 (GPU并行)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    results = {}
    
    # Exp
    print("测试 SpikeFP32Exp...", end=" ", flush=True)
    t0 = time.time()
    match, total = test_exp(device, 100)
    t1 = time.time()
    results['Exp'] = (match, total, t1-t0)
    print(f"{match}/{total} ({100*match/total:.1f}%) - {t1-t0:.2f}s")
    
    # Sqrt
    print("测试 SpikeFP32Sqrt...", end=" ", flush=True)
    t0 = time.time()
    match, total = test_sqrt(device, 100)
    t1 = time.time()
    results['Sqrt'] = (match, total, t1-t0)
    print(f"{match}/{total} ({100*match/total:.1f}%) - {t1-t0:.2f}s")
    
    # Sigmoid
    print("测试 SpikeFP32Sigmoid...", end=" ", flush=True)
    t0 = time.time()
    match, total = test_sigmoid(device, 100)
    t1 = time.time()
    results['Sigmoid'] = (match, total, t1-t0)
    print(f"{match}/{total} ({100*match/total:.1f}%) - {t1-t0:.2f}s")
    
    # SiLU
    print("测试 SpikeFP32SiLU...", end=" ", flush=True)
    t0 = time.time()
    match, total = test_silu(device, 100)
    t1 = time.time()
    results['SiLU'] = (match, total, t1-t0)
    print(f"{match}/{total} ({100*match/total:.1f}%) - {t1-t0:.2f}s")
    
    # Embedding
    print("测试 SpikeFP32Embedding...", end=" ", flush=True)
    t0 = time.time()
    match, total = test_embedding(device)
    t1 = time.time()
    results['Embedding'] = (match, total, t1-t0)
    print(f"{match}/{total} ({100*match/total:.1f}%) - {t1-t0:.2f}s")
    
    # 汇总
    print()
    print("="*60)
    print("汇总")
    print("="*60)
    
    for name, (match, total, elapsed) in results.items():
        rate = 100 * match / total
        status = "100%位精确" if rate == 100 else f"{rate:.1f}%"
        print(f"  {name}: {status}")
    
    print("="*60)


if __name__ == "__main__":
    main()

