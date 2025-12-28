"""
纯FP8 Linear测试：用纯FP8计算作为参考（端到端浮点验证）
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_Fast
)
from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder


def pure_fp8_matmul_sequential(x_fp8, w_fp8):
    """纯FP8顺序累加 matmul"""
    batch, in_f = x_fp8.shape
    out_f = w_fp8.shape[0]
    device = x_fp8.device
    
    result = torch.zeros(batch, out_f, dtype=torch.float8_e4m3fn, device=device)
    
    for b in range(batch):
        for o in range(out_f):
            # 计算所有乘积（每个乘积立即转FP8）
            products = []
            for i in range(in_f):
                p = x_fp8[b, i].float() * w_fp8[o, i].float()
                p_fp8 = p.to(torch.float8_e4m3fn)
                products.append(p_fp8)
            
            # 顺序累加（每步立即转FP8）
            acc = products[0]
            for i in range(1, in_f):
                acc = (acc.float() + products[i].float()).to(torch.float8_e4m3fn)
            
            result[b, o] = acc
    
    return result


def pure_fp8_matmul_tree(x_fp8, w_fp8):
    """纯FP8树形累加 matmul"""
    batch, in_f = x_fp8.shape
    out_f = w_fp8.shape[0]
    device = x_fp8.device
    
    result = torch.zeros(batch, out_f, dtype=torch.float8_e4m3fn, device=device)
    
    for b in range(batch):
        for o in range(out_f):
            # 计算所有乘积
            products = []
            for i in range(in_f):
                p = x_fp8[b, i].float() * w_fp8[o, i].float()
                p_fp8 = p.to(torch.float8_e4m3fn)
                products.append(p_fp8)
            
            # 树形累加
            current = products
            while len(current) > 1:
                next_level = []
                for i in range(0, len(current), 2):
                    if i + 1 < len(current):
                        s = (current[i].float() + current[i+1].float()).to(torch.float8_e4m3fn)
                        next_level.append(s)
                    else:
                        next_level.append(current[i])
                current = next_level
            
            result[b, o] = current[0]
    
    return result


def main():
    print("="*70)
    print("纯FP8 Linear测试：SNN vs 纯FP8计算")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Run a small debug case
    configs = [
        (4, 2, 2), # Small case to debug
    ]
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    print("\n" + "="*70)
    print("Sequential模式 vs 纯FP8顺序累加")
    print("="*70)
    
    for in_f, out_f, batch in configs:
        torch.manual_seed(in_f * out_f + batch)
        
        x_float = torch.randn(batch, in_f, device=device) * 1.0 # Standard range
        w_float = torch.randn(out_f, in_f, device=device) * 1.0
        
        x_fp8 = x_float.to(torch.float8_e4m3fn)
        w_fp8 = w_float.to(torch.float8_e4m3fn)
        x_fp8_f32 = x_fp8.float()
        w_fp8_f32 = w_fp8.float()
        
        # 纯FP8参考
        y_ref = pure_fp8_matmul_sequential(x_fp8, w_fp8)
        y_ref_float = y_ref.float()
        
        # SNN
        x_pulse = encoder(x_fp8_f32)
        snn_linear = SpikeFP8Linear_Fast(in_f, out_f, mode='sequential').to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        y_snn_pulse = snn_linear(x_pulse)
        
        # 使用框架解码器
        decoder = PulseFloatingPointDecoder().to(device)
        decoder.reset()
        y_snn_float = decoder(y_snn_pulse)
        
        # 直接比较浮点数
        match = torch.isclose(y_snn_float, y_ref_float, rtol=1e-5, atol=1e-6) | (y_snn_float == y_ref_float)
        n_match = match.sum().item()
        n_total = y_snn_float.numel()
        
        print(f"Match: {n_match}/{n_total}")
        
        if n_match < n_total:
            print("\n!!! MISMATCH DETAILS !!!")
            for b in range(batch):
                for o in range(out_f):
                    if not match[b, o]:
                        print(f"Idx ({b},{o}): Ref={y_ref_float[b,o].item():.6f}, SNN={y_snn_float[b,o].item():.6f}")
                        print(f"    Diff: {y_snn_float[b,o].item() - y_ref_float[b,o].item():.6f}")

if __name__ == "__main__":
    main()
