"""
纯FP8 Linear测试：用纯FP8计算作为参考
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
        y_ref_bytes = y_ref.view(torch.uint8).int()
        y_ref_float = y_ref.float()
        
        # SNN
        x_pulse = encoder(x_fp8_f32)
        snn_linear = SpikeFP8Linear_Fast(in_f, out_f, mode='sequential').to(device)
        snn_linear.set_weight_from_float(w_fp8_f32, encoder)
        snn_linear.reset()
        y_snn_pulse = snn_linear(x_pulse)
        y_snn_bytes = pulse_to_fp8_bytes(y_snn_pulse)
        
        # Decode SNN manually for float comparison
        s = y_snn_pulse[..., 0]
        e = y_snn_pulse[..., 1:5]
        m = y_snn_pulse[..., 5:8]
        sign = (-1) ** s
        e_val = torch.zeros_like(s)
        for i in range(4): e_val += e[..., i] * (2**(3-i))
        m_val = torch.zeros_like(s)
        for i in range(3): m_val += m[..., i] * (2**(2-i))
        is_sub = (e_val == 0)
        val_norm = sign * (2 ** (e_val - 7)) * (1 + m_val / 8.0)
        val_sub = sign * (2 ** -6) * (m_val / 8.0)
        y_snn_float = torch.where(is_sub, val_sub, val_norm)
        # Fix zero: e=0, m=0 -> 0
        y_snn_float = torch.where((e_val==0) & (m_val==0), torch.zeros_like(y_snn_float), y_snn_float)
        
        match = (y_ref_bytes == y_snn_bytes)
        n_match = match.sum().item()
        n_total = y_snn_bytes.numel()
        
        print(f"Match: {n_match}/{n_total}")
        
        if n_match < n_total:
            print("\n!!! MISMATCH DETAILS !!!")
            for b in range(batch):
                for o in range(out_f):
                    if not match[b, o]:
                        print(f"Idx ({b},{o}): Ref={y_ref_float[b,o].item():.4f} (Int:{y_ref_bytes[b,o].item()}), SNN={y_snn_float[b,o].item():.4f} (Int:{y_snn_bytes[b,o].item()})")
                        print(f"    Diff: {y_snn_float[b,o].item() - y_ref_float[b,o].item()}")

if __name__ == "__main__":
    main()
