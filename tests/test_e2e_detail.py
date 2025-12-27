"""
详细调试：找出SNN与手动顺序累加不匹配的原因
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Multiplier,
    SpikeFP8Adder_Spatial,
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


def debug_failing_case():
    """调试一个失败案例"""
    print("\n" + "="*70)
    print("调试失败案例")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    mul = SpikeFP8Multiplier().to(device)
    adder = SpikeFP8Adder_Spatial().to(device)
    
    # 使用产生差异的种子
    torch.manual_seed(4 * 2 + 20)  # in=4, out=2, batch=20
    
    in_features = 4
    out_features = 2
    batch_size = 20
    
    w_float = torch.randn(out_features, in_features, device=device) * 0.3
    x_float = torch.randn(batch_size, in_features, device=device) * 0.3
    
    w_fp8 = float_to_fp8_tensor(w_float)
    x_fp8 = float_to_fp8_tensor(x_float)
    w_fp8_f32 = w_fp8.float()
    x_fp8_f32 = x_fp8.float()
    
    # 只测试第一个失败样本 [0,1]
    b, o = 0, 1
    
    print(f"\n样本 [{b},{o}]:")
    print(f"x[{b}] (FP8): {x_fp8_f32[b].tolist()}")
    print(f"w[{o}] (FP8): {w_fp8_f32[o].tolist()}")
    
    # ===== 手动FP8顺序累加 =====
    print("\n--- 手动FP8顺序累加 ---")
    products_ref = []
    for i in range(in_features):
        p = x_fp8_f32[b, i] * w_fp8_f32[o, i]
        p_fp8 = float_to_fp8_tensor(p.unsqueeze(0))
        products_ref.append(p_fp8)
        p_bytes = fp8_tensor_to_bytes(p_fp8).item()
        print(f"  p{i}: {p.item():.6f} -> FP8 bytes={p_bytes}")
    
    acc = products_ref[0].float().squeeze()
    print(f"\n  acc[0] = {acc.item():.6f}")
    for i in range(1, in_features):
        acc_sum = acc + products_ref[i].float().squeeze()
        acc_fp8 = float_to_fp8_tensor(acc_sum.unsqueeze(0))
        acc = acc_fp8.float().squeeze()
        print(f"  acc[{i}] = {acc.item():.6f} (bytes={fp8_tensor_to_bytes(acc_fp8).item()})")
    
    ref_bytes = fp8_tensor_to_bytes(float_to_fp8_tensor(acc.unsqueeze(0))).item()
    print(f"\n手动结果 bytes: {ref_bytes}")
    
    # ===== SNN计算 =====
    print("\n--- SNN计算 ---")
    x_pulse = encoder(x_fp8_f32[b])  # [4, 8]
    w_pulse = encoder(w_fp8_f32[o])  # [4, 8]
    
    # 检查编码
    print("\n编码检查:")
    x_orig_bytes = fp8_tensor_to_bytes(x_fp8[b])
    w_orig_bytes = fp8_tensor_to_bytes(w_fp8[o])
    x_enc_bytes = pulse_to_fp8_bytes(x_pulse)
    w_enc_bytes = pulse_to_fp8_bytes(w_pulse)
    
    x_match = (x_enc_bytes == x_orig_bytes).all().item()
    w_match = (w_enc_bytes == w_orig_bytes).all().item()
    print(f"  x 编码匹配: {'✓' if x_match else '✗'}")
    print(f"  w 编码匹配: {'✓' if w_match else '✗'}")
    
    if not x_match:
        for i in range(in_features):
            if x_enc_bytes[i].item() != x_orig_bytes[i].item():
                print(f"    x[{i}]: 原始={x_orig_bytes[i].item()}, 编码={x_enc_bytes[i].item()}")
    if not w_match:
        for i in range(in_features):
            if w_enc_bytes[i].item() != w_orig_bytes[i].item():
                print(f"    w[{i}]: 原始={w_orig_bytes[i].item()}, 编码={w_enc_bytes[i].item()}")
    
    # SNN乘积
    print("\n乘积检查:")
    snn_products = []
    for i in range(in_features):
        mul.reset()
        p = mul(x_pulse[i], w_pulse[i])
        snn_products.append(p)
        p_bytes = pulse_to_fp8_bytes(p.unsqueeze(0)).item()
        ref_p_bytes = fp8_tensor_to_bytes(products_ref[i]).item()
        match = '✓' if p_bytes == ref_p_bytes else '✗'
        print(f"  p{i}: SNN={p_bytes}, 参考={ref_p_bytes} {match}")
    
    # SNN顺序累加
    print("\n累加检查:")
    acc_snn = snn_products[0]
    print(f"  acc[0]: bytes={pulse_to_fp8_bytes(acc_snn.unsqueeze(0)).item()}")
    
    for i in range(1, in_features):
        adder.reset()
        
        # 获取累加前的值
        acc_bytes_before = pulse_to_fp8_bytes(acc_snn.unsqueeze(0)).item()
        p_bytes = pulse_to_fp8_bytes(snn_products[i].unsqueeze(0)).item()
        
        acc_snn = adder(acc_snn, snn_products[i])
        acc_bytes = pulse_to_fp8_bytes(acc_snn.unsqueeze(0)).item()
        
        # 计算参考值
        ref_acc_fp8 = float_to_fp8_tensor(
            products_ref[0].float().squeeze() if i == 1 else 
            float_to_fp8_tensor(sum(products_ref[:i], products_ref[0]).float()).float().squeeze()
        )
        # 重新计算参考
        ref_acc = products_ref[0].float().squeeze()
        for j in range(1, i+1):
            ref_acc = float_to_fp8_tensor((ref_acc + products_ref[j].float().squeeze()).unsqueeze(0)).float().squeeze()
        ref_acc_bytes = fp8_tensor_to_bytes(float_to_fp8_tensor(ref_acc.unsqueeze(0))).item()
        
        match = '✓' if acc_bytes == ref_acc_bytes else '✗'
        print(f"  acc[{i}]: SNN={acc_bytes}, 参考={ref_acc_bytes} {match}")
        if acc_bytes != ref_acc_bytes:
            print(f"        输入: acc={acc_bytes_before} + p{i}={p_bytes}")
    
    snn_bytes = pulse_to_fp8_bytes(acc_snn.unsqueeze(0)).item()
    print(f"\nSNN结果 bytes: {snn_bytes}")
    print(f"参考结果 bytes: {ref_bytes}")
    print(f"匹配: {'✓' if snn_bytes == ref_bytes else '✗'}")


if __name__ == "__main__":
    debug_failing_case()
