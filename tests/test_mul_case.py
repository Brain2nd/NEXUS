"""
调试乘法器特定案例（端到端浮点验证）
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Multiplier,
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


def analyze_fp8(byte_val):
    """分析FP8字节值"""
    s = (byte_val >> 7) & 1
    e = (byte_val >> 3) & 0xF
    m = byte_val & 0x7
    
    if e == 0 and m == 0:
        val = 0.0
    elif e == 0:  # subnormal
        val = (m / 8.0) * (2 ** -6)
    else:  # normal
        val = (1 + m / 8.0) * (2 ** (e - 7))
    
    if s:
        val = -val
    
    return s, e, m, val


def test_specific_multiply():
    """测试具体的乘法案例"""
    print("\n" + "="*70)
    print("测试: -0.05078125 × -0.21875")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    mul = SpikeFP8Multiplier().to(device)
    
    a_val = -0.05078125
    b_val = -0.21875
    
    a = torch.tensor([a_val], device=device)
    b = torch.tensor([b_val], device=device)
    
    # PyTorch FP8
    a_fp8 = float_to_fp8_tensor(a)
    b_fp8 = float_to_fp8_tensor(b)
    
    a_bytes = fp8_tensor_to_bytes(a_fp8).item()
    b_bytes = fp8_tensor_to_bytes(b_fp8).item()
    
    s_a, e_a, m_a, v_a = analyze_fp8(a_bytes)
    s_b, e_b, m_b, v_b = analyze_fp8(b_bytes)
    
    print(f"\na = {a_val}")
    print(f"  FP8 bytes: {a_bytes} = 0b{a_bytes:08b}")
    print(f"  S={s_a}, E={e_a}, M={m_a}")
    print(f"  FP8 value: {v_a}")
    
    print(f"\nb = {b_val}")
    print(f"  FP8 bytes: {b_bytes} = 0b{b_bytes:08b}")
    print(f"  S={s_b}, E={e_b}, M={m_b}")
    print(f"  FP8 value: {v_b}")
    
    # 理论乘积
    product_exact = v_a * v_b
    print(f"\n理论乘积: {product_exact}")
    
    # PyTorch FP8乘积
    product_fp8 = float_to_fp8_tensor(torch.tensor([product_exact], device=device))
    product_bytes = fp8_tensor_to_bytes(product_fp8).item()
    s_p, e_p, m_p, v_p = analyze_fp8(product_bytes)
    
    print(f"\nPyTorch FP8乘积:")
    print(f"  bytes: {product_bytes} = 0b{product_bytes:08b}")
    print(f"  S={s_p}, E={e_p}, M={m_p}")
    print(f"  value: {v_p}")
    
    # SNN乘法
    a_fp8_f32 = a_fp8.float()
    b_fp8_f32 = b_fp8.float()
    
    a_pulse = encoder(a_fp8_f32)
    b_pulse = encoder(b_fp8_f32)
    
    print(f"\n编码检查:")
    print(f"  a_pulse bytes: {pulse_to_fp8_bytes(a_pulse).item()}")
    print(f"  b_pulse bytes: {pulse_to_fp8_bytes(b_pulse).item()}")
    
    mul.reset()
    p_snn = mul(a_pulse[0], b_pulse[0])
    p_snn_bytes = pulse_to_fp8_bytes(p_snn.unsqueeze(0)).item()
    s_snn, e_snn, m_snn, v_snn = analyze_fp8(p_snn_bytes)
    
    print(f"\nSNN乘积:")
    print(f"  bytes: {p_snn_bytes} = 0b{p_snn_bytes:08b}")
    print(f"  S={s_snn}, E={e_snn}, M={m_snn}")
    print(f"  value: {v_snn}")
    
    print(f"\n期望 bytes: {product_bytes}, 实际 bytes: {p_snn_bytes}")
    print(f"匹配: {'✓' if product_bytes == p_snn_bytes else '✗'}")
    
    # 分析乘法过程
    print("\n--- 乘法过程分析 ---")
    print(f"符号: {s_a} XOR {s_b} = {s_a ^ s_b}")
    
    # 指数: E_a + E_b - bias
    e_sum = e_a + e_b
    e_result = e_sum - 7  # bias = 7
    print(f"指数: {e_a} + {e_b} - 7 = {e_result}")
    
    # 尾数: (1.M_a) × (1.M_b)
    m_a_full = 1 + m_a / 8.0  # 1.xxx
    m_b_full = 1 + m_b / 8.0
    m_product = m_a_full * m_b_full
    print(f"尾数: {m_a_full} × {m_b_full} = {m_product}")
    
    # 如果尾数 >= 2，需要右移并指数+1
    if m_product >= 2:
        m_product_normalized = m_product / 2
        e_result += 1
        print(f"  规格化后: {m_product_normalized}, E={e_result}")
    else:
        m_product_normalized = m_product
    
    # 提取3位尾数
    m_frac = m_product_normalized - 1  # 去掉隐含的1
    m_result = int(m_frac * 8 + 0.5)  # 舍入
    print(f"  尾数小数部分: {m_frac}, M={m_result}")
    
    print(f"\n理论结果: S={s_a ^ s_b}, E={e_result}, M={m_result}")


def test_more_cases():
    """测试更多乘法案例"""
    print("\n" + "="*70)
    print("更多乘法案例测试")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    mul = SpikeFP8Multiplier().to(device)
    
    test_cases = [
        (-0.05078125, -0.21875),  # 问题案例
        (0.5, 0.5),
        (0.25, 0.25),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.125, 0.125),
        (0.0625, 0.0625),  # 小值
    ]
    
    print("\n| a | b | PyTorch bytes | SNN bytes | 匹配 |")
    print("|------|------|---------------|-----------|------|")
    
    for a_val, b_val in test_cases:
        a = torch.tensor([a_val], device=device)
        b = torch.tensor([b_val], device=device)
        
        a_fp8 = float_to_fp8_tensor(a)
        b_fp8 = float_to_fp8_tensor(b)
        
        product = a_fp8.float() * b_fp8.float()
        product_fp8 = float_to_fp8_tensor(product)
        product_bytes = fp8_tensor_to_bytes(product_fp8).item()
        
        a_pulse = encoder(a_fp8.float())
        b_pulse = encoder(b_fp8.float())
        
        mul.reset()
        p_snn = mul(a_pulse[0], b_pulse[0])
        p_snn_bytes = pulse_to_fp8_bytes(p_snn.unsqueeze(0)).item()
        
        match = "✓" if product_bytes == p_snn_bytes else "✗"
        print(f"| {a_val:7.4f} | {b_val:7.4f} | {product_bytes:13} | {p_snn_bytes:9} | {match} |")


if __name__ == "__main__":
    test_specific_multiply()
    test_more_cases()

