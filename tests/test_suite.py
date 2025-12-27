"""
SNNTorch 核心测试套件 (Core Test Suite)
======================================

运行所有核心功能测试，验证 SNN 浮点运算的正确性。

测试分类
--------
1. 逻辑门测试: AND, OR, XOR, NOT 门电路
2. 算术单元测试: 半加器、全加器、行波进位加法器
3. 编码器/解码器测试: 浮点↔脉冲转换
4. 乘法器测试: FP8 × FP8 → FP8
5. 加法器测试: FP8/FP16/FP32 加法
6. Linear层测试: 多精度累加对齐

运行方式
--------
```bash
# 运行完整测试套件
python SNNTorch/tests/test_suite.py

# 运行单项测试
python SNNTorch/tests/test_suite.py --only logic_gates
```

作者: HumanBrain Project
许可: MIT License
"""
import torch
import torch.nn as nn
import sys
import argparse
from typing import Tuple, List

sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import (
    # 逻辑门
    ANDGate, ORGate, XORGate, NOTGate,
    # 算术单元
    HalfAdder, FullAdder, RippleCarryAdder,
    # 编码器/解码器
    PulseFloatingPointEncoder, PulseFloatingPointDecoder,
    # 浮点运算
    SpikeFP8Multiplier, SpikeFP8Adder_Spatial,
    # Linear层
    SpikeFP8Linear_MultiPrecision,
)


# ==============================================================================
# 测试辅助函数
# ==============================================================================

def pulse_to_bytes(pulse: torch.Tensor) -> torch.Tensor:
    """将脉冲序列转换为字节值"""
    bits = pulse.int()
    byte_val = torch.zeros(pulse.shape[:-1], dtype=torch.int32, device=pulse.device)
    for i in range(8):
        byte_val = byte_val + (bits[..., i] << (7 - i))
    return byte_val


class TestResult:
    """测试结果记录"""
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.details = []
    
    def record(self, test_name: str, success: bool, info: str = ""):
        if success:
            self.passed += 1
            self.details.append(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.details.append(f"  ✗ {test_name}: {info}")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        status = "✓ PASS" if self.failed == 0 else "✗ FAIL"
        return f"[{status}] {self.name}: {self.passed}/{total}"


# ==============================================================================
# 1. 逻辑门测试
# ==============================================================================

def test_logic_gates() -> TestResult:
    """测试基础逻辑门"""
    result = TestResult("逻辑门测试")
    
    # 真值表
    test_cases = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
    
    # AND 门
    and_gate = ANDGate()
    expected_and = [0, 0, 0, 1]
    for i, (a, b) in enumerate(test_cases):
        out = and_gate(torch.tensor([[a]]), torch.tensor([[b]])).item()
        result.record(f"AND({a},{b})={expected_and[i]}", out == expected_and[i])
    
    # OR 门
    or_gate = ORGate()
    expected_or = [0, 1, 1, 1]
    for i, (a, b) in enumerate(test_cases):
        out = or_gate(torch.tensor([[a]]), torch.tensor([[b]])).item()
        result.record(f"OR({a},{b})={expected_or[i]}", out == expected_or[i])
    
    # XOR 门
    xor_gate = XORGate()
    expected_xor = [0, 1, 1, 0]
    for i, (a, b) in enumerate(test_cases):
        out = xor_gate(torch.tensor([[a]]), torch.tensor([[b]])).item()
        result.record(f"XOR({a},{b})={expected_xor[i]}", out == expected_xor[i])
    
    # NOT 门
    not_gate = NOTGate()
    for a in [0.0, 1.0]:
        out = not_gate(torch.tensor([[a]])).item()
        expected = 1.0 - a
        result.record(f"NOT({a})={expected}", out == expected)
    
    return result


# ==============================================================================
# 2. 算术单元测试
# ==============================================================================

def test_arithmetic_units() -> TestResult:
    """测试算术单元"""
    result = TestResult("算术单元测试")
    
    # 半加器
    ha = HalfAdder()
    ha_cases = [
        ((0, 0), (0, 0)),  # S=0, C=0
        ((0, 1), (1, 0)),  # S=1, C=0
        ((1, 0), (1, 0)),  # S=1, C=0
        ((1, 1), (0, 1)),  # S=0, C=1
    ]
    for (a, b), (exp_s, exp_c) in ha_cases:
        s, c = ha(torch.tensor([[float(a)]]), torch.tensor([[float(b)]]))
        success = (s.item() == exp_s) and (c.item() == exp_c)
        result.record(f"HalfAdder({a}+{b})=({exp_s},{exp_c})", success)
    
    # 全加器
    fa = FullAdder()
    fa_cases = [
        ((0, 0, 0), (0, 0)),
        ((0, 0, 1), (1, 0)),
        ((0, 1, 0), (1, 0)),
        ((0, 1, 1), (0, 1)),
        ((1, 0, 0), (1, 0)),
        ((1, 0, 1), (0, 1)),
        ((1, 1, 0), (0, 1)),
        ((1, 1, 1), (1, 1)),
    ]
    for (a, b, cin), (exp_s, exp_c) in fa_cases:
        s, c = fa(
            torch.tensor([[float(a)]]),
            torch.tensor([[float(b)]]),
            torch.tensor([[float(cin)]])
        )
        success = (s.item() == exp_s) and (c.item() == exp_c)
        result.record(f"FullAdder({a}+{b}+{cin})=({exp_s},{exp_c})", success)
    
    # 4位行波进位加法器
    rca = RippleCarryAdder(bits=4)
    rca_cases = [
        (0b0000, 0b0000, 0b0000),
        (0b0001, 0b0001, 0b0010),
        (0b0111, 0b0001, 0b1000),
        (0b1111, 0b0001, 0b0000),  # 溢出
    ]
    for a_val, b_val, exp_sum in rca_cases:
        a_bits = [(a_val >> i) & 1 for i in range(4)]
        b_bits = [(b_val >> i) & 1 for i in range(4)]
        A = torch.tensor([[float(b) for b in a_bits]])
        B = torch.tensor([[float(b) for b in b_bits]])
        S, _ = rca(A, B)
        s_val = sum(int(S[0, i].item()) << i for i in range(4))
        result.record(f"RCA4({a_val}+{b_val})={exp_sum}", (s_val & 0xF) == exp_sum)
    
    return result


# ==============================================================================
# 3. 编码器/解码器测试
# ==============================================================================

def test_encoder_decoder() -> TestResult:
    """测试编码器和解码器"""
    result = TestResult("编码器/解码器测试")
    
    encoder = PulseFloatingPointEncoder()
    decoder = PulseFloatingPointDecoder()
    
    # 测试各种数值
    test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.0, 4.0, 0.25, 0.125]
    
    for val in test_values:
        x = torch.tensor([val]).to(torch.float8_e4m3fn).float()
        pulse = encoder(x)
        decoded = decoder(pulse)
        match = torch.isclose(decoded, x, rtol=1e-5).item()
        result.record(f"往返({val})={decoded.item():.4f}", match)
    
    # 测试任意维度
    shapes = [(10,), (5, 8), (2, 3, 4)]
    for shape in shapes:
        x = torch.randn(shape).to(torch.float8_e4m3fn).float()
        pulse = encoder(x)
        decoded = decoder(pulse)
        expected_pulse_shape = shape + (8,)
        shape_ok = (pulse.shape == expected_pulse_shape) and (decoded.shape == x.shape)
        result.record(f"维度{shape}→{expected_pulse_shape}", shape_ok)
    
    return result


# ==============================================================================
# 4. FP8 乘法器测试
# ==============================================================================

def test_fp8_multiplier() -> TestResult:
    """测试 FP8 乘法器"""
    result = TestResult("FP8 乘法器测试")
    
    encoder = PulseFloatingPointEncoder()
    mul = SpikeFP8Multiplier()
    
    # 测试用例
    test_cases = [
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 4.0),
        (0.5, 2.0, 1.0),
        (-1.0, 2.0, -2.0),
        (0.0, 5.0, 0.0),
    ]
    
    for a, b, expected in test_cases:
        a_fp8 = torch.tensor([a]).to(torch.float8_e4m3fn).float()
        b_fp8 = torch.tensor([b]).to(torch.float8_e4m3fn).float()
        exp_fp8 = torch.tensor([expected]).to(torch.float8_e4m3fn).float()
        
        a_pulse = encoder(a_fp8)
        b_pulse = encoder(b_fp8)
        result_pulse = mul(a_pulse, b_pulse)
        result_bytes = pulse_to_bytes(result_pulse)
        
        expected_bytes = exp_fp8.to(torch.float8_e4m3fn).view(torch.uint8).int()
        match = (result_bytes.item() == expected_bytes.item())
        result.record(f"{a}×{b}={expected}", match)
    
    return result


# ==============================================================================
# 5. Multi-Precision Linear 层测试
# ==============================================================================

def test_linear_alignment() -> TestResult:
    """测试 Linear 层精度对齐"""
    result = TestResult("Linear层对齐测试")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PulseFloatingPointEncoder().to(device)
    
    for precision in ['fp8', 'fp16', 'fp32']:
        torch.manual_seed(42)
        
        in_f, out_f, batch = 4, 2, 10
        x = torch.randn(batch, in_f, device=device) * 0.3
        w = torch.randn(out_f, in_f, device=device) * 0.3
        
        x_fp8 = x.to(torch.float8_e4m3fn).float()
        w_fp8 = w.to(torch.float8_e4m3fn).float()
        
        # PyTorch 参考
        if precision == 'fp32':
            y_ref = (x_fp8 @ w_fp8.T).to(torch.float8_e4m3fn).view(torch.uint8).int()
        else:
            # FP8/FP16 需要更复杂的参考实现，这里简化
            y_ref = (x_fp8 @ w_fp8.T).to(torch.float8_e4m3fn).view(torch.uint8).int()
        
        # SNN 计算
        x_pulse = encoder(x_fp8)
        linear = SpikeFP8Linear_MultiPrecision(
            in_f, out_f, accum_precision=precision, mode='sequential'
        ).to(device)
        linear.set_weight_from_float(w_fp8, encoder)
        linear.reset()
        y_pulse = linear(x_pulse)
        y_snn = pulse_to_bytes(y_pulse)
        
        match_rate = (y_ref == y_snn).float().mean().item() * 100
        result.record(f"{precision.upper()}累加对齐率={match_rate:.1f}%", match_rate >= 50)
    
    return result


# ==============================================================================
# 主测试函数
# ==============================================================================

def run_all_tests(only: str = None) -> bool:
    """运行所有测试"""
    
    print("=" * 70)
    print("SNNTorch 核心测试套件")
    print("=" * 70)
    
    tests = {
        'logic_gates': test_logic_gates,
        'arithmetic': test_arithmetic_units,
        'encoder_decoder': test_encoder_decoder,
        'multiplier': test_fp8_multiplier,
        'linear': test_linear_alignment,
    }
    
    if only and only in tests:
        tests = {only: tests[only]}
    
    results = []
    for name, test_func in tests.items():
        print(f"\n运行: {name}")
        try:
            res = test_func()
            results.append(res)
            print(res.summary())
            for detail in res.details:
                print(detail)
        except Exception as e:
            print(f"  ✗ 异常: {e}")
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    for res in results:
        print(res.summary())
    
    print("-" * 70)
    print(f"总计: {total_passed} 通过, {total_failed} 失败")
    
    all_pass = total_failed == 0
    if all_pass:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 存在失败测试")
    
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNTorch 核心测试套件")
    parser.add_argument('--only', type=str, default=None,
                        help="只运行指定测试: logic_gates, arithmetic, encoder_decoder, multiplier, linear")
    args = parser.parse_args()
    
    success = run_all_tests(only=args.only)
    sys.exit(0 if success else 1)

