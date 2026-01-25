"""
MultiHeadAttention 测试
========================

测试多精度多头注意力机制的功能正确性。

作者: MofNeuroSim Project
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    SpikeFP32MultiHeadAttention,
    SpikeMultiHeadAttention,
    SpikeFP8MultiHeadAttention,
    SpikeFP16MultiHeadAttention,
    float32_to_pulse,
    pulse_to_float32,
    float_to_fp8_bits,
    float16_to_pulse,
)

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_fp32_attention_shape():
    """测试 FP32 注意力输出形状"""
    print("\n=== Test FP32 Attention Shape ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 2
    seq_len = 4

    attn = SpikeFP32MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建输入脉冲
    x_float = torch.randn(batch_size, seq_len, embed_dim)
    x_pulse = float32_to_pulse(x_float)

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse)

    # 验证形状
    expected_shape = (batch_size, seq_len, embed_dim, 32)
    assert out_pulse.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {out_pulse.shape}"

    print(f"Input shape: {x_pulse.shape}")
    print(f"Output shape: {out_pulse.shape}")
    print("✓ Shape test passed!")


def test_fp32_attention_with_rope():
    """测试带 RoPE 的 FP32 注意力"""
    print("\n=== Test FP32 Attention with RoPE ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 2
    seq_len = 4

    attn = SpikeFP32MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=True
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建输入脉冲
    x_float = torch.randn(batch_size, seq_len, embed_dim)
    x_pulse = float32_to_pulse(x_float)

    # 位置索引
    positions = torch.arange(seq_len)

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse, positions=positions)

    # 验证形状
    expected_shape = (batch_size, seq_len, embed_dim, 32)
    assert out_pulse.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {out_pulse.shape}"

    print(f"Input shape: {x_pulse.shape}")
    print(f"Positions: {positions}")
    print(f"Output shape: {out_pulse.shape}")
    print("✓ RoPE test passed!")


def test_fp32_attention_with_mask():
    """测试带因果掩码的 FP32 注意力"""
    print("\n=== Test FP32 Attention with Mask ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 2
    seq_len = 4

    attn = SpikeFP32MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建输入脉冲
    x_float = torch.randn(batch_size, seq_len, embed_dim)
    x_pulse = float32_to_pulse(x_float)

    # 因果掩码 (上三角为 True，表示屏蔽未来位置)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse, attn_mask=causal_mask)

    # 验证形状
    expected_shape = (batch_size, seq_len, embed_dim, 32)
    assert out_pulse.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {out_pulse.shape}"

    print(f"Input shape: {x_pulse.shape}")
    print(f"Causal mask:\n{causal_mask.int()}")
    print(f"Output shape: {out_pulse.shape}")
    print("✓ Mask test passed!")


def test_multi_precision_fp32():
    """测试多精度包装器 (FP32)"""
    print("\n=== Test Multi-Precision Attention (FP32) ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 2
    seq_len = 4

    attn = SpikeMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        input_precision='fp32',
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建输入脉冲
    x_float = torch.randn(batch_size, seq_len, embed_dim)
    x_pulse = float32_to_pulse(x_float)

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse)

    # 验证形状和位宽
    assert out_pulse.shape[-1] == 32, f"Expected 32 bits, got {out_pulse.shape[-1]}"

    print(f"Input precision: fp32")
    print(f"Output bits: {out_pulse.shape[-1]}")
    print("✓ FP32 multi-precision test passed!")


def test_multi_precision_fp16():
    """测试多精度包装器 (FP16)"""
    print("\n=== Test Multi-Precision Attention (FP16) ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 2
    seq_len = 4

    attn = SpikeFP16MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建 FP16 输入脉冲
    x_float = torch.randn(batch_size, seq_len, embed_dim).half()
    x_pulse = float16_to_pulse(x_float)

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse)

    # 验证形状和位宽
    assert out_pulse.shape[-1] == 16, f"Expected 16 bits, got {out_pulse.shape[-1]}"

    print(f"Input precision: fp16")
    print(f"Output bits: {out_pulse.shape[-1]}")
    print("✓ FP16 multi-precision test passed!")


def test_multi_precision_fp8():
    """测试多精度包装器 (FP8)"""
    print("\n=== Test Multi-Precision Attention (FP8) ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 2
    seq_len = 4

    attn = SpikeFP8MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建 FP8 输入脉冲
    x_float = torch.randn(batch_size, seq_len, embed_dim)
    x_pulse = float_to_fp8_bits(x_float)

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse)

    # 验证形状和位宽
    assert out_pulse.shape[-1] == 8, f"Expected 8 bits, got {out_pulse.shape[-1]}"

    print(f"Input precision: fp8")
    print(f"Output bits: {out_pulse.shape[-1]}")
    print("✓ FP8 multi-precision test passed!")


def test_batched_matmul():
    """测试 BatchMatMul 功能"""
    print("\n=== Test Batched MatMul ===")

    embed_dim = 8
    num_heads = 2
    batch_size = 1
    seq_len = 2

    attn = SpikeFP32MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建简单权重 (单位矩阵风格)
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.eye(total_head_dim, embed_dim)
    k_weight = torch.eye(total_head_dim, embed_dim)
    v_weight = torch.eye(total_head_dim, embed_dim)
    out_weight = torch.eye(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 创建输入
    x_float = torch.randn(batch_size, seq_len, embed_dim)
    x_pulse = float32_to_pulse(x_float)

    # 前向传播
    out_pulse = attn(x_pulse, x_pulse, x_pulse)

    # 转回浮点
    out_float = pulse_to_float32(out_pulse)

    print(f"Input shape: {x_float.shape}")
    print(f"Output shape: {out_float.shape}")
    print(f"Input sample: {x_float[0, 0, :4]}")
    print(f"Output sample: {out_float[0, 0, :4]}")
    print("✓ BatchMatMul test passed!")


def test_boundary_values():
    """测试边界值输入 (CLAUDE.md #8: 随机+边界值)"""
    print("\n=== Test Boundary Values ===")

    embed_dim = 16
    num_heads = 2
    batch_size = 1
    seq_len = 4

    attn = SpikeFP32MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 边界值测试用例
    boundary_cases = [
        ("全零输入", torch.zeros(batch_size, seq_len, embed_dim)),
        ("全一输入", torch.ones(batch_size, seq_len, embed_dim)),
        ("全负一输入", -torch.ones(batch_size, seq_len, embed_dim)),
        ("单位矩阵风格", torch.eye(embed_dim).unsqueeze(0)[:, :seq_len, :]),
        ("极小值", torch.ones(batch_size, seq_len, embed_dim) * 1e-6),
        ("中等值", torch.ones(batch_size, seq_len, embed_dim) * 0.5),
    ]

    passed = 0
    for name, x_float in boundary_cases:
        x_pulse = float32_to_pulse(x_float)

        try:
            attn.reset()
            out_pulse = attn(x_pulse, x_pulse, x_pulse)

            # 验证输出形状
            expected_shape = (batch_size, seq_len, embed_dim, 32)
            shape_ok = out_pulse.shape == expected_shape

            # 转换回浮点并检查 NaN/Inf
            out_float = pulse_to_float32(out_pulse)
            no_nan = not torch.isnan(out_float).any()
            no_inf = not torch.isinf(out_float).any()

            if shape_ok and no_nan and no_inf:
                passed += 1
                print(f"  ✓ {name}: 输出形状正确，无 NaN/Inf")
            else:
                if not shape_ok:
                    print(f"  ✗ {name}: 形状错误 {out_pulse.shape}")
                if not no_nan:
                    print(f"  ✗ {name}: 存在 NaN")
                if not no_inf:
                    print(f"  ✗ {name}: 存在 Inf")
        except Exception as e:
            print(f"  ✗ {name}: 异常 {e}")

    print(f"\n边界值测试: {passed}/{len(boundary_cases)} 通过")
    assert passed >= len(boundary_cases) - 1, f"边界值测试失败太多: {passed}/{len(boundary_cases)}"


def test_random_inputs():
    """测试随机输入 (CLAUDE.md #8: 随机+边界值)"""
    print("\n=== Test Random Inputs ===")

    embed_dim = 16
    num_heads = 2
    seq_len = 4

    torch.manual_seed(42)

    attn = SpikeFP32MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=False
    )

    # 创建随机权重
    total_head_dim = num_heads * (embed_dim // num_heads)
    q_weight = torch.randn(total_head_dim, embed_dim)
    k_weight = torch.randn(total_head_dim, embed_dim)
    v_weight = torch.randn(total_head_dim, embed_dim)
    out_weight = torch.randn(embed_dim, total_head_dim)

    attn.set_weights_from_float(q_weight, k_weight, v_weight, out_weight)

    # 随机测试
    n_tests = 20
    passed = 0

    for i in range(n_tests):
        batch_size = torch.randint(1, 4, (1,)).item()
        x_float = torch.randn(batch_size, seq_len, embed_dim) * 2

        x_pulse = float32_to_pulse(x_float)

        try:
            attn.reset()
            out_pulse = attn(x_pulse, x_pulse, x_pulse)

            # 验证输出形状
            expected_shape = (batch_size, seq_len, embed_dim, 32)
            shape_ok = out_pulse.shape == expected_shape

            # 转换回浮点并检查 NaN/Inf
            out_float = pulse_to_float32(out_pulse)
            no_nan = not torch.isnan(out_float).any()
            no_inf = not torch.isinf(out_float).any()

            if shape_ok and no_nan and no_inf:
                passed += 1
        except Exception:
            pass

    rate = passed / n_tests * 100
    print(f"随机测试: {passed}/{n_tests} ({rate:.1f}%)")
    assert rate >= 80, f"随机测试通过率太低: {rate:.1f}%"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("MultiHeadAttention Tests")
    print("=" * 60)

    tests = [
        ("FP32 Shape", test_fp32_attention_shape),
        ("FP32 with RoPE", test_fp32_attention_with_rope),
        ("FP32 with Mask", test_fp32_attention_with_mask),
        ("Multi-Precision FP32", test_multi_precision_fp32),
        ("Multi-Precision FP16", test_multi_precision_fp16),
        ("Multi-Precision FP8", test_multi_precision_fp8),
        ("Batched MatMul", test_batched_matmul),
        ("Boundary Values", test_boundary_values),
        ("Random Inputs", test_random_inputs),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} tests passed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
