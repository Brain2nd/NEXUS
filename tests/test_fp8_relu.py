"""
FP8 ReLU 测试 - 验证 SpikeFP8ReLU 符合 SNN 原则且功能正确
=============================================================

测试内容:
1. 基本功能: 正数保持，负数变零
2. SNN 原则合规性验证
3. 批量处理
4. GPU 并行

作者: HumanBrain Project
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import struct

# 使用统一的转换函数
from atomic_ops import float_to_fp8_bits, fp8_bits_to_float


def test_basic_relu():
    """测试基本 ReLU 功能"""
    from atomic_ops import SpikeFP8ReLU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    relu = SpikeFP8ReLU().to(device)
    
    test_cases = [
        (1.0, 1.0, "正数保持"),
        (2.5, 2.5, "正数保持"),
        (0.0, 0.0, "零保持"),
        (-1.0, 0.0, "负数变零"),
        (-2.5, 0.0, "负数变零"),
        (0.5, 0.5, "小正数保持"),
        (-0.5, 0.0, "小负数变零"),
    ]
    
    passed = 0
    for val, expected, desc in test_cases:
        # 转换为 FP8 脉冲
        x_tensor = torch.tensor([val], dtype=torch.float32)
        x_pulse = float_to_fp8_bits(x_tensor, device)
        
        # 执行 ReLU
        relu.reset()
        result_pulse = relu(x_pulse)
        
        # 转换回 float
        result_float = fp8_bits_to_float(result_pulse).item()
        
        # FP8 量化后的期望值
        expected_fp8 = x_tensor.to(torch.float8_e4m3fn).to(torch.float32).item()
        expected_relu = max(0.0, expected_fp8)
        
        if abs(result_float - expected_relu) < 1e-6:
            passed += 1
            print(f"  ✓ {desc}: ReLU({val}) = {result_float}")
        else:
            print(f"  ✗ {desc}: ReLU({val}) = {result_float}, 期望 {expected_relu}")
    
    print(f"\n基本 ReLU: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_batch_relu():
    """测试批量 ReLU"""
    from atomic_ops import SpikeFP8ReLU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    relu = SpikeFP8ReLU().to(device)
    
    # 生成随机测试数据
    np.random.seed(42)
    batch_size = 100
    vals = (np.random.randn(batch_size) * 5).astype(np.float32)
    
    x_tensor = torch.from_numpy(vals)
    x_pulse = float_to_fp8_bits(x_tensor, device)
    
    # 执行 ReLU
    relu.reset()
    result_pulse = relu(x_pulse)
    result_float = fp8_bits_to_float(result_pulse).cpu().numpy()
    
    # 计算期望结果
    x_fp8 = x_tensor.to(torch.float8_e4m3fn).to(torch.float32).numpy()
    expected = np.maximum(0, x_fp8)
    
    # 验证
    matches = np.sum(np.abs(result_float - expected) < 1e-6)
    
    print(f"\n批量测试 ({batch_size} 样本):")
    print(f"  正确匹配: {matches}/{batch_size} ({100*matches/batch_size:.1f}%)")
    print(f"  正数数量: {np.sum(vals > 0)}")
    print(f"  负数数量: {np.sum(vals < 0)}")
    
    return matches == batch_size


def test_sign_bit_logic():
    """测试符号位逻辑"""
    from atomic_ops import SpikeFP8ReLU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    relu = SpikeFP8ReLU().to(device)
    
    # 手动构造脉冲: 符号位 = 1 (负数)
    neg_pulse = torch.tensor([[1., 0., 1., 1., 1., 0., 0., 0.]], device=device)  # 负数
    pos_pulse = torch.tensor([[0., 0., 1., 1., 1., 0., 0., 0.]], device=device)  # 正数
    
    relu.reset()
    neg_result = relu(neg_pulse)
    
    relu.reset()
    pos_result = relu(pos_pulse)
    
    # 负数应该全部变为 0
    neg_is_zero = torch.allclose(neg_result, torch.zeros_like(neg_result))
    # 正数应该保持不变
    pos_unchanged = torch.allclose(pos_result, pos_pulse)
    
    print(f"\n符号位逻辑测试:")
    print(f"  ✓ 负数 (符号位=1) → 全零: {neg_is_zero}")
    print(f"  ✓ 正数 (符号位=0) → 保持: {pos_unchanged}")
    
    return neg_is_zero and pos_unchanged


def test_snn_principle_compliance():
    """测试 SNN 原则合规性 - 验证模块结构"""
    from atomic_ops.fp8_relu import SpikeFP8ReLU
    
    # 检查模块是否使用了正确的组件
    relu = SpikeFP8ReLU()
    
    checks = []
    
    # 检查是否有 VecNOT 实例
    has_vec_not = hasattr(relu, 'vec_not') and relu.vec_not is not None
    checks.append(("使用 VecNOT", has_vec_not))
    
    # 检查是否有 VecAND 实例
    has_vec_and = hasattr(relu, 'vec_and') and relu.vec_and is not None
    checks.append(("使用 VecAND", has_vec_and))
    
    # 检查 forward 方法是否调用了 vec_not 和 vec_and
    import inspect
    forward_source = inspect.getsource(relu.forward)
    uses_vec_not = 'vec_not' in forward_source
    uses_vec_and = 'vec_and' in forward_source
    checks.append(("forward 调用 vec_not", uses_vec_not))
    checks.append(("forward 调用 vec_and", uses_vec_and))
    
    # 检查 forward 中是否有违规的直接运算
    # 只检查赋值语句中的违规
    has_direct_sub = False
    has_direct_mul = False
    for line in forward_source.split('\n'):
        code_part = line.split('#')[0].strip()
        # 检查赋值语句 (包含 = 但不是 ==)
        if '=' in code_part and '==' not in code_part:
            # 右侧是否有 (1 - 或 1-
            right_side = code_part.split('=', 1)[-1] if '=' in code_part else ''
            if '1 -' in right_side or '1-' in right_side:
                has_direct_sub = True
            if ' * ' in right_side and 'self.' not in right_side.split('*')[0]:
                has_direct_mul = True
    
    checks.append(("无直接 (1-x) 运算", not has_direct_sub))
    checks.append(("无直接乘法运算", not has_direct_mul))
    
    all_passed = all(c[1] for c in checks)
    
    print("\n" + ("✓" if all_passed else "✗") + " SNN 原则合规性检查:")
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")
    
    return all_passed


def test_multidim_relu():
    """测试多维输入"""
    from atomic_ops import SpikeFP8ReLU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    relu = SpikeFP8ReLU().to(device)
    
    # 测试 [batch, seq, 8] 形状
    batch, seq = 4, 16
    vals = torch.randn(batch, seq, dtype=torch.float32) * 3
    
    x_pulse = float_to_fp8_bits(vals, device)  # [4, 16, 8]
    
    relu.reset()
    result_pulse = relu(x_pulse)
    result_float = fp8_bits_to_float(result_pulse)
    
    # 期望结果
    x_fp8 = vals.to(torch.float8_e4m3fn).to(torch.float32)
    expected = torch.maximum(torch.zeros_like(x_fp8), x_fp8)
    
    matches = torch.sum(torch.abs(result_float.cpu() - expected) < 1e-6).item()
    total = batch * seq
    
    print(f"\n多维测试 [{batch}, {seq}]:")
    print(f"  正确匹配: {matches}/{total} ({100*matches/total:.1f}%)")
    
    return matches == total


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("FP8 ReLU 测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    tests = [
        ("基本 ReLU", test_basic_relu),
        ("批量 ReLU", test_batch_relu),
        ("符号位逻辑", test_sign_bit_logic),
        ("SNN 原则合规", test_snn_principle_compliance),
        ("多维输入", test_multidim_relu),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {name} 通过")
            else:
                failed += 1
                print(f"✗ {name} 失败")
        except Exception as e:
            print(f"✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
