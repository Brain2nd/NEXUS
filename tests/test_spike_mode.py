"""
SpikeMode 测试套件
==================

验证 SpikeMode 双模式控制系统的正确性。

测试内容:
1. 全局模式切换
2. 上下文管理器
3. 实例级模式覆盖
4. BIT_EXACT 模式下的状态重置
5. TEMPORAL 模式下的残差保留
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import SpikeMode, ANDGate, VecAND

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_global_mode():
    """测试全局模式设置和获取"""
    # 默认应该是 BIT_EXACT
    assert SpikeMode.get_mode() == SpikeMode.BIT_EXACT
    assert SpikeMode.is_bit_exact()
    assert not SpikeMode.is_temporal()

    # 切换到 TEMPORAL
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    assert SpikeMode.get_mode() == SpikeMode.TEMPORAL
    assert SpikeMode.is_temporal()
    assert not SpikeMode.is_bit_exact()

    # 切换回 BIT_EXACT
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)
    assert SpikeMode.get_mode() == SpikeMode.BIT_EXACT

    print("[PASS] 全局模式测试通过")


def test_context_manager():
    """测试上下文管理器"""
    # 确保从 BIT_EXACT 开始
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)
    assert SpikeMode.is_bit_exact()

    # 使用 temporal() 上下文
    with SpikeMode.temporal():
        assert SpikeMode.is_temporal()
        assert SpikeMode.get_mode() == SpikeMode.TEMPORAL

    # 退出后应恢复
    assert SpikeMode.is_bit_exact()

    # 使用 bit_exact() 上下文
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    with SpikeMode.bit_exact():
        assert SpikeMode.is_bit_exact()
    assert SpikeMode.is_temporal()

    # 嵌套上下文
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)
    with SpikeMode.temporal():
        assert SpikeMode.is_temporal()
        with SpikeMode.bit_exact():
            assert SpikeMode.is_bit_exact()
        assert SpikeMode.is_temporal()
    assert SpikeMode.is_bit_exact()

    print("[PASS] 上下文管理器测试通过")


def test_should_reset():
    """测试 should_reset 逻辑"""
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    # 无实例模式 -> 跟随全局
    assert SpikeMode.should_reset(None) == True

    # 实例模式覆盖
    assert SpikeMode.should_reset(SpikeMode.TEMPORAL) == False
    assert SpikeMode.should_reset(SpikeMode.BIT_EXACT) == True

    # 切换全局模式
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    assert SpikeMode.should_reset(None) == False

    # 实例模式仍然覆盖
    assert SpikeMode.should_reset(SpikeMode.BIT_EXACT) == True

    # 恢复
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    print("[PASS] should_reset 测试通过")


def test_bit_exact_mode():
    """测试 BIT_EXACT 模式下门电路正确性"""
    print(f"Device: {device}")
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    gate = ANDGate().to(device)

    # 多次调用应该得到一致结果
    a = torch.tensor([1.0], device=device)
    b = torch.tensor([1.0], device=device)

    result1 = gate(a, b)
    result2 = gate(a, b)
    result3 = gate(a, b)

    assert result1.item() == 1.0
    assert result2.item() == 1.0
    assert result3.item() == 1.0

    # 不同输入也应正确
    a = torch.tensor([0.0], device=device)
    b = torch.tensor([1.0], device=device)
    result = gate(a, b)
    assert result.item() == 0.0

    print("[PASS] BIT_EXACT 模式测试通过")


def test_instance_mode_override():
    """测试实例级模式覆盖"""
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    # 创建一个 TEMPORAL 模式的门
    gate_temporal = ANDGate(mode=SpikeMode.TEMPORAL).to(device)

    # 创建一个 BIT_EXACT 模式的门
    gate_exact = ANDGate(mode=SpikeMode.BIT_EXACT).to(device)

    a = torch.tensor([1.0], device=device)
    b = torch.tensor([1.0], device=device)

    # 两者都应该给出正确结果
    assert gate_temporal(a, b).item() == 1.0
    assert gate_exact(a, b).item() == 1.0

    print("[PASS] 实例模式覆盖测试通过")


def test_vec_gates():
    """测试向量化门电路的 SpikeMode 支持"""
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    gate = VecAND().to(device)

    # 批量输入
    a = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device=device)
    b = torch.tensor([[1.0, 1.0, 0.0, 0.0]], device=device)

    result = gate(a, b)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    assert torch.allclose(result, expected)

    # 多次调用
    result2 = gate(a, b)
    assert torch.allclose(result2, expected)

    print("[PASS] 向量化门电路测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("SpikeMode 测试套件")
    print("=" * 60)

    # 保存原始全局模式
    original_mode = SpikeMode.get_global_mode()

    try:
        test_global_mode()
        test_context_manager()
        test_should_reset()
        test_bit_exact_mode()
        test_instance_mode_override()
        test_vec_gates()

        print("-" * 60)
        print("所有测试通过!")
        print("=" * 60)

    finally:
        # 恢复原始模式
        SpikeMode.set_global_mode(original_mode)


if __name__ == "__main__":
    run_all_tests()
