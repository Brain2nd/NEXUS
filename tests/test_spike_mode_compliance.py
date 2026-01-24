"""
验证 SpikeMode 机制是否正确实现：
- BIT_EXACT 模式：forward 后 v=None
- TEMPORAL 模式：forward 后 v 保留
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.spike_mode import SpikeMode
from atomic_ops.core.vec_logic_gates import VecAND, VecOR, VecXOR
from atomic_ops.core.logic_gates import ANDGate, ORGate, NOTGate


def test_bit_exact_mode():
    """BIT_EXACT 模式：forward 结束后 v 应该被清理"""
    print("=" * 60)
    print("测试 BIT_EXACT 模式")
    print("=" * 60)

    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)
    assert SpikeMode.should_reset() == True, "BIT_EXACT 模式应该返回 True"

    gates = [
        ("VecAND", VecAND()),
        ("VecOR", VecOR()),
        ("ANDGate", ANDGate()),
        ("ORGate", ORGate()),
    ]

    all_pass = True
    for name, gate in gates:
        a = torch.randint(0, 2, (10, 8), dtype=torch.float32)
        b = torch.randint(0, 2, (10, 8), dtype=torch.float32)

        result = gate(a, b)

        # 检查神经元 v 状态
        if hasattr(gate, 'node'):
            v_state = gate.node.v
        else:
            v_state = None
            for name_child, child in gate.named_modules():
                if hasattr(child, 'v') and child.v is not None:
                    v_state = child.v
                    break

        passed = v_state is None
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: v={'None' if v_state is None else 'Tensor'} - {status}")

        if not passed:
            all_pass = False

    return all_pass


def test_temporal_mode():
    """TEMPORAL 模式：forward 结束后 v 应该保留"""
    print("\n" + "=" * 60)
    print("测试 TEMPORAL 模式")
    print("=" * 60)

    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    assert SpikeMode.should_reset() == False, "TEMPORAL 模式应该返回 False"

    gates = [
        ("VecAND", VecAND()),
        ("VecOR", VecOR()),
        ("ANDGate", ANDGate()),
        ("ORGate", ORGate()),
    ]

    all_pass = True
    for name, gate in gates:
        a = torch.randint(0, 2, (10, 8), dtype=torch.float32)
        b = torch.randint(0, 2, (10, 8), dtype=torch.float32)

        result = gate(a, b)

        # 检查神经元 v 状态
        if hasattr(gate, 'node'):
            v_state = gate.node.v
        else:
            v_state = None
            for name_child, child in gate.named_modules():
                if hasattr(child, 'v'):
                    v_state = child.v
                    break

        passed = v_state is not None
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: v={'Tensor' if v_state is not None else 'None'} - {status}")

        if not passed:
            all_pass = False

    # 恢复默认模式
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    return all_pass


def test_context_manager():
    """测试上下文管理器"""
    print("\n" + "=" * 60)
    print("测试上下文管理器")
    print("=" * 60)

    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    gate = VecAND()
    a = torch.randint(0, 2, (10, 8), dtype=torch.float32)
    b = torch.randint(0, 2, (10, 8), dtype=torch.float32)

    # 默认 BIT_EXACT
    _ = gate(a, b)
    bit_exact_v = gate.node.v
    print(f"  默认 BIT_EXACT: v={'None' if bit_exact_v is None else 'Tensor'}")

    # 临时切换到 TEMPORAL
    with SpikeMode.temporal():
        _ = gate(a, b)
        temporal_v = gate.node.v
        print(f"  with temporal(): v={'None' if temporal_v is None else 'Tensor'}")

    # 退出后恢复 BIT_EXACT
    gate.reset()  # 手动清理以测试
    _ = gate(a, b)
    restored_v = gate.node.v
    print(f"  退出后 BIT_EXACT: v={'None' if restored_v is None else 'Tensor'}")

    passed = (bit_exact_v is None and temporal_v is not None and restored_v is None)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  上下文管理器测试: {status}")

    return passed


def test_instance_mode_override():
    """测试实例级模式覆盖"""
    print("\n" + "=" * 60)
    print("测试实例级模式覆盖")
    print("=" * 60)

    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    # 创建一个强制 TEMPORAL 的实例
    gate_temporal = VecAND(mode=SpikeMode.TEMPORAL)
    # 创建一个跟随全局的实例
    gate_global = VecAND()

    a = torch.randint(0, 2, (10, 8), dtype=torch.float32)
    b = torch.randint(0, 2, (10, 8), dtype=torch.float32)

    _ = gate_temporal(a, b)
    _ = gate_global(a, b)

    temporal_v = gate_temporal.node.v
    global_v = gate_global.node.v

    print(f"  实例 mode=TEMPORAL: v={'None' if temporal_v is None else 'Tensor'}")
    print(f"  实例 mode=None (跟随全局 BIT_EXACT): v={'None' if global_v is None else 'Tensor'}")

    passed = (temporal_v is not None and global_v is None)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  实例级覆盖测试: {status}")

    return passed


if __name__ == "__main__":
    results = []

    results.append(("BIT_EXACT 模式", test_bit_exact_mode()))
    results.append(("TEMPORAL 模式", test_temporal_mode()))
    results.append(("上下文管理器", test_context_manager()))
    results.append(("实例级覆盖", test_instance_mode_override()))

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + ("全部测试通过！" if all_pass else "存在失败的测试！"))
