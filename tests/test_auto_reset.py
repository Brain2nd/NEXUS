"""
测试自动 reset 机制是否生效
"""
import torch
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.vec_logic_gates import VecAND
from atomic_ops.core.spike_mode import SpikeMode


def mem_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def force_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_vecand_auto_reset():
    """测试 VecAND 的自动 reset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"SpikeMode: {SpikeMode.get_mode()}")
    print(f"should_reset(): {SpikeMode.should_reset()}")

    gate = VecAND().to(device)

    force_gc()
    print(f"\n初始显存: {mem_mb():.2f} MB")

    # 测试多次调用
    for i in range(5):
        a = torch.randint(0, 2, (1000, 1000), device=device, dtype=torch.float32)
        b = torch.randint(0, 2, (1000, 1000), device=device, dtype=torch.float32)

        result = gate(a, b)

        # 检查神经元 v 的状态
        v_state = gate.node.v
        v_info = f"v={v_state.shape if v_state is not None else None}"

        print(f"  迭代 {i}: 显存={mem_mb():.2f} MB, {v_info}")

        del a, b, result

    force_gc()
    print(f"\nGC后显存: {mem_mb():.2f} MB")

    # 检查最终 v 状态
    print(f"最终 gate.node.v: {gate.node.v}")


def test_multiple_gates():
    """测试多个门电路的显存累积"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print("测试多个门电路")
    print(f"{'='*50}")

    gates = [VecAND().to(device) for _ in range(10)]

    force_gc()
    print(f"初始显存: {mem_mb():.2f} MB")

    for iteration in range(3):
        for i, gate in enumerate(gates):
            a = torch.randint(0, 2, (500, 500), device=device, dtype=torch.float32)
            b = torch.randint(0, 2, (500, 500), device=device, dtype=torch.float32)
            result = gate(a, b)
            del a, b, result

        print(f"  迭代 {iteration}: 显存={mem_mb():.2f} MB")

    force_gc()
    print(f"GC后显存: {mem_mb():.2f} MB")

    # 检查每个门的 v 状态
    v_states = [g.node.v for g in gates]
    none_count = sum(1 for v in v_states if v is None)
    print(f"门电路 v=None 数量: {none_count}/{len(gates)}")


def test_fp32_linear():
    """测试 SpikeFP32Linear 的显存累积"""
    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision as SpikeFP32Linear
    from atomic_ops import float32_to_pulse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"测试 SpikeFP32Linear (device={device})")
    print(f"{'='*50}")

    linear = SpikeFP32Linear(64, 32).to(device)
    weight = torch.randn(32, 64, device=device)
    linear.set_weight_from_float(weight)

    force_gc()
    baseline = mem_mb()
    print(f"初始显存: {baseline:.2f} MB")

    for i in range(5):
        x = torch.randn(10, 64, device=device)
        x_pulse = float32_to_pulse(x, device=device)

        result = linear(x_pulse)

        mem = mem_mb()
        delta = mem - baseline
        print(f"  迭代 {i}: 显存={mem:.2f} MB (Δ{delta:+.2f} MB)")

        del x, x_pulse, result

    force_gc()
    final = mem_mb()
    print(f"GC后显存: {final:.2f} MB (Δ{final-baseline:+.2f} MB)")


def test_fp32_multiplier():
    """测试 SpikeFP32Multiplier 的显存累积"""
    from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
    from atomic_ops import float32_to_pulse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"测试 SpikeFP32Multiplier (device={device})")
    print(f"{'='*50}")

    mul = SpikeFP32Multiplier().to(device)

    force_gc()
    baseline = mem_mb()
    print(f"初始显存: {baseline:.2f} MB")

    for i in range(5):
        a = float32_to_pulse(torch.randn(100, device=device), device=device)
        b = float32_to_pulse(torch.randn(100, device=device), device=device)

        result = mul(a, b)

        mem = mem_mb()
        delta = mem - baseline
        print(f"  迭代 {i}: 显存={mem:.2f} MB (Δ{delta:+.2f} MB)")

        del a, b, result

    force_gc()
    final = mem_mb()
    print(f"GC后显存: {final:.2f} MB (Δ{final-baseline:+.2f} MB)")


if __name__ == "__main__":
    test_vecand_auto_reset()
    test_multiple_gates()
    test_fp32_multiplier()
    test_fp32_linear()
