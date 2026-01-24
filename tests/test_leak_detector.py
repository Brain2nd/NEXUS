"""
检测哪些神经元没有被自动 reset
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.spike_mode import SpikeMode


def find_leaking_neurons(module, prefix=""):
    """找出所有有残留 v 状态的神经元及其路径"""
    leaks = []
    for name, child in module.named_modules():
        if hasattr(child, 'v') and child.v is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            leaks.append((full_name, type(child).__name__, child.v.shape))
    return leaks


def test_fp32_adder_leaks():
    """检测 SpikeFP32Adder 的泄漏位置"""
    from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
    from atomic_ops import float32_to_pulse

    print("=" * 60)
    print("SpikeFP32Adder 泄漏检测")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adder = SpikeFP32Adder().to(device)

    a = float32_to_pulse(torch.randn(10, device=device), device=device)
    b = float32_to_pulse(torch.randn(10, device=device), device=device)
    result = adder(a, b)

    leaks = find_leaking_neurons(adder)
    print(f"泄漏神经元数量: {len(leaks)}")

    # 按类型统计
    type_counts = {}
    for path, type_name, shape in leaks:
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print("\n按类型统计:")
    for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {type_name}: {count}")

    print("\n前20个泄漏位置:")
    for i, (path, type_name, shape) in enumerate(leaks[:20]):
        print(f"  {path}: {type_name}, shape={shape}")


def test_fp32_mul_leaks():
    """检测 SpikeFP32Multiplier 的泄漏位置"""
    from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
    from atomic_ops import float32_to_pulse

    print("\n" + "=" * 60)
    print("SpikeFP32Multiplier 泄漏检测")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP32Multiplier().to(device)

    a = float32_to_pulse(torch.randn(10, device=device), device=device)
    b = float32_to_pulse(torch.randn(10, device=device), device=device)
    result = mul(a, b)

    leaks = find_leaking_neurons(mul)
    print(f"泄漏神经元数量: {len(leaks)}")

    # 按类型统计
    type_counts = {}
    for path, type_name, shape in leaks:
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print("\n按类型统计:")
    for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {type_name}: {count}")

    # 按路径前缀分组
    prefix_counts = {}
    for path, type_name, shape in leaks:
        parts = path.split('.')
        if len(parts) > 1:
            prefix = parts[0]
        else:
            prefix = "(root)"
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    print("\n按组件分组:")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        print(f"  {prefix}: {count}")

    print("\n前20个泄漏位置:")
    for i, (path, type_name, shape) in enumerate(leaks[:20]):
        print(f"  {path}: {type_name}, shape={shape}")


if __name__ == "__main__":
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)
    test_fp32_adder_leaks()
    test_fp32_mul_leaks()
