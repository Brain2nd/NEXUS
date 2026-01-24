"""
深度检测自动 reset 机制 - 检查所有嵌套组件的膜电位状态
"""
import torch
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.spike_mode import SpikeMode
from atomic_ops.core.neurons import SimpleLIFNode, SimpleIFNode


def count_membrane_tensors(module, prefix=""):
    """递归计数所有神经元的膜电位 v 状态

    返回: (total_neurons, neurons_with_v_tensor, total_v_memory_bytes)
    """
    total_neurons = 0
    neurons_with_v = 0
    total_memory = 0

    for name, child in module.named_modules():
        # 检查是否有 v 属性（神经元）
        if hasattr(child, 'v'):
            total_neurons += 1
            v = child.v
            if v is not None:
                neurons_with_v += 1
                total_memory += v.numel() * v.element_size()
                if prefix:
                    print(f"  {prefix}.{name}: v.shape={v.shape}, memory={v.numel() * v.element_size()} bytes")

    return total_neurons, neurons_with_v, total_memory


def test_basic_gates():
    """测试基础门电路的自动 reset"""
    from atomic_ops.core.vec_logic_gates import VecAND, VecOR, VecXOR, VecMUX, VecORTree, VecANDTree

    print("=" * 60)
    print("测试基础门电路")
    print("=" * 60)
    print(f"SpikeMode: {SpikeMode.get_mode()}")
    print(f"should_reset(): {SpikeMode.should_reset()}")

    gates = {
        'VecAND': VecAND(),
        'VecOR': VecOR(),
        'VecXOR': VecXOR(),
        'VecMUX': VecMUX(),
        'VecORTree': VecORTree(),
        'VecANDTree': VecANDTree(),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name, gate in gates.items():
        gate = gate.to(device)

    # 测试每个门
    for name, gate in gates.items():
        a = torch.randint(0, 2, (10, 32), device=device, dtype=torch.float32)
        b = torch.randint(0, 2, (10, 32), device=device, dtype=torch.float32)

        if name in ['VecAND', 'VecOR', 'VecXOR']:
            result = gate(a, b)
        elif name == 'VecMUX':
            sel = torch.randint(0, 2, (10, 32), device=device, dtype=torch.float32)
            result = gate(sel, a, b)
        else:  # Tree
            result = gate(a)

        total, with_v, mem = count_membrane_tensors(gate)
        status = "✓ PASS" if with_v == 0 else "✗ FAIL"
        print(f"  {name}: neurons={total}, with_v={with_v}, memory={mem} bytes - {status}")

        if with_v > 0:
            count_membrane_tensors(gate, prefix=name)


def test_fp32_adder():
    """测试 SpikeFP32Adder 的自动 reset"""
    from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
    from atomic_ops import float32_to_pulse

    print("\n" + "=" * 60)
    print("测试 SpikeFP32Adder")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adder = SpikeFP32Adder().to(device)

    # 运行多次
    for i in range(3):
        a = float32_to_pulse(torch.randn(10, device=device), device=device)
        b = float32_to_pulse(torch.randn(10, device=device), device=device)
        result = adder(a, b)

        total, with_v, mem = count_membrane_tensors(adder)
        status = "✓ PASS" if with_v == 0 else "✗ FAIL"
        print(f"  迭代 {i}: neurons={total}, with_v={with_v}, memory={mem} bytes - {status}")

        del a, b, result


def test_fp32_multiplier():
    """测试 SpikeFP32Multiplier 的自动 reset"""
    from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
    from atomic_ops import float32_to_pulse

    print("\n" + "=" * 60)
    print("测试 SpikeFP32Multiplier")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mul = SpikeFP32Multiplier().to(device)

    # 运行多次
    for i in range(3):
        a = float32_to_pulse(torch.randn(10, device=device), device=device)
        b = float32_to_pulse(torch.randn(10, device=device), device=device)
        result = mul(a, b)

        total, with_v, mem = count_membrane_tensors(mul)
        status = "✓ PASS" if with_v == 0 else "✗ FAIL"
        print(f"  迭代 {i}: neurons={total}, with_v={with_v}, memory={mem} bytes - {status}")

        del a, b, result


def test_fp32_linear():
    """测试 SpikeFP32Linear 的自动 reset"""
    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision as SpikeFP32Linear
    from atomic_ops import float32_to_pulse

    print("\n" + "=" * 60)
    print("测试 SpikeFP32Linear")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear = SpikeFP32Linear(64, 32).to(device)
    weight = torch.randn(32, 64, device=device)
    linear.set_weight_from_float(weight)

    # 运行多次
    for i in range(3):
        x = torch.randn(4, 64, device=device)
        x_pulse = float32_to_pulse(x, device=device)
        result = linear(x_pulse)

        total, with_v, mem = count_membrane_tensors(linear)
        status = "✓ PASS" if with_v == 0 else "✗ FAIL"
        print(f"  迭代 {i}: neurons={total}, with_v={with_v}, memory={mem} bytes - {status}")

        del x, x_pulse, result


def test_attention():
    """测试 SpikeAttention 的自动 reset"""
    try:
        from atomic_ops.attention.attention import SpikeQwen3Attention
        from atomic_ops import float32_to_pulse
    except ImportError as e:
        print(f"\n跳过 Attention 测试: {e}")
        return

    print("\n" + "=" * 60)
    print("测试 SpikeQwen3Attention")
    print("=" * 60)

    # 创建一个简单的 config
    class Config:
        hidden_size = 64
        num_attention_heads = 2
        num_key_value_heads = 2
        head_dim = 32
        rms_norm_eps = 1e-6
        rope_theta = 10000.0
        max_position_embeddings = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    try:
        attn = SpikeQwen3Attention(config, layer_idx=0).to(device)

        # 运行
        batch, seq_len = 2, 8
        hidden = torch.randn(batch, seq_len, config.hidden_size, device=device)
        hidden_pulse = float32_to_pulse(hidden.view(-1), device=device).view(batch, seq_len, config.hidden_size, 32)

        # 预计算 position embeddings (简化)
        cos = torch.ones(seq_len, config.head_dim // 2, device=device)
        sin = torch.zeros(seq_len, config.head_dim // 2, device=device)
        position_embeddings = (cos, sin)

        result = attn(hidden_pulse, position_embeddings)

        total, with_v, mem = count_membrane_tensors(attn)
        status = "✓ PASS" if with_v == 0 else "✗ FAIL"
        print(f"  neurons={total}, with_v={with_v}, memory={mem} bytes - {status}")

        if with_v > 0:
            print("  有残留膜电位的组件:")
            count_membrane_tensors(attn, prefix="attn")

    except Exception as e:
        print(f"  Attention 测试失败: {e}")


def find_non_reset_components():
    """扫描所有组件找出没有正确 reset 的"""
    from atomic_ops.core.vec_logic_gates import (
        VecAND, VecOR, VecNOT, VecXOR, VecMUX,
        VecORTree, VecANDTree, VecAdder, VecSubtractor
    )

    print("\n" + "=" * 60)
    print("扫描所有 VecAdder 组件")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adder = VecAdder(bits=32).to(device)

    a = torch.randint(0, 2, (10, 32), device=device, dtype=torch.float32)
    b = torch.randint(0, 2, (10, 32), device=device, dtype=torch.float32)

    s, cout = adder(a, b)

    total, with_v, mem = count_membrane_tensors(adder)
    status = "✓ PASS" if with_v == 0 else "✗ FAIL"
    print(f"  VecAdder(32): neurons={total}, with_v={with_v}, memory={mem} bytes - {status}")

    if with_v > 0:
        print("  有残留膜电位的组件:")
        for name, child in adder.named_modules():
            if hasattr(child, 'v') and child.v is not None:
                print(f"    {name}: v.shape={child.v.shape}")


if __name__ == "__main__":
    # 确保是 BIT_EXACT 模式
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    test_basic_gates()
    find_non_reset_components()
    test_fp32_adder()
    test_fp32_multiplier()
    test_fp32_linear()
    test_attention()
