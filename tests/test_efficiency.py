"""
实验三：资源效率统计 (Strict Mode)

统计SNN FP8系统的资源开销：
1. 各组件神经元数量 (IFNode 精确计数)
2. 单次FP8运算的脉冲发放次数
3. Linear层的总脉冲发放
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import torch.nn as nn
from SNNTorch.atomic_ops import (
    ANDGate, ORGate, XORGate, NOTGate,
    HalfAdder, FullAdder, RippleCarryAdder,
    MUXGate, ArrayMultiplier4x4_Strict,
    SpikeFP8Multiplier, SpikeFP8Adder_Spatial,
    SpikeFP8Linear_Fast,
    PriorityEncoder8, BarrelShifter8, ExponentAdjuster, Denormalizer,
    NewNormalizationUnit, TemporalExponentGenerator, DelayNode
)
from SNNTorch.atomic_ops.fp8_adder_spatial import (
    Comparator4Bit, Comparator3Bit, BarrelShifterRight12,
    BarrelShifterLeft8, LeadingZeroDetector8, 
    Adder12Bit, Subtractor12Bit
)


def count_if_neurons(module):
    """统计模块中的IF神经元数量（准确计数）"""
    count = 0
    
    # 遍历所有子模块
    for name, child in module.named_modules():
        class_name = child.__class__.__name__
        
        # SpikingJelly的IF/LIF神经元
        if 'IFNode' in class_name or 'LIFNode' in class_name:
            count += 1
        
        # 我们自定义的SimpleLIF
        if 'SimpleLIFNode' in class_name:
            count += 1
    
    return count


def count_gate_neurons(module):
    """统计模块中的逻辑门数量（只统计直接子模块，避免双重计数）"""
    gate_counts = {
        'ANDGate': 0,
        'ORGate': 0,
        'XORGate': 0,
        'NOTGate': 0,
        'MUXGate': 0,
        'HalfAdder': 0,
        'FullAdder': 0,
    }
    
    # 神经元数/门（理论最小值）
    neurons_per_gate = {
        'ANDGate': 1,
        'ORGate': 1,
        'XORGate': 2,
        'NOTGate': 0,  # NOT不需要神经元
        'MUXGate': 3,  # 2 AND + 1 OR
        'HalfAdder': 3,  # 1 XOR + 1 AND
        'FullAdder': 5,  # 2 XOR + 2 AND + 1 OR
    }
    
    # 只统计直接子模块，避免双重计数
    for name, child in module.named_children():
        class_name = child.__class__.__name__
        for gate_type in gate_counts.keys():
            if class_name == gate_type:
                gate_counts[gate_type] += 1
    
    total_neurons = sum(gate_counts[k] * neurons_per_gate[k] for k in gate_counts)
    
    return gate_counts, total_neurons


def analyze_component_neurons():
    """分析各组件的神经元数量"""
    print("\n" + "="*70)
    print("实验3.1: 各组件神经元数量统计（IFNode精确计数）")
    print("="*70)
    
    components = [
        ("ANDGate", ANDGate()),
        ("ORGate", ORGate()),
        ("XORGate", XORGate()),
        ("NOTGate", NOTGate()),
        ("MUXGate", MUXGate()),
        ("HalfAdder", HalfAdder()),
        ("FullAdder", FullAdder()),
        ("RippleCarryAdder(4-bit)", RippleCarryAdder(bits=4)),
        ("ArrayMultiplier4x4", ArrayMultiplier4x4_Strict()),
        ("PriorityEncoder8", PriorityEncoder8()),
        ("BarrelShifter8", BarrelShifter8()),
        ("ExponentAdjuster", ExponentAdjuster()),
        ("Denormalizer", Denormalizer()),
        ("NewNormalizationUnit", NewNormalizationUnit()),
        ("TemporalExponentGenerator", TemporalExponentGenerator()),
        ("DelayNode", DelayNode()),
        ("SpikeFP8Multiplier", SpikeFP8Multiplier()),
        ("SpikeFP8Adder_Spatial", SpikeFP8Adder_Spatial()),
    ]
    
    results = {}
    
    print("\n| 组件名称                  | IFNode数 | 直接子门 |")
    print("|---------------------------|----------|----------|")
    
    for name, module in components:
        if_count = count_if_neurons(module)
        gate_counts, gate_neurons = count_gate_neurons(module)
        
        # 筛选非零门（只显示直接子模块）
        gate_str = ", ".join([f"{k}:{v}" for k, v in gate_counts.items() if v > 0])
        if not gate_str:
            gate_str = "-"
        
        results[name] = if_count
        
        print(f"| {name:<25} | {if_count:>8} | {gate_str[:30]:<30} |")
    
    return results


def count_spikes_in_forward(module, inputs, device):
    """统计一次前向传播的脉冲发放次数"""
    total_spikes = 0
    
    # Hook函数记录输出脉冲
    spike_counts = []
    
    def hook_fn(m, inp, out):
        if isinstance(out, torch.Tensor):
            spike_counts.append(out.sum().item())
        elif isinstance(out, tuple):
            for o in out:
                if isinstance(o, torch.Tensor):
                    spike_counts.append(o.sum().item())
    
    # 注册hook到所有子模块
    hooks = []
    for child in module.modules():
        if hasattr(child, 'forward') and child != module:
            h = child.register_forward_hook(hook_fn)
            hooks.append(h)
    
    # 运行前向传播
    module.reset() if hasattr(module, 'reset') else None
    _ = module(*inputs)
    
    # 移除hooks
    for h in hooks:
        h.remove()
    
    return sum(spike_counts)


def analyze_spike_activity():
    """分析脉冲发放活动"""
    print("\n" + "="*70)
    print("实验3.2: 单次运算脉冲发放统计")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # FP8加法器测试
    adder = SpikeFP8Adder_Spatial().to(device)
    
    # 随机FP8输入
    a_bits = torch.randint(0, 2, (8,), device=device).float()
    b_bits = torch.randint(0, 2, (8,), device=device).float()
    
    # 统计输入脉冲
    input_spikes = a_bits.sum().item() + b_bits.sum().item()
    
    # 运行加法并统计
    adder.reset()
    result = adder(a_bits, b_bits)
    output_spikes = result.sum().item()
    
    print(f"\nFP8加法器 (单次运算):")
    print(f"  输入脉冲数: {input_spikes:.0f}")
    print(f"  输出脉冲数: {output_spikes:.0f}")
    
    # FP8乘法器测试
    mul = SpikeFP8Multiplier().to(device)
    mul.reset()
    result_mul = mul(a_bits, b_bits)
    output_spikes_mul = result_mul.sum().item()
    
    print(f"\nFP8乘法器 (单次运算):")
    print(f"  输入脉冲数: {input_spikes:.0f}")
    print(f"  输出脉冲数: {output_spikes_mul:.0f}")
    
    # 批量测试统计平均值
    print("\n--- 批量测试 (100次) ---")
    
    add_input_total = 0
    add_output_total = 0
    mul_output_total = 0
    
    for _ in range(100):
        a = torch.randint(0, 2, (8,), device=device).float()
        b = torch.randint(0, 2, (8,), device=device).float()
        
        add_input_total += a.sum().item() + b.sum().item()
        
        adder.reset()
        result = adder(a, b)
        add_output_total += result.sum().item()
        
        mul.reset()
        result = mul(a, b)
        mul_output_total += result.sum().item()
    
    print(f"\nFP8加法器 (平均):")
    print(f"  平均输入脉冲: {add_input_total/100:.1f}")
    print(f"  平均输出脉冲: {add_output_total/100:.1f}")
    
    print(f"\nFP8乘法器 (平均):")
    print(f"  平均输出脉冲: {mul_output_total/100:.1f}")


def analyze_linear_layer_cost(neuron_counts):
    """分析Linear层的资源开销"""
    print("\n" + "="*70)
    print("实验3.3: Linear层资源开销分析")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取实际神经元数量
    mul_neurons = neuron_counts.get('SpikeFP8Multiplier', 0)
    add_neurons = neuron_counts.get('SpikeFP8Adder_Spatial', 0)
    
    print(f"基准神经元数: Multiplier={mul_neurons}, Adder={add_neurons}")
    
    # 不同维度配置
    configs = [
        (4, 2),
        (8, 4),
        (16, 8),
        (32, 16),
        (64, 32),
        (128, 64),
    ]
    
    print("\n| Din | Dout | 乘法器数 | 加法器数 | 树形层数 | 估计神经元数 |")
    print("|-----|------|----------|----------|----------|--------------|")
    
    for d_in, d_out in configs:
        # 计算资源
        num_multipliers = d_in * d_out
        
        # 树形累加层数
        import math
        tree_layers = math.ceil(math.log2(d_in)) if d_in > 1 else 0
        
        # 每层加法器数量
        n = d_in
        total_adders = 0
        for _ in range(tree_layers):
            adder_count = (n + 1) // 2 * d_out
            total_adders += adder_count
            n = (n + 1) // 2
        
        # 估计神经元数（基于IFNode精确计数）
        est_neurons = num_multipliers * mul_neurons + total_adders * add_neurons
        
        print(f"| {d_in:3} | {d_out:4} | {num_multipliers:8} | {total_adders:8} | {tree_layers:8} | {est_neurons:12,} |")


def analyze_memory_footprint():
    """分析内存占用"""
    print("\n" + "="*70)
    print("实验3.4: 内存占用分析")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    components = [
        ("SpikeFP8Multiplier", SpikeFP8Multiplier()),
        ("SpikeFP8Adder_Spatial", SpikeFP8Adder_Spatial()),
    ]
    
    print("\n| 组件                    | 参数量 | 缓冲区数 |")
    print("|-------------------------|--------|----------|")
    
    for name, module in components:
        module = module.to(device)
        
        # 统计参数
        num_params = sum(p.numel() for p in module.parameters())
        
        # 统计缓冲区
        num_buffers = sum(b.numel() for b in module.buffers())
        
        print(f"| {name:<23} | {num_params:>6} | {num_buffers:>8} |")


def main():
    print("="*70)
    print("实验三：资源效率统计 (Strict)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 运行所有分析
    neuron_counts = analyze_component_neurons()
    analyze_spike_activity()
    analyze_linear_layer_cost(neuron_counts)
    analyze_memory_footprint()
    
    # 总结
    print("\n" + "="*70)
    print("实验三总结")
    print("="*70)
    print("统计完成。请记录以上数据用于文档更新。")


if __name__ == "__main__":
    main()
