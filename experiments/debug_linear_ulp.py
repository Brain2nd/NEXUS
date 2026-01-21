"""
SpikeFP32Linear ULP 调试脚本
============================

逐步隔离问题：
1. 最简单的矩阵乘法测试
2. 检查乘法器是否正确
3. 检查加法器是否正确
4. 检查累加过程是否正确
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import float32_to_pulse, pulse_to_float32, SpikeFP32Linear
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return device


def compute_ulp(ref, snn):
    """计算 ULP 误差"""
    ref_flat = ref.detach().contiguous().view(-1)
    snn_flat = snn.detach().contiguous().view(-1)

    ref_bits = ref_flat.view(torch.int32).long()
    snn_bits = snn_flat.view(torch.int32).long()
    ulp = (ref_bits - snn_bits).abs()

    return {
        'max_ulp': ulp.max().item(),
        'mean_ulp': ulp.float().mean().item(),
        'zero_ulp_rate': (ulp == 0).float().mean().item(),
    }


def test_single_multiply(device):
    """测试单个 FP32 乘法"""
    print("\n" + "="*60, flush=True)
    print("[TEST 1] Single FP32 Multiplication", flush=True)
    print("="*60, flush=True)

    mul = SpikeFP32Multiplier().to(device)

    # 测试值
    test_pairs = [
        (1.0, 2.0),
        (0.5, 0.5),
        (-1.0, 2.0),
        (1.5, -3.0),
        (0.123, 0.456),
        (100.0, 0.01),
    ]

    all_pass = True
    for a_val, b_val in test_pairs:
        a = torch.tensor([a_val], dtype=torch.float32, device=device)
        b = torch.tensor([b_val], dtype=torch.float32, device=device)

        # PyTorch 参考
        ref = a * b

        # SNN
        a_pulse = float32_to_pulse(a)
        b_pulse = float32_to_pulse(b)
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)
        result = pulse_to_float32(result_pulse)

        ulp = compute_ulp(ref, result)
        status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
        if ulp['max_ulp'] > 0:
            all_pass = False
        print(f"  {a_val} * {b_val} = {ref.item():.6f} vs {result.item():.6f}, ULP={ulp['max_ulp']} [{status}]", flush=True)

    return all_pass


def test_single_add(device):
    """测试单个 FP32 加法"""
    print("\n" + "="*60, flush=True)
    print("[TEST 2] Single FP32 Addition", flush=True)
    print("="*60, flush=True)

    add = SpikeFP32Adder().to(device)

    test_pairs = [
        (1.0, 2.0),
        (0.5, 0.5),
        (-1.0, 2.0),
        (1.5, -3.0),
        (0.123, 0.456),
        (100.0, 0.01),
    ]

    all_pass = True
    for a_val, b_val in test_pairs:
        a = torch.tensor([a_val], dtype=torch.float32, device=device)
        b = torch.tensor([b_val], dtype=torch.float32, device=device)

        # PyTorch 参考
        ref = a + b

        # SNN
        a_pulse = float32_to_pulse(a)
        b_pulse = float32_to_pulse(b)
        add.reset()
        result_pulse = add(a_pulse, b_pulse)
        result = pulse_to_float32(result_pulse)

        ulp = compute_ulp(ref, result)
        status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
        if ulp['max_ulp'] > 0:
            all_pass = False
        print(f"  {a_val} + {b_val} = {ref.item():.6f} vs {result.item():.6f}, ULP={ulp['max_ulp']} [{status}]", flush=True)

    return all_pass


def test_dot_product_manual(device, size=4):
    """手动测试点积（不使用 SpikeFP32Linear）"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 3] Manual Dot Product (size={size})", flush=True)
    print("="*60, flush=True)

    mul = SpikeFP32Multiplier().to(device)
    add = SpikeFP32Adder().to(device)

    # 随机向量
    torch.manual_seed(42)
    x = torch.randn(size, dtype=torch.float32, device=device)
    w = torch.randn(size, dtype=torch.float32, device=device)

    # PyTorch 参考
    ref = (x * w).sum()
    print(f"  PyTorch dot product: {ref.item():.6f}", flush=True)

    # SNN 手动计算
    x_pulse = float32_to_pulse(x)
    w_pulse = float32_to_pulse(w)

    # 逐元素乘法
    mul.reset()
    products_pulse = mul(x_pulse, w_pulse)
    products = pulse_to_float32(products_pulse)
    print(f"  Products: {products.tolist()}", flush=True)
    print(f"  PyTorch products: {(x * w).tolist()}", flush=True)

    # 累加 (使用手动reset - 和之前一样)
    acc_pulse = products_pulse[0:1]  # 第一个
    for i in range(1, size):
        add.reset()
        acc_pulse = add(acc_pulse, products_pulse[i:i+1])

    result = pulse_to_float32(acc_pulse)
    print(f"  SNN dot product (with reset): {result.item():.6f}", flush=True)

    ulp = compute_ulp(ref.unsqueeze(0), result)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP (with reset): {ulp['max_ulp']} [{status}]", flush=True)

    return ulp['max_ulp'] == 0


def test_dot_product_no_reset(device, size=4):
    """手动测试点积（不手动reset，依赖SpikeMode自动reset）"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 3b] Manual Dot Product NO RESET (size={size})", flush=True)
    print("="*60, flush=True)

    mul = SpikeFP32Multiplier().to(device)
    add = SpikeFP32Adder().to(device)

    # 随机向量
    torch.manual_seed(42)
    x = torch.randn(size, dtype=torch.float32, device=device)
    w = torch.randn(size, dtype=torch.float32, device=device)

    # PyTorch 参考
    ref = (x * w).sum()
    print(f"  PyTorch dot product: {ref.item():.6f}", flush=True)

    # SNN 手动计算
    x_pulse = float32_to_pulse(x)
    w_pulse = float32_to_pulse(w)

    # 逐元素乘法 - 只reset一次在开始
    mul.reset()
    products_pulse = mul(x_pulse, w_pulse)

    # 累加 (不手动reset - 依赖SpikeMode自动reset)
    add.reset()  # 只在开始reset一次
    acc_pulse = products_pulse[0:1]  # 第一个
    for i in range(1, size):
        # 不调用 add.reset()，依赖SpikeMode
        acc_pulse = add(acc_pulse, products_pulse[i:i+1])

    result = pulse_to_float32(acc_pulse)
    print(f"  SNN dot product (no reset): {result.item():.6f}", flush=True)

    ulp = compute_ulp(ref.unsqueeze(0), result)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP (no reset): {ulp['max_ulp']} [{status}]", flush=True)

    return ulp['max_ulp'] == 0


def test_dot_product_larger(device, size=64):
    """测试更大的点积"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 3c] Large Dot Product (size={size})", flush=True)
    print("="*60, flush=True)

    mul = SpikeFP32Multiplier().to(device)
    add = SpikeFP32Adder().to(device)

    # 随机向量
    torch.manual_seed(42)
    x = torch.randn(size, dtype=torch.float32, device=device)
    w = torch.randn(size, dtype=torch.float32, device=device)

    # PyTorch 参考
    ref = (x * w).sum()
    print(f"  PyTorch dot product: {ref.item():.6f}", flush=True)

    # SNN 手动计算
    x_pulse = float32_to_pulse(x)
    w_pulse = float32_to_pulse(w)

    # 逐元素乘法
    mul.reset()
    products_pulse = mul(x_pulse, w_pulse)

    # 累加 (使用手动reset)
    acc_pulse = products_pulse[0:1]
    for i in range(1, size):
        add.reset()
        acc_pulse = add(acc_pulse, products_pulse[i:i+1])

    result = pulse_to_float32(acc_pulse)
    print(f"  SNN dot product: {result.item():.6f}", flush=True)

    ulp = compute_ulp(ref.unsqueeze(0), result)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP: {ulp['max_ulp']} [{status}]", flush=True)

    return ulp['max_ulp'] == 0


def test_spike_linear_1x1(device):
    """测试 SpikeFP32Linear 最简单情况: 1x1"""
    print("\n" + "="*60, flush=True)
    print("[TEST 4] SpikeFP32Linear 1x1", flush=True)
    print("="*60, flush=True)

    # PyTorch Linear
    pt_linear = nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        pt_linear.weight.fill_(2.0)

    # 输入
    x = torch.tensor([[3.0]], dtype=torch.float32, device=device)

    # PyTorch 参考
    ref = pt_linear(x)
    print(f"  Input: {x.item()}", flush=True)
    print(f"  Weight: {pt_linear.weight.item()}", flush=True)
    print(f"  PyTorch output: {ref.item()}", flush=True)

    # SNN
    snn_linear = SpikeFP32Linear(1, 1, accum_precision='fp32').to(device)
    snn_linear.set_weight_from_float(pt_linear.weight.data)
    snn_linear.reset()

    x_pulse = float32_to_pulse(x)
    y_pulse = snn_linear(x_pulse)
    y_snn = pulse_to_float32(y_pulse)

    print(f"  SNN output: {y_snn.item()}", flush=True)

    ulp = compute_ulp(ref, y_snn)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP: {ulp['max_ulp']} [{status}]", flush=True)

    return ulp['max_ulp'] == 0


def test_spike_linear_small(device, in_f=4, out_f=2):
    """测试 SpikeFP32Linear 小矩阵"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 5] SpikeFP32Linear {in_f}x{out_f}", flush=True)
    print("="*60, flush=True)

    torch.manual_seed(42)

    # PyTorch Linear
    pt_linear = nn.Linear(in_f, out_f, bias=False).to(device)

    # 输入
    x = torch.randn(1, in_f, dtype=torch.float32, device=device)

    # PyTorch 参考
    ref = pt_linear(x)
    print(f"  Input shape: {x.shape}", flush=True)
    print(f"  Weight shape: {pt_linear.weight.shape}", flush=True)
    print(f"  PyTorch output: {ref.squeeze().tolist()}", flush=True)

    # SNN
    snn_linear = SpikeFP32Linear(in_f, out_f, accum_precision='fp32').to(device)
    snn_linear.set_weight_from_float(pt_linear.weight.data)
    snn_linear.reset()

    x_pulse = float32_to_pulse(x)
    print(f"  x_pulse shape: {x_pulse.shape}", flush=True)

    y_pulse = snn_linear(x_pulse)
    print(f"  y_pulse shape: {y_pulse.shape}", flush=True)

    y_snn = pulse_to_float32(y_pulse)
    print(f"  SNN output: {y_snn.squeeze().tolist()}", flush=True)

    ulp = compute_ulp(ref, y_snn)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP: max={ulp['max_ulp']}, 0-ULP rate={ulp['zero_ulp_rate']*100:.1f}% [{status}]", flush=True)

    # 详细比较
    if ulp['max_ulp'] > 0:
        print("\n  Detailed comparison:", flush=True)
        ref_flat = ref.flatten()
        snn_flat = y_snn.flatten()
        for i in range(min(10, ref_flat.numel())):
            ref_val = ref_flat[i].item()
            snn_val = snn_flat[i].item()
            ref_bits = ref_flat[i].view(torch.int32).item()
            snn_bits = snn_flat[i].view(torch.int32).item()
            ulp_i = abs(ref_bits - snn_bits)
            print(f"    [{i}] ref={ref_val:.6f} (0x{ref_bits:08X}) vs snn={snn_val:.6f} (0x{snn_bits:08X}), ULP={ulp_i}", flush=True)

    return ulp['max_ulp'] == 0


def compute_ref_exact_order(x, weight, device):
    """计算精确匹配 SNN 累加顺序的参考值

    SNN 计算顺序: products[0] + products[1] + ... + products[n-1]
    这和 PyTorch 的 mm 可能不同（mm 可能使用 FMA、blocked accumulation 等）
    """
    batch, in_f = x.shape
    out_f, _ = weight.shape

    ref = torch.zeros(batch, out_f, device=device, dtype=torch.float32)
    for b in range(batch):
        for o in range(out_f):
            # 先计算所有乘积
            products = x[b, :] * weight[o, :]  # [in_f]
            # 按照 SNN 的顺序累加: acc = p[0], acc = acc + p[1], ...
            acc = products[0]
            for i in range(1, in_f):
                acc = acc + products[i]
            ref[b, o] = acc
    return ref


def test_manual_linear(device, in_f=64, out_f=32, batch=10):
    """手动实现 Linear 层的计算逻辑，对比 SpikeFP32Linear"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 6a] Manual Linear Implementation {in_f}x{out_f} (batch={batch})", flush=True)
    print("="*60, flush=True)

    from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
    from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder

    torch.manual_seed(42)

    # 创建组件
    mul = SpikeFP32Multiplier().to(device)
    add = SpikeFP32Adder().to(device)

    # 创建权重 [out_f, in_f]
    weight = torch.randn(out_f, in_f, dtype=torch.float32, device=device)

    # 创建输入 [batch, in_f]
    x = torch.randn(batch, in_f, dtype=torch.float32, device=device)

    # PyTorch 参考 (torch.mm - 可能有不同计算顺序)
    ref_mm = torch.mm(x, weight.T)

    # 精确顺序参考 (匹配 SNN 累加顺序)
    ref_exact = compute_ref_exact_order(x, weight, device)

    print(f"  PyTorch mm output shape: {ref_mm.shape}", flush=True)
    print(f"  mm vs exact order diff: max={torch.abs(ref_mm - ref_exact).max().item():.2e}", flush=True)

    # 手动 SNN 计算
    x_pulse = float32_to_pulse(x)  # [batch, in_f, 32]
    weight_pulse = float32_to_pulse(weight)  # [out_f, in_f, 32]

    print(f"  x_pulse shape: {x_pulse.shape}", flush=True)
    print(f"  weight_pulse shape: {weight_pulse.shape}", flush=True)

    # 广播乘法: [batch, 1, in_f, 32] x [out_f, in_f, 32] -> [batch, out_f, in_f, 32]
    x_expanded = x_pulse.unsqueeze(1)  # [batch, 1, in_f, 32]
    print(f"  x_expanded shape: {x_expanded.shape}", flush=True)

    mul.reset()
    products_pulse = mul(x_expanded, weight_pulse)  # [batch, out_f, in_f, 32]
    print(f"  products_pulse shape: {products_pulse.shape}", flush=True)

    # 累加: 对 in_f 维度求和
    # products_pulse: [batch, out_f, in_f, 32]
    acc = products_pulse[:, :, 0, :]  # [batch, out_f, 32]
    for i in range(1, in_f):
        add.reset()
        acc = add(acc, products_pulse[:, :, i, :])

    y_snn = pulse_to_float32(acc)
    print(f"  SNN output shape: {y_snn.shape}", flush=True)

    # 与 torch.mm 比较
    ulp_mm = compute_ulp(ref_mm, y_snn)
    print(f"  vs torch.mm: ULP max={ulp_mm['max_ulp']}, 0-ULP={ulp_mm['zero_ulp_rate']*100:.1f}%", flush=True)

    # 与精确顺序参考比较
    ulp_exact = compute_ulp(ref_exact, y_snn)
    print(f"  vs exact order: ULP max={ulp_exact['max_ulp']}, 0-ULP={ulp_exact['zero_ulp_rate']*100:.1f}%", flush=True)

    status = "PASS" if ulp_exact['max_ulp'] == 0 else "FAIL"
    print(f"  [{status}]", flush=True)

    # 如果失败，找出最大误差位置
    if ulp_exact['max_ulp'] > 0:
        ref_bits = ref_exact.view(-1).view(torch.int32).long()
        snn_bits = y_snn.view(-1).view(torch.int32).long()
        ulp_all = (ref_bits - snn_bits).abs()
        max_idx = ulp_all.argmax().item()
        batch_idx = max_idx // out_f
        out_idx = max_idx % out_f

        print(f"\n  Max ULP at batch={batch_idx}, out={out_idx}", flush=True)
        print(f"    ref={ref_exact.view(-1)[max_idx].item():.6f} (0x{ref_bits[max_idx].item():08X})", flush=True)
        print(f"    snn={y_snn.view(-1)[max_idx].item():.6f} (0x{snn_bits[max_idx].item():08X})", flush=True)

    return ulp_exact['max_ulp'] == 0


def test_broadcast_multiply(device, in_f=64, batch=10):
    """测试广播乘法是否正确"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 6b] Broadcast Multiply Test (batch={batch}, in_f={in_f})", flush=True)
    print("="*60, flush=True)

    from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier

    torch.manual_seed(42)

    mul = SpikeFP32Multiplier().to(device)

    # 创建输入 [batch, in_f]
    x = torch.randn(batch, in_f, dtype=torch.float32, device=device)
    # 创建权重 (单行) [in_f]
    w = torch.randn(in_f, dtype=torch.float32, device=device)

    # PyTorch 参考: 广播乘法
    ref_products = x * w  # [batch, in_f]
    print(f"  PyTorch products shape: {ref_products.shape}", flush=True)

    # SNN 计算
    x_pulse = float32_to_pulse(x)  # [batch, in_f, 32]
    w_pulse = float32_to_pulse(w)  # [in_f, 32]

    mul.reset()
    products_pulse = mul(x_pulse, w_pulse)  # [batch, in_f, 32]
    products_snn = pulse_to_float32(products_pulse)
    print(f"  SNN products shape: {products_snn.shape}", flush=True)

    # ULP 比较
    ulp = compute_ulp(ref_products, products_snn)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP: max={ulp['max_ulp']}, 0-ULP={ulp['zero_ulp_rate']*100:.1f}% [{status}]", flush=True)

    if ulp['max_ulp'] > 0:
        ref_bits = ref_products.view(-1).view(torch.int32).long()
        snn_bits = products_snn.view(-1).view(torch.int32).long()
        ulp_all = (ref_bits - snn_bits).abs()
        max_idx = ulp_all.argmax().item()
        print(f"\n  Max ULP at index {max_idx}", flush=True)
        print(f"    ref={ref_products.view(-1)[max_idx].item():.6f} (0x{ref_bits[max_idx].item():08X})", flush=True)
        print(f"    snn={products_snn.view(-1)[max_idx].item():.6f} (0x{snn_bits[max_idx].item():08X})", flush=True)

    return ulp['max_ulp'] == 0


def test_sequential_addition_trace(device, n_values=20):
    """逐步追踪顺序累加的错误"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 6e] Sequential Addition Trace (n={n_values})", flush=True)
    print("="*60, flush=True)

    from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder

    torch.manual_seed(42)

    add = SpikeFP32Adder().to(device)

    # 创建测试值
    values = torch.randn(n_values, dtype=torch.float32, device=device)
    print(f"  Values: {values[:5].tolist()} ...", flush=True)

    # 转换为脉冲
    values_pulse = float32_to_pulse(values)  # [n_values, 32]

    # 顺序累加 - 逐步追踪
    acc_ref = values[0].clone()
    acc_snn_pulse = values_pulse[0:1, :].clone()  # [1, 32]

    print("\n  Step-by-step trace:", flush=True)
    print(f"  {'Step':<5} {'Ref':>15} {'SNN':>15} {'ULP':>6}", flush=True)
    print(f"  {'-'*42}", flush=True)

    for i in range(1, n_values):
        # PyTorch
        acc_ref = acc_ref + values[i]

        # SNN
        add.reset()
        acc_snn_pulse = add(acc_snn_pulse, values_pulse[i:i+1, :])
        acc_snn = pulse_to_float32(acc_snn_pulse).item()

        # ULP
        ref_bits = acc_ref.view(torch.int32).item()
        snn_bits = torch.tensor(acc_snn).view(torch.int32).item()
        ulp = abs(ref_bits - snn_bits)

        if ulp > 0 or i <= 5 or i >= n_values - 3:
            print(f"  {i+1:<5} {acc_ref.item():>15.6f} {acc_snn:>15.6f} {ulp:>6}", flush=True)

    final_ulp = compute_ulp(acc_ref.unsqueeze(0), pulse_to_float32(acc_snn_pulse))
    print(f"\n  Final ULP: {final_ulp['max_ulp']}", flush=True)

    return final_ulp['max_ulp'] == 0


def test_batch_accumulation(device, in_f=64):
    """测试批量累加 - 对比单个累加和批量累加"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 6d] Batch Accumulation (in_f={in_f})", flush=True)
    print("="*60, flush=True)

    from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder

    torch.manual_seed(42)

    add = SpikeFP32Adder().to(device)

    # 创建测试数据 - 10 组独立的 64 个数要累加
    batch = 10
    values = torch.randn(batch, in_f, dtype=torch.float32, device=device)

    # PyTorch 参考: 顺序累加
    ref = torch.zeros(batch, device=device, dtype=torch.float32)
    for b in range(batch):
        acc = values[b, 0]
        for i in range(1, in_f):
            acc = acc + values[b, i]
        ref[b] = acc
    print(f"  PyTorch ref shape: {ref.shape}", flush=True)

    # SNN: 逐批单独累加 (batch=1)
    snn_single = torch.zeros(batch, device=device, dtype=torch.float32)
    values_pulse = float32_to_pulse(values)  # [batch, in_f, 32]
    for b in range(batch):
        acc = values_pulse[b:b+1, 0, :]  # [1, 32]
        for i in range(1, in_f):
            add.reset()
            acc = add(acc, values_pulse[b:b+1, i, :])
        snn_single[b] = pulse_to_float32(acc).item()

    ulp_single = compute_ulp(ref, snn_single)
    print(f"  SNN (batch=1 loop): ULP max={ulp_single['max_ulp']}, 0-ULP={ulp_single['zero_ulp_rate']*100:.1f}%", flush=True)

    # SNN: 批量累加 (batch=10)
    acc = values_pulse[:, 0, :]  # [batch, 32]
    for i in range(1, in_f):
        add.reset()
        acc = add(acc, values_pulse[:, i, :])
    snn_batch = pulse_to_float32(acc)

    ulp_batch = compute_ulp(ref, snn_batch)
    print(f"  SNN (batch=10): ULP max={ulp_batch['max_ulp']}, 0-ULP={ulp_batch['zero_ulp_rate']*100:.1f}%", flush=True)

    # 对比两种 SNN 方法
    ulp_compare = compute_ulp(snn_single, snn_batch)
    print(f"  SNN single vs batch: ULP max={ulp_compare['max_ulp']}", flush=True)

    return ulp_batch['max_ulp'] == 0


def test_single_row_matmul(device, in_f=64, batch=10):
    """测试单行矩阵乘法 (相当于一个输出神经元)"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 6c] Single Row Matmul (batch={batch}, in_f={in_f})", flush=True)
    print("="*60, flush=True)

    from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
    from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder

    torch.manual_seed(42)

    mul = SpikeFP32Multiplier().to(device)
    add = SpikeFP32Adder().to(device)

    # 创建输入 [batch, in_f]
    x = torch.randn(batch, in_f, dtype=torch.float32, device=device)
    # 创建权重 (单行) [in_f]
    w = torch.randn(in_f, dtype=torch.float32, device=device)

    # PyTorch 参考: 精确顺序累加 (向量化版本)
    products_ref = x * w  # [batch, in_f]
    acc_ref = products_ref[:, 0]  # [batch]
    for i in range(1, in_f):
        acc_ref = acc_ref + products_ref[:, i]  # PyTorch 向量化加法
    ref = acc_ref
    print(f"  PyTorch (exact order) shape: {ref.shape}", flush=True)

    # SNN 计算
    x_pulse = float32_to_pulse(x)  # [batch, in_f, 32]
    w_pulse = float32_to_pulse(w)  # [in_f, 32]

    mul.reset()
    products_pulse = mul(x_pulse, w_pulse)  # [batch, in_f, 32]

    # 验证乘积是否正确
    products_snn = pulse_to_float32(products_pulse)
    products_ulp = compute_ulp(products_ref, products_snn)
    print(f"  Products ULP: max={products_ulp['max_ulp']}", flush=True)

    # 累加 - 逐步追踪
    acc = products_pulse[:, 0, :]  # [batch, 32]
    acc_ref_trace = products_ref[:, 0]  # [batch]

    error_found = False
    for i in range(1, in_f):
        add.reset()
        acc = add(acc, products_pulse[:, i, :])
        acc_ref_trace = acc_ref_trace + products_ref[:, i]

        # 每10步检查一次
        if (i + 1) % 10 == 0 or i == in_f - 1:
            acc_snn = pulse_to_float32(acc)
            step_ulp = compute_ulp(acc_ref_trace, acc_snn)
            if step_ulp['max_ulp'] > 0 and not error_found:
                print(f"  Step {i+1}: ULP max={step_ulp['max_ulp']}, 0-ULP={step_ulp['zero_ulp_rate']*100:.1f}%", flush=True)
                error_found = True

    y_snn = pulse_to_float32(acc)
    print(f"  SNN output shape: {y_snn.shape}", flush=True)

    # ULP 比较
    ulp = compute_ulp(ref, y_snn)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP: max={ulp['max_ulp']}, 0-ULP={ulp['zero_ulp_rate']*100:.1f}% [{status}]", flush=True)

    if ulp['max_ulp'] > 0:
        ref_bits = ref.view(-1).view(torch.int32).long()
        snn_bits = y_snn.view(-1).view(torch.int32).long()
        ulp_all = (ref_bits - snn_bits).abs()
        max_idx = ulp_all.argmax().item()
        print(f"\n  Max ULP at batch {max_idx}", flush=True)
        print(f"    ref={ref[max_idx].item():.6f} (0x{ref_bits[max_idx].item():08X})", flush=True)
        print(f"    snn={y_snn[max_idx].item():.6f} (0x{snn_bits[max_idx].item():08X})", flush=True)

    return ulp['max_ulp'] == 0


def test_spike_linear_medium(device, in_f=64, out_f=32, batch=10):
    """测试 SpikeFP32Linear 中等矩阵"""
    print("\n" + "="*60, flush=True)
    print(f"[TEST 6] SpikeFP32Linear {in_f}x{out_f} (batch={batch})", flush=True)
    print("="*60, flush=True)

    torch.manual_seed(42)

    # PyTorch Linear
    pt_linear = nn.Linear(in_f, out_f, bias=False).to(device)

    # 输入
    x = torch.randn(batch, in_f, dtype=torch.float32, device=device)

    # PyTorch 参考
    ref = pt_linear(x)
    print(f"  Input shape: {x.shape}", flush=True)
    print(f"  Weight shape: {pt_linear.weight.shape}", flush=True)
    print(f"  Output shape: {ref.shape}", flush=True)

    # SNN
    snn_linear = SpikeFP32Linear(in_f, out_f, accum_precision='fp32').to(device)
    snn_linear.set_weight_from_float(pt_linear.weight.data)
    snn_linear.reset()

    x_pulse = float32_to_pulse(x)
    y_pulse = snn_linear(x_pulse)
    y_snn = pulse_to_float32(y_pulse)

    ulp = compute_ulp(ref, y_snn)
    status = "PASS" if ulp['max_ulp'] == 0 else "FAIL"
    print(f"  ULP: max={ulp['max_ulp']}, 0-ULP rate={ulp['zero_ulp_rate']*100:.1f}% [{status}]", flush=True)

    # 如果失败，找出最大误差位置
    if ulp['max_ulp'] > 0:
        ref_bits = ref.view(-1).view(torch.int32).long()
        snn_bits = y_snn.view(-1).view(torch.int32).long()
        ulp_all = (ref_bits - snn_bits).abs()
        max_idx = ulp_all.argmax().item()
        batch_idx = max_idx // out_f
        out_idx = max_idx % out_f

        print(f"\n  Max ULP at batch={batch_idx}, out={out_idx}", flush=True)
        print(f"    ref={ref.view(-1)[max_idx].item():.6f} (0x{ref_bits[max_idx].item():08X})", flush=True)
        print(f"    snn={y_snn.view(-1)[max_idx].item():.6f} (0x{snn_bits[max_idx].item():08X})", flush=True)

    return ulp['max_ulp'] == 0


def main():
    print("="*60, flush=True)
    print("SpikeFP32Linear ULP Debug", flush=True)
    print("="*60, flush=True)

    device = get_device()

    results = []

    # 1. 测试乘法器
    results.append(("Single Multiply", test_single_multiply(device)))

    # 2. 测试加法器
    results.append(("Single Add", test_single_add(device)))

    # 3. 手动点积 (with reset)
    results.append(("Manual Dot Product (reset)", test_dot_product_manual(device)))

    # 3b. 手动点积 (no reset)
    results.append(("Manual Dot Product (no reset)", test_dot_product_no_reset(device)))

    # 3c. 大点积 (size=64)
    results.append(("Large Dot Product (64)", test_dot_product_larger(device)))

    # 4. SpikeFP32Linear 1x1
    results.append(("SpikeFP32Linear 1x1", test_spike_linear_1x1(device)))

    # 5. SpikeFP32Linear 小矩阵
    results.append(("SpikeFP32Linear 4x2", test_spike_linear_small(device)))

    # 6. 手动 Linear 实现
    results.append(("Manual Linear 64x32", test_manual_linear(device)))

    # 6b. 广播乘法测试
    results.append(("Broadcast Multiply", test_broadcast_multiply(device)))

    # 6c. 单行矩阵乘法测试
    results.append(("Single Row Matmul", test_single_row_matmul(device)))

    # 6d. 批量累加测试
    results.append(("Batch Accumulation", test_batch_accumulation(device)))

    # 6e. 顺序累加追踪
    results.append(("Sequential Add Trace", test_sequential_addition_trace(device)))

    # 7. SpikeFP32Linear 中等矩阵
    results.append(("SpikeFP32Linear 64x32", test_spike_linear_medium(device)))

    # 汇总
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]", flush=True)

    all_pass = all(p for _, p in results)
    if all_pass:
        print("\nAll tests passed!", flush=True)
    else:
        print("\nSome tests FAILED - need to investigate!", flush=True)


if __name__ == '__main__':
    main()
