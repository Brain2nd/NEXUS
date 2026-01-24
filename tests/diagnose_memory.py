"""
显存累积诊断测试 - 手动清理版
==============================

精确定位"滚雪球"显存累积问题的来源。
在每个操作后手动调用 reset() 来诊断问题。

运行方式:
    python tests/diagnose_memory.py
    python tests/diagnose_memory.py --test gate
    python tests/diagnose_memory.py --test linear
    python tests/diagnose_memory.py --test layer
    python tests/diagnose_memory.py --test trace  # 详细张量追踪
"""
import torch
import gc
import sys
import os
import weakref

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mem_mb():
    """返回当前 allocated 和 reserved 显存 (MB)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        resv = torch.cuda.memory_reserved() / 1024 / 1024
        return alloc, resv
    return 0, 0


def force_gc():
    """强制垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_mem(label):
    alloc, resv = mem_mb()
    print(f"  {label}: alloc={alloc:.2f}MB, reserved={resv:.2f}MB")


def get_all_cuda_tensors():
    """获取所有当前在 GPU 上的张量及其大小"""
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.numel() * obj.element_size() / 1024 / 1024
                cuda_tensors.append({
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'size_mb': size_mb,
                    'device': str(obj.device),
                    'requires_grad': obj.requires_grad,
                    'id': id(obj),
                    'is_leaf': obj.is_leaf,
                    'grad_fn': type(obj.grad_fn).__name__ if obj.grad_fn else None,
                })
        except Exception:
            pass
    return cuda_tensors


def identify_tensor_owners(model, target_shapes=None):
    """识别模型中哪些参数/缓冲区拥有特定形状的张量"""
    owners = []

    # 检查 parameters
    for name, param in model.named_parameters():
        if target_shapes is None or tuple(param.shape) in target_shapes:
            size_mb = param.numel() * param.element_size() / 1024 / 1024
            owners.append({
                'type': 'parameter',
                'name': name,
                'shape': tuple(param.shape),
                'size_mb': size_mb,
                'id': id(param.data),
            })

    # 检查 buffers
    for name, buf in model.named_buffers():
        if buf is not None and (target_shapes is None or tuple(buf.shape) in target_shapes):
            size_mb = buf.numel() * buf.element_size() / 1024 / 1024
            owners.append({
                'type': 'buffer',
                'name': name,
                'shape': tuple(buf.shape),
                'size_mb': size_mb,
                'id': id(buf),
            })

    return owners


def find_orphan_tensors(model):
    """找出不属于模型参数/缓冲区的 CUDA 张量（潜在泄漏）"""
    # 收集所有模型拥有的张量 ID
    owned_ids = set()
    for param in model.parameters():
        owned_ids.add(id(param.data))
    for buf in model.buffers():
        if buf is not None:
            owned_ids.add(id(buf))

    # 收集所有 CUDA 张量
    all_cuda = get_all_cuda_tensors()

    # 找出孤儿张量
    orphans = [t for t in all_cuda if t['id'] not in owned_ids]
    owned = [t for t in all_cuda if t['id'] in owned_ids]

    return orphans, owned


def count_neuron_v_tensors(module, prefix=""):
    """统计模块中所有神经元的 self.v 膜电位张量"""
    from atomic_ops.core.neurons import SimpleIFNode, SimpleLIFNode

    v_count = 0
    v_total_mb = 0
    v_details = []

    for name, child in module.named_modules():
        if isinstance(child, (SimpleIFNode, SimpleLIFNode)):
            if child.v is not None:
                size_mb = child.v.numel() * child.v.element_size() / 1024 / 1024
                v_count += 1
                v_total_mb += size_mb
                v_details.append({
                    'name': f"{prefix}{name}" if name else prefix,
                    'shape': tuple(child.v.shape),
                    'size_mb': size_mb,
                })

    return v_count, v_total_mb, v_details


def print_neuron_v_stats(module, label=""):
    """打印神经元膜电位统计"""
    v_count, v_total_mb, v_details = count_neuron_v_tensors(module)

    print(f"\n{'='*60}")
    print(f"神经元膜电位 (self.v) 统计 {label}")
    print(f"{'='*60}")
    print(f"总计: {v_count} 个神经元有 self.v, 占用 {v_total_mb:.2f} MB")

    # 按形状分组
    shape_groups = {}
    for d in v_details:
        key = d['shape']
        if key not in shape_groups:
            shape_groups[key] = {'count': 0, 'total_mb': 0}
        shape_groups[key]['count'] += 1
        shape_groups[key]['total_mb'] += d['size_mb']

    print(f"\n按形状分组:")
    print(f"{'Shape':<40} {'Count':<10} {'Total(MB)'}")
    print("-" * 60)
    for shape, info in sorted(shape_groups.items(), key=lambda x: -x[1]['total_mb']):
        print(f"{str(shape):<40} {info['count']:<10} {info['total_mb']:.4f}")

    return v_count, v_total_mb


def release_neuron_v_tensors(module):
    """释放模块中所有神经元的 self.v 膜电位张量"""
    from atomic_ops.core.neurons import SimpleIFNode, SimpleLIFNode

    released = 0
    for child in module.modules():
        if isinstance(child, (SimpleIFNode, SimpleLIFNode)):
            if child.v is not None:
                child.v = None  # 真正释放，而不是 zero_()
                released += 1
    return released


def print_cuda_tensors(label="", top_n=20, min_mb=0.1):
    """打印当前 GPU 张量列表"""
    tensors = get_all_cuda_tensors()
    tensors.sort(key=lambda x: x['size_mb'], reverse=True)

    total_mb = sum(t['size_mb'] for t in tensors)
    print(f"\n{'='*60}")
    print(f"CUDA 张量列表 {label}")
    print(f"{'='*60}")
    print(f"总计: {len(tensors)} 个张量, {total_mb:.2f} MB")
    print(f"\n按大小排序 (显示 >{min_mb:.2f}MB 的前 {top_n} 个):")
    print(f"{'Shape':<30} {'Dtype':<15} {'Size(MB)':<12} {'Grad'}")
    print("-" * 70)

    shown = 0
    for t in tensors:
        if t['size_mb'] >= min_mb and shown < top_n:
            grad_str = "✓" if t['requires_grad'] else ""
            print(f"{str(t['shape']):<30} {t['dtype']:<15} {t['size_mb']:<12.4f} {grad_str}")
            shown += 1

    # 按形状分组统计
    shape_groups = {}
    for t in tensors:
        key = (t['shape'], t['dtype'])
        if key not in shape_groups:
            shape_groups[key] = {'count': 0, 'total_mb': 0}
        shape_groups[key]['count'] += 1
        shape_groups[key]['total_mb'] += t['size_mb']

    print(f"\n按形状分组统计:")
    print(f"{'Shape':<30} {'Dtype':<15} {'Count':<8} {'Total(MB)'}")
    print("-" * 70)
    sorted_groups = sorted(shape_groups.items(), key=lambda x: x[1]['total_mb'], reverse=True)
    for (shape, dtype), info in sorted_groups[:15]:
        if info['total_mb'] >= min_mb:
            print(f"{str(shape):<30} {dtype:<15} {info['count']:<8} {info['total_mb']:.4f}")

    return tensors


def compare_tensors(before, after, label=""):
    """比较两次张量快照的差异"""
    before_ids = {t['id'] for t in before}
    after_ids = {t['id'] for t in after}

    new_ids = after_ids - before_ids
    removed_ids = before_ids - after_ids

    after_map = {t['id']: t for t in after}
    before_map = {t['id']: t for t in before}

    new_tensors = [after_map[tid] for tid in new_ids]
    new_tensors.sort(key=lambda x: x['size_mb'], reverse=True)

    removed_tensors = [before_map[tid] for tid in removed_ids]

    print(f"\n{'='*60}")
    print(f"张量变化 {label}")
    print(f"{'='*60}")
    print(f"新增: {len(new_tensors)} 个, 移除: {len(removed_tensors)} 个")

    new_total = sum(t['size_mb'] for t in new_tensors)
    removed_total = sum(t['size_mb'] for t in removed_tensors)
    print(f"新增总计: {new_total:.2f} MB, 移除总计: {removed_total:.2f} MB")
    print(f"净变化: {new_total - removed_total:+.2f} MB")

    if new_tensors:
        print(f"\n新增张量 (前10个):")
        print(f"{'Shape':<30} {'Dtype':<15} {'Size(MB)':<12}")
        print("-" * 60)
        for t in new_tensors[:10]:
            print(f"{str(t['shape']):<30} {t['dtype']:<15} {t['size_mb']:.4f}")

    return new_tensors, removed_tensors


def reset_all_children(module, verbose=False):
    """递归重置模块及其所有子模块"""
    count = 0
    for child in module.modules():
        if hasattr(child, 'reset'):
            try:
                child.reset()
                count += 1
            except Exception as e:
                if verbose:
                    print(f"    警告: {type(child).__name__}.reset() 失败: {e}")
    return count


def test_gate_memory():
    """测试基本门电路的显存累积"""
    print("\n" + "="*60)
    print("测试1: 基本门电路 (VecAND) 显存累积")
    print("="*60)

    from atomic_ops.core.vec_logic_gates import VecAND
    from atomic_ops.core.spike_mode import SpikeMode

    device = torch.device('cuda')

    # 测试有 max_param_shape 和没有的区别
    print(f"\nSpikeMode: {SpikeMode.get_mode()}")

    print("\n--- 无 max_param_shape ---")
    gate = VecAND().to(device)
    force_gc()
    print_mem("创建后")
    baseline = mem_mb()[0]

    for i in range(5):
        a = torch.randn(1000, 32, device=device)
        b = torch.randn(1000, 32, device=device)
        c = gate(a, b)
        alloc = mem_mb()[0]
        print(f"  forward {i+1}: {alloc:.2f}MB (Δ{alloc-baseline:+.2f}MB)")

        # 手动 reset
        gate.reset()
        del a, b, c
        force_gc()
        alloc = mem_mb()[0]
        print(f"  reset+GC: {alloc:.2f}MB (Δ{alloc-baseline:+.2f}MB)")

    print("\n--- 有 max_param_shape=(32,) ---")
    gate2 = VecAND(max_param_shape=(32,)).to(device)
    force_gc()
    print_mem("创建后")
    baseline = mem_mb()[0]

    for i in range(5):
        a = torch.randn(1000, 32, device=device)
        b = torch.randn(1000, 32, device=device)
        c = gate2(a, b)
        alloc = mem_mb()[0]
        print(f"  forward {i+1}: {alloc:.2f}MB (Δ{alloc-baseline:+.2f}MB)")

        gate2.reset()
        del a, b, c
        force_gc()
        alloc = mem_mb()[0]
        print(f"  reset+GC: {alloc:.2f}MB (Δ{alloc-baseline:+.2f}MB)")


def test_linear_memory():
    """测试 Linear 层的显存累积"""
    print("\n" + "="*60)
    print("测试2: SpikeFP32Linear 显存累积")
    print("="*60)

    from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision
    from atomic_ops import float32_to_pulse

    device = torch.device('cuda')

    in_features = 64
    out_features = 64
    batch_size = 1
    seq_len = 4

    linear = SpikeFP32Linear_MultiPrecision(in_features, out_features).to(device)
    weight = torch.randn(out_features, in_features, device=device)
    linear.set_weight_from_float(weight)

    force_gc()
    print_mem("初始化后")
    baseline = mem_mb()[0]

    # 创建输入
    x_float = torch.randn(batch_size, seq_len, in_features, device=device)
    x_pulse = float32_to_pulse(x_float)
    print(f"  输入形状: {x_pulse.shape}")

    for i in range(5):
        print(f"\n--- Forward {i+1} ---")
        mem_before = mem_mb()[0]

        with torch.no_grad():
            y = linear(x_pulse)

        mem_after = mem_mb()[0]
        print(f"  forward: {mem_before:.2f}MB -> {mem_after:.2f}MB (Δ{mem_after-mem_before:+.2f}MB)")

        # 手动 reset
        count = reset_all_children(linear)
        print(f"  reset {count} 个模块")

        del y
        force_gc()

        mem_after_gc = mem_mb()[0]
        print(f"  reset+GC: {mem_after_gc:.2f}MB (相对基线: Δ{mem_after_gc-baseline:+.2f}MB)")


def test_layer_memory():
    """测试完整层的显存累积"""
    print("\n" + "="*60)
    print("测试3: SNN 层显存累积（手动 reset 版）")
    print("="*60)

    from transformers import Qwen3Config, Qwen3ForCausalLM
    from models.reference.modeling_qwen3 import SpikeQwen3Config, SpikeQwen3ForCausalLM

    device = torch.device('cuda')

    # 极小配置
    vocab_size = 1000
    hidden_size = 64
    intermediate_size = 128
    num_layers = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    seq_len = 4
    batch_size = 1

    print(f"配置: hidden={hidden_size}, intermediate={intermediate_size}")

    # 创建模型
    hf_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    hf_model = Qwen3ForCausalLM(hf_config).to(device).eval()

    snn_config = SpikeQwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    snn_model = SpikeQwen3ForCausalLM(snn_config).to(device)

    # 复制权重
    print("\n复制权重...")
    snn_model.model.set_embedding_weight(hf_model.model.embed_tokens.weight.data)
    snn_model.lm_head.set_weight_from_float(hf_model.lm_head.weight.data)
    snn_model.model.norm.weight.data = hf_model.model.norm.weight.data.clone()

    layer = snn_model.model.layers[0]
    hf_layer = hf_model.model.layers[0]

    layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
    layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

    layer.self_attn.set_weights_from_float(
        hf_layer.self_attn.q_proj.weight.data,
        hf_layer.self_attn.k_proj.weight.data,
        hf_layer.self_attn.v_proj.weight.data,
        hf_layer.self_attn.o_proj.weight.data,
        hf_layer.self_attn.q_norm.weight.data,
        hf_layer.self_attn.k_norm.weight.data,
    )

    layer.mlp.set_weights_from_float(
        hf_layer.mlp.gate_proj.weight.data,
        hf_layer.mlp.up_proj.weight.data,
        hf_layer.mlp.down_proj.weight.data,
    )

    del hf_model
    force_gc()
    print_mem("权重复制完成")

    # 准备输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    positions = torch.arange(seq_len, device=device)

    # 全局 reset
    snn_model.reset()
    force_gc()
    print_mem("全局 reset 后（基线）")
    baseline = mem_mb()[0]

    print("\n" + "="*60)
    print("逐操作测试（每操作后手动 reset + GC）")
    print("="*60)

    attn = layer.self_attn

    with torch.no_grad():
        # Embedding
        print("\n--- Embedding ---")
        hidden_states = snn_model.model.embed_tokens(input_ids)
        print(f"  shape: {list(hidden_states.shape)}")
        print_mem("forward 后")

        count = reset_all_children(snn_model.model.embed_tokens)
        force_gc()
        print(f"  reset({count})+GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

        # Input LayerNorm
        print("\n--- input_layernorm ---")
        residual = hidden_states.clone()  # 保留副本
        hidden_states = layer.input_layernorm(hidden_states)
        print_mem("forward 后")

        count = reset_all_children(layer.input_layernorm)
        force_gc()
        print(f"  reset({count})+GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

        # Q_proj
        print("\n--- Q_proj ---")
        Q = attn.q_proj(hidden_states)
        print_mem("forward 后")

        count = reset_all_children(attn.q_proj)
        force_gc()
        print(f"  reset({count})+GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

        # 测试：删除 Q 后
        del Q
        force_gc()
        print(f"  del Q + GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

        # 再次 Q_proj
        print("\n--- Q_proj (第二次) ---")
        Q = attn.q_proj(hidden_states)
        print_mem("forward 后")

        count = reset_all_children(attn.q_proj)
        force_gc()
        print(f"  reset({count})+GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

        # 测试 gate_proj (MLP)
        print("\n--- gate_proj ---")
        del Q
        force_gc()

        gate = layer.mlp.gate_proj(hidden_states)
        print_mem("forward 后")

        count = reset_all_children(layer.mlp.gate_proj)
        force_gc()
        print(f"  reset({count})+GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

        del gate
        force_gc()
        print(f"  del gate + GC 后: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"\n最终显存: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")


def test_tensor_trace():
    """详细追踪每个操作的张量分配"""
    print("\n" + "="*60)
    print("测试4: 详细张量追踪")
    print("="*60)

    from transformers import Qwen3Config, Qwen3ForCausalLM
    from models.reference.modeling_qwen3 import SpikeQwen3Config, SpikeQwen3ForCausalLM

    device = torch.device('cuda')

    # 极小配置
    vocab_size = 1000
    hidden_size = 64
    intermediate_size = 128
    num_layers = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    seq_len = 4
    batch_size = 1

    print(f"配置: hidden={hidden_size}, intermediate={intermediate_size}")

    # 创建模型
    hf_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    hf_model = Qwen3ForCausalLM(hf_config).to(device).eval()

    snn_config = SpikeQwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    snn_model = SpikeQwen3ForCausalLM(snn_config).to(device)

    # 复制权重
    print("\n复制权重...")
    snn_model.model.set_embedding_weight(hf_model.model.embed_tokens.weight.data)
    snn_model.lm_head.set_weight_from_float(hf_model.lm_head.weight.data)
    snn_model.model.norm.weight.data = hf_model.model.norm.weight.data.clone()

    layer = snn_model.model.layers[0]
    hf_layer = hf_model.model.layers[0]

    layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
    layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

    layer.self_attn.set_weights_from_float(
        hf_layer.self_attn.q_proj.weight.data,
        hf_layer.self_attn.k_proj.weight.data,
        hf_layer.self_attn.v_proj.weight.data,
        hf_layer.self_attn.o_proj.weight.data,
        hf_layer.self_attn.q_norm.weight.data,
        hf_layer.self_attn.k_norm.weight.data,
    )

    layer.mlp.set_weights_from_float(
        hf_layer.mlp.gate_proj.weight.data,
        hf_layer.mlp.up_proj.weight.data,
        hf_layer.mlp.down_proj.weight.data,
    )

    del hf_model
    force_gc()
    print_mem("权重复制完成")

    # 准备输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # 全局 reset
    snn_model.reset()
    force_gc()

    print("\n" + "="*60)
    print("初始张量快照")
    print("="*60)
    baseline_tensors = print_cuda_tensors("(基线)", top_n=30, min_mb=0.01)
    baseline = mem_mb()[0]

    # 检查整个模型的 max_param_shape 使用情况
    print("\n" + "="*60)
    print("神经元参数分配机制统计（全模型）")
    print("="*60)
    from atomic_ops.core.neurons import SimpleIFNode, SimpleLIFNode
    all_neurons = [m for m in snn_model.modules()
                   if isinstance(m, (SimpleIFNode, SimpleLIFNode))]
    with_prealloc = sum(1 for n in all_neurons if n.max_param_shape is not None)
    with_lazy = len(all_neurons) - with_prealloc
    print(f"总神经元数: {len(all_neurons)}")
    print(f"预加载切片机制 (max_param_shape): {with_prealloc}")
    print(f"旧懒加载机制 (已废弃): {with_lazy}  ← 问题根源!")

    # 区分模型拥有的张量 vs 孤儿张量
    print("\n" + "="*60)
    print("孤儿张量分析 (不属于 model.parameters/buffers)")
    print("="*60)
    orphans, owned = find_orphan_tensors(snn_model)
    orphan_mb = sum(t['size_mb'] for t in orphans)
    owned_mb = sum(t['size_mb'] for t in owned)
    print(f"模型拥有: {len(owned)} 个张量, {owned_mb:.2f} MB")
    print(f"孤儿张量: {len(orphans)} 个张量, {orphan_mb:.2f} MB")

    if orphans:
        print(f"\n孤儿张量按大小排序 (前20个):")
        print(f"{'Shape':<30} {'Dtype':<15} {'Size(MB)':<10} {'Grad':<6} {'Leaf':<6} {'GradFn'}")
        print("-" * 90)
        orphans.sort(key=lambda x: x['size_mb'], reverse=True)
        for t in orphans[:20]:
            grad_str = "✓" if t['requires_grad'] else ""
            leaf_str = "✓" if t.get('is_leaf') else ""
            grad_fn = t.get('grad_fn', '')
            print(f"{str(t['shape']):<30} {t['dtype']:<15} {t['size_mb']:<10.4f} {grad_str:<6} {leaf_str:<6} {grad_fn}")

    print("\n" + "="*60)
    print("逐操作张量追踪")
    print("="*60)

    attn = layer.self_attn

    with torch.no_grad():
        # Embedding
        print("\n\n>>> EMBEDDING <<<")
        before = get_all_cuda_tensors()
        before_orphans, _ = find_orphan_tensors(snn_model)
        hidden_states = snn_model.model.embed_tokens(input_ids)
        after = get_all_cuda_tensors()
        after_orphans, _ = find_orphan_tensors(snn_model)
        compare_tensors(before, after, "Embedding")
        print(f"孤儿张量变化: {len(before_orphans)} -> {len(after_orphans)} (Δ{len(after_orphans)-len(before_orphans):+d})")
        print_mem("forward 后")

        reset_all_children(snn_model.model.embed_tokens)
        force_gc()
        after_gc = get_all_cuda_tensors()
        compare_tensors(after, after_gc, "reset+GC 后")

        # Input LayerNorm - 详细追踪
        print("\n\n>>> INPUT_LAYERNORM (详细追踪) <<<")
        residual = hidden_states.clone()

        # 统计 RMSNorm 中的神经元
        from atomic_ops.core.neurons import SimpleIFNode, SimpleLIFNode
        neurons = [m for m in layer.input_layernorm.modules()
                   if isinstance(m, (SimpleIFNode, SimpleLIFNode))]
        print(f"RMSNorm 中的神经元数量: {len(neurons)}")

        # 检查参数分配机制
        with_prealloc = sum(1 for n in neurons if n.max_param_shape is not None)
        with_lazy = len(neurons) - with_prealloc
        print(f"  预加载切片机制: {with_prealloc}")
        print(f"  旧懒加载机制: {with_lazy}  ← 问题根源!")

        # 统计参数张量
        params_before = []
        for name, m in layer.input_layernorm.named_modules():
            if isinstance(m, (SimpleIFNode, SimpleLIFNode)):
                if m.v is not None:
                    params_before.append(('v', name, tuple(m.v.shape)))
                if hasattr(m, '_v_threshold') and m._v_threshold is not None:
                    params_before.append(('threshold', name, tuple(m._v_threshold.shape)))
                if hasattr(m, '_beta') and m._beta is not None:
                    params_before.append(('beta', name, tuple(m._beta.shape)))
        print(f"Forward 前神经元状态张量: {len(params_before)}")

        before = get_all_cuda_tensors()
        hidden_states = layer.input_layernorm(hidden_states)
        after = get_all_cuda_tensors()

        # 统计 forward 后
        params_after = []
        v_shapes = {}
        threshold_shapes = {}
        beta_shapes = {}
        for name, m in layer.input_layernorm.named_modules():
            if isinstance(m, (SimpleIFNode, SimpleLIFNode)):
                if m.v is not None:
                    shape = tuple(m.v.shape)
                    v_shapes[shape] = v_shapes.get(shape, 0) + 1
                    params_after.append(('v', name, shape))
                if hasattr(m, '_v_threshold') and m._v_threshold is not None:
                    shape = tuple(m._v_threshold.shape)
                    threshold_shapes[shape] = threshold_shapes.get(shape, 0) + 1
                    params_after.append(('threshold', name, shape))
                if hasattr(m, '_beta') and m._beta is not None:
                    shape = tuple(m._beta.shape)
                    beta_shapes[shape] = beta_shapes.get(shape, 0) + 1
                    params_after.append(('beta', name, shape))

        print(f"Forward 后神经元状态张量: {len(params_after)} (新增 {len(params_after)-len(params_before)})")
        print(f"\n膜电位 (v) 形状分布:")
        for shape, count in sorted(v_shapes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {shape}: {count} 个")
        print(f"\n阈值 (threshold) 形状分布:")
        for shape, count in sorted(threshold_shapes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {shape}: {count} 个")
        print(f"\nbeta 形状分布:")
        for shape, count in sorted(beta_shapes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {shape}: {count} 个")

        compare_tensors(before, after, "input_layernorm")

        reset_all_children(layer.input_layernorm)
        force_gc()

        # Q_proj - 这是主要的内存增长点
        print("\n\n>>> Q_PROJ (关键观察点) <<<")
        before = get_all_cuda_tensors()
        before_orphans, _ = find_orphan_tensors(snn_model)
        Q = attn.q_proj(hidden_states)
        after = get_all_cuda_tensors()
        after_orphans, _ = find_orphan_tensors(snn_model)
        new_tensors, _ = compare_tensors(before, after, "Q_proj forward")
        print(f"\nQ 输出形状: {Q.shape}")
        print(f"孤儿张量变化: {len(before_orphans)} -> {len(after_orphans)} (Δ{len(after_orphans)-len(before_orphans):+d})")
        orphan_mb_before = sum(t['size_mb'] for t in before_orphans)
        orphan_mb_after = sum(t['size_mb'] for t in after_orphans)
        print(f"孤儿内存变化: {orphan_mb_before:.2f}MB -> {orphan_mb_after:.2f}MB (Δ{orphan_mb_after-orphan_mb_before:+.2f}MB)")
        print_mem("forward 后")

        reset_all_children(attn.q_proj)
        del Q
        force_gc()

        after_gc = get_all_cuda_tensors()
        after_gc_orphans, _ = find_orphan_tensors(snn_model)
        compare_tensors(after, after_gc, "reset+del Q+GC 后")
        print(f"GC后孤儿张量: {len(after_gc_orphans)} (Δ{len(after_gc_orphans)-len(after_orphans):+d})")

        # gate_proj
        print("\n\n>>> GATE_PROJ (MLP) <<<")
        before = get_all_cuda_tensors()
        before_orphans, _ = find_orphan_tensors(snn_model)
        gate = layer.mlp.gate_proj(hidden_states)
        after = get_all_cuda_tensors()
        after_orphans, _ = find_orphan_tensors(snn_model)
        new_tensors, _ = compare_tensors(before, after, "gate_proj forward")
        print(f"\ngate 输出形状: {gate.shape}")
        print(f"孤儿张量变化: {len(before_orphans)} -> {len(after_orphans)} (Δ{len(after_orphans)-len(before_orphans):+d})")
        orphan_mb_before = sum(t['size_mb'] for t in before_orphans)
        orphan_mb_after = sum(t['size_mb'] for t in after_orphans)
        print(f"孤儿内存变化: {orphan_mb_before:.2f}MB -> {orphan_mb_after:.2f}MB (Δ{orphan_mb_after-orphan_mb_before:+.2f}MB)")
        print_mem("forward 后")

        reset_all_children(layer.mlp.gate_proj)
        del gate
        force_gc()

    print("\n" + "="*60)
    print("神经元膜电位累积诊断")
    print("="*60)

    # 统计膜电位
    print_neuron_v_stats(snn_model, "(forward 后)")

    # 尝试真正释放膜电位
    print("\n>>> 尝试真正释放 self.v (设为 None) <<<")
    released = release_neuron_v_tensors(snn_model)
    print(f"释放了 {released} 个神经元的 self.v")
    force_gc()
    print_mem("释放 self.v + GC 后")

    print_neuron_v_stats(snn_model, "(释放后)")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

    # 最终快照
    final_tensors = print_cuda_tensors("(最终)", top_n=30, min_mb=0.01)
    compare_tensors(baseline_tensors, final_tensors, "总变化 (基线 vs 最终)")

    print(f"\n最终显存: Δ{mem_mb()[0]-baseline:+.2f}MB vs baseline")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["gate", "linear", "layer", "trace", "all"], default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA 设备")
        sys.exit(1)

    if args.test == "gate" or args.test == "all":
        test_gate_memory()

    if args.test == "linear" or args.test == "all":
        test_linear_memory()

    if args.test == "layer" or args.test == "all":
        test_layer_memory()

    if args.test == "trace":
        test_tensor_trace()
