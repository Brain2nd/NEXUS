"""
Qwen3 SNN End-to-End Tests (HuggingFace Baseline)
===================================================

Baseline: HuggingFace Qwen/Qwen3-0.6B pretrained model
Target: 0 ULP error (bit-exact with HuggingFace)

详细日志版本 - 打印每层初始化时间和前向传播细节
包含细粒度显存监控和神经元状态统计

Author: MofNeuroSim Project
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import SpikeQwen3Config, SpikeQwen3ForCausalLM
from atomic_ops import float32_to_pulse, pulse_to_float32


# ==============================================================================
# 显存追踪器
# ==============================================================================
class MemoryTracker:
    """追踪显存变化，检测累积"""
    def __init__(self):
        self.history = []
        self.baseline = 0
        self.layer_stats = {}  # 每层统计

    def reset_baseline(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.baseline = torch.cuda.memory_allocated() / 1024 / 1024
        self.history = []
        self.layer_stats = {}

    def record(self, label):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024 / 1024
            delta = alloc - self.baseline
            self.history.append((label, alloc, delta))
            return alloc, delta
        return 0, 0

    def record_layer_start(self, layer_idx):
        """记录层开始时的显存"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.layer_stats[layer_idx] = {
                'start': torch.cuda.memory_allocated() / 1024 / 1024,
                'peak': 0,
                'end': 0,
            }

    def record_layer_end(self, layer_idx):
        """记录层结束时的显存"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if layer_idx in self.layer_stats:
                self.layer_stats[layer_idx]['end'] = torch.cuda.memory_allocated() / 1024 / 1024
                self.layer_stats[layer_idx]['peak'] = torch.cuda.max_memory_allocated() / 1024 / 1024

    def print_summary(self):
        if not self.history:
            print("\n[显存追踪] 无记录数据")
            return

        print("\n" + "=" * 80)
        print("显存累积分析摘要")
        print("=" * 80)
        print(f"{'操作':<40} {'当前(MB)':<12} {'相对基线(MB)':<15}")
        print("-" * 80)

        # 找出增长最大的操作
        max_delta = 0
        max_op = ""
        for label, alloc, delta in self.history:
            if delta > max_delta:
                max_delta = delta
                max_op = label

        for label, alloc, delta in self.history:
            marker = " <<<" if delta == max_delta and delta > 1 else ""
            print(f"{label:<40} {alloc:<12.2f} {delta:+<15.2f}{marker}")

        print("-" * 80)
        if max_delta > 1:
            print(f"峰值增长: {max_op} (+{max_delta:.2f}MB)")

        # 层统计
        if self.layer_stats:
            print("\n" + "-" * 80)
            print("逐层显存统计:")
            print(f"{'Layer':<10} {'开始(MB)':<12} {'结束(MB)':<12} {'增量(MB)':<12}")
            print("-" * 80)
            total_growth = 0
            for idx in sorted(self.layer_stats.keys()):
                stats = self.layer_stats[idx]
                growth = stats['end'] - stats['start']
                total_growth += growth
                marker = " ⚠️" if growth > 10 else ""
                print(f"Layer {idx:<4} {stats['start']:<12.1f} {stats['end']:<12.1f} {growth:+<12.1f}{marker}")
            print("-" * 80)
            print(f"总增长: {total_growth:+.1f}MB")


mem_tracker = MemoryTracker()


# ==============================================================================
# 神经元状态检查器
# ==============================================================================
def count_neuron_states(module, prefix=""):
    """统计模块中所有神经元的 v 状态

    Returns:
        (total_neurons, neurons_with_v, total_v_memory_bytes)
    """
    total = 0
    with_v = 0
    memory = 0

    for name, child in module.named_modules():
        if hasattr(child, 'v'):
            total += 1
            if child.v is not None:
                with_v += 1
                memory += child.v.numel() * child.v.element_size()

    return total, with_v, memory


def check_neuron_cleanup(module, label=""):
    """检查神经元是否被正确清理"""
    total, with_v, memory = count_neuron_states(module)
    memory_mb = memory / 1024 / 1024

    if with_v == 0:
        status = "✓ 已清理"
    else:
        status = f"⚠️ 残留 {with_v}/{total} ({memory_mb:.2f}MB)"

    return total, with_v, memory_mb, status


# ==============================================================================
# 工具函数
# ==============================================================================
def force_gc():
    """强制垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem_mb():
    """获取当前GPU显存(MB)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def print_gpu_mem(label=""):
    """打印GPU显存状态"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   {label} GPU: {alloc:.1f}MB (reserved: {reserved:.1f}MB, peak: {peak:.1f}MB)")


def compute_ulp_error_fp32(snn_result, ref_result):
    """Compute FP32 ULP error."""
    snn_bits = snn_result.view(torch.int32)
    ref_bits = ref_result.view(torch.int32)
    ulp_diff = (snn_bits - ref_bits).abs()
    return {
        'max_ulp': ulp_diff.max().item(),
        'mean_ulp': ulp_diff.float().mean().item(),
        'zero_ulp_rate': (ulp_diff == 0).float().mean().item() * 100,
    }


def timed(name, show_mem=False, track=True):
    """计时上下文管理器 - 可选显示显存变化"""
    class Timer:
        def __init__(self, name, show_mem, track):
            self.name = name
            self.show_mem = show_mem
            self.track = track
            self.mem_before = 0

        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.mem_before = torch.cuda.memory_allocated() / 1024 / 1024
            self.t0 = time.time()
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            suffix = ""

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                delta = mem_after - self.mem_before

                if self.track:
                    mem_tracker.record(self.name)

                if self.show_mem:
                    delta_str = f"Δ{delta:+.1f}MB" if abs(delta) > 0.1 else ""
                    suffix = f" [{mem_after:.1f}MB {delta_str}]"

            print(f"      {self.name}: {elapsed:.3f}s{suffix}", flush=True)

    return Timer(name, show_mem, track)


def print_shape(name, tensor):
    """打印张量形状"""
    if hasattr(tensor, 'shape'):
        print(f"         {name}: {list(tensor.shape)}", flush=True)


def test_qwen3_e2e(model_name="Qwen/Qwen3-0.6B", verbose=True):
    """
    End-to-end test: SNN vs HuggingFace pretrained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # 1. Load HuggingFace pretrained model
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"1. Loading HuggingFace model: {model_name}")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    hf_model.eval()
    print(f"   HuggingFace model loaded!")

    hf_config = hf_model.config
    print(f"   vocab_size: {hf_config.vocab_size}")
    print(f"   hidden_size: {hf_config.hidden_size}")
    print(f"   intermediate_size: {hf_config.intermediate_size}")
    print(f"   num_hidden_layers: {hf_config.num_hidden_layers}")
    print(f"   num_attention_heads: {hf_config.num_attention_heads}")
    print(f"   num_key_value_heads: {hf_config.num_key_value_heads}")
    head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)
    print(f"   head_dim: {head_dim}")

    # =========================================================================
    # 2. Create SNN model - 逐层创建并打印时间
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"2. Creating SNN model (layer by layer)")
    print(f"{'='*60}")

    snn_config = SpikeQwen3Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
    )

    # 导入组件
    from atomic_ops import (
        SpikeFP32Embedding, SpikeFP32Linear, SpikeFP32RMSNormFullFP64,
        SpikeFP32Adder, SpikeFP32RoPE
    )
    from models import SpikeQwen3Attention, SpikeQwen3MLP, SpikeQwen3DecoderLayer

    # 直接创建完整模型（避免重复创建临时组件浪费内存）
    print(f"\n   Creating SpikeQwen3ForCausalLM...")
    print(f"   (vocab={snn_config.vocab_size}, hidden={snn_config.hidden_size}, layers={snn_config.num_hidden_layers})")
    t0 = time.time()
    snn_model = SpikeQwen3ForCausalLM(snn_config)
    print(f"   Full model created in {time.time()-t0:.1f}s", flush=True)

    print(f"\n   Moving to {device}...", flush=True)
    t0 = time.time()
    snn_model = snn_model.to(device)
    print(f"   SNN model on {device} in {time.time()-t0:.1f}s", flush=True)

    # 打印内存使用
    if device.type == 'cuda':
        print(f"   GPU 内存: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")

    # =========================================================================
    # 3. Transfer weights - 逐层打印
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"3. Transferring weights from HuggingFace to SNN")
    print(f"{'='*60}")

    with timed(f"Embedding weight [{hf_model.model.embed_tokens.weight.shape}]"):
        snn_model.model.set_embedding_weight(hf_model.model.embed_tokens.weight.data)

    with timed(f"LM head weight [{hf_model.lm_head.weight.shape}]"):
        snn_model.lm_head.set_weight_from_float(hf_model.lm_head.weight.data)

    with timed(f"Final norm weight [{hf_model.model.norm.weight.shape}]"):
        snn_model.model.norm.weight.data = hf_model.model.norm.weight.data.clone()

    for i in range(hf_config.num_hidden_layers):
        layer_start = time.time()
        snn_layer = snn_model.model.layers[i]
        hf_layer = hf_model.model.layers[i]

        snn_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
        snn_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

        snn_layer.self_attn.set_weights_from_float(
            hf_layer.self_attn.q_proj.weight.data,
            hf_layer.self_attn.k_proj.weight.data,
            hf_layer.self_attn.v_proj.weight.data,
            hf_layer.self_attn.o_proj.weight.data,
            hf_layer.self_attn.q_norm.weight.data if hasattr(hf_layer.self_attn, 'q_norm') else None,
            hf_layer.self_attn.k_norm.weight.data if hasattr(hf_layer.self_attn, 'k_norm') else None,
        )

        snn_layer.mlp.set_weights_from_float(
            hf_layer.mlp.gate_proj.weight.data,
            hf_layer.mlp.up_proj.weight.data,
            hf_layer.mlp.down_proj.weight.data,
        )

        print(f"      Layer {i:2d}: {time.time() - layer_start:.3f}s", flush=True)

    print(f"   All {hf_config.num_hidden_layers} layers transferred!")

    if device.type == 'cuda':
        print(f"   GPU 内存 (权重加载后): {torch.cuda.memory_allocated()/1024/1024:.1f} MB")

    # =========================================================================
    # 4. Test with real tokenized input
    # =========================================================================
    prompt = "Hello"
    print(f"\n{'='*60}")
    print(f"4. Testing with input: '{prompt}'")
    print(f"{'='*60}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    seq_len = input_ids.shape[1]

    print(f"   Token IDs: {input_ids.tolist()}")
    print(f"   Sequence length: {seq_len}")

    # =========================================================================
    # 5. HuggingFace forward pass
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"5. HuggingFace forward pass")
    print(f"{'='*60}")
    with torch.no_grad():
        t0 = time.time()
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits
        print(f"   耗时: {time.time()-t0:.3f}s")
    print(f"   Output shape: {list(hf_logits.shape)}")

    # 释放 HuggingFace 模型以节省显存
    print(f"\n   释放 HuggingFace 模型...")
    hf_logits_cpu = hf_logits.cpu().clone()  # 保存结果到 CPU
    del hf_model, hf_outputs, hf_logits
    force_gc()
    if device.type == 'cuda':
        print(f"   GPU 内存 (释放HF后): {torch.cuda.memory_allocated()/1024/1024:.1f} MB")

    # =========================================================================
    # 6. SNN forward pass - 详细日志 + 细粒度显存监控
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"6. SNN forward pass (详细日志 + 显存监控)")
    print(f"{'='*60}")

    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    # 重置模型状态
    print(f"\n   重置模型状态...")
    t0 = time.time()
    snn_model.reset()
    print(f"   reset() 完成: {time.time()-t0:.3f}s")

    # 重置显存追踪基线
    force_gc()
    mem_tracker.reset_baseline()
    print_gpu_mem("GC后基线")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # 使用详细模式的前向传播
    print(f"\n   开始前向传播...")
    print(f"   input_ids: {list(input_ids.shape)}")
    print(f"   attention_mask: {list(attention_mask.shape)}")
    print(f"\n   格式: 操作名: 耗时 [当前显存 Δ变化]")
    print("-" * 60)

    forward_start = time.time()

    with torch.no_grad():
        # Embedding
        with timed("Embedding lookup", show_mem=True):
            hidden_pulse = snn_model.model.embed_tokens(input_ids.squeeze(0))
        print(f"         hidden_pulse: {list(hidden_pulse.shape)}")

        positions = torch.arange(seq_len, device=device)

        # 逐层前向传播
        num_layers = len(snn_model.model.layers)
        for layer_idx, layer in enumerate(snn_model.model.layers):
            print(f"\n   {'='*50}")
            print(f"   [Layer {layer_idx}/{num_layers}]")
            print(f"   {'='*50}")
            layer_start = time.time()
            mem_tracker.record_layer_start(layer_idx)
            mem_before_layer = gpu_mem_mb()

            # Input LayerNorm
            with timed("input_layernorm", show_mem=True):
                normed = layer.input_layernorm(hidden_pulse)
            print(f"         normed: {list(normed.shape)}")

            # Self Attention - 详细分解
            attn = layer.self_attn
            print(f"      [Attention]")

            with timed("  Q projection", show_mem=True):
                Q = attn.q_proj(normed)
            print(f"         Q: {list(Q.shape)}")

            with timed("  K projection", show_mem=True):
                K = attn.k_proj(normed)
            print(f"         K: {list(K.shape)}")

            with timed("  V projection", show_mem=True):
                V = attn.v_proj(normed)
            print(f"         V: {list(V.shape)}")

            # Reshape to heads
            batch_size = 1
            with timed("  reshape to heads", show_mem=True):
                Q = attn._reshape_to_heads(Q, batch_size, seq_len, attn.num_attention_heads)
                K = attn._reshape_to_heads(K, batch_size, seq_len, attn.num_key_value_heads)
                V = attn._reshape_to_heads(V, batch_size, seq_len, attn.num_key_value_heads)
            print(f"         Q (heads): {list(Q.shape)}")
            print(f"         K (heads): {list(K.shape)}")
            print(f"         V (heads): {list(V.shape)}")

            # QK Norm
            with timed("  Q norm", show_mem=True):
                Q = attn._apply_head_norm(Q, attn.q_norm)

            with timed("  K norm", show_mem=True):
                K = attn._apply_head_norm(K, attn.k_norm)

            # RoPE
            with timed("  RoPE (Q)", show_mem=True):
                Q = attn._apply_rope(Q, positions)

            with timed("  RoPE (K)", show_mem=True):
                K = attn._apply_rope(K, positions)

            # GQA repeat
            if attn.num_key_value_groups > 1:
                with timed("  GQA repeat KV", show_mem=True):
                    K = attn._repeat_kv(K)
                    V = attn._repeat_kv(V)
                print(f"         K (after GQA): {list(K.shape)}")
                print(f"         V (after GQA): {list(V.shape)}")

            # QK matmul
            with timed("  QK^T matmul", show_mem=True):
                attn_scores = attn._batched_matmul_qk(Q, K)
            print(f"         attn_scores: {list(attn_scores.shape)}")

            # Scale
            with timed("  scale", show_mem=True):
                attn_scores = attn.scale_mul(attn_scores, attn.scale_pulse)

            # Mask
            with timed("  apply mask", show_mem=True):
                attn_scores = attn._apply_mask(attn_scores, attention_mask)

            # Softmax
            with timed("  softmax", show_mem=True):
                original_shape = attn_scores.shape
                attn_scores_flat = attn_scores.reshape(-1, seq_len, 32)
                attn_weights_flat = attn.softmax(attn_scores_flat)
                attn_weights = attn_weights_flat.reshape(original_shape)
            print(f"         attn_weights: {list(attn_weights.shape)}")

            # AV matmul
            with timed("  Attn @ V matmul", show_mem=True):
                attn_output = attn._batched_matmul_av(attn_weights, V)
            print(f"         attn_output: {list(attn_output.shape)}")

            # Merge heads + output proj
            with timed("  merge heads + o_proj", show_mem=True):
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(batch_size, seq_len, attn.num_attention_heads * attn.head_dim, 32)
                attn_output = attn.o_proj(attn_output)
            print(f"         attn_output (final): {list(attn_output.shape)}")

            # Residual 1
            with timed("  residual add 1", show_mem=True):
                hidden_pulse = layer.residual_add1(hidden_pulse, attn_output)

            # Post LayerNorm
            with timed("  post_attention_layernorm", show_mem=True):
                normed2 = layer.post_attention_layernorm(hidden_pulse)

            # MLP
            print(f"      [MLP]")
            mlp = layer.mlp

            with timed("  gate_proj", show_mem=True):
                gate = mlp.gate_proj(normed2)
            print(f"         gate: {list(gate.shape)}")

            with timed("  up_proj", show_mem=True):
                up = mlp.up_proj(normed2)
            print(f"         up: {list(up.shape)}")

            with timed("  SiLU activation", show_mem=True):
                gate_act = mlp.silu(gate)

            with timed("  gate * up", show_mem=True):
                hidden_states = mlp.mul(gate_act, up)

            with timed("  down_proj", show_mem=True):
                mlp_output = mlp.down_proj(hidden_states)
            print(f"         mlp_output: {list(mlp_output.shape)}")

            # Residual 2
            with timed("  residual add 2", show_mem=True):
                hidden_pulse = layer.residual_add2(hidden_pulse, mlp_output)

            # ============ 层结束统计 ============
            mem_after_layer = gpu_mem_mb()
            layer_mem_growth = mem_after_layer - mem_before_layer

            # 检查神经元状态
            total_n, with_v, v_mem_mb, cleanup_status = check_neuron_cleanup(layer, f"Layer{layer_idx}")

            # 删除本层中间变量
            del Q, K, V, attn_scores, attn_weights, attn_output, attn_scores_flat, attn_weights_flat
            del normed, normed2, gate, up, gate_act, hidden_states, mlp_output

            # GC
            force_gc()
            mem_after_gc = gpu_mem_mb()

            mem_tracker.record_layer_end(layer_idx)

            # 打印层统计
            print(f"\n      --- Layer {layer_idx} 统计 ---")
            print(f"      耗时: {time.time() - layer_start:.3f}s")
            print(f"      显存: {mem_before_layer:.1f}MB → {mem_after_layer:.1f}MB (Δ{layer_mem_growth:+.1f}MB)")
            print(f"      GC后: {mem_after_gc:.1f}MB (释放 {mem_after_layer - mem_after_gc:.1f}MB)")
            print(f"      神经元: {cleanup_status}")

        # Final norm
        print(f"\n   {'='*50}")
        print(f"   [Final Components]")
        print(f"   {'='*50}")

        with timed("final_norm", show_mem=True):
            hidden_pulse = snn_model.model.norm(hidden_pulse)

        # LM head
        with timed("lm_head", show_mem=True):
            logits_pulse = snn_model.lm_head(hidden_pulse)
        print(f"         logits_pulse: {list(logits_pulse.shape)}")

    forward_time = time.time() - forward_start
    print(f"\n   前向传播总计: {forward_time:.3f}s")
    print_gpu_mem("Forward结束")

    # 解码
    with timed("pulse_to_float32", show_mem=True):
        snn_logits = pulse_to_float32(logits_pulse)
    print(f"   Output shape: {list(snn_logits.shape)}")

    # 最终神经元状态检查
    print(f"\n   --- 全模型神经元状态检查 ---")
    total_n, with_v, v_mem_mb, cleanup_status = check_neuron_cleanup(snn_model)
    print(f"   总神经元: {total_n}")
    print(f"   状态: {cleanup_status}")
    if with_v > 0:
        print(f"   ⚠️ 残留显存: {v_mem_mb:.2f}MB")

    # =========================================================================
    # 7. Compare results
    # =========================================================================
    print(f"\n{'='*60}")
    print("7. Results")
    print(f"{'='*60}")

    diff = (snn_logits.cpu() - hf_logits_cpu).abs()
    print(f"Max absolute error:  {diff.max().item():.6e}")
    print(f"Mean absolute error: {diff.mean().item():.6e}")

    ulp_stats = compute_ulp_error_fp32(snn_logits.cpu(), hf_logits_cpu)
    print(f"Max ULP error:       {ulp_stats['max_ulp']}")
    print(f"Mean ULP error:      {ulp_stats['mean_ulp']:.2f}")
    print(f"0-ULP rate:          {ulp_stats['zero_ulp_rate']:.1f}%")

    hf_pred = hf_logits_cpu[0, -1].argmax().item()
    snn_pred = snn_logits[0, -1].argmax().item()
    print(f"\nNext token prediction:")
    print(f"  HuggingFace: {hf_pred} -> '{tokenizer.decode([hf_pred])}'")
    print(f"  SNN:         {snn_pred} -> '{tokenizer.decode([snn_pred])}'")
    print(f"  Match: {hf_pred == snn_pred}")

    # =========================================================================
    # 8. 显存统计摘要
    # =========================================================================
    print_gpu_mem("最终")
    mem_tracker.print_summary()

    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"  模型: {model_name}")
    print(f"  层数: {hf_config.num_hidden_layers}")
    print(f"  序列长度: {seq_len}")
    print(f"  前向耗时: {forward_time:.3f}s")
    print(f"  ULP匹配率: {ulp_stats['zero_ulp_rate']:.1f}%")
    print(f"  预测匹配: {hf_pred == snn_pred}")

    total_n, with_v, v_mem_mb, _ = check_neuron_cleanup(snn_model)
    if with_v == 0:
        print(f"  内存清理: ✓ 所有神经元已清理")
    else:
        print(f"  内存清理: ⚠️ {with_v}/{total_n} 神经元有残留 ({v_mem_mb:.2f}MB)")

    return ulp_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Qwen3 SNN E2E Test')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B')
    args = parser.parse_args()

    test_qwen3_e2e(args.model)
