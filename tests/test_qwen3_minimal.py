"""
Qwen3 SNN 极小模型快速验证测试
==============================

使用 HuggingFace 官方 Qwen2 模型作为参考，验证 SNN 实现的正确性。
使用极小配置快速验证前向传播是否可以跑通。

配置:
- vocab_size: 1000
- hidden_size: 64
- intermediate_size: 128
- num_layers: 2
- num_attention_heads: 4
- num_key_value_heads: 2
- head_dim: 16
- seq_len: 4

运行方式:
    conda activate SNN
    python tests/test_qwen3_minimal.py
"""
import torch
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen3Config, Qwen3ForCausalLM
from atomic_ops import float32_to_pulse, pulse_to_float32


# ==============================================================================
# 显存追踪器
# ==============================================================================
class MemoryTracker:
    """追踪显存变化，检测累积"""
    def __init__(self):
        self.history = []
        self.baseline = 0

    def reset_baseline(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.baseline = torch.cuda.memory_allocated() / 1024 / 1024
        self.history = []

    def record(self, label):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024 / 1024
            delta = alloc - self.baseline
            self.history.append((label, alloc, delta))
            return alloc, delta
        return 0, 0

    def print_summary(self):
        if not self.history:
            return
        print("\n" + "="*70)
        print("显存累积分析摘要")
        print("="*70)
        print(f"{'操作':<30} {'当前(MB)':<12} {'相对基线(MB)':<15}")
        print("-"*70)

        # 找出增长最大的操作
        max_delta = 0
        max_op = ""
        for label, alloc, delta in self.history:
            if delta > max_delta:
                max_delta = delta
                max_op = label

        for label, alloc, delta in self.history:
            marker = " <<<" if delta == max_delta and delta > 1 else ""
            print(f"{label:<30} {alloc:<12.2f} {delta:+<15.2f}{marker}")

        print("-"*70)
        if max_delta > 1:
            print(f"最大增长: {max_op} (+{max_delta:.2f}MB)")

        # 检查是否有持续累积
        if len(self.history) > 5:
            recent = [h[2] for h in self.history[-5:]]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                print("警告: 检测到持续显存累积!")


mem_tracker = MemoryTracker()


def gpu_mem_str():
    """返回 GPU 显存使用情况字符串"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        return f"[GPU: {allocated:.1f}MB alloc, {reserved:.1f}MB reserved, {max_allocated:.1f}MB peak]"
    return ""


def print_gpu_mem(label=""):
    """打印 GPU 显存状态"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   {label} GPU 显存: {allocated:.1f}MB (reserved: {reserved:.1f}MB, peak: {max_allocated:.1f}MB)")


def force_gc():
    """强制垃圾回收并清空 CUDA 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def timed(name, show_mem=False, track=True):
    """计时上下文管理器 - 在操作开始和结束时都打印，可选显示显存"""
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
                if self.show_mem:
                    torch.cuda.reset_peak_memory_stats()
            print(f"      → {self.name}...", end='', flush=True)
            self.t0 = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                delta = mem_after - self.mem_before

                if self.track:
                    mem_tracker.record(self.name)

                if self.show_mem:
                    # 显示 delta 和累积
                    delta_str = f"Δ{delta:+.1f}MB" if abs(delta) > 0.1 else ""
                    mem_str = f"[{mem_after:.1f}MB {delta_str}]"
                    print(f" done ({elapsed:.3f}s) {mem_str}", flush=True)
                else:
                    print(f" done ({elapsed:.3f}s)", flush=True)
            else:
                print(f" done ({elapsed:.3f}s)", flush=True)
    return Timer(name, show_mem, track)


def shape_str(t):
    """返回张量形状字符串"""
    if hasattr(t, 'shape'):
        return str(list(t.shape))
    return str(t)


# ==============================================================================
# 主测试
# ==============================================================================

def test_minimal():
    """极小模型快速验证测试 - 使用 HuggingFace 官方模型作为参考"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 极小配置 (与 Qwen2Config 兼容)
    vocab_size = 1000  # 最小词表
    hidden_size = 64
    intermediate_size = 128
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads  # 16
    rms_norm_eps = 1e-6
    rope_theta = 10000.0
    seq_len = 4
    batch_size = 1

    print(f"\n{'='*60}")
    print("极小模型配置")
    print(f"{'='*60}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  seq_len: {seq_len}")

    # =========================================================================
    # 1. 创建 HuggingFace Qwen2 参考模型
    # =========================================================================
    print(f"\n{'='*60}")
    print("1. 创建 HuggingFace Qwen2 参考模型")
    print(f"{'='*60}")

    hf_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,  # 显式设置，否则默认是 128
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        tie_word_embeddings=False,
    )

    with timed("Qwen3ForCausalLM", show_mem=True):
        hf_model = Qwen3ForCausalLM(hf_config).to(device)
    hf_model.eval()
    print(f"   HuggingFace 模型参数量: {sum(p.numel() for p in hf_model.parameters()):,}")
    print_gpu_mem("HF模型加载后")

    # =========================================================================
    # 2. 创建 SNN 模型
    # =========================================================================
    print(f"\n{'='*60}")
    print("2. 创建 SNN 模型")
    print(f"{'='*60}")

    # 使用新的 reference 实现（单文件 SNN 化版本）
    from models.reference.modeling_qwen3 import SpikeQwen3Config, SpikeQwen3ForCausalLM

    snn_config = SpikeQwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
    )

    with timed("SpikeQwen3ForCausalLM", show_mem=True):
        snn_model = SpikeQwen3ForCausalLM(snn_config)

    with timed(f"Move to {device}", show_mem=True):
        snn_model = snn_model.to(device)

    print_gpu_mem("SNN模型加载后")

    # =========================================================================
    # 3. 复制权重 (HuggingFace -> SNN)
    # =========================================================================
    print(f"\n{'='*60}")
    print("3. 复制权重 (HuggingFace -> SNN)")
    print(f"{'='*60}")

    with timed("Embedding"):
        snn_model.model.set_embedding_weight(hf_model.model.embed_tokens.weight.data)

    with timed("LM head"):
        snn_model.lm_head.set_weight_from_float(hf_model.lm_head.weight.data)

    with timed("Final norm"):
        snn_model.model.norm.weight.data = hf_model.model.norm.weight.data.clone()

    for i in range(num_layers):
        with timed(f"Layer {i}"):
            snn_layer = snn_model.model.layers[i]
            hf_layer = hf_model.model.layers[i]

            # Norm weights
            snn_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
            snn_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

            # Attention weights
            snn_layer.self_attn.set_weights_from_float(
                hf_layer.self_attn.q_proj.weight.data,
                hf_layer.self_attn.k_proj.weight.data,
                hf_layer.self_attn.v_proj.weight.data,
                hf_layer.self_attn.o_proj.weight.data,
                hf_layer.self_attn.q_norm.weight.data,
                hf_layer.self_attn.k_norm.weight.data,
            )

            # MLP weights
            snn_layer.mlp.set_weights_from_float(
                hf_layer.mlp.gate_proj.weight.data,
                hf_layer.mlp.up_proj.weight.data,
                hf_layer.mlp.down_proj.weight.data,
            )

    print("   权重复制完成!")
    print_gpu_mem("权重复制后")

    # =========================================================================
    # 4. HuggingFace 前向传播
    # =========================================================================
    print(f"\n{'='*60}")
    print("4. HuggingFace 前向传播")
    print(f"{'='*60}")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    print(f"   input_ids: {shape_str(input_ids)}")

    with torch.no_grad():
        with timed("HuggingFace forward"):
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits
    print(f"   hf_logits: {shape_str(hf_logits)}")

    # =========================================================================
    # 5. SNN 前向传播
    # =========================================================================
    print(f"\n{'='*60}")
    print("5. SNN 前向传播")
    print(f"{'='*60}")

    print_gpu_mem("Forward前")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    print("   执行 snn_model.reset()...", end='', flush=True)
    t0 = time.time()
    snn_model.reset()
    print(f" done ({time.time() - t0:.3f}s)", flush=True)
    print_gpu_mem("Reset后")

    attention_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=1
    ).bool()
    positions = torch.arange(seq_len, device=device)

    # =========================================================================
    # 逐层逐操作详细统计
    # =========================================================================
    print(f"\n{'='*60}")
    print("5.1 逐层逐操作详细统计")
    print("="*60)
    print("    格式: [当前显存 Δ变化]")
    print("="*60)

    # 重置显存追踪基线
    force_gc()
    mem_tracker.reset_baseline()
    print_gpu_mem("GC后基线")

    with torch.no_grad():
        # Embedding
        with timed("Embedding", show_mem=True):
            hidden_states = snn_model.model.embed_tokens(input_ids)
        print(f"      shape: {list(hidden_states.shape)}")

        # 逐层处理
        for layer_idx, layer in enumerate(snn_model.model.layers):
            print(f"\n   --- Layer {layer_idx} ---")

            # Input LayerNorm
            with timed(f"L{layer_idx} input_layernorm", show_mem=True):
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

            # === Attention Block ===
            print(f"      [Attention]")
            attn = layer.self_attn
            bs, sl = hidden_states.shape[0], hidden_states.shape[1]

            # Q/K/V Projections (无手动 reset，测试自动清理)
            with timed(f"L{layer_idx} Q_proj", show_mem=True):
                Q = attn.q_proj(hidden_states)
            with timed(f"L{layer_idx} K_proj", show_mem=True):
                K = attn.k_proj(hidden_states)
            with timed(f"L{layer_idx} V_proj", show_mem=True):
                V = attn.v_proj(hidden_states)

            # Reshape
            with timed(f"L{layer_idx} reshape_heads", show_mem=True):
                Q = attn._reshape_to_heads(Q, bs, sl, attn.num_attention_heads)
                K = attn._reshape_to_heads(K, bs, sl, attn.num_key_value_heads)
                V = attn._reshape_to_heads(V, bs, sl, attn.num_key_value_heads)

            # QK Norm
            with timed(f"L{layer_idx} Q_norm", show_mem=True):
                Q = attn._apply_head_norm(Q, attn.q_norm)
            with timed(f"L{layer_idx} K_norm", show_mem=True):
                K = attn._apply_head_norm(K, attn.k_norm)

            # RoPE
            with timed(f"L{layer_idx} RoPE_Q", show_mem=True):
                Q = attn._apply_rope(Q, positions)
            with timed(f"L{layer_idx} RoPE_K", show_mem=True):
                K = attn._apply_rope(K, positions)

            # GQA repeat
            if attn.num_key_value_groups > 1:
                with timed(f"L{layer_idx} GQA_repeat", show_mem=True):
                    K = attn._repeat_kv(K)
                    V = attn._repeat_kv(V)

            # QK matmul
            with timed(f"L{layer_idx} QK_matmul", show_mem=True):
                attn_scores = attn._batched_matmul_qk(Q, K)

            # Scale
            with timed(f"L{layer_idx} scale", show_mem=True):
                attn_scores = attn.scale_mul(attn_scores, attn.scale_pulse)

            # Mask
            with timed(f"L{layer_idx} mask", show_mem=True):
                attn_scores = attn._apply_mask(attn_scores, attention_mask)

            # Softmax
            with timed(f"L{layer_idx} softmax", show_mem=True):
                orig_shape = attn_scores.shape
                attn_flat = attn_scores.reshape(-1, sl, 32)
                attn_weights = attn.softmax(attn_flat).reshape(orig_shape)

            # AV matmul
            with timed(f"L{layer_idx} AV_matmul", show_mem=True):
                attn_out = attn._batched_matmul_av(attn_weights, V)

            # Merge heads & O_proj
            with timed(f"L{layer_idx} O_proj", show_mem=True):
                attn_out = attn_out.transpose(1, 2).reshape(bs, sl, -1, 32)
                attn_out = attn.o_proj(attn_out)

            # Residual add (attention)
            with timed(f"L{layer_idx} residual_attn", show_mem=True):
                hidden_states = layer.residual_add_attn(residual, attn_out)

            # === MLP Block ===
            print(f"      [MLP]")

            # Post-attention LayerNorm
            with timed(f"L{layer_idx} post_attn_ln", show_mem=True):
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)

            # MLP: gate_proj + SiLU (无手动 reset)
            with timed(f"L{layer_idx} gate_proj", show_mem=True):
                gate = layer.mlp.gate_proj(hidden_states)
            with timed(f"L{layer_idx} silu", show_mem=True):
                gate = layer.mlp.act_fn(gate)

            # MLP: up_proj
            with timed(f"L{layer_idx} up_proj", show_mem=True):
                up = layer.mlp.up_proj(hidden_states)

            # MLP: gate * up
            with timed(f"L{layer_idx} gate*up", show_mem=True):
                hidden_mlp = layer.mlp.mul(gate, up)

            # MLP: down_proj
            with timed(f"L{layer_idx} down_proj", show_mem=True):
                mlp_out = layer.mlp.down_proj(hidden_mlp)

            # Residual add (MLP)
            with timed(f"L{layer_idx} residual_mlp", show_mem=True):
                hidden_states = layer.residual_add_mlp(residual, mlp_out)

        # Final Norm
        with timed("final_norm", show_mem=True):
            hidden_states = snn_model.model.norm(hidden_states)

        # LM Head
        with timed("lm_head", show_mem=True):
            snn_logits_pulse = snn_model.lm_head(hidden_states)

    print_gpu_mem("Forward后")

    # 转换回 float
    snn_logits = pulse_to_float32(snn_logits_pulse)
    print(f"   snn_logits: {shape_str(snn_logits)}")
    print_gpu_mem("Decode后")

    # =========================================================================
    # 6. 比较结果
    # =========================================================================
    print(f"\n{'='*60}")
    print("6. 结果比较")
    print(f"{'='*60}")

    diff = (snn_logits - hf_logits).abs()
    print(f"   Max abs error:  {diff.max().item():.6e}")
    print(f"   Mean abs error: {diff.mean().item():.6e}")

    # ULP 分析
    snn_bits = snn_logits.view(torch.int32)
    hf_bits = hf_logits.view(torch.int32)
    ulp_diff = (snn_bits - hf_bits).abs()
    print(f"   Max ULP error:  {ulp_diff.max().item()}")
    print(f"   Mean ULP error: {ulp_diff.float().mean().item():.2f}")
    print(f"   0-ULP rate:     {(ulp_diff == 0).float().mean().item() * 100:.1f}%")

    # 预测比较
    hf_pred = hf_logits[0, -1].argmax().item()
    snn_pred = snn_logits[0, -1].argmax().item()
    print(f"\n   预测 token:")
    print(f"     HuggingFace: {hf_pred}")
    print(f"     SNN:         {snn_pred}")
    print(f"     Match:       {hf_pred == snn_pred}")

    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")

    # 打印显存累积分析
    mem_tracker.print_summary()

    return hf_pred == snn_pred


if __name__ == "__main__":
    success = test_minimal()
    sys.exit(0 if success else 1)
