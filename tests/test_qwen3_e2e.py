"""
Qwen3 SNN End-to-End Tests (HuggingFace Baseline)
===================================================

Baseline: HuggingFace Qwen/Qwen3-0.6B pretrained model
Target: 0 ULP error (bit-exact with HuggingFace)

Author: MofNeuroSim Project
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import SpikeQwen3Config, SpikeQwen3ForCausalLM
from atomic_ops import float32_to_pulse, pulse_to_float32


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


def test_qwen3_e2e(model_name="Qwen/Qwen3-0.6B"):
    """
    End-to-end test: SNN vs HuggingFace pretrained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # 1. Load HuggingFace pretrained model
    # =========================================================================
    print(f"\n1. Loading HuggingFace model: {model_name}")
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

    # =========================================================================
    # 2. Create SNN model with identical config
    # =========================================================================
    print(f"\n2. Creating SNN model (identical config)")
    head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)

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
    print(f"   SNN config: {snn_config.num_hidden_layers} layers, hidden_size={snn_config.hidden_size}", flush=True)

    # 逐步创建模型以定位卡顿
    import time
    from models.qwen3_model import SpikeQwen3Model
    from atomic_ops import SpikeFP32Linear

    print(f"   Creating embedding...", flush=True)
    t0 = time.time()
    from atomic_ops import SpikeFP32Embedding
    embed = SpikeFP32Embedding(snn_config.vocab_size, snn_config.hidden_size)
    print(f"   Embedding created in {time.time()-t0:.1f}s", flush=True)

    print(f"   Creating first decoder layer (component by component)...", flush=True)
    from atomic_ops import SpikeFP32RMSNormFullFP64, SpikeFP32Adder
    from models.qwen3_attention import SpikeQwen3Attention
    from models.qwen3_mlp import SpikeQwen3MLP

    print(f"      input_layernorm...", flush=True)
    t0 = time.time()
    ln1 = SpikeFP32RMSNormFullFP64(snn_config.hidden_size, snn_config.rms_norm_eps)
    print(f"      input_layernorm: {time.time()-t0:.1f}s", flush=True)

    print(f"      self_attn...", flush=True)
    t0 = time.time()
    attn = SpikeQwen3Attention(snn_config)
    print(f"      self_attn: {time.time()-t0:.1f}s", flush=True)

    print(f"      post_attention_layernorm...", flush=True)
    t0 = time.time()
    ln2 = SpikeFP32RMSNormFullFP64(snn_config.hidden_size, snn_config.rms_norm_eps)
    print(f"      post_attention_layernorm: {time.time()-t0:.1f}s", flush=True)

    print(f"      mlp...", flush=True)
    t0 = time.time()
    mlp = SpikeQwen3MLP(snn_config.hidden_size, snn_config.intermediate_size)
    print(f"      mlp: {time.time()-t0:.1f}s", flush=True)

    print(f"      residual adders...", flush=True)
    t0 = time.time()
    add1 = SpikeFP32Adder()
    add2 = SpikeFP32Adder()
    print(f"      residual adders: {time.time()-t0:.1f}s", flush=True)

    print(f"   First layer components OK. Now creating full model...", flush=True)

    print(f"   Creating full model...", flush=True)
    t0 = time.time()
    snn_model = SpikeQwen3ForCausalLM(snn_config)
    print(f"   Full model created in {time.time()-t0:.1f}s", flush=True)

    print(f"   Moving to {device}...", flush=True)
    t0 = time.time()
    snn_model = snn_model.to(device)
    print(f"   SNN model on {device} in {time.time()-t0:.1f}s", flush=True)

    # =========================================================================
    # 3. Transfer ALL weights from HuggingFace to SNN
    # =========================================================================
    print(f"\n3. Transferring weights from HuggingFace to SNN")

    print(f"   Embedding [{hf_model.model.embed_tokens.weight.shape}]...")
    snn_model.model.set_embedding_weight(hf_model.model.embed_tokens.weight.data)

    print(f"   LM head [{hf_model.lm_head.weight.shape}]...")
    snn_model.lm_head.set_weight_from_float(hf_model.lm_head.weight.data)

    print(f"   Final norm [{hf_model.model.norm.weight.shape}]...")
    snn_model.model.norm.weight.data = hf_model.model.norm.weight.data.clone()

    for i in range(hf_config.num_hidden_layers):
        print(f"   Layer {i}/{hf_config.num_hidden_layers}...", end='\r')
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

    print(f"   All {hf_config.num_hidden_layers} layers transferred!")

    # =========================================================================
    # 4. Test with real tokenized input
    # =========================================================================
    prompt = "Hello, how are you?"
    print(f"\n4. Testing with input: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    seq_len = input_ids.shape[1]

    print(f"   Token IDs: {input_ids.tolist()}")
    print(f"   Sequence length: {seq_len}")

    # =========================================================================
    # 5. HuggingFace forward pass
    # =========================================================================
    print(f"\n5. Running HuggingFace forward pass...")
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits
    print(f"   Output shape: {hf_logits.shape}")

    # =========================================================================
    # 6. SNN forward pass
    # =========================================================================
    print(f"\n6. Running SNN forward pass...")
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    snn_model.reset()
    snn_logits_pulse = snn_model(input_ids, attention_mask=attention_mask)
    snn_logits = pulse_to_float32(snn_logits_pulse)
    print(f"   Output shape: {snn_logits.shape}")

    # =========================================================================
    # 7. Compare results
    # =========================================================================
    print("\n" + "="*60)
    print("Results")
    print("="*60)

    diff = (snn_logits - hf_logits).abs()
    print(f"Max absolute error:  {diff.max().item():.6e}")
    print(f"Mean absolute error: {diff.mean().item():.6e}")

    ulp_stats = compute_ulp_error_fp32(snn_logits, hf_logits)
    print(f"Max ULP error:       {ulp_stats['max_ulp']}")
    print(f"Mean ULP error:      {ulp_stats['mean_ulp']:.2f}")
    print(f"0-ULP rate:          {ulp_stats['zero_ulp_rate']:.1f}%")

    hf_pred = hf_logits[0, -1].argmax().item()
    snn_pred = snn_logits[0, -1].argmax().item()
    print(f"\nNext token prediction:")
    print(f"  HuggingFace: {hf_pred} -> '{tokenizer.decode([hf_pred])}'")
    print(f"  SNN:         {snn_pred} -> '{tokenizer.decode([snn_pred])}'")
    print(f"  Match: {hf_pred == snn_pred}")

    return ulp_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Qwen3 SNN E2E Test')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B')
    args = parser.parse_args()

    test_qwen3_e2e(args.model)
