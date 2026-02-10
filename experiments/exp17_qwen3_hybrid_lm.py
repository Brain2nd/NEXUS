#!/usr/bin/env python3
"""
实验 17: Qwen3 Embedding + SNN + LM Head 混合语言模型
===============================================================
架构:
  Input Text → Qwen3 Tokenizer
    → Qwen3 Embedding (frozen, from pretrained)
    → [SNN Network - trainable]
    → Qwen3 LM Head (frozen, from pretrained)
    → Output Logits

数据集: tiny_shakespeare (HuggingFace)

核心设计:
1. Qwen3 Embedding 和 LM Head 从预训练模型迁移，冻结不训练
2. 中间 SNN 网络使用 TEMPORAL 模式训练
3. 使用 SPSA 零阶优化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[import] torch, numpy...", flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from copy import deepcopy

print("[import] transformers...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[import] datasets...", flush=True)
from datasets import load_dataset

print("[import] atomic_ops...", flush=True)
from atomic_ops import (
    SpikeFP32Linear_MultiPrecision,
    SimpleLIFNode,
    SpikeMode,
    float32_to_pulse,
    pulse_to_float32,
)
print("[import] done", flush=True)

# ============================================================
# 配置
# ============================================================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"[device] {DEVICE}", flush=True)

# Qwen3 模型配置 (使用最小的 0.6B)
QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

# 数据集配置
DATASET_NAME = "karpathy/tiny_shakespeare"
MAX_SEQ_LEN = None  # None = 使用数据的自然长度（分词后）
BATCH_SIZE = 8

# SNN 配置
SNN_HIDDEN_SIZE = 128  # SNN 隐藏层大小（远小于 Qwen 的 hidden_size）

# 训练配置
N_EPOCHS = 50
N_STEPS_PER_EPOCH = 100  # 每 epoch 的步数
TEST_INTERVAL = 5

# SPSA 配置
C_W = 0.02
A_W = 0.002
C_BETA = 0.02
A_BETA = 0.005
C_VTH = 0.02
A_VTH = 0.005

MOMENTUM = 0.9
GRAD_CLIP = 10.0

ALPHA = 0.602
GAMMA_SPSA = 0.101

# Warmup 配置
WARMUP_EPOCHS = 5  # warmup 阶段的 epoch 数

INIT_BETA = 0.1


# ============================================================
# 数据加载
# ============================================================
def load_tiny_shakespeare(tokenizer, max_seq_len=None, train_ratio=0.9):
    """加载 tiny_shakespeare 数据集

    Args:
        tokenizer: 分词器
        max_seq_len: 最大序列长度。如果为 None，使用合理的默认值 (128)
        train_ratio: 训练集比例

    Returns:
        train_samples, test_samples: 训练和测试样本列表
    """
    print(f"[data] Loading {DATASET_NAME}...", flush=True)

    dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

    # tiny_shakespeare 只有 train split，需要手动分割
    text = dataset['train']['text'][0]  # 整个文本是一个字符串

    print(f"[data] Total text length: {len(text)} chars", flush=True)

    # Tokenize
    print("[data] Tokenizing...", flush=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"[data] Total tokens: {len(tokens)}", flush=True)

    # 如果没有指定 max_seq_len，使用合理的默认值
    if max_seq_len is None:
        max_seq_len = 128  # 对于语言模型任务，128 是合理的序列长度

    print(f"[data] Using sequence length: {max_seq_len}", flush=True)

    # 创建训练样本 (sliding window)
    stride = max_seq_len // 2
    samples = []
    for i in range(0, len(tokens) - max_seq_len - 1, stride):
        input_ids = tokens[i:i + max_seq_len]
        labels = tokens[i + 1:i + max_seq_len + 1]
        samples.append((input_ids, labels))

    print(f"[data] Created {len(samples)} samples (stride={stride})", flush=True)

    # 分割训练集和测试集
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    print(f"[data] Train: {len(train_samples)}, Test: {len(test_samples)}", flush=True)

    return train_samples, test_samples, max_seq_len


def get_batch(samples, batch_size, device):
    """获取随机 batch"""
    indices = np.random.choice(len(samples), batch_size, replace=False)

    input_ids_list = []
    labels_list = []
    for idx in indices:
        input_ids, labels = samples[idx]
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    return input_ids, labels


# ============================================================
# 混合模型定义
# ============================================================
class SNNMiddleLayer(nn.Module):
    """SNN 中间层 - 可训练部分

    输入: [batch, seq_len, hidden_size] (float, 来自 Qwen Embedding)
    输出: [batch, seq_len, hidden_size] (float, 送入 LM Head)

    内部使用 SNN 门电路进行处理。
    由于 Qwen 的 hidden_size 很大 (1024+)，我们使用投影层降维。
    """

    def __init__(self, qwen_hidden_size, snn_hidden_size, neuron_template=None):
        super().__init__()
        self.qwen_hidden_size = qwen_hidden_size
        self.snn_hidden_size = snn_hidden_size

        # 投影层: qwen_hidden -> snn_hidden (使用标准 nn.Linear，因为这是降维)
        # 这些投影层会被训练
        self.proj_down = nn.Linear(qwen_hidden_size, snn_hidden_size, bias=False)
        self.proj_up = nn.Linear(snn_hidden_size, qwen_hidden_size, bias=False)

        # SNN 层
        self.snn_linear1 = SpikeFP32Linear_MultiPrecision(
            snn_hidden_size, snn_hidden_size,
            accum_precision='fp32', neuron_template=neuron_template
        )
        self.snn_linear2 = SpikeFP32Linear_MultiPrecision(
            snn_hidden_size, snn_hidden_size,
            accum_precision='fp32', neuron_template=neuron_template
        )

        # 残差连接缩放
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def set_snn_weights(self, w1, w2):
        self.snn_linear1.set_weight_from_float(w1)
        self.snn_linear2.set_weight_from_float(w2)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, qwen_hidden_size]
        Returns:
            [batch, seq_len, qwen_hidden_size]

        TEMPORAL 模式: 逐时间步处理，让膜电位残余在时间步之间累积
        """
        batch, seq_len, _ = hidden_states.shape

        # 保存残差
        residual = hidden_states

        # 投影到 SNN 维度
        x = self.proj_down(hidden_states)  # [batch, seq_len, snn_hidden]

        # TEMPORAL: 逐时间步处理
        out_list = []
        for t in range(seq_len):
            print(f"      [token] {t+1}/{seq_len}", flush=True)
            x_t = x[:, t, :]  # [batch, snn_hidden]

            # 转换为脉冲
            x_pulse = float32_to_pulse(x_t, device=hidden_states.device)

            # SNN 处理 (膜电位会在时间步之间累积)
            h_pulse = self.snn_linear1(x_pulse)
            out_pulse = self.snn_linear2(h_pulse)

            # 转回 float
            out_t = pulse_to_float32(out_pulse)  # [batch, snn_hidden]
            out_list.append(out_t)

        # 堆叠所有时间步
        out = torch.stack(out_list, dim=1)  # [batch, seq_len, snn_hidden]

        # 投影回 Qwen 维度
        out = self.proj_up(out)  # [batch, seq_len, qwen_hidden]

        # 残差连接
        out = residual + self.residual_scale * out

        return out

    def reset(self):
        self.snn_linear1.reset()
        self.snn_linear2.reset()


class Qwen3HybridLM(nn.Module):
    """Qwen3 + SNN 混合语言模型

    架构:
    - Qwen3 Embedding (frozen)
    - SNN Middle Layer (trainable)
    - Qwen3 LM Head (frozen)
    """

    def __init__(self, model_name, snn_hidden_size, neuron_template=None):
        super().__init__()

        print(f"[model] Loading Qwen3 from {model_name}...", flush=True)

        # 加载 Qwen3 模型
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        # 提取组件
        self.embed_tokens = qwen_model.model.embed_tokens
        self.lm_head = qwen_model.lm_head

        # 获取 hidden_size
        self.hidden_size = qwen_model.config.hidden_size
        self.vocab_size = qwen_model.config.vocab_size

        print(f"[model] Qwen3 hidden_size: {self.hidden_size}", flush=True)
        print(f"[model] Qwen3 vocab_size: {self.vocab_size}", flush=True)

        # 冻结 Embedding 和 LM Head
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        print("[model] Frozen: embed_tokens, lm_head", flush=True)

        # 创建 SNN 中间层
        self.snn_layer = SNNMiddleLayer(
            self.hidden_size, snn_hidden_size, neuron_template=neuron_template
        )

        # 释放 Qwen 模型的其他部分以节省显存
        del qwen_model

        print(f"[model] SNN layer: {self.hidden_size} -> {snn_hidden_size} -> {self.hidden_size}", flush=True)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embedding (frozen)
        hidden_states = self.embed_tokens(input_ids)

        # SNN processing (trainable)
        hidden_states = self.snn_layer(hidden_states)

        # LM Head (frozen)
        logits = self.lm_head(hidden_states)

        return logits

    def reset(self):
        self.snn_layer.reset()


# ============================================================
# 参数管理
# ============================================================
def get_trainable_params(model):
    """获取可训练参数"""
    snn_layer = model.snn_layer

    # 投影层权重
    proj_down_w = snn_layer.proj_down.weight.data.clone()
    proj_up_w = snn_layer.proj_up.weight.data.clone()

    # SNN 权重
    w1_pulse = snn_layer.snn_linear1.weight_pulse
    w2_pulse = snn_layer.snn_linear2.weight_pulse

    if w1_pulse is not None:
        w1_float = pulse_to_float32(w1_pulse.float())
        w2_float = pulse_to_float32(w2_pulse.float())
    else:
        w1_float = None
        w2_float = None

    # 残差缩放
    residual_scale = snn_layer.residual_scale.data.clone()

    return {
        'proj_down': proj_down_w,
        'proj_up': proj_up_w,
        'snn_w1': w1_float,
        'snn_w2': w2_float,
        'residual_scale': residual_scale,
    }


def set_trainable_params(model, params):
    """设置可训练参数"""
    snn_layer = model.snn_layer

    snn_layer.proj_down.weight.data.copy_(params['proj_down'])
    snn_layer.proj_up.weight.data.copy_(params['proj_up'])

    if params['snn_w1'] is not None:
        snn_layer.set_snn_weights(params['snn_w1'], params['snn_w2'])

    snn_layer.residual_scale.data.copy_(params['residual_scale'])


def get_lif_params(model):
    """收集所有 LIF 节点的 β 和 V_th"""
    all_beta = []
    all_vth = []
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            all_beta.append(module._beta.data.clone().flatten())
            all_vth.append(module._v_threshold.data.clone().flatten())
    if all_beta:
        return torch.cat(all_beta), torch.cat(all_vth)
    return torch.tensor([]), torch.tensor([])


def set_lif_beta(model, beta_flat, device):
    """设置 β"""
    idx = 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            n = module._beta.numel()
            module._beta.data.copy_(
                beta_flat[idx:idx+n].view(module._beta.shape).to(device))
            idx += n


def set_lif_vth(model, vth_flat, device):
    """设置 V_th"""
    idx = 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_v_threshold'):
            n = module._v_threshold.numel()
            module._v_threshold.data.copy_(
                vth_flat[idx:idx+n].view(module._v_threshold.shape).to(device))
            idx += n


# ============================================================
# 评估函数
# ============================================================
def compute_perplexity(loss):
    """从交叉熵损失计算 Perplexity

    PPL = exp(CE_loss)
    """
    return np.exp(min(loss, 100))  # 截断防止溢出


def evaluate_batch(model, input_ids, labels, device):
    """评估单个 batch

    Returns:
        loss: 交叉熵损失
        acc: token 准确率
        ppl: Perplexity
    """
    model.eval()
    model.reset()

    with torch.no_grad():
        logits = model(input_ids)

        # 计算 loss (标准 LM loss: 预测下一个 token)
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len] - 对应 input_ids 右移一位
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = F.cross_entropy(logits_flat, labels_flat)

        # 计算准确率
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        acc = correct / total

        # 计算 Perplexity
        ppl = compute_perplexity(loss.item())

    return loss.item(), acc, ppl


def evaluate_dataset(model, samples, device, n_batches=10, verbose=False):
    """评估数据集

    Returns:
        avg_loss: 平均交叉熵损失
        avg_acc: 平均准确率
        avg_ppl: 平均 Perplexity
    """
    total_loss = 0
    total_acc = 0
    total_ppl = 0

    for i in range(n_batches):
        if verbose:
            print(f"  [eval] batch {i+1}/{n_batches}...", flush=True)
        input_ids, labels = get_batch(samples, BATCH_SIZE, device)
        loss, acc, ppl = evaluate_batch(model, input_ids, labels, device)
        total_loss += loss
        total_acc += acc
        total_ppl += ppl

    return total_loss / n_batches, total_acc / n_batches, total_ppl / n_batches


# ============================================================
# SPSA 优化
# ============================================================
def spsa_step(model, train_samples, device,
              c_w, c_beta, c_vth, a_w, a_beta, a_vth,
              momentum_buf=None, mu=0.9):
    """SPSA 优化步骤"""

    # 获取当前参数
    params_orig = get_trainable_params(model)
    beta_orig, vth_orig = get_lif_params(model)

    # 获取 batch
    input_ids, labels = get_batch(train_samples, BATCH_SIZE, device)

    # 投影权重展平
    proj_down_flat = params_orig['proj_down'].flatten()
    proj_up_flat = params_orig['proj_up'].flatten()

    # 生成扰动
    delta_proj_down = torch.sign(torch.randn_like(proj_down_flat))
    delta_proj_up = torch.sign(torch.randn_like(proj_up_flat))
    delta_residual = torch.sign(torch.randn_like(params_orig['residual_scale']))

    if len(beta_orig) > 0:
        beta_orig = beta_orig.to(device)
        vth_orig = vth_orig.to(device)
        delta_beta = torch.sign(torch.randn_like(beta_orig))
        delta_vth = torch.sign(torch.randn_like(vth_orig))
    else:
        delta_beta = None
        delta_vth = None

    # SNN 权重扰动
    if params_orig['snn_w1'] is not None:
        snn_w1_flat = params_orig['snn_w1'].flatten()
        snn_w2_flat = params_orig['snn_w2'].flatten()
        delta_snn_w1 = torch.sign(torch.randn_like(snn_w1_flat))
        delta_snn_w2 = torch.sign(torch.randn_like(snn_w2_flat))
    else:
        delta_snn_w1 = None
        delta_snn_w2 = None

    # ===== 正向扰动 =====
    params_plus = deepcopy(params_orig)
    params_plus['proj_down'] = (proj_down_flat + c_w * delta_proj_down).view(params_orig['proj_down'].shape)
    params_plus['proj_up'] = (proj_up_flat + c_w * delta_proj_up).view(params_orig['proj_up'].shape)
    params_plus['residual_scale'] = params_orig['residual_scale'] + c_w * delta_residual

    if delta_snn_w1 is not None:
        params_plus['snn_w1'] = (snn_w1_flat + c_w * delta_snn_w1).view(params_orig['snn_w1'].shape)
        params_plus['snn_w2'] = (snn_w2_flat + c_w * delta_snn_w2).view(params_orig['snn_w2'].shape)

    set_trainable_params(model, params_plus)

    if delta_beta is not None:
        beta_plus = (beta_orig + c_beta * delta_beta).clamp(0.001, 0.999)
        vth_plus = (vth_orig + c_vth * delta_vth).clamp(0.1, 3.0)
        set_lif_beta(model, beta_plus, device)
        set_lif_vth(model, vth_plus, device)

    loss_plus, _, _ = evaluate_batch(model, input_ids, labels, device)

    # ===== 负向扰动 =====
    params_minus = deepcopy(params_orig)
    params_minus['proj_down'] = (proj_down_flat - c_w * delta_proj_down).view(params_orig['proj_down'].shape)
    params_minus['proj_up'] = (proj_up_flat - c_w * delta_proj_up).view(params_orig['proj_up'].shape)
    params_minus['residual_scale'] = params_orig['residual_scale'] - c_w * delta_residual

    if delta_snn_w1 is not None:
        params_minus['snn_w1'] = (snn_w1_flat - c_w * delta_snn_w1).view(params_orig['snn_w1'].shape)
        params_minus['snn_w2'] = (snn_w2_flat - c_w * delta_snn_w2).view(params_orig['snn_w2'].shape)

    set_trainable_params(model, params_minus)

    if delta_beta is not None:
        beta_minus = (beta_orig - c_beta * delta_beta).clamp(0.001, 0.999)
        vth_minus = (vth_orig - c_vth * delta_vth).clamp(0.1, 3.0)
        set_lif_beta(model, beta_minus, device)
        set_lif_vth(model, vth_minus, device)

    loss_minus, _, _ = evaluate_batch(model, input_ids, labels, device)

    # 计算 diff
    diff = loss_plus - loss_minus

    if abs(diff) < 1e-12 or np.isnan(diff):
        # 恢复原始参数
        set_trainable_params(model, params_orig)
        if delta_beta is not None:
            set_lif_beta(model, beta_orig, device)
            set_lif_vth(model, vth_orig, device)
        loss, acc = evaluate_batch(model, input_ids, labels, device)
        return loss, acc, momentum_buf, diff

    # 梯度估计
    grad_proj_down = diff / (2 * c_w) * (1.0 / delta_proj_down)
    grad_proj_up = diff / (2 * c_w) * (1.0 / delta_proj_up)
    grad_residual = diff / (2 * c_w) * (1.0 / delta_residual)

    if delta_snn_w1 is not None:
        grad_snn_w1 = diff / (2 * c_w) * (1.0 / delta_snn_w1)
        grad_snn_w2 = diff / (2 * c_w) * (1.0 / delta_snn_w2)

    if delta_beta is not None:
        grad_beta = diff / (2 * c_beta) * (1.0 / delta_beta)
        grad_vth = diff / (2 * c_vth) * (1.0 / delta_vth)

    # 梯度裁剪
    grad_proj_down = torch.clamp(grad_proj_down, -GRAD_CLIP, GRAD_CLIP)
    grad_proj_up = torch.clamp(grad_proj_up, -GRAD_CLIP, GRAD_CLIP)
    grad_residual = torch.clamp(grad_residual, -GRAD_CLIP, GRAD_CLIP)

    if delta_snn_w1 is not None:
        grad_snn_w1 = torch.clamp(grad_snn_w1, -GRAD_CLIP, GRAD_CLIP)
        grad_snn_w2 = torch.clamp(grad_snn_w2, -GRAD_CLIP, GRAD_CLIP)

    if delta_beta is not None:
        grad_beta = torch.clamp(grad_beta, -GRAD_CLIP, GRAD_CLIP)
        grad_vth = torch.clamp(grad_vth, -GRAD_CLIP, GRAD_CLIP)

    # 动量更新
    if momentum_buf is None:
        momentum_buf = {
            'proj_down': grad_proj_down.clone(),
            'proj_up': grad_proj_up.clone(),
            'residual': grad_residual.clone(),
        }
        if delta_snn_w1 is not None:
            momentum_buf['snn_w1'] = grad_snn_w1.clone()
            momentum_buf['snn_w2'] = grad_snn_w2.clone()
        if delta_beta is not None:
            momentum_buf['beta'] = grad_beta.clone()
            momentum_buf['vth'] = grad_vth.clone()
    else:
        momentum_buf['proj_down'] = mu * momentum_buf['proj_down'] + (1 - mu) * grad_proj_down
        momentum_buf['proj_up'] = mu * momentum_buf['proj_up'] + (1 - mu) * grad_proj_up
        momentum_buf['residual'] = mu * momentum_buf['residual'] + (1 - mu) * grad_residual

        if delta_snn_w1 is not None:
            momentum_buf['snn_w1'] = mu * momentum_buf['snn_w1'] + (1 - mu) * grad_snn_w1
            momentum_buf['snn_w2'] = mu * momentum_buf['snn_w2'] + (1 - mu) * grad_snn_w2

        if delta_beta is not None:
            momentum_buf['beta'] = mu * momentum_buf['beta'] + (1 - mu) * grad_beta
            momentum_buf['vth'] = mu * momentum_buf['vth'] + (1 - mu) * grad_vth

    # 参数更新
    new_params = deepcopy(params_orig)
    new_proj_down = proj_down_flat - a_w * momentum_buf['proj_down']
    new_proj_up = proj_up_flat - a_w * momentum_buf['proj_up']
    new_residual = params_orig['residual_scale'] - a_w * momentum_buf['residual']

    new_params['proj_down'] = new_proj_down.view(params_orig['proj_down'].shape)
    new_params['proj_up'] = new_proj_up.view(params_orig['proj_up'].shape)
    new_params['residual_scale'] = new_residual

    if delta_snn_w1 is not None:
        new_snn_w1 = snn_w1_flat - a_w * momentum_buf['snn_w1']
        new_snn_w2 = snn_w2_flat - a_w * momentum_buf['snn_w2']
        new_params['snn_w1'] = new_snn_w1.view(params_orig['snn_w1'].shape)
        new_params['snn_w2'] = new_snn_w2.view(params_orig['snn_w2'].shape)

    set_trainable_params(model, new_params)

    if delta_beta is not None:
        new_beta = (beta_orig - a_beta * momentum_buf['beta']).clamp(0.001, 0.999)
        new_vth = (vth_orig - a_vth * momentum_buf['vth']).clamp(0.1, 3.0)
        set_lif_beta(model, new_beta, device)
        set_lif_vth(model, new_vth, device)

    # 评估更新后
    loss, acc = evaluate_batch(model, input_ids, labels, device)

    return loss, acc, momentum_buf, diff


# ============================================================
# 文本生成函数 (带采样策略)
# ============================================================
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """过滤 logits 使用 top-k 和/或 nucleus (top-p) 采样

    Args:
        logits: 形状为 [batch, vocab_size] 的 logits
        top_k: 保留概率最高的 k 个 token (0 表示不使用)
        top_p: 保留累积概率超过 p 的最小 token 集合 (1.0 表示不使用)

    Returns:
        过滤后的 logits，被过滤的位置设为 filter_value
    """
    top_k = min(top_k, logits.size(-1))  # 安全检查

    if top_k > 0:
        # 移除概率低于 top-k 阈值的 token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 右移以保留第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将排序后的索引映射回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def generate_text(model, tokenizer, prompt, device,
                  max_new_tokens=50, temperature=1.0, top_k=0, top_p=1.0):
    """使用指定采样策略生成文本

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 提示文本
        device: 设备
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数 (>1 更随机, <1 更确定)
        top_k: top-k 采样参数 (0 表示不使用)
        top_p: nucleus 采样参数 (1.0 表示不使用)

    Returns:
        生成的文本
    """
    model.eval()
    model.reset()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 前向传播
            logits = model(generated)

            # 取最后一个位置的 logits
            next_token_logits = logits[:, -1, :].clone()

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 应用 top-k 和 top-p 过滤
            if top_k > 0 or top_p < 1.0:
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
            else:
                filtered_logits = next_token_logits

            # 采样或贪婪选择
            if temperature == 1.0 and top_k == 0 and top_p == 1.0:
                # 贪婪解码
                next_token = filtered_logits.argmax(dim=-1, keepdim=True)
            else:
                # 采样
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否生成了 EOS token
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ============================================================
# 检查点保存
# ============================================================
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'exp17_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(model, epoch, loss, acc, trajectory):
    """保存检查点"""
    params = get_trainable_params(model)
    beta, vth = get_lif_params(model)

    ckpt = {
        'epoch': epoch,
        'loss': loss,
        'acc': acc,
        'params': {
            'proj_down_norm': params['proj_down'].norm().item(),
            'proj_up_norm': params['proj_up'].norm().item(),
            'residual_scale': params['residual_scale'].item(),
        },
        'lif': {
            'beta_mean': beta.mean().item() if len(beta) > 0 else 0,
            'vth_mean': vth.mean().item() if len(vth) > 0 else 0,
        },
        'trajectory': trajectory,
    }

    ckpt_path = os.path.join(CKPT_DIR, f'epoch_{epoch:03d}.json')
    with open(ckpt_path, 'w') as f:
        json.dump(ckpt, f, indent=2)
    print(f"  [ckpt] Saved to {ckpt_path}", flush=True)


# ============================================================
# 训练主函数
# ============================================================
def train_hybrid_lm():
    print("\n" + "=" * 70)
    print("实验 17: Qwen3 Embedding + SNN + LM Head 混合语言模型")
    print("=" * 70)

    # 加载 tokenizer
    print(f"[tokenizer] Loading from {QWEN_MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)

    # 加载数据
    train_samples, test_samples, actual_seq_len = load_tiny_shakespeare(tokenizer, MAX_SEQ_LEN)

    # 创建模型
    print("[model] Creating hybrid model...", flush=True)
    template = SimpleLIFNode(beta=INIT_BETA)
    model = Qwen3HybridLM(QWEN_MODEL_NAME, SNN_HIDDEN_SIZE, neuron_template=template)
    model = model.to(DEVICE)

    # 初始化 SNN 权重
    print("[model] Initializing SNN weights...", flush=True)
    snn_w1_shape = (SNN_HIDDEN_SIZE, SNN_HIDDEN_SIZE)
    snn_w2_shape = (SNN_HIDDEN_SIZE, SNN_HIDDEN_SIZE)

    torch.manual_seed(42)
    snn_w1 = torch.randn(snn_w1_shape, device=DEVICE) * 0.1
    snn_w2 = torch.randn(snn_w2_shape, device=DEVICE) * 0.1
    model.snn_layer.set_snn_weights(snn_w1, snn_w2)

    # 统计参数 (正确计算)
    frozen_params = sum(p.numel() for p in model.embed_tokens.parameters()) + \
                    sum(p.numel() for p in model.lm_head.parameters())
    snn_params = sum(p.numel() for p in model.snn_layer.parameters())
    total_params = frozen_params + snn_params

    print(f"\n[params] Total: {total_params:,}", flush=True)
    print(f"[params] Frozen (Qwen3): {frozen_params:,}", flush=True)
    print(f"[params] Trainable (SNN): {snn_params:,}", flush=True)
    print(f"[data] Sequence length: {actual_seq_len}", flush=True)
    print(f"[data] SPSA updates per epoch: {N_STEPS_PER_EPOCH}", flush=True)

    # 初始评估
    print("\n[eval] Initial evaluation...", flush=True)
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    print("[eval] Evaluating train set...", flush=True)
    train_loss, train_acc, train_ppl = evaluate_dataset(model, train_samples, DEVICE, verbose=True)
    print("[eval] Evaluating test set...", flush=True)
    test_loss, test_acc, test_ppl = evaluate_dataset(model, test_samples, DEVICE, verbose=True)

    print(f"[init] Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, PPL={train_ppl:.2f}", flush=True)
    print(f"[init] Test:  Loss={test_loss:.4f}, Acc={test_acc:.4f}, PPL={test_ppl:.2f}", flush=True)

    # 训练记录 (参照 exp16 格式)
    trajectory = {
        'epoch': [0],
        'train_loss': [train_loss],
        'train_acc': [train_acc],
        'train_ppl': [train_ppl],
        'test_loss': [test_loss],
        'test_acc': [test_acc],
        'test_ppl': [test_ppl],
        'spsa_count': [0],  # SPSA 累计次数 (参照 exp16)
    }

    # 训练循环
    momentum_buf = None
    total_spsa_count = 0

    print("\n" + "=" * 70)
    print("Training with SPSA")
    print("=" * 70)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # Warmup + 学习率衰减
        if epoch <= WARMUP_EPOCHS:
            # Linear warmup: 从 0 线性增长到 1
            warmup_scale = epoch / WARMUP_EPOCHS
        else:
            warmup_scale = 1.0

        # 标准 SPSA 衰减 + warmup
        c_w_k = warmup_scale * C_W / (epoch + 1) ** GAMMA_SPSA
        c_beta_k = warmup_scale * C_BETA / (epoch + 1) ** GAMMA_SPSA
        c_vth_k = warmup_scale * C_VTH / (epoch + 1) ** GAMMA_SPSA
        a_w_k = warmup_scale * A_W / (epoch + 1) ** ALPHA
        a_beta_k = warmup_scale * A_BETA / (epoch + 1) ** ALPHA
        a_vth_k = warmup_scale * A_VTH / (epoch + 1) ** ALPHA

        # 多步 SPSA
        epoch_loss = 0
        epoch_acc = 0

        for step in range(N_STEPS_PER_EPOCH):
            loss, acc, momentum_buf, diff = spsa_step(
                model, train_samples, DEVICE,
                c_w_k, c_beta_k, c_vth_k, a_w_k, a_beta_k, a_vth_k,
                momentum_buf, MOMENTUM
            )
            epoch_loss += loss
            epoch_acc += acc
            total_spsa_count += 1

        epoch_loss /= N_STEPS_PER_EPOCH
        epoch_acc /= N_STEPS_PER_EPOCH
        epoch_ppl = compute_perplexity(epoch_loss)

        elapsed = time.time() - t0

        # 记录
        trajectory['epoch'].append(epoch)
        trajectory['train_loss'].append(epoch_loss)
        trajectory['train_acc'].append(epoch_acc)
        trajectory['train_ppl'].append(epoch_ppl)
        trajectory['spsa_count'].append(total_spsa_count)

        # 打印进度 (参照 exp16 格式)
        if epoch <= 10 or epoch % 10 == 0:
            warmup_str = " [warmup]" if epoch <= WARMUP_EPOCHS else ""
            print(f"Ep {epoch:3d}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, PPL={epoch_ppl:.2f}, "
                  f"SPSA本epoch={N_STEPS_PER_EPOCH}, 累计={total_spsa_count} [{elapsed:.1f}s]{warmup_str}", flush=True)

        # 测试集评估 (每 TEST_INTERVAL 个 epoch)
        if epoch % TEST_INTERVAL == 0:
            test_loss, test_acc, test_ppl = evaluate_dataset(model, test_samples, DEVICE)
            trajectory['test_loss'].append(test_loss)
            trajectory['test_acc'].append(test_acc)
            trajectory['test_ppl'].append(test_ppl)
            print(f"  [test] Loss={test_loss:.4f}, Acc={test_acc:.4f}, PPL={test_ppl:.2f}", flush=True)

        # 保存检查点 (每 50 epoch，参照 exp16)
        if epoch % 50 == 0:
            save_checkpoint(model, epoch, epoch_loss, epoch_acc, trajectory)

    # 最终评估
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    train_loss, train_acc, train_ppl = evaluate_dataset(model, train_samples, DEVICE, n_batches=20)
    test_loss, test_acc, test_ppl = evaluate_dataset(model, test_samples, DEVICE, n_batches=20)

    print(f"[final] Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, PPL={train_ppl:.2f}", flush=True)
    print(f"[final] Test:  Loss={test_loss:.4f}, Acc={test_acc:.4f}, PPL={test_ppl:.2f}", flush=True)
    print(f"[final] Total SPSA updates: {total_spsa_count}", flush=True)

    # 保存结果 (参照 exp16 格式)
    result_path = os.path.join(os.path.dirname(__file__), 'exp17_results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'config': {
                'qwen_model': QWEN_MODEL_NAME,
                'dataset': DATASET_NAME,
                'seq_len': actual_seq_len,
                'batch_size': BATCH_SIZE,
                'snn_hidden_size': SNN_HIDDEN_SIZE,
                'n_epochs': N_EPOCHS,
                'n_steps_per_epoch': N_STEPS_PER_EPOCH,
                'warmup_epochs': WARMUP_EPOCHS,
            },
            'total_spsa_updates': total_spsa_count,
            'trajectory': trajectory,
            'final': {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_ppl': train_ppl,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_ppl': test_ppl,
            }
        }, f, indent=2)
    print(f"\n[save] Results saved to {result_path}", flush=True)

    # 生成文本示例
    print("\n" + "=" * 70)
    print("Generation Example")
    print("=" * 70)

    prompt = "ROMEO:"
    print(f"Prompt: {prompt}")

    # 使用不同采样策略生成
    for strategy_name, params in [
        ("Greedy", {"temperature": 1.0, "top_k": 0, "top_p": 1.0}),
        ("Temperature=0.8", {"temperature": 0.8, "top_k": 0, "top_p": 1.0}),
        ("Top-k=50", {"temperature": 1.0, "top_k": 50, "top_p": 1.0}),
        ("Top-p=0.9", {"temperature": 1.0, "top_k": 0, "top_p": 0.9}),
    ]:
        generated_text = generate_text(
            model, tokenizer, prompt, DEVICE,
            max_new_tokens=50, **params
        )
        print(f"\n[{strategy_name}]:")
        print(f"  {generated_text}")

    return trajectory


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    trajectory = train_hybrid_lm()
    print("\n[完成]", flush=True)
