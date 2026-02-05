#!/usr/bin/env python3
"""
实验 14: Sequential MNIST 真实数据集验证 (Logit 重参数化版本)
===============================================================
使用完整的 Sequential MNIST (784步) 验证 NEXUS TEMPORAL 模式的时序学习能力。

核心改进：Logit 重参数化
- β = sigmoid(w_beta)，在 Logit 空间优化
- V_th = softplus(w_vth)，保证正定
- 允许 SPSA 在无约束空间大幅度探索，跨越"死区"
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
from copy import deepcopy

print("[import] torchvision...", flush=True)
from torchvision import datasets, transforms

print("[import] matplotlib...", flush=True)
import matplotlib.pyplot as plt

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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] {DEVICE}", flush=True)

# Sequential MNIST 配置
SEQ_LEN = 784  # 28x28 展平
N_FEATURES = 1  # 每步1个像素
N_CLASSES = 10  # 0-9数字
HIDDEN_SIZE = 32  # 隐藏层大小

# 训练配置
N_TRAIN_SAMPLES = 100  # 先用小规模验证 (可调大)
N_TEST_SAMPLES = 20
N_EPOCHS = 50  # 增加 epoch 数以观察隧穿现象
BATCH_SIZE = 1  # 逐样本处理 (SNN 时序依赖)

# ============================================================
# SPSA 配置 (Logit 重参数化)
# ============================================================
# 权重扰动 (线性空间)
C_W = 0.05
A_W = 0.01

# β 生成元扰动 (Logit 空间) - 核心改进！
# w_beta 初始化为 -4.0 → β ≈ 0.018
# 扰动 ±3.0 后 w_beta ∈ [-7, -1] → β ∈ [0.001, 0.27]
# 稀疏大幅变异：10% 概率扰动 ±8.0 → β 可达 0.98
C_BETA_LOGIT = 3.0      # 常规扰动幅度
C_BETA_LOGIT_LARGE = 8.0  # 大幅变异扰动
LARGE_PERTURB_PROB = 0.1   # 大幅变异概率
A_BETA_LOGIT = 1.0      # Logit 空间学习率

# V_th 生成元扰动 (Softplus 空间)
C_VTH_LOGIT = 0.5
A_VTH_LOGIT = 0.1

MOMENTUM = 0.9
GRAD_CLIP = 10.0

# SPSA 衰减参数
ALPHA = 0.602       # 学习率衰减指数
GAMMA_SPSA = 0.101  # 扰动幅度衰减指数


# ============================================================
# 模型定义
# ============================================================
class SimpleSpikeMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, neuron_template=None):
        super().__init__()
        self.linear1 = SpikeFP32Linear_MultiPrecision(
            in_features, hidden_features, accum_precision='fp32', neuron_template=neuron_template)
        self.linear2 = SpikeFP32Linear_MultiPrecision(
            hidden_features, out_features, accum_precision='fp32', neuron_template=neuron_template)

    def set_weights(self, w1, w2):
        self.linear1.set_weight_from_float(w1)
        self.linear2.set_weight_from_float(w2)

    def forward(self, x_pulse):
        return self.linear2(self.linear1(x_pulse))

    def reset(self):
        self.linear1.reset()
        self.linear2.reset()


# ============================================================
# 数据加载
# ============================================================
def load_sequential_mnist(n_train=1000, n_test=200):
    """加载 Sequential MNIST 数据集"""
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 子采样
    train_indices = torch.randperm(len(train_dataset))[:n_train]
    test_indices = torch.randperm(len(test_dataset))[:n_test]

    # 转换为序列格式: (n_samples, seq_len, n_features)
    train_sequences = []
    train_labels = []
    for idx in train_indices:
        img, label = train_dataset[idx]
        seq = img.view(-1, 1)  # (784, 1)
        train_sequences.append(seq)
        train_labels.append(label)

    test_sequences = []
    test_labels = []
    for idx in test_indices:
        img, label = test_dataset[idx]
        seq = img.view(-1, 1)  # (784, 1)
        test_sequences.append(seq)
        test_labels.append(label)

    return (torch.stack(train_sequences), torch.tensor(train_labels),
            torch.stack(test_sequences), torch.tensor(test_labels))


# ============================================================
# 重参数化工具函数
# ============================================================
def logit(x):
    """sigmoid 的逆函数: logit(x) = log(x / (1-x))"""
    x = torch.clamp(x, 1e-7, 1 - 1e-7)  # 数值稳定
    return torch.log(x / (1 - x))


def softplus_inv(x):
    """softplus 的逆函数: softplus_inv(x) = log(exp(x) - 1)"""
    x = torch.clamp(x, 1e-7, 100)  # 数值稳定
    return torch.log(torch.exp(x) - 1 + 1e-7)


def _get_weight_floats(model):
    w1_pulse = model.linear1.weight_pulse
    w2_pulse = model.linear2.weight_pulse
    w1_float = pulse_to_float32(w1_pulse.float())
    w2_float = pulse_to_float32(w2_pulse.float())
    return w1_float, w2_float


def _set_weight_floats(model, w1_float, w2_float):
    model.set_weights(w1_float, w2_float)


def _get_lif_params_raw(model):
    """获取原始物理参数 β 和 V_th"""
    betas, vths = [], []
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            betas.append(module._beta.data.flatten())
            vths.append(module._v_threshold.data.flatten())
    return torch.cat(betas), torch.cat(vths)


def _get_lif_params_logit(model):
    """获取 Logit 空间的生成元参数 w_beta 和 w_vth"""
    beta, vth = _get_lif_params_raw(model)
    w_beta = logit(beta)      # β → w_beta
    w_vth = softplus_inv(vth)  # V_th → w_vth
    return w_beta, w_vth


def _set_lif_params_from_logit(model, w_beta, w_vth):
    """从 Logit 空间参数设置物理参数"""
    # 映射回物理空间
    beta = torch.sigmoid(w_beta)           # w_beta → β ∈ (0, 1)
    vth = F.softplus(w_vth) + 0.01         # w_vth → V_th > 0 (加小量避免零阈值)

    idx_beta, idx_vth = 0, 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            n_beta = module._beta.numel()
            n_vth = module._v_threshold.numel()
            module._beta.data.copy_(beta[idx_beta:idx_beta+n_beta].view(module._beta.shape))
            module._v_threshold.data.copy_(vth[idx_vth:idx_vth+n_vth].view(module._v_threshold.shape))
            idx_beta += n_beta
            idx_vth += n_vth


def compute_spectral_radius(model):
    """计算权重矩阵的谱半径"""
    w1, w2 = _get_weight_floats(model)
    try:
        eigvals1 = torch.linalg.eigvalsh((w1 @ w1.T).float())
        eigvals2 = torch.linalg.eigvalsh((w2 @ w2.T).float())
        rho1 = torch.sqrt(eigvals1.max()).item()
        rho2 = torch.sqrt(eigvals2.max()).item()
        return max(rho1, rho2)
    except:
        return float('nan')


# ============================================================
# 评估函数
# ============================================================
def evaluate_model(model, labels, device, max_samples=None, precomputed_pulses=None, phase='train', verbose=True):
    """评估模型"""
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    n_samples = len(labels) if max_samples is None else min(len(labels), max_samples)
    batch_labels = labels[:n_samples].to(device)
    all_pulses = precomputed_pulses[:n_samples]
    seq_len = all_pulses.shape[1]

    model.reset()

    for t in range(seq_len):
        if verbose and t % 100 == 0:
            print(f"    [{phase}] t={t}/{seq_len}", flush=True)
        x_pulse = all_pulses[:, t, :, :]
        out_pulse = model(x_pulse)

    out_float = pulse_to_float32(out_pulse)

    if verbose:
        print(f"    [{phase}] out_pulse: shape={out_pulse.shape}, sum={out_pulse.sum():.2f}", flush=True)
        print(f"    [{phase}] out_float: min={out_float.min():.4f}, max={out_float.max():.4f}, mean={out_float.mean():.4f}", flush=True)
        print(f"    [{phase}] out_float[0]: {out_float[0].tolist()}", flush=True)

    preds = out_float.argmax(dim=-1)
    correct = (preds == batch_labels).sum().item()

    # NaN 防护
    if torch.isnan(out_float).any() or torch.isinf(out_float).any():
        loss = 1e6  # 大惩罚但不是 inf
    else:
        loss = torch.nn.functional.cross_entropy(out_float, batch_labels).item()
        if np.isnan(loss) or np.isinf(loss):
            loss = 1e6

    acc = correct / n_samples
    return loss, acc


# ============================================================
# SPSA 优化 (Logit 重参数化)
# ============================================================
def spsa_step_logit_reparameterized(model, labels, device, precomputed_pulses,
                                     c_W, c_beta_logit, c_vth_logit,
                                     a_W, a_beta_logit, a_vth_logit,
                                     w1_shape, w2_shape,
                                     momentum_buf=None, mu=0.9, max_samples=50,
                                     epoch=1):
    """Logit 重参数化 SPSA 优化步

    核心思想：在 Logit 空间做扰动，实现"量子隧穿"
    """
    # 获取当前参数
    w1, w2 = _get_weight_floats(model)
    w_flat = torch.cat([w1.flatten(), w2.flatten()])
    w_beta, w_vth = _get_lif_params_logit(model)

    # 保存原始参数
    w_flat_orig = w_flat.clone()
    w_beta_orig = w_beta.clone()
    w_vth_orig = w_vth.clone()

    # 生成扰动方向 (Rademacher)
    delta_W = torch.sign(torch.randn_like(w_flat))
    delta_beta = torch.sign(torch.randn_like(w_beta))
    delta_vth = torch.sign(torch.randn_like(w_vth))

    # 稀疏大幅变异：10% 概率使用大扰动
    if np.random.random() < LARGE_PERTURB_PROB:
        c_beta_actual = C_BETA_LOGIT_LARGE
        print(f"    [SPSA] ★ 大幅变异触发！c_beta_logit={c_beta_actual}", flush=True)
    else:
        c_beta_actual = c_beta_logit

    # 正向扰动 (在 Logit 空间)
    w_plus = w_flat + c_W * delta_W
    w_beta_plus = w_beta + c_beta_actual * delta_beta
    w_vth_plus = w_vth + c_vth_logit * delta_vth

    w1_plus = w_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_plus = w_plus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_plus, w2_plus)
    _set_lif_params_from_logit(model, w_beta_plus, w_vth_plus)

    # 打印扰动后的物理参数范围
    beta_plus_physical = torch.sigmoid(w_beta_plus)
    print(f"    [SPSA] β+ range: [{beta_plus_physical.min():.4f}, {beta_plus_physical.max():.4f}]", flush=True)

    loss_plus, acc_plus = evaluate_model(model, labels, device, max_samples=max_samples,
                                          precomputed_pulses=precomputed_pulses, phase='train', verbose=False)

    # 负向扰动
    w_minus = w_flat - c_W * delta_W
    w_beta_minus = w_beta - c_beta_actual * delta_beta
    w_vth_minus = w_vth - c_vth_logit * delta_vth

    w1_minus = w_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_minus = w_minus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_minus, w2_minus)
    _set_lif_params_from_logit(model, w_beta_minus, w_vth_minus)

    beta_minus_physical = torch.sigmoid(w_beta_minus)
    print(f"    [SPSA] β- range: [{beta_minus_physical.min():.4f}, {beta_minus_physical.max():.4f}]", flush=True)

    loss_minus, acc_minus = evaluate_model(model, labels, device, max_samples=max_samples,
                                            precomputed_pulses=precomputed_pulses, phase='train', verbose=False)

    # 计算 diff
    diff = loss_plus - loss_minus
    print(f"    [SPSA] loss+={loss_plus:.6f}, loss-={loss_minus:.6f}, diff={diff:.6f}", flush=True)
    print(f"    [SPSA] acc+={acc_plus:.2%}, acc-={acc_minus:.2%}", flush=True)

    # diff 阈值检查
    if abs(diff) < 1e-15 or np.isnan(diff):
        print(f"    [SPSA] diff 异常，恢复原参数", flush=True)
        w1_orig = w_flat_orig[:w1_shape[0]*w1_shape[1]].view(w1_shape)
        w2_orig = w_flat_orig[w1_shape[0]*w1_shape[1]:].view(w2_shape)
        _set_weight_floats(model, w1_orig, w2_orig)
        _set_lif_params_from_logit(model, w_beta_orig, w_vth_orig)
        loss, acc = evaluate_model(model, labels, device, max_samples=max_samples,
                                    precomputed_pulses=precomputed_pulses, phase='train', verbose=False)
        return loss, acc, momentum_buf

    # 梯度估计 (在 Logit 空间)
    grad_W = diff / (2 * c_W) * (1.0 / delta_W)
    grad_beta = diff / (2 * c_beta_actual) * (1.0 / delta_beta)
    grad_vth = diff / (2 * c_vth_logit) * (1.0 / delta_vth)

    print(f"    [SPSA] grad_W: mean={grad_W.abs().mean():.6f}, max={grad_W.abs().max():.6f}", flush=True)
    print(f"    [SPSA] grad_beta(logit): mean={grad_beta.abs().mean():.6f}, max={grad_beta.abs().max():.6f}", flush=True)

    # 梯度裁剪
    grad_W = torch.clamp(grad_W, -GRAD_CLIP, GRAD_CLIP)
    grad_beta = torch.clamp(grad_beta, -GRAD_CLIP, GRAD_CLIP)
    grad_vth = torch.clamp(grad_vth, -GRAD_CLIP, GRAD_CLIP)

    # 动量更新
    if momentum_buf is None:
        momentum_buf = {
            'W': grad_W.clone(),
            'beta': grad_beta.clone(),
            'vth': grad_vth.clone()
        }
    else:
        momentum_buf['W'] = mu * momentum_buf['W'] + (1 - mu) * grad_W
        momentum_buf['beta'] = mu * momentum_buf['beta'] + (1 - mu) * grad_beta
        momentum_buf['vth'] = mu * momentum_buf['vth'] + (1 - mu) * grad_vth

    # 参数更新 (在 Logit 空间)
    new_W = w_flat_orig - a_W * momentum_buf['W']
    new_w_beta = w_beta_orig - a_beta_logit * momentum_buf['beta']
    new_w_vth = w_vth_orig - a_vth_logit * momentum_buf['vth']

    w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params_from_logit(model, new_w_beta, new_w_vth)

    # 打印更新后的物理参数
    beta_new_physical = torch.sigmoid(new_w_beta)
    vth_new_physical = F.softplus(new_w_vth) + 0.01
    print(f"    [SPSA] β_new range: [{beta_new_physical.min():.4f}, {beta_new_physical.max():.4f}]", flush=True)
    print(f"    [SPSA] Vth_new range: [{vth_new_physical.min():.4f}, {vth_new_physical.max():.4f}]", flush=True)

    # 评估更新后的损失
    loss, acc = evaluate_model(model, labels, device, max_samples=max_samples,
                                precomputed_pulses=precomputed_pulses, phase='train', verbose=False)

    return loss, acc, momentum_buf


# ============================================================
# 训练主函数
# ============================================================
def train_sequential_mnist():
    print("\n" + "=" * 70)
    print("实验 14: Sequential MNIST (Logit 重参数化 SPSA)")
    print("=" * 70)

    # 加载数据
    print(f"[data] Loading MNIST (train={N_TRAIN_SAMPLES}, test={N_TEST_SAMPLES})...", flush=True)
    train_seq, train_labels, test_seq, test_labels = load_sequential_mnist(
        n_train=N_TRAIN_SAMPLES, n_test=N_TEST_SAMPLES)
    train_seq = train_seq.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_seq = test_seq.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    print(f"[data] Train: {train_seq.shape}, Test: {test_seq.shape}", flush=True)

    # 预编码
    print("[encode] Pre-encoding training data...", flush=True)
    train_pulse = float32_to_pulse(train_seq.reshape(-1, 1), device=DEVICE)
    train_pulse = train_pulse.reshape(N_TRAIN_SAMPLES, SEQ_LEN, N_FEATURES, 32)
    print("[encode] Pre-encoding test data...", flush=True)
    test_pulse = float32_to_pulse(test_seq.reshape(-1, 1), device=DEVICE)
    test_pulse = test_pulse.reshape(N_TEST_SAMPLES, SEQ_LEN, N_FEATURES, 32)
    print(f"[encode] Done. Train pulse: {train_pulse.shape}, Test pulse: {test_pulse.shape}", flush=True)

    # 创建模型 - 使用默认神经元参数，让SPSA发现最优值
    model = SimpleSpikeMLP(N_FEATURES, HIDDEN_SIZE, N_CLASSES).to(DEVICE)

    # 权重初始化
    w1_shape = (HIDDEN_SIZE, N_FEATURES)
    w2_shape = (N_CLASSES, HIDDEN_SIZE)

    torch.manual_seed(42)
    w1_init = torch.randn(w1_shape) * 0.5
    w2_full = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)
    q, _ = torch.linalg.qr(w2_full)
    w2_init = q[:N_CLASSES, :] * 0.5
    model.set_weights(w1_init, w2_init)

    # 高泄漏启动：将 β 初始化为 0.01 (每步衰减 99%)
    # 根据 CHAOS_DYNAMICS_REPORT 结论：高泄漏启动避免饱和，SPSA 可探索到最优 β≈0.90
    HIGH_LEAK_BETA = 0.01
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            module._beta.data.fill_(HIGH_LEAK_BETA)

    # 打印初始参数
    beta_init, vth_init = _get_lif_params_raw(model)
    w_beta_init, w_vth_init = _get_lif_params_logit(model)
    w1, w2 = _get_weight_floats(model)

    print(f"[params] W1: mean={w1.mean():.4f}, std={w1.std():.4f}, shape={tuple(w1.shape)}", flush=True)
    print(f"[params] W2: mean={w2.mean():.4f}, std={w2.std():.4f}, shape={tuple(w2.shape)}", flush=True)
    print(f"[params] β (physical): mean={beta_init.mean():.4f}, range=[{beta_init.min():.4f}, {beta_init.max():.4f}]", flush=True)
    print(f"[params] w_β (logit): mean={w_beta_init.mean():.4f}, range=[{w_beta_init.min():.4f}, {w_beta_init.max():.4f}]", flush=True)
    print(f"[params] V_th (physical): mean={vth_init.mean():.4f}, range=[{vth_init.min():.4f}, {vth_init.max():.4f}]", flush=True)

    # 初始评估
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    loss0, acc0 = evaluate_model(model, train_labels, DEVICE, precomputed_pulses=train_pulse, phase='train', verbose=True)
    rho0 = compute_spectral_radius(model)
    print(f"[init] Loss={loss0:.4f}, Acc={acc0:.2%}, ρ={rho0:.4f}", flush=True)

    # 训练记录
    trajectory = {
        'epoch': [0],
        'loss': [loss0],
        'acc': [acc0],
        'beta_mean': [beta_init.mean().item()],
        'beta_max': [beta_init.max().item()],
        'test_loss': [],
        'test_acc': [],
    }

    # 训练循环
    momentum_buf = None
    import time

    prev_loss = loss0
    prev_beta, _ = _get_lif_params_raw(model)

    print("\n" + "=" * 70)
    print("Training with Logit-Reparameterized SPSA")
    print(f"  c_W={C_W}, c_β_logit={C_BETA_LOGIT} (large={C_BETA_LOGIT_LARGE}, prob={LARGE_PERTURB_PROB})")
    print(f"  a_W={A_W}, a_β_logit={A_BETA_LOGIT}")
    print("=" * 70)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # 学习率衰减
        c_W_k = C_W / (epoch + 1) ** GAMMA_SPSA
        c_beta_k = C_BETA_LOGIT / (epoch + 1) ** GAMMA_SPSA
        c_vth_k = C_VTH_LOGIT / (epoch + 1) ** GAMMA_SPSA
        a_W_k = A_W / (epoch + 1) ** ALPHA
        a_beta_k = A_BETA_LOGIT / (epoch + 1) ** ALPHA
        a_vth_k = A_VTH_LOGIT / (epoch + 1) ** ALPHA

        loss, acc, momentum_buf = spsa_step_logit_reparameterized(
            model, train_labels, DEVICE, train_pulse,
            c_W_k, c_beta_k, c_vth_k, a_W_k, a_beta_k, a_vth_k,
            w1_shape, w2_shape,
            momentum_buf, MOMENTUM,
            max_samples=min(50, N_TRAIN_SAMPLES),
            epoch=epoch
        )

        elapsed = time.time() - t0

        # 获取当前参数
        beta_cur, vth_cur = _get_lif_params_raw(model)
        rho = compute_spectral_radius(model)

        # 计算 delta
        delta_loss = loss - prev_loss
        delta_beta = (beta_cur - prev_beta).abs()

        # 记录
        trajectory['epoch'].append(epoch)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)
        trajectory['beta_mean'].append(beta_cur.mean().item())
        trajectory['beta_max'].append(beta_cur.max().item())

        # 打印进度
        print(f"Ep {epoch:3d}: Loss={loss:.4f} (Δ={delta_loss:+.4f}), Acc={acc:.2%}, ρ={rho:.4f} [{elapsed:.1f}s]", flush=True)
        print(f"         β: mean={beta_cur.mean():.4f}, max={beta_cur.max():.4f}, Δmax={delta_beta.max():.4f}", flush=True)

        # 检测隧穿事件
        if beta_cur.max() > 0.9 and prev_beta.max() < 0.5:
            print(f"         ★★★ 隧穿事件！β 从 {prev_beta.max():.4f} 跃迁到 {beta_cur.max():.4f} ★★★", flush=True)

        prev_loss = loss
        prev_beta = beta_cur.clone()

        # 每 5 轮评估测试集
        if epoch % 5 == 0:
            test_loss, test_acc = evaluate_model(model, test_labels, DEVICE,
                                                  precomputed_pulses=test_pulse, phase='test', verbose=False)
            trajectory['test_loss'].append(test_loss)
            trajectory['test_acc'].append(test_acc)
            print(f"  [test] Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 最终测试
    test_loss, test_acc = evaluate_model(model, test_labels, DEVICE,
                                          precomputed_pulses=test_pulse, phase='test', verbose=True)
    print(f"\n[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 保存结果
    save_path = os.path.join(os.path.dirname(__file__), 'exp14_sequential_mnist_data.json')
    with open(save_path, 'w') as f:
        json.dump({
            'config': {
                'n_train': N_TRAIN_SAMPLES,
                'n_test': N_TEST_SAMPLES,
                'seq_len': SEQ_LEN,
                'hidden_size': HIDDEN_SIZE,
                'n_classes': N_CLASSES,
                'n_epochs': N_EPOCHS,
                'spsa': {
                    'c_W': C_W, 'c_beta_logit': C_BETA_LOGIT,
                    'c_beta_logit_large': C_BETA_LOGIT_LARGE,
                    'large_perturb_prob': LARGE_PERTURB_PROB,
                    'a_W': A_W, 'a_beta_logit': A_BETA_LOGIT
                }
            },
            'trajectory': trajectory,
            'final_test': {'loss': test_loss, 'acc': test_acc}
        }, f, indent=2)
    print(f"[save] Results saved to {save_path}", flush=True)

    return trajectory


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    trajectory = train_sequential_mnist()
    print("\n[完成]", flush=True)
