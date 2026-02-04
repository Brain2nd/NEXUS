#!/usr/bin/env python3
"""
实验 14: Sequential MNIST 真实数据集验证
=========================================
使用完整的 Sequential MNIST (784步) 验证 NEXUS TEMPORAL 模式的时序学习能力。

基于实验 13 的各向异性 SPSA 优化策略。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[import] torch, numpy...", flush=True)
import torch
import torch.nn as nn
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
N_EPOCHS = 20
BATCH_SIZE = 1  # 逐样本处理 (SNN 时序依赖)

# SPSA 配置 (各向异性)
C_W, C_BETA, C_VTH = 0.05, 0.001, 0.1
A_W, A_BETA, A_VTH = 0.01, 0.0001, 0.001
MOMENTUM = 0.9
GRAD_CLIP = 5.0


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
# 参数工具函数
# ============================================================
def _get_weight_floats(model):
    w1_pulse = model.linear1.weight_pulse
    w2_pulse = model.linear2.weight_pulse
    w1_float = pulse_to_float32(w1_pulse.float())
    w2_float = pulse_to_float32(w2_pulse.float())
    return w1_float, w2_float


def _set_weight_floats(model, w1_float, w2_float):
    model.set_weights(w1_float, w2_float)


def _get_lif_params(model):
    betas, vths = [], []
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            betas.append(module._beta.data.flatten())
            vths.append(module._v_threshold.data.flatten())
    return torch.cat(betas), torch.cat(vths)


def _set_lif_params(model, all_beta, all_vth):
    idx_beta, idx_vth = 0, 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            n_beta = module._beta.numel()
            n_vth = module._v_threshold.numel()
            module._beta.data.copy_(all_beta[idx_beta:idx_beta+n_beta].view(module._beta.shape))
            module._v_threshold.data.copy_(all_vth[idx_vth:idx_vth+n_vth].view(module._v_threshold.shape))
            idx_beta += n_beta
            idx_vth += n_vth


def clamp_physics(model):
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            module._beta.data.clamp_(0.01, 0.999)
            module._v_threshold.data.clamp_(0.1, 100.0)


def compute_spectral_radius(model):
    """计算权重矩阵的谱半径 (最大特征值模)

    对于非方阵 W，计算 W @ W.T 的最大特征值再开方
    """
    w1, w2 = _get_weight_floats(model)
    try:
        # 对非方阵，计算 W @ W.T 的特征值，再开方
        eigvals1 = torch.linalg.eigvalsh((w1 @ w1.T).float())  # w1: (32,1) -> (32,32)
        eigvals2 = torch.linalg.eigvalsh((w2 @ w2.T).float())  # w2: (10,32) -> (10,10)
        rho1 = torch.sqrt(eigvals1.max()).item()
        rho2 = torch.sqrt(eigvals2.max()).item()
        return max(rho1, rho2)
    except:
        return float('nan')


# ============================================================
# 评估函数
# ============================================================
def evaluate_model(model, labels, device, max_samples=None, precomputed_pulses=None, phase='train', verbose=True):
    """评估模型在数据集上的表现 (IEEE 754 解码 + Cross-Entropy)

    Args:
        model: 模型
        labels: 标签 tensor
        device: 设备
        max_samples: 最大样本数
        precomputed_pulses: 预编码的脉冲 (n_samples, seq_len, n_features, 32)
        phase: 阶段标识 ('train' 或 'test')
        verbose: 是否打印进度
    """
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    n_samples = len(labels) if max_samples is None else min(len(labels), max_samples)

    # 取子集
    batch_labels = labels[:n_samples].to(device)  # (n_samples,)
    all_pulses = precomputed_pulses[:n_samples]  # (n_samples, 784, 1, 32)
    seq_len = all_pulses.shape[1]

    # 重置模型状态
    model.reset()

    # 逐时间步前向传播 (所有样本并行)
    for t in range(seq_len):
        if verbose and t % 100 == 0:
            print(f"    [{phase}] t={t}/{seq_len}", flush=True)
        x_pulse = all_pulses[:, t, :, :]  # (n_samples, 1, 32)
        out_pulse = model(x_pulse)

    # IEEE 754 解码：脉冲 -> 浮点数
    out_float = pulse_to_float32(out_pulse)  # (n_samples, 10)

    # DEBUG: 检查输出值
    if verbose:
        print(f"    [{phase}] out_pulse: shape={out_pulse.shape}, sum={out_pulse.sum():.2f}", flush=True)
        print(f"    [{phase}] out_float: min={out_float.min():.4f}, max={out_float.max():.4f}, mean={out_float.mean():.4f}", flush=True)
        print(f"    [{phase}] out_float[0]: {out_float[0].tolist()}", flush=True)

    # 预测
    preds = out_float.argmax(dim=-1)  # (n_samples,)
    correct = (preds == batch_labels).sum().item()

    # Cross-Entropy Loss
    loss = torch.nn.functional.cross_entropy(out_float, batch_labels)

    acc = correct / n_samples

    return loss.item(), acc


# ============================================================
# SPSA 优化 (各向异性)
# ============================================================
def spsa_step_anisotropic(model, labels, device, precomputed_pulses,
                          c_W, c_beta, c_vth, a_W, a_beta, a_vth,
                          w1_shape, w2_shape, n_beta, n_vth,
                          momentum_buf=None, mu=0.9, max_samples=50):
    """各向异性 SPSA 优化步"""

    # 获取当前参数
    w1, w2 = _get_weight_floats(model)
    w_flat = torch.cat([w1.flatten(), w2.flatten()])
    all_beta, all_vth = _get_lif_params(model)

    # 生成扰动方向 (Rademacher)
    delta_W = torch.sign(torch.randn_like(w_flat))
    delta_beta = torch.sign(torch.randn_like(all_beta))
    delta_vth = torch.sign(torch.randn_like(all_vth))

    # 正向扰动
    w_plus = w_flat + c_W * delta_W
    beta_plus = all_beta + c_beta * delta_beta
    vth_plus = all_vth + c_vth * delta_vth

    w1_plus = w_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_plus = w_plus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_plus, w2_plus)
    _set_lif_params(model, beta_plus, vth_plus)
    clamp_physics(model)

    loss_plus, acc_plus = evaluate_model(model, labels, device, max_samples=max_samples, precomputed_pulses=precomputed_pulses, phase='train', verbose=False)

    # 负向扰动
    w_minus = w_flat - c_W * delta_W
    beta_minus = all_beta - c_beta * delta_beta
    vth_minus = all_vth - c_vth * delta_vth

    w1_minus = w_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_minus = w_minus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_minus, w2_minus)
    _set_lif_params(model, beta_minus, vth_minus)
    clamp_physics(model)

    loss_minus, acc_minus = evaluate_model(model, labels, device, max_samples=max_samples, precomputed_pulses=precomputed_pulses, phase='train', verbose=False)

    # DEBUG: 打印 loss 差异
    print(f"    [SPSA] loss+={loss_plus:.6f}, loss-={loss_minus:.6f}, diff={loss_plus-loss_minus:.6f}", flush=True)
    print(f"    [SPSA] acc+={acc_plus:.2%}, acc-={acc_minus:.2%}", flush=True)
    print(f"    [SPSA] w_perturb: ±{c_W:.4f}, β_perturb: ±{c_beta:.6f}, vth_perturb: ±{c_vth:.4f}", flush=True)

    # 梯度估计
    grad_W = (loss_plus - loss_minus) / (2 * c_W) * delta_W
    grad_beta = (loss_plus - loss_minus) / (2 * c_beta) * delta_beta
    grad_vth = (loss_plus - loss_minus) / (2 * c_vth) * delta_vth

    print(f"    [SPSA] grad_W: mean={grad_W.abs().mean():.6f}, max={grad_W.abs().max():.6f}", flush=True)

    # 梯度裁剪
    grad_W = torch.clamp(grad_W, -GRAD_CLIP, GRAD_CLIP)
    grad_beta = torch.clamp(grad_beta, -GRAD_CLIP, GRAD_CLIP)
    grad_vth = torch.clamp(grad_vth, -GRAD_CLIP, GRAD_CLIP)

    # 动量更新
    if momentum_buf is None:
        momentum_buf = {
            'W': torch.zeros_like(w_flat),
            'beta': torch.zeros_like(all_beta),
            'vth': torch.zeros_like(all_vth)
        }

    momentum_buf['W'] = mu * momentum_buf['W'] + grad_W
    momentum_buf['beta'] = mu * momentum_buf['beta'] + grad_beta
    momentum_buf['vth'] = mu * momentum_buf['vth'] + grad_vth

    # 参数更新
    new_W = w_flat - a_W * momentum_buf['W']
    new_beta = all_beta - a_beta * momentum_buf['beta']
    new_vth = all_vth - a_vth * momentum_buf['vth']

    w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params(model, new_beta, new_vth)
    clamp_physics(model)

    # 评估更新后的损失
    loss, acc = evaluate_model(model, labels, device, max_samples=max_samples, precomputed_pulses=precomputed_pulses, phase='train', verbose=False)

    return loss, acc, momentum_buf


# ============================================================
# 训练主函数
# ============================================================
def train_sequential_mnist():
    print("\n" + "=" * 70)
    print("实验 14: Sequential MNIST 真实数据集验证")
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

    # 预编码所有输入 (避免训练过程中重复编码)
    print("[encode] Pre-encoding training data...", flush=True)
    train_pulse = float32_to_pulse(train_seq.reshape(-1, 1), device=DEVICE)
    train_pulse = train_pulse.reshape(N_TRAIN_SAMPLES, SEQ_LEN, N_FEATURES, 32)
    print("[encode] Pre-encoding test data...", flush=True)
    test_pulse = float32_to_pulse(test_seq.reshape(-1, 1), device=DEVICE)
    test_pulse = test_pulse.reshape(N_TEST_SAMPLES, SEQ_LEN, N_FEATURES, 32)
    print(f"[encode] Done. Train pulse: {train_pulse.shape}, Test pulse: {test_pulse.shape}", flush=True)

    # 创建模型
    template = SimpleLIFNode(
        v_threshold=1.0,
        beta=0.01  # 高泄漏启动
    )
    model = SimpleSpikeMLP(N_FEATURES, HIDDEN_SIZE, N_CLASSES, neuron_template=template).to(DEVICE)

    # 正交权重初始化
    w1_shape = (HIDDEN_SIZE, N_FEATURES)
    w2_shape = (N_CLASSES, HIDDEN_SIZE)

    torch.manual_seed(42)
    # W1: (32, 1) - 无法正交，用标准初始化
    w1_init = torch.randn(w1_shape) * 0.5
    # W2: (10, 32) - 部分正交
    w2_full = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)
    q, _ = torch.linalg.qr(w2_full)
    w2_init = q[:N_CLASSES, :] * 0.5

    model.set_weights(w1_init, w2_init)

    # 初始化 β 和 V_th
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            module._beta.data.uniform_(0.01, 0.02)
            module._v_threshold.data.normal_(1.0, 0.4).clamp_(0.5, 1.5)

    # 获取参数信息
    all_beta, all_vth = _get_lif_params(model)
    n_beta, n_vth = all_beta.numel(), all_vth.numel()
    w1, w2 = _get_weight_floats(model)

    print(f"[params] W1: mean={w1.mean():.4f}, std={w1.std():.4f}, shape={tuple(w1.shape)}", flush=True)
    print(f"[params] W2: mean={w2.mean():.4f}, std={w2.std():.4f}, shape={tuple(w2.shape)}", flush=True)
    print(f"[params] β: mean={all_beta.mean():.4f}, range=[{all_beta.min():.4f}, {all_beta.max():.4f}], n={n_beta}", flush=True)
    print(f"[params] V_th: mean={all_vth.mean():.4f}, range=[{all_vth.min():.4f}, {all_vth.max():.4f}], n={n_vth}", flush=True)

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
        'test_loss': [],
        'test_acc': [],
    }

    # 训练循环
    momentum_buf = None
    import time

    # 记录上一轮的值用于计算 delta
    prev_loss = loss0
    prev_w1, prev_w2 = _get_weight_floats(model)
    prev_beta, prev_vth = _get_lif_params(model)

    print("\n" + "=" * 70)
    print("Training with Anisotropic SPSA")
    print(f"  c_W={C_W}, c_β={C_BETA}, c_Vth={C_VTH}")
    print(f"  a_W={A_W}, a_β={A_BETA}, a_Vth={A_VTH}")
    print("=" * 70)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        loss, acc, momentum_buf = spsa_step_anisotropic(
            model, train_labels, DEVICE, train_pulse,
            C_W, C_BETA, C_VTH, A_W, A_BETA, A_VTH,
            w1_shape, w2_shape, n_beta, n_vth,
            momentum_buf, MOMENTUM,
            max_samples=min(50, N_TRAIN_SAMPLES)  # 每步用部分样本估计梯度
        )

        elapsed = time.time() - t0

        # 获取当前参数
        w1_cur, w2_cur = _get_weight_floats(model)
        beta_cur, vth_cur = _get_lif_params(model)
        rho = compute_spectral_radius(model)

        # 计算 delta
        delta_loss = loss - prev_loss

        diff_w1 = (w1_cur - prev_w1).abs()
        diff_w2 = (w2_cur - prev_w2).abs()
        diff_beta = (beta_cur - prev_beta).abs()
        diff_vth = (vth_cur - prev_vth).abs()

        # 平均变化
        delta_w1_mean = diff_w1.mean().item()
        delta_w2_mean = diff_w2.mean().item()
        delta_beta_mean = diff_beta.mean().item()
        delta_vth_mean = diff_vth.mean().item()

        # 最大变化及其位置
        w1_max_val, w1_max_idx = diff_w1.flatten().max(dim=0)
        w2_max_val, w2_max_idx = diff_w2.flatten().max(dim=0)
        beta_max_val, beta_max_idx = diff_beta.max(dim=0)
        vth_max_val, vth_max_idx = diff_vth.max(dim=0)

        # 记录
        trajectory['epoch'].append(epoch)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)

        # 打印进度 (含 delta)
        print(f"Ep {epoch:3d}: Loss={loss:.4f} (Δ={delta_loss:+.4f}), Acc={acc:.2%}, ρ={rho:.4f} [{elapsed:.1f}s]", flush=True)
        print(f"         ΔW1: mean={delta_w1_mean:.6f}, max={w1_max_val.item():.6f}@{w1_max_idx.item()}", flush=True)
        print(f"         ΔW2: mean={delta_w2_mean:.6f}, max={w2_max_val.item():.6f}@{w2_max_idx.item()}", flush=True)
        print(f"         Δβ:  mean={delta_beta_mean:.6f}, max={beta_max_val.item():.6f}@{beta_max_idx.item()}", flush=True)
        print(f"         ΔVth: mean={delta_vth_mean:.6f}, max={vth_max_val.item():.6f}@{vth_max_idx.item()}", flush=True)

        # 更新 prev 值
        prev_loss = loss
        prev_w1, prev_w2 = w1_cur.clone(), w2_cur.clone()
        prev_beta, prev_vth = beta_cur.clone(), vth_cur.clone()

        # 每 5 轮评估测试集
        if epoch % 5 == 0:
            test_loss, test_acc = evaluate_model(model, test_labels, DEVICE, precomputed_pulses=test_pulse, phase='test', verbose=False)
            trajectory['test_loss'].append(test_loss)
            trajectory['test_acc'].append(test_acc)
            print(f"  [test] Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 最终测试
    test_loss, test_acc = evaluate_model(model, test_labels, DEVICE, precomputed_pulses=test_pulse, phase='test', verbose=True)
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
                'spsa': {'c_W': C_W, 'c_beta': C_BETA, 'c_vth': C_VTH,
                         'a_W': A_W, 'a_beta': A_BETA, 'a_vth': A_VTH}
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
