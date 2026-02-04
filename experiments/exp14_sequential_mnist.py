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
    w1, w2 = _get_weight_floats(model)
    try:
        eigvals = torch.linalg.eigvals(w1.float())
        return torch.max(torch.abs(eigvals)).item()
    except:
        return float('nan')


# ============================================================
# 评估函数
# ============================================================
def evaluate_model(model, sequences, labels, device, max_samples=None):
    """评估模型在数据集上的表现"""
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    n_samples = len(sequences) if max_samples is None else min(len(sequences), max_samples)

    total_loss = 0.0
    correct = 0

    # 10类的 anchor (one-hot 风格)
    anchors = torch.eye(N_CLASSES, device=device)

    for i in range(n_samples):
        model.reset()
        seq = sequences[i].to(device)  # (784, 1)
        label = labels[i].item()

        # 逐时间步前向传播
        for t in range(seq.shape[0]):
            x_t = seq[t:t+1].unsqueeze(0)  # (1, 1, 1) -> (batch=1, features=1)
            x_t = x_t.squeeze(0)  # (1, 1)
            x_pulse = float32_to_pulse(x_t, device=device)
            out_pulse = model(x_pulse)

        # 最后时间步的输出
        out_float = pulse_to_float32(out_pulse).squeeze(0)  # (10,)

        if torch.isnan(out_float).any() or torch.isinf(out_float).any():
            loss_i = 1e6
            pred = -1
        else:
            target = anchors[label]
            loss_i = torch.mean((out_float - target) ** 2).item()
            pred = out_float.argmax().item()

        total_loss += loss_i
        if pred == label:
            correct += 1

    avg_loss = total_loss / n_samples
    acc = correct / n_samples

    return avg_loss, acc


# ============================================================
# SPSA 优化 (各向异性)
# ============================================================
def spsa_step_anisotropic(model, sequences, labels, device,
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

    loss_plus, _ = evaluate_model(model, sequences, labels, device, max_samples=max_samples)

    # 负向扰动
    w_minus = w_flat - c_W * delta_W
    beta_minus = all_beta - c_beta * delta_beta
    vth_minus = all_vth - c_vth * delta_vth

    w1_minus = w_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_minus = w_minus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_minus, w2_minus)
    _set_lif_params(model, beta_minus, vth_minus)
    clamp_physics(model)

    loss_minus, _ = evaluate_model(model, sequences, labels, device, max_samples=max_samples)

    # 梯度估计
    grad_W = (loss_plus - loss_minus) / (2 * c_W) * delta_W
    grad_beta = (loss_plus - loss_minus) / (2 * c_beta) * delta_beta
    grad_vth = (loss_plus - loss_minus) / (2 * c_vth) * delta_vth

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
    loss, acc = evaluate_model(model, sequences, labels, device, max_samples=max_samples)

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
    n_W = w1.numel() + w2.numel()

    print(f"[params] W: {n_W}, β: {n_beta}, V_th: {n_vth}", flush=True)

    # 初始评估
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    loss0, acc0 = evaluate_model(model, train_seq, train_labels, DEVICE)
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

    print("\n" + "=" * 70)
    print("Training with Anisotropic SPSA")
    print(f"  c_W={C_W}, c_β={C_BETA}, c_Vth={C_VTH}")
    print(f"  a_W={A_W}, a_β={A_BETA}, a_Vth={A_VTH}")
    print("=" * 70)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        loss, acc, momentum_buf = spsa_step_anisotropic(
            model, train_seq, train_labels, DEVICE,
            C_W, C_BETA, C_VTH, A_W, A_BETA, A_VTH,
            w1_shape, w2_shape, n_beta, n_vth,
            momentum_buf, MOMENTUM,
            max_samples=min(50, N_TRAIN_SAMPLES)  # 每步用部分样本估计梯度
        )

        elapsed = time.time() - t0

        # 记录
        trajectory['epoch'].append(epoch)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)

        # 打印进度
        rho = compute_spectral_radius(model)
        beta_mean, vth_mean = _get_lif_params(model)
        print(f"Ep {epoch:3d}: Loss={loss:.4f}, Acc={acc:.2%}, ρ={rho:.4f}, "
              f"β={beta_mean.mean():.4f}, V_th={vth_mean.mean():.4f} [{elapsed:.1f}s]", flush=True)

        # 每 5 轮评估测试集
        if epoch % 5 == 0:
            test_loss, test_acc = evaluate_model(model, test_seq, test_labels, DEVICE)
            trajectory['test_loss'].append(test_loss)
            trajectory['test_acc'].append(test_acc)
            print(f"  [test] Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 最终测试
    test_loss, test_acc = evaluate_model(model, test_seq, test_labels, DEVICE)
    print(f"\n[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 保存结果
    save_path = '/Users/tangzhengzheng/Desktop/NEXUS/experiments/exp14_sequential_mnist_data.json'
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
