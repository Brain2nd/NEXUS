#!/usr/bin/env python3
"""
实验 16 Group B: 逐步 SPSA 优化（每个时间步单独优化）
===============================================================
对照实验：验证"逐步信用分配"方式（类似 LLM 的 BPTT）

训练方式：
- 每个时间步 t 预测 t+1
- 每一步计算 loss_t = MSE(pred_t, target_{t+1})
- 每一步都执行 SPSA 更新
- 需要保存/恢复膜电位状态

关键技术：
- 膜电位状态保存/恢复
- 每步更新后使用 teacher forcing 继续前向
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
import time

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

# 数据集配置
DATASET_NAME = "FloodModeling1"
N_FEATURES = 1
N_OUTPUT = 1

# 网络配置
HIDDEN_SIZE = 16

# 训练配置
N_EPOCHS = 100
TEST_INTERVAL = 10

# SPSA 配置 - 为每步优化调小学习率
C_W = 0.01       # 更小的扰动
A_W = 0.0005     # 更小的步长（因为更新频率高 265x）
C_BETA = 0.02
A_BETA = 0.002
C_VTH = 0.02
A_VTH = 0.002

GRAD_CLIP = 5.0
ALPHA = 0.602
GAMMA_SPSA = 0.101
INIT_BETA = 0.01
TARGET_SPECTRAL_RADIUS = 1.0

# 每步更新间隔（不是每步都更新，减少计算量）
STEP_UPDATE_INTERVAL = 10  # 每 10 步更新一次


# ============================================================
# 数据加载
# ============================================================
def load_flood_data():
    from aeon.datasets import load_regression
    print(f"[data] Loading {DATASET_NAME}...", flush=True)
    X_train, y_train = load_regression(DATASET_NAME, split="train")
    X_test, y_test = load_regression(DATASET_NAME, split="test")

    train_seq = X_train.transpose(0, 2, 1)
    test_seq = X_test.transpose(0, 2, 1)

    mean_x = train_seq.mean()
    std_x = train_seq.std()
    train_seq = (train_seq - mean_x) / (std_x + 1e-8)
    test_seq = (test_seq - mean_x) / (std_x + 1e-8)

    mean_y = y_train.mean()
    std_y = y_train.std()
    y_train_norm = (y_train - mean_y) / (std_y + 1e-8)
    y_test_norm = (y_test - mean_y) / (std_y + 1e-8)

    train_seq = torch.tensor(train_seq, dtype=torch.float32)
    test_seq = torch.tensor(test_seq, dtype=torch.float32)
    train_labels = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(-1)
    test_labels = torch.tensor(y_test_norm, dtype=torch.float32).unsqueeze(-1)

    return train_seq, train_labels, test_seq, test_labels


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
# 膜电位状态管理
# ============================================================
def save_membrane_states(model):
    """保存所有 LIF 节点的膜电位状态"""
    states = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            if module.v is not None:
                states[name] = module.v.clone()
            else:
                states[name] = None
    return states


def restore_membrane_states(model, states):
    """恢复所有 LIF 节点的膜电位状态"""
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and name in states:
            if states[name] is not None:
                module.v = states[name].clone()
            else:
                module.v = None


# ============================================================
# 参数管理
# ============================================================
def _get_weight_floats(model):
    w1_pulse = model.linear1.weight_pulse
    w2_pulse = model.linear2.weight_pulse
    return pulse_to_float32(w1_pulse.float()), pulse_to_float32(w2_pulse.float())


def _set_weight_floats(model, w1_float, w2_float):
    model.set_weights(w1_float, w2_float)


def _get_lif_params(model):
    all_beta, all_vth = [], []
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            all_beta.append(module._beta.data.clone().flatten())
            all_vth.append(module._v_threshold.data.clone().flatten())
    return torch.cat(all_beta), torch.cat(all_vth)


def _set_lif_beta(model, beta_flat, device):
    idx = 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            n = module._beta.numel()
            module._beta.data.copy_(beta_flat[idx:idx+n].view(module._beta.shape).to(device))
            idx += n


def _set_lif_vth(model, vth_flat, device):
    idx = 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_v_threshold'):
            n = module._v_threshold.numel()
            module._v_threshold.data.copy_(vth_flat[idx:idx+n].view(module._v_threshold.shape).to(device))
            idx += n


def init_orthogonal_weights(shape, target_rho=1.0, device='cpu'):
    A = torch.randn(shape[0], shape[1], device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    w = U @ Vh
    svs = torch.linalg.svdvals(w)
    current_rho = svs[0].item()
    if current_rho > 0:
        w = w * (target_rho / current_rho)
    return w


# ============================================================
# 检查点保存
# ============================================================
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'exp16_groupB_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(epoch, loss, trajectory):
    ckpt = {'epoch': epoch, 'loss': loss, 'trajectory': trajectory}
    ckpt_path = os.path.join(CKPT_DIR, f'epoch_{epoch:03d}.json')
    with open(ckpt_path, 'w') as f:
        json.dump(ckpt, f, indent=2)
    print(f"    [ckpt] Saved to {ckpt_path}", flush=True)


# ============================================================
# Group B: 单步评估
# ============================================================
def evaluate_single_step(model, x_pulse, target_float, device):
    """
    单步评估：给定输入脉冲，预测下一步值
    返回 loss 和输出
    """
    out_pulse = model(x_pulse)
    out_float = pulse_to_float32(out_pulse)

    if torch.isnan(out_float).any() or torch.isinf(out_float).any():
        loss = 100.0
    else:
        loss = F.mse_loss(out_float, target_float).item()
        if np.isnan(loss) or np.isinf(loss):
            loss = 100.0

    return loss, out_float, out_pulse


def evaluate_batch(model, device, precomputed_pulses, raw_seq_float):
    """用于测试的批量评估（不更新参数）"""
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    seq_len = precomputed_pulses.shape[1]
    model.reset()

    all_outputs = []
    total_spikes = 0
    total_neurons = 0

    for t in range(seq_len - 1):
        x_pulse = precomputed_pulses[:, t, :, :]
        out_pulse = model(x_pulse)
        total_spikes += (out_pulse > 0).sum().item()
        total_neurons += out_pulse.numel()
        out_float = pulse_to_float32(out_pulse)
        all_outputs.append(out_float)

    fr = total_spikes / total_neurons if total_neurons > 0 else 0
    all_outputs = torch.stack(all_outputs, dim=1)
    target = raw_seq_float[:, 1:, :].to(device)

    if torch.isnan(all_outputs).any() or torch.isinf(all_outputs).any():
        loss = 100.0
    else:
        loss = F.mse_loss(all_outputs, target).item()
        if np.isnan(loss) or np.isinf(loss):
            loss = 100.0

    return loss, fr


# ============================================================
# Group B: 逐步 SPSA 优化
# ============================================================
def spsa_step_per_timestep(model, device, precomputed_pulses, raw_seq_float,
                           c_W, c_beta, c_vth, a_W, a_beta, a_vth,
                           w1_shape, w2_shape, epoch):
    """
    Group B: 逐步 SPSA - 每个时间步（或每隔几步）执行一次 SPSA 更新

    关键点：
    1. 遍历时间步
    2. 每隔 STEP_UPDATE_INTERVAL 步执行一次 SPSA
    3. SPSA 时保存膜电位状态，扰动参数评估，恢复状态后更新
    4. 继续正常前向传播
    """
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    seq_len = precomputed_pulses.shape[1]
    model.reset()

    total_loss = 0.0
    total_spikes = 0
    total_neurons = 0
    spsa_count = 0

    for t in range(seq_len - 1):
        x_pulse = precomputed_pulses[:, t, :, :]
        target_float = raw_seq_float[:, t+1, :].to(device)

        # 是否执行 SPSA 更新
        if t % STEP_UPDATE_INTERVAL == 0:
            # 保存当前膜电位状态
            v_states = save_membrane_states(model)

            # 获取当前参数
            w1, w2 = _get_weight_floats(model)
            w_flat = torch.cat([w1.flatten(), w2.flatten()])
            w_flat_orig = w_flat.clone()

            beta_orig, vth_orig = _get_lif_params(model)
            beta_orig = beta_orig.to(device)
            vth_orig = vth_orig.to(device)

            # 生成扰动
            delta_W = torch.sign(torch.randn_like(w_flat))
            delta_beta = torch.sign(torch.randn_like(beta_orig))
            delta_vth = torch.sign(torch.randn_like(vth_orig))

            # +扰动评估
            w_plus = w_flat + c_W * delta_W
            w1_plus = w_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
            w2_plus = w_plus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
            _set_weight_floats(model, w1_plus, w2_plus)
            _set_lif_beta(model, (beta_orig + c_beta * delta_beta).clamp(0.001, 0.999), device)
            _set_lif_vth(model, (vth_orig + c_vth * delta_vth).clamp(0.1, 3.0), device)
            restore_membrane_states(model, v_states)  # 恢复膜电位状态

            loss_plus, _, _ = evaluate_single_step(model, x_pulse, target_float, device)

            # -扰动评估
            w_minus = w_flat - c_W * delta_W
            w1_minus = w_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
            w2_minus = w_minus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
            _set_weight_floats(model, w1_minus, w2_minus)
            _set_lif_beta(model, (beta_orig - c_beta * delta_beta).clamp(0.001, 0.999), device)
            _set_lif_vth(model, (vth_orig - c_vth * delta_vth).clamp(0.1, 3.0), device)
            restore_membrane_states(model, v_states)  # 恢复膜电位状态

            loss_minus, _, _ = evaluate_single_step(model, x_pulse, target_float, device)

            diff = loss_plus - loss_minus

            if abs(diff) >= 1e-12 and not np.isnan(diff):
                # 梯度估计
                grad_W = diff / (2 * c_W) * (1.0 / delta_W)
                grad_beta = diff / (2 * c_beta) * (1.0 / delta_beta)
                grad_vth = diff / (2 * c_vth) * (1.0 / delta_vth)

                grad_W = torch.clamp(grad_W, -GRAD_CLIP, GRAD_CLIP)
                grad_beta = torch.clamp(grad_beta, -GRAD_CLIP, GRAD_CLIP)
                grad_vth = torch.clamp(grad_vth, -GRAD_CLIP, GRAD_CLIP)

                # 更新参数
                new_W = w_flat_orig - a_W * grad_W
                new_beta = (beta_orig - a_beta * grad_beta).clamp(0.001, 0.999)
                new_vth = (vth_orig - a_vth * grad_vth).clamp(0.1, 3.0)

                w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
                w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
                _set_weight_floats(model, w1_new, w2_new)
                _set_lif_beta(model, new_beta, device)
                _set_lif_vth(model, new_vth, device)
            else:
                # 恢复原始参数
                w1_orig = w_flat_orig[:w1_shape[0]*w1_shape[1]].view(w1_shape)
                w2_orig = w_flat_orig[w1_shape[0]*w1_shape[1]:].view(w2_shape)
                _set_weight_floats(model, w1_orig, w2_orig)
                _set_lif_beta(model, beta_orig, device)
                _set_lif_vth(model, vth_orig, device)

            # 恢复膜电位状态后继续前向
            restore_membrane_states(model, v_states)
            spsa_count += 1

        # 正常前向传播（更新膜电位）
        out_pulse = model(x_pulse)
        out_float = pulse_to_float32(out_pulse)

        total_spikes += (out_pulse > 0).sum().item()
        total_neurons += out_pulse.numel()

        step_loss = F.mse_loss(out_float, target_float).item()
        if not np.isnan(step_loss) and not np.isinf(step_loss):
            total_loss += step_loss

    avg_loss = total_loss / (seq_len - 1)
    fr = total_spikes / total_neurons if total_neurons > 0 else 0

    return avg_loss, fr, spsa_count


# ============================================================
# 训练主函数
# ============================================================
def train_groupB():
    print("\n" + "=" * 70)
    print("实验 16 Group B: 逐步 SPSA（每个时间步单独优化）")
    print(f"  更新间隔: 每 {STEP_UPDATE_INTERVAL} 步执行一次 SPSA")
    print("=" * 70)

    train_seq, train_labels, test_seq, test_labels = load_flood_data()
    n_train = len(train_labels)
    n_test = len(test_labels)
    actual_seq_len = train_seq.shape[1]

    train_seq = train_seq.to(DEVICE)
    test_seq = test_seq.to(DEVICE)

    print(f"[data] Train: {train_seq.shape}, Test: {test_seq.shape}", flush=True)
    print(f"[data] Sequence length: {actual_seq_len}", flush=True)
    print(f"[data] SPSA updates per epoch: ~{(actual_seq_len - 1) // STEP_UPDATE_INTERVAL}", flush=True)

    # 预编码
    print("[encode] Pre-encoding...", flush=True)
    train_pulse = float32_to_pulse(train_seq.reshape(-1, N_FEATURES), device=DEVICE)
    train_pulse = train_pulse.reshape(n_train, actual_seq_len, N_FEATURES, 32)
    test_pulse = float32_to_pulse(test_seq.reshape(-1, N_FEATURES), device=DEVICE)
    test_pulse = test_pulse.reshape(n_test, actual_seq_len, N_FEATURES, 32)

    # 创建模型
    template = SimpleLIFNode(beta=INIT_BETA)
    model = SimpleSpikeMLP(N_FEATURES, HIDDEN_SIZE, N_OUTPUT, neuron_template=template).to(DEVICE)

    # 初始化权重（使用相同的随机种子确保公平对比）
    w1_shape = (HIDDEN_SIZE, N_FEATURES)
    w2_shape = (N_OUTPUT, HIDDEN_SIZE)
    torch.manual_seed(42)
    w1_init = init_orthogonal_weights(w1_shape, TARGET_SPECTRAL_RADIUS, device=DEVICE)
    w2_init = init_orthogonal_weights(w2_shape, TARGET_SPECTRAL_RADIUS, device=DEVICE)
    model.set_weights(w1_init, w2_init)

    # β 噪声
    rng = np.random.RandomState(42)
    for module in model.modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.normal(0, 0.01, module._beta.shape), dtype=torch.float32, device=DEVICE)
            module._beta.data.add_(noise).clamp_(0.001, 0.999)

    # 初始评估
    loss0, fr0 = evaluate_batch(model, DEVICE, train_pulse, train_seq)
    print(f"\n[init] Loss={loss0:.4f}, FR={fr0:.4f}", flush=True)

    trajectory = {'epoch': [0], 'loss': [loss0], 'fr': [fr0], 'spsa_count': [0], 'test_loss': []}

    total_spsa_count = 0

    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # 学习率衰减
        decay = (epoch + 1) ** GAMMA_SPSA
        c_W_k = C_W / decay
        c_beta_k = C_BETA / decay
        c_vth_k = C_VTH / decay

        decay_a = (epoch + 1) ** ALPHA
        a_W_k = A_W / decay_a
        a_beta_k = A_BETA / decay_a
        a_vth_k = A_VTH / decay_a

        loss, fr, spsa_count = spsa_step_per_timestep(
            model, DEVICE, train_pulse, train_seq,
            c_W_k, c_beta_k, c_vth_k, a_W_k, a_beta_k, a_vth_k,
            w1_shape, w2_shape, epoch
        )

        total_spsa_count += spsa_count
        elapsed = time.time() - t0

        trajectory['epoch'].append(epoch)
        trajectory['loss'].append(loss)
        trajectory['fr'].append(fr)
        trajectory['spsa_count'].append(total_spsa_count)

        if epoch <= 10 or epoch % 10 == 0:
            print(f"Ep {epoch:3d}: Loss={loss:.4f}, FR={fr:.4f}, "
                  f"SPSA本epoch={spsa_count}, 累计={total_spsa_count} [{elapsed:.1f}s]", flush=True)

        if epoch % TEST_INTERVAL == 0:
            test_loss, test_fr = evaluate_batch(model, DEVICE, test_pulse, test_seq)
            trajectory['test_loss'].append(test_loss)
            print(f"  [test] Loss={test_loss:.4f}, FR={test_fr:.4f}", flush=True)

        if epoch % 50 == 0:
            save_checkpoint(epoch, loss, trajectory)

    # 最终测试
    test_loss, test_fr = evaluate_batch(model, DEVICE, test_pulse, test_seq)
    print(f"\n[Final] Train Loss={loss:.4f}, Test Loss={test_loss:.4f}")
    print(f"Total SPSA updates: {total_spsa_count}")

    # 保存结果
    save_path = os.path.join(os.path.dirname(__file__), 'exp16_groupB_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'group': 'B',
            'method': 'step_spsa',
            'description': '每个时间步单独 SPSA 优化',
            'step_update_interval': STEP_UPDATE_INTERVAL,
            'total_spsa_updates': total_spsa_count,
            'final_train_loss': loss,
            'final_test_loss': test_loss,
            'trajectory': trajectory,
        }, f, indent=2)
    print(f"[save] {save_path}", flush=True)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(trajectory['epoch'], trajectory['loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Group B: Training Loss')
    axes[0].set_yscale('log')

    axes[1].plot(trajectory['epoch'], trajectory['fr'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Firing Rate')
    axes[1].set_title('Firing Rate')

    axes[2].plot(trajectory['epoch'], trajectory['spsa_count'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Cumulative SPSA Updates')
    axes[2].set_title('SPSA Update Count')

    plt.tight_layout()
    plt.savefig(save_path.replace('.json', '.png'), dpi=150)
    plt.close()

    return trajectory


if __name__ == '__main__':
    train_groupB()
    print("\n[完成] Group B", flush=True)
