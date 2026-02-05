#!/usr/bin/env python3
"""
实验 14: ECG200 真实时间序列分类 (各向异性 SPSA)
===============================================================
使用 UCR Archive 的 ECG200 数据集验证 NEXUS TEMPORAL 模式的时序学习能力。

数据集特点：
- 真实心电图数据
- 序列长度: 96 时间步
- 2 类分类 (正常 vs 异常)
- 每个时间步 1 个浮点特征

与实验13的处理方式一致：
- 短序列 (T=96) 适合 β ≈ 0.01~0.9 的探索
- 各向异性 SPSA (不同参数不同学习率)
- 浮点输入 → 编码 → SNN → 解码 → 浮点输出
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

# ECG200 数据集配置
DATASET_NAME = "ECG200"
SEQ_LEN = 96  # ECG200 序列长度
N_FEATURES = 1  # 每步1个特征
N_CLASSES = 2  # 2类分类

# 网络配置
HIDDEN_SIZE = 32

# 训练配置
N_EPOCHS = 100
TEST_INTERVAL = 5

# ============================================================
# SPSA 配置 (各向异性，与实验13 Group B一致)
# ============================================================
# 权重扰动
C_W = 0.05
A_W = 0.01

# β 扰动 (物理空间，小步长)
C_BETA = 0.001
A_BETA = 0.0001

# V_th 扰动
C_VTH = 0.1
A_VTH = 0.001

MOMENTUM = 0.9
GRAD_CLIP = 10.0

# SPSA 衰减参数
ALPHA = 0.602
GAMMA_SPSA = 0.101

# β 初始化 (与实验13一致)
INIT_BETA = 0.01  # 高泄漏启动


# ============================================================
# 数据加载 (使用 aeon 库加载真实 UCR 数据集)
# ============================================================
def load_ecg_data():
    """使用 aeon 库加载 ECG200 真实数据集"""
    from aeon.datasets import load_classification

    print(f"[data] Loading {DATASET_NAME} using aeon library...", flush=True)

    # 加载训练集和测试集
    X_train, y_train = load_classification(DATASET_NAME, split="train")
    X_test, y_test = load_classification(DATASET_NAME, split="test")

    print(f"[data] Raw shapes: X_train={X_train.shape}, X_test={X_test.shape}", flush=True)

    # aeon 返回格式: (n_samples, n_channels, seq_len)
    # 转换为: (n_samples, seq_len, n_channels)
    train_seq = X_train.transpose(0, 2, 1)  # [N, T, C]
    test_seq = X_test.transpose(0, 2, 1)

    # 标签转换为整数 0/1
    unique_labels = np.unique(y_train)
    label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
    train_labels = np.array([label_map[l] for l in y_train])
    test_labels = np.array([label_map[l] for l in y_test])

    print(f"[data] Labels: {unique_labels} -> {label_map}", flush=True)
    print(f"[data] Train: {len(train_labels)} samples, Test: {len(test_labels)} samples", flush=True)

    # 标准化
    mean = train_seq.mean()
    std = train_seq.std()
    train_seq = (train_seq - mean) / (std + 1e-8)
    test_seq = (test_seq - mean) / (std + 1e-8)

    # 转换为 tensor
    train_seq = torch.tensor(train_seq, dtype=torch.float32)
    test_seq = torch.tensor(test_seq, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_seq, train_labels, test_seq, test_labels


# ============================================================
# 模型定义 (与实验13一致)
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
# 辅助函数
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
    """获取 β 和 V_th 参数"""
    betas, vths = [], []
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            betas.append(module._beta.data.flatten())
            vths.append(module._v_threshold.data.flatten())
    if betas:
        return torch.cat(betas), torch.cat(vths)
    return torch.tensor([]), torch.tensor([])


def _set_lif_params(model, beta, vth):
    """设置 β 和 V_th 参数"""
    # 确保物理约束
    beta = torch.clamp(beta, 1e-6, 1 - 1e-6)
    vth = torch.clamp(vth, 0.01, 100.0)

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
def evaluate_model(model, labels, device, precomputed_pulses, phase='train', verbose=False):
    """评估模型"""
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    n_samples = len(labels)
    batch_labels = labels.to(device)
    all_pulses = precomputed_pulses
    seq_len = all_pulses.shape[1]

    model.reset()

    # 时序处理
    for t in range(seq_len):
        if verbose and t % 20 == 0:
            print(f"    [{phase}] t={t}/{seq_len}", flush=True)
        x_pulse = all_pulses[:, t, :, :]  # [N, D, 32]
        out_pulse = model(x_pulse)

    out_float = pulse_to_float32(out_pulse)

    if verbose:
        print(f"    [{phase}] out_pulse: shape={out_pulse.shape}, sum={out_pulse.sum():.2f}", flush=True)
        print(f"    [{phase}] out_float: min={out_float.min():.4f}, max={out_float.max():.4f}, mean={out_float.mean():.4f}", flush=True)

    preds = out_float.argmax(dim=-1)
    correct = (preds == batch_labels).sum().item()

    # NaN 防护
    if torch.isnan(out_float).any() or torch.isinf(out_float).any():
        loss = 1e6
    else:
        loss = F.cross_entropy(out_float, batch_labels).item()
        if np.isnan(loss) or np.isinf(loss):
            loss = 1e6

    acc = correct / n_samples
    return loss, acc


# ============================================================
# SPSA 优化 (各向异性，物理空间)
# ============================================================
def spsa_step_anisotropic(model, labels, device, precomputed_pulses,
                          c_W, c_beta, c_vth,
                          a_W, a_beta, a_vth,
                          w1_shape, w2_shape,
                          momentum_buf=None, mu=0.9):
    """各向异性 SPSA 优化步（与实验13 Group B一致）"""
    # 获取当前参数
    w1, w2 = _get_weight_floats(model)
    w_flat = torch.cat([w1.flatten(), w2.flatten()])
    beta, vth = _get_lif_params(model)

    # 保存原始参数
    w_flat_orig = w_flat.clone()
    beta_orig = beta.clone()
    vth_orig = vth.clone()

    # 生成扰动方向 (Rademacher)
    delta_W = torch.sign(torch.randn_like(w_flat))
    delta_beta = torch.sign(torch.randn_like(beta))
    delta_vth = torch.sign(torch.randn_like(vth))

    # 正向扰动
    w_plus = w_flat + c_W * delta_W
    beta_plus = torch.clamp(beta + c_beta * delta_beta, 1e-6, 1 - 1e-6)
    vth_plus = torch.clamp(vth + c_vth * delta_vth, 0.01, 100.0)

    w1_plus = w_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_plus = w_plus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_plus, w2_plus)
    _set_lif_params(model, beta_plus, vth_plus)

    loss_plus, acc_plus = evaluate_model(model, labels, device, precomputed_pulses, phase='train', verbose=False)

    # 负向扰动
    w_minus = w_flat - c_W * delta_W
    beta_minus = torch.clamp(beta - c_beta * delta_beta, 1e-6, 1 - 1e-6)
    vth_minus = torch.clamp(vth - c_vth * delta_vth, 0.01, 100.0)

    w1_minus = w_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_minus = w_minus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_minus, w2_minus)
    _set_lif_params(model, beta_minus, vth_minus)

    loss_minus, acc_minus = evaluate_model(model, labels, device, precomputed_pulses, phase='train', verbose=False)

    # 计算 diff
    diff = loss_plus - loss_minus

    # diff 阈值检查
    if abs(diff) < 1e-15 or np.isnan(diff):
        w1_orig = w_flat_orig[:w1_shape[0]*w1_shape[1]].view(w1_shape)
        w2_orig = w_flat_orig[w1_shape[0]*w1_shape[1]:].view(w2_shape)
        _set_weight_floats(model, w1_orig, w2_orig)
        _set_lif_params(model, beta_orig, vth_orig)
        loss, acc = evaluate_model(model, labels, device, precomputed_pulses, phase='train', verbose=False)
        return loss, acc, momentum_buf, diff, None

    # 梯度估计
    grad_W = diff / (2 * c_W) * (1.0 / delta_W)
    grad_beta = diff / (2 * c_beta) * (1.0 / delta_beta)
    grad_vth = diff / (2 * c_vth) * (1.0 / delta_vth)

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

    # 参数更新
    delta_W_actual = a_W * momentum_buf['W']
    delta_beta_actual = a_beta * momentum_buf['beta']
    delta_vth_actual = a_vth * momentum_buf['vth']

    new_W = w_flat_orig - delta_W_actual
    new_beta = torch.clamp(beta_orig - delta_beta_actual, 1e-6, 1 - 1e-6)
    new_vth = torch.clamp(vth_orig - delta_vth_actual, 0.01, 100.0)

    w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params(model, new_beta, new_vth)

    # 评估更新后的损失
    loss, acc = evaluate_model(model, labels, device, precomputed_pulses, phase='train', verbose=False)

    # 返回更新量统计
    delta_stats = {
        'W': (delta_W_actual.abs().mean().item(), delta_W_actual.abs().max().item()),
        'beta': (delta_beta_actual.abs().mean().item(), delta_beta_actual.abs().max().item()),
        'vth': (delta_vth_actual.abs().mean().item(), delta_vth_actual.abs().max().item()),
    }

    return loss, acc, momentum_buf, diff, delta_stats


# ============================================================
# 训练主函数
# ============================================================
def train_ecg_timeseries():
    print("\n" + "=" * 70)
    print(f"实验 14: {DATASET_NAME} 真实时间序列分类")
    print("(各向异性 SPSA，与实验13 Group B一致)")
    print("=" * 70)

    # 加载数据
    print(f"[data] Loading {DATASET_NAME}...", flush=True)
    train_seq, train_labels, test_seq, test_labels = load_ecg_data()

    n_train = len(train_labels)
    n_test = len(test_labels)
    actual_seq_len = train_seq.shape[1]

    train_seq = train_seq.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_seq = test_seq.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    print(f"[data] Train: {train_seq.shape}, labels: {train_labels.shape}", flush=True)
    print(f"[data] Test: {test_seq.shape}, labels: {test_labels.shape}", flush=True)
    print(f"[data] Sequence length: {actual_seq_len}, Classes: {N_CLASSES}", flush=True)

    # 预编码
    print("[encode] Pre-encoding training data...", flush=True)
    train_flat = train_seq.reshape(-1, N_FEATURES)
    train_pulse_flat = float32_to_pulse(train_flat, device=DEVICE)
    train_pulse = train_pulse_flat.reshape(n_train, actual_seq_len, N_FEATURES, 32)

    print("[encode] Pre-encoding test data...", flush=True)
    test_flat = test_seq.reshape(-1, N_FEATURES)
    test_pulse_flat = float32_to_pulse(test_flat, device=DEVICE)
    test_pulse = test_pulse_flat.reshape(n_test, actual_seq_len, N_FEATURES, 32)

    print(f"[encode] Done. Train pulse: {train_pulse.shape}, Test pulse: {test_pulse.shape}", flush=True)

    # 创建模型
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

    # β 初始化
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            module._beta.data.fill_(INIT_BETA)

    # 打印初始参数
    beta_init, vth_init = _get_lif_params(model)
    w1, w2 = _get_weight_floats(model)

    print(f"[params] W: {w1.numel() + w2.numel()}, β: {beta_init.numel()}, V_th: {vth_init.numel()}", flush=True)
    print(f"[params] W1: mean={w1.mean():.4f}, std={w1.std():.4f}", flush=True)
    print(f"[params] β: mean={beta_init.mean():.4f}, range=[{beta_init.min():.4f}, {beta_init.max():.4f}]", flush=True)
    print(f"[params] V_th: mean={vth_init.mean():.4f}, range=[{vth_init.min():.4f}, {vth_init.max():.4f}]", flush=True)

    # 初始评估
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
    loss0, acc0 = evaluate_model(model, train_labels, DEVICE, train_pulse, phase='train', verbose=True)
    rho0 = compute_spectral_radius(model)
    print(f"[init] Loss={loss0:.4f}, Acc={acc0:.2%}, ρ={rho0:.4f}", flush=True)

    # 训练记录
    trajectory = {
        'epoch': [0],
        'loss': [loss0],
        'acc': [acc0],
        'beta_mean': [beta_init.mean().item()],
        'beta_max': [beta_init.max().item()],
        'vth_mean': [vth_init.mean().item()],
        'spectral_radius': [rho0],
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

        # 学习率衰减
        c_W_k = C_W / (epoch + 1) ** GAMMA_SPSA
        c_beta_k = C_BETA / (epoch + 1) ** GAMMA_SPSA
        c_vth_k = C_VTH / (epoch + 1) ** GAMMA_SPSA
        a_W_k = A_W / (epoch + 1) ** ALPHA
        a_beta_k = A_BETA / (epoch + 1) ** ALPHA
        a_vth_k = A_VTH / (epoch + 1) ** ALPHA

        loss, acc, momentum_buf, diff, delta_stats = spsa_step_anisotropic(
            model, train_labels, DEVICE, train_pulse,
            c_W_k, c_beta_k, c_vth_k, a_W_k, a_beta_k, a_vth_k,
            w1_shape, w2_shape,
            momentum_buf, MOMENTUM
        )

        elapsed = time.time() - t0

        # 获取当前参数
        beta_cur, vth_cur = _get_lif_params(model)
        rho = compute_spectral_radius(model)

        # 记录
        trajectory['epoch'].append(epoch)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)
        trajectory['beta_mean'].append(beta_cur.mean().item())
        trajectory['beta_max'].append(beta_cur.max().item())
        trajectory['vth_mean'].append(vth_cur.mean().item())
        trajectory['spectral_radius'].append(rho)

        # 计算发放率
        fr = 0.5  # 简化，实际应计算

        # 打印进度
        if epoch <= 5 or epoch % 10 == 0:
            print(f"Ep {epoch:3d}: Loss={loss:.4f}, Acc={acc:.2%}, "
                  f"W=‖{(w1.norm()**2 + w2.norm()**2).sqrt():.2f}‖, "
                  f"β={beta_cur.mean():.4f}±{beta_cur.std():.4f}, "
                  f"V_th={vth_cur.mean():.4f}±{vth_cur.std():.4f}, "
                  f"ρ={rho:.4f}, diff={diff:.2e} [{elapsed:.0f}s]", flush=True)
            # 打印 delta 统计
            if delta_stats:
                print(f"       Δ: W(mean={delta_stats['W'][0]:.2e}, max={delta_stats['W'][1]:.2e}), "
                      f"β(mean={delta_stats['beta'][0]:.2e}, max={delta_stats['beta'][1]:.2e}), "
                      f"V_th(mean={delta_stats['vth'][0]:.2e}, max={delta_stats['vth'][1]:.2e})", flush=True)

        # 定期评估测试集
        if epoch % TEST_INTERVAL == 0:
            test_loss, test_acc = evaluate_model(model, test_labels, DEVICE, test_pulse, phase='test', verbose=False)
            trajectory['test_loss'].append(test_loss)
            trajectory['test_acc'].append(test_acc)
            print(f"  [test] Ep {epoch}: Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 最终测试
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    test_loss, test_acc = evaluate_model(model, test_labels, DEVICE, test_pulse, phase='test', verbose=True)
    print(f"\n[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2%}", flush=True)

    # 总结
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Initial β: {beta_init.mean():.6f} → Final β: {beta_cur.mean():.6f} (Δ={beta_cur.mean() - beta_init.mean():.6f})")
    print(f"Initial V_th: {vth_init.mean():.4f} → Final V_th: {vth_cur.mean():.4f} (Δ={vth_cur.mean() - vth_init.mean():.4f})")
    print(f"Initial Loss: {loss0:.4f} → Final Loss: {loss:.4f}")
    print(f"Initial Acc: {acc0:.2%} → Final Acc: {acc:.2%}")
    print(f"Test Acc: {test_acc:.2%}")

    # 保存结果
    save_path = os.path.join(os.path.dirname(__file__), 'exp14_ecg_timeseries_data.json')
    with open(save_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': DATASET_NAME,
                'seq_len': actual_seq_len,
                'n_features': N_FEATURES,
                'n_classes': N_CLASSES,
                'hidden_size': HIDDEN_SIZE,
                'n_train': n_train,
                'n_test': n_test,
                'n_epochs': N_EPOCHS,
                'init_beta': INIT_BETA,
                'spsa': {
                    'c_W': C_W, 'c_beta': C_BETA, 'c_vth': C_VTH,
                    'a_W': A_W, 'a_beta': A_BETA, 'a_vth': A_VTH
                }
            },
            'trajectory': trajectory,
            'final': {
                'train_loss': loss,
                'train_acc': acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'beta_mean': beta_cur.mean().item(),
                'vth_mean': vth_cur.mean().item()
            }
        }, f, indent=2)
    print(f"\n[save] Results saved to {save_path}", flush=True)

    # 绘图
    plot_results(trajectory, save_path.replace('.json', '.png'))

    return trajectory


def plot_results(trajectory, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = trajectory['epoch']

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, trajectory['loss'], 'b-', label='Train Loss')
    if trajectory['test_loss']:
        test_epochs = [e for e in epochs if e % TEST_INTERVAL == 0 and e > 0]
        ax.plot(test_epochs[:len(trajectory['test_loss'])], trajectory['test_loss'], 'r--', label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, trajectory['acc'], 'b-', label='Train Acc')
    if trajectory['test_acc']:
        test_epochs = [e for e in epochs if e % TEST_INTERVAL == 0 and e > 0]
        ax.plot(test_epochs[:len(trajectory['test_acc'])], trajectory['test_acc'], 'r--', label='Test Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta evolution
    ax = axes[1, 0]
    ax.plot(epochs, trajectory['beta_mean'], 'g-', label='β mean')
    ax.plot(epochs, trajectory['beta_max'], 'g--', alpha=0.5, label='β max')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('β')
    ax.set_title('β Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral radius
    ax = axes[1, 1]
    ax.plot(epochs, trajectory['spectral_radius'], 'm-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ρ')
    ax.set_title('Spectral Radius')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Saved to {save_path}", flush=True)


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    trajectory = train_ecg_timeseries()
    print("\n[完成]", flush=True)
