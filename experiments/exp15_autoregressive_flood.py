#!/usr/bin/env python3
"""
实验 15: 自回归时序预测 - FloodModeling1 数据集
===============================================================
核心改进：
1. β 初始化为 0.01（通过 neuron_template）
2. V_th 初始化保持门电路默认值（AND=1.5, OR=0.5, NOT=1.0）
3. 优化 W、β 和 V_th 三个参数
4. 真正的自回归预测：时间步 t 预测时间步 t+1（类似 LLM 的 next-token prediction）
5. 膜电位累积 ≈ LLM 的上下文记忆
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
N_FEATURES = 1  # 单维输入
N_OUTPUT = 1    # 单维输出（回归）

# 网络配置
HIDDEN_SIZE = 16

# 训练配置
N_EPOCHS = 200
TEST_INTERVAL = 10

# ============================================================
# SPSA 配置 - 优化 W、β 和 V_th
# ============================================================
# 权重
C_W = 0.02
A_W = 0.002

# β: 时间衰减参数（直接在物理空间优化）
C_BETA = 0.05
A_BETA = 0.01

# V_th: 阈值参数（直接在物理空间优化）
C_VTH = 0.05
A_VTH = 0.01

MOMENTUM = 0.9
GRAD_CLIP = 5.0

# SPSA 衰减参数
ALPHA = 0.602
GAMMA_SPSA = 0.101

# 初始 β 值（与 exp13 一致）
INIT_BETA = 0.01

# 权重谱半径目标
TARGET_SPECTRAL_RADIUS = 1.0


# ============================================================
# 数据加载
# ============================================================
def load_flood_data():
    """使用 aeon 库加载 FloodModeling1 回归数据集"""
    from aeon.datasets import load_regression

    print(f"[data] Loading {DATASET_NAME} using aeon library...", flush=True)

    X_train, y_train = load_regression(DATASET_NAME, split="train")
    X_test, y_test = load_regression(DATASET_NAME, split="test")

    print(f"[data] Raw shapes: X_train={X_train.shape}, X_test={X_test.shape}", flush=True)
    print(f"[data] y_train shape={y_train.shape}, range=[{y_train.min():.4f}, {y_train.max():.4f}]", flush=True)

    # [N, channels, T] -> [N, T, channels]
    train_seq = X_train.transpose(0, 2, 1)
    test_seq = X_test.transpose(0, 2, 1)

    # 标准化 X
    mean_x = train_seq.mean()
    std_x = train_seq.std()
    train_seq = (train_seq - mean_x) / (std_x + 1e-8)
    test_seq = (test_seq - mean_x) / (std_x + 1e-8)

    # 标准化 Y (回归目标)
    mean_y = y_train.mean()
    std_y = y_train.std()
    y_train_norm = (y_train - mean_y) / (std_y + 1e-8)
    y_test_norm = (y_test - mean_y) / (std_y + 1e-8)

    print(f"[data] Normalized y_train: range=[{y_train_norm.min():.4f}, {y_train_norm.max():.4f}]", flush=True)

    train_seq = torch.tensor(train_seq, dtype=torch.float32)
    test_seq = torch.tensor(test_seq, dtype=torch.float32)
    train_labels = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    test_labels = torch.tensor(y_test_norm, dtype=torch.float32).unsqueeze(-1)

    return train_seq, train_labels, test_seq, test_labels, (mean_y, std_y)


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
# 参数管理（与 exp13/exp14 一致）
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
    """收集所有 LIF 节点的 β 和 V_th"""
    all_beta = []
    all_vth = []
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            all_beta.append(module._beta.data.clone().flatten())
            all_vth.append(module._v_threshold.data.clone().flatten())
    return torch.cat(all_beta), torch.cat(all_vth)


def _set_lif_beta(model, beta_flat, device):
    """设置 β"""
    idx = 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_beta'):
            n = module._beta.numel()
            module._beta.data.copy_(
                beta_flat[idx:idx+n].view(module._beta.shape).to(device))
            idx += n


def _set_lif_vth(model, vth_flat, device):
    """设置 V_th"""
    idx = 0
    for module in model.modules():
        if isinstance(module, SimpleLIFNode) and hasattr(module, '_v_threshold'):
            n = module._v_threshold.numel()
            module._v_threshold.data.copy_(
                vth_flat[idx:idx+n].view(module._v_threshold.shape).to(device))
            idx += n


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


def compute_firing_rate(out_pulse):
    """计算发放率"""
    return (out_pulse > 0).float().mean().item()


def init_orthogonal_weights(shape, target_rho=1.0, device='cpu'):
    """正交初始化，控制谱半径"""
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
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'exp15_flood_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(model, epoch, loss, trajectory, prefix='epoch'):
    """保存模型检查点"""
    w1, w2 = _get_weight_floats(model)
    beta, vth = _get_lif_params(model)

    ckpt = {
        'epoch': epoch,
        'loss': loss,
        'weights': {
            'w1': w1.cpu().numpy().tolist(),
            'w2': w2.cpu().numpy().tolist(),
        },
        'params': {
            'beta_mean': beta.mean().item(),
            'beta_std': beta.std().item(),
            'vth_mean': vth.mean().item(),
            'vth_std': vth.std().item(),
        },
    }
    ckpt_path = os.path.join(CKPT_DIR, f'{prefix}_epoch{epoch:03d}.json')
    with open(ckpt_path, 'w') as f:
        json.dump(ckpt, f, indent=2)
    print(f"    [ckpt] Saved to {ckpt_path}", flush=True)


# ============================================================
# 自回归评估函数（真正的 next-step prediction）
# ============================================================
def evaluate_autoregressive(model, labels, device, precomputed_pulses, raw_seq_float, phase='train', verbose=False):
    """
    真正的自回归评估：时间步 t 的输出预测时间步 t+1 的输入值

    类似 LLM 的 next-token prediction:
    - 输入:  [x0, x1, x2, ..., x_{T-2}]
    - 目标:  [x1, x2, x3, ..., x_{T-1}]
    - 膜电位累积 ≈ LLM 的上下文记忆

    Args:
        labels: [N, 1] 最终回归目标（仅用于最后一步评估）
        precomputed_pulses: [N, T, D, 32] 预编码的脉冲序列
        raw_seq_float: [N, T, D] 原始浮点序列（用于构建目标）

    Returns:
        loss: 所有时间步的平均 MSE
        final_pred: 最后一步的预测值
        fr: 平均发放率
    """
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    n_samples = precomputed_pulses.shape[0]
    seq_len = precomputed_pulses.shape[1]

    model.reset()

    # 累积统计
    total_spikes = 0
    total_neurons = 0
    all_outputs = []

    # 遍历 t=0 到 t=T-2，预测 t+1
    for t in range(seq_len - 1):
        if verbose and t % 50 == 0:
            print(f"    [{phase}] t={t}/{seq_len-1}", flush=True)

        x_pulse = precomputed_pulses[:, t, :, :]  # [N, D, 32]
        out_pulse = model(x_pulse)                 # [N, 1, 32]

        # 统计发放
        total_spikes += (out_pulse > 0).sum().item()
        total_neurons += out_pulse.numel()

        # 解码输出
        out_float = pulse_to_float32(out_pulse)  # [N, 1]
        all_outputs.append(out_float)

    # 计算发放率
    fr = total_spikes / total_neurons if total_neurons > 0 else 0

    # 堆叠所有时间步输出: [N, T-1, 1]
    all_outputs = torch.stack(all_outputs, dim=1)

    if verbose:
        print(f"    [{phase}] all_outputs shape: {all_outputs.shape}", flush=True)
        print(f"    [{phase}] firing_rate: {fr:.4f}", flush=True)

    # 真正的自回归目标：下一个时间步的值
    # target[t] = raw_seq_float[t+1]
    target = raw_seq_float[:, 1:, :].to(device)  # [N, T-1, D]

    # NaN 防护
    if torch.isnan(all_outputs).any() or torch.isinf(all_outputs).any():
        loss = 100.0
        final_pred = all_outputs[:, -1, :]
    else:
        # MSE: 预测下一步
        loss = F.mse_loss(all_outputs, target).item()
        if np.isnan(loss) or np.isinf(loss):
            loss = 100.0
        final_pred = all_outputs[:, -1, :]

    return loss, final_pred, fr


# ============================================================
# SPSA 优化 (优化 W、β 和 V_th)
# ============================================================
def spsa_step(model, labels, device, precomputed_pulses, raw_seq_float,
              c_W, c_beta, c_vth, a_W, a_beta, a_vth,
              w1_shape, w2_shape,
              momentum_buf=None, mu=0.9):
    """优化 W、β 和 V_th 的 SPSA 步骤"""

    # 获取当前参数
    w1, w2 = _get_weight_floats(model)
    w_flat = torch.cat([w1.flatten(), w2.flatten()])
    w_flat_orig = w_flat.clone()

    beta_orig, vth_orig = _get_lif_params(model)
    beta_orig = beta_orig.to(device)
    vth_orig = vth_orig.to(device)

    # 生成扰动方向
    delta_W = torch.sign(torch.randn_like(w_flat))
    delta_beta = torch.sign(torch.randn_like(beta_orig))
    delta_vth = torch.sign(torch.randn_like(vth_orig))

    # ===== 正向扰动 =====
    w_plus = w_flat + c_W * delta_W
    w1_plus = w_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_plus = w_plus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_plus, w2_plus)

    beta_plus = (beta_orig + c_beta * delta_beta).clamp(0.001, 0.999)
    _set_lif_beta(model, beta_plus, device)

    vth_plus = (vth_orig + c_vth * delta_vth).clamp(0.1, 3.0)
    _set_lif_vth(model, vth_plus, device)

    loss_plus, _, fr_plus = evaluate_autoregressive(model, labels, device, precomputed_pulses, raw_seq_float)

    # ===== 负向扰动 =====
    w_minus = w_flat - c_W * delta_W
    w1_minus = w_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_minus = w_minus[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_minus, w2_minus)

    beta_minus = (beta_orig - c_beta * delta_beta).clamp(0.001, 0.999)
    _set_lif_beta(model, beta_minus, device)

    vth_minus = (vth_orig - c_vth * delta_vth).clamp(0.1, 3.0)
    _set_lif_vth(model, vth_minus, device)

    loss_minus, _, fr_minus = evaluate_autoregressive(model, labels, device, precomputed_pulses, raw_seq_float)

    # 计算 diff
    diff = loss_plus - loss_minus

    # diff 阈值检查
    if abs(diff) < 1e-12 or np.isnan(diff):
        # 恢复原始参数
        w1_orig = w_flat_orig[:w1_shape[0]*w1_shape[1]].view(w1_shape)
        w2_orig = w_flat_orig[w1_shape[0]*w1_shape[1]:].view(w2_shape)
        _set_weight_floats(model, w1_orig, w2_orig)
        _set_lif_beta(model, beta_orig, device)
        _set_lif_vth(model, vth_orig, device)
        loss, pred, fr = evaluate_autoregressive(model, labels, device, precomputed_pulses, raw_seq_float)
        return loss, fr, momentum_buf, diff, None

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
            'vth': grad_vth.clone(),
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
    new_beta = (beta_orig - delta_beta_actual).clamp(0.001, 0.999)
    new_vth = (vth_orig - delta_vth_actual).clamp(0.1, 3.0)

    w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_beta(model, new_beta, device)
    _set_lif_vth(model, new_vth, device)

    # 评估更新后
    loss, pred, fr = evaluate_autoregressive(model, labels, device, precomputed_pulses, raw_seq_float)

    # 更新量统计
    delta_stats = {
        'W': (delta_W_actual.abs().mean().item(), delta_W_actual.abs().max().item()),
        'beta': (delta_beta_actual.abs().mean().item(), delta_beta_actual.abs().max().item()),
        'vth': (delta_vth_actual.abs().mean().item(), delta_vth_actual.abs().max().item()),
    }

    return loss, fr, momentum_buf, diff, delta_stats


# ============================================================
# 训练主函数
# ============================================================
def train_flood_autoregressive():
    print("\n" + "=" * 70)
    print(f"实验 15: {DATASET_NAME} 自回归时序预测 - 门电路默认阈值")
    print("(β=0.01 初始化 + V_th 保持门电路默认值 + 每步输出监督)")
    print("=" * 70)

    # 加载数据
    print(f"[data] Loading {DATASET_NAME}...", flush=True)
    train_seq, train_labels, test_seq, test_labels, (mean_y, std_y) = load_flood_data()

    n_train = len(train_labels)
    n_test = len(test_labels)
    actual_seq_len = train_seq.shape[1]

    train_seq = train_seq.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_seq = test_seq.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    print(f"[data] Train: {train_seq.shape}, labels: {train_labels.shape}", flush=True)
    print(f"[data] Test: {test_seq.shape}, labels: {test_labels.shape}", flush=True)
    print(f"[data] Sequence length: {actual_seq_len}", flush=True)

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

    # 创建模型 - 使用 SimpleLIFNode(beta=0.01) 模板，V_th 保持门电路默认值
    template = SimpleLIFNode(beta=INIT_BETA)
    model = SimpleSpikeMLP(N_FEATURES, HIDDEN_SIZE, N_OUTPUT, neuron_template=template).to(DEVICE)

    # 正交权重初始化
    w1_shape = (HIDDEN_SIZE, N_FEATURES)
    w2_shape = (N_OUTPUT, HIDDEN_SIZE)

    torch.manual_seed(42)
    w1_init = init_orthogonal_weights(w1_shape, TARGET_SPECTRAL_RADIUS, device=DEVICE)
    w2_init = init_orthogonal_weights(w2_shape, TARGET_SPECTRAL_RADIUS, device=DEVICE)
    model.set_weights(w1_init, w2_init)

    # 为 β 添加小噪声（与 exp13 一致）
    rng = np.random.RandomState(42)
    for module in model.modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.normal(0, 0.01, module._beta.shape),
                                 dtype=torch.float32, device=DEVICE)
            module._beta.data.add_(noise)
            module._beta.data.clamp_(0.001, 0.999)

    # 打印初始参数
    beta_init, vth_init = _get_lif_params(model)
    w1, w2 = _get_weight_floats(model)
    rho0 = compute_spectral_radius(model)

    print(f"\n[init] 参数统计:", flush=True)
    print(f"  W: {w1.numel() + w2.numel()} params, ρ={rho0:.4f}", flush=True)
    print(f"  β: {beta_init.numel()} params", flush=True)
    print(f"    mean={beta_init.mean():.4f}, range=[{beta_init.min():.4f}, {beta_init.max():.4f}]", flush=True)
    print(f"  V_th: {vth_init.numel()} params (门电路默认值初始化，参与优化)", flush=True)
    print(f"    mean={vth_init.mean():.4f}, range=[{vth_init.min():.4f}, {vth_init.max():.4f}]", flush=True)

    # Pre-check
    print("\n[pre-check] 检查初始活动...", flush=True)
    loss0, pred0, fr0 = evaluate_autoregressive(model, train_labels, DEVICE, train_pulse, train_seq, phase='train', verbose=True)

    print(f"\n[pre-check] 初始状态:", flush=True)
    print(f"  MSE Loss={loss0:.4f}, FR={fr0:.4f}", flush=True)

    # 训练记录
    trajectory = {
        'epoch': [0],
        'loss': [loss0],
        'firing_rate': [fr0],
        'beta_mean': [beta_init.mean().item()],
        'beta_std': [beta_init.std().item()],
        'vth_mean': [vth_init.mean().item()],
        'spectral_radius': [rho0],
        'diff': [0.0],
        'test_loss': [],
    }

    # 保存初始检查点
    save_checkpoint(model, 0, loss0, trajectory, prefix='init')

    # 训练循环
    momentum_buf = None
    best_loss = loss0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("Training with Anisotropic SPSA (W + β + V_th)")
    print(f"  c_W={C_W}, c_β={C_BETA}, c_vth={C_VTH}")
    print(f"  a_W={A_W}, a_β={A_BETA}, a_vth={A_VTH}")
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

        loss, fr, momentum_buf, diff, delta_stats = spsa_step(
            model, train_labels, DEVICE, train_pulse, train_seq,
            c_W_k, c_beta_k, c_vth_k, a_W_k, a_beta_k, a_vth_k,
            w1_shape, w2_shape,
            momentum_buf, MOMENTUM
        )

        elapsed = time.time() - t0

        # 获取当前参数
        beta_cur, vth_cur = _get_lif_params(model)
        rho = compute_spectral_radius(model)

        # 记录最佳
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch

        # 记录
        trajectory['epoch'].append(epoch)
        trajectory['loss'].append(loss)
        trajectory['firing_rate'].append(fr)
        trajectory['beta_mean'].append(beta_cur.mean().item())
        trajectory['beta_std'].append(beta_cur.std().item())
        trajectory['vth_mean'].append(vth_cur.mean().item())
        trajectory['spectral_radius'].append(rho)
        trajectory['diff'].append(diff if diff is not None else 0.0)

        # 打印进度
        if epoch <= 10 or epoch % 20 == 0:
            print(f"Ep {epoch:3d}: MSE={loss:.4f}, FR={fr:.4f}, "
                  f"β={beta_cur.mean():.4f}±{beta_cur.std():.4f}, "
                  f"ρ={rho:.4f}, diff={diff:.2e} [{elapsed:.1f}s]", flush=True)
            if delta_stats:
                print(f"       Δ: W({delta_stats['W'][0]:.2e}), β({delta_stats['beta'][0]:.2e}), vth({delta_stats['vth'][0]:.2e})", flush=True)

        # 定期评估测试集
        if epoch % TEST_INTERVAL == 0:
            test_loss, test_pred, test_fr = evaluate_autoregressive(
                model, test_labels, DEVICE, test_pulse, test_seq, phase='test')
            trajectory['test_loss'].append(test_loss)
            print(f"  [test] Ep {epoch}: MSE={test_loss:.4f}, FR={test_fr:.4f}", flush=True)

        # 保存检查点
        if epoch % 50 == 0:
            save_checkpoint(model, epoch, loss, trajectory, prefix='epoch')

    # 最终测试
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    test_loss, test_pred, test_fr = evaluate_autoregressive(
        model, test_labels, DEVICE, test_pulse, test_seq, phase='test', verbose=True)
    print(f"\n[Final Test] MSE={test_loss:.4f}, FR={test_fr:.4f}", flush=True)

    # 总结
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Initial β: {beta_init.mean():.4f} → Final β: {beta_cur.mean():.4f} (Δ={beta_cur.mean() - beta_init.mean():.4f})")
    print(f"V_th (fixed): {vth_init.mean():.4f} (门电路默认值)")
    print(f"Initial FR: {fr0:.4f} → Final FR: {fr:.4f}")
    print(f"Initial MSE: {loss0:.4f} → Final MSE: {loss:.4f}")
    print(f"Best Train MSE: {best_loss:.4f} at epoch {best_epoch}")
    print(f"Test MSE: {test_loss:.4f}")

    # 保存结果
    save_path = os.path.join(os.path.dirname(__file__), 'exp15_flood_autoregressive_data.json')
    with open(save_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': DATASET_NAME,
                'seq_len': actual_seq_len,
                'n_features': N_FEATURES,
                'hidden_size': HIDDEN_SIZE,
                'n_train': n_train,
                'n_test': n_test,
                'n_epochs': N_EPOCHS,
                'init_beta': INIT_BETA,
                'target_spectral_radius': TARGET_SPECTRAL_RADIUS,
                'spsa': {
                    'c_W': C_W, 'c_beta': C_BETA,
                    'a_W': A_W, 'a_beta': A_BETA
                }
            },
            'trajectory': trajectory,
            'final': {
                'train_loss': loss,
                'test_loss': test_loss,
                'firing_rate': fr,
                'beta_mean': beta_cur.mean().item(),
                'beta_std': beta_cur.std().item(),
                'vth_mean': vth_cur.mean().item(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
            }
        }, f, indent=2)
    print(f"\n[save] Results saved to {save_path}", flush=True)

    # 绘图
    plot_results(trajectory, save_path.replace('.json', '.png'))

    return trajectory


def plot_results(trajectory, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = trajectory['epoch']

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, trajectory['loss'], 'b-', label='Train MSE')
    if trajectory['test_loss']:
        test_epochs = [e for e in epochs if e > 0 and e % TEST_INTERVAL == 0]
        ax.plot(test_epochs[:len(trajectory['test_loss'])], trajectory['test_loss'], 'r--', label='Test MSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Firing Rate
    ax = axes[0, 1]
    ax.plot(epochs, trajectory['firing_rate'], 'g-')
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Healthy range')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Firing Rate')
    ax.set_title('Firing Rate Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta evolution
    ax = axes[0, 2]
    ax.plot(epochs, trajectory['beta_mean'], 'b-', label='β mean')
    ax.fill_between(epochs,
                    [m - s for m, s in zip(trajectory['beta_mean'], trajectory['beta_std'])],
                    [m + s for m, s in zip(trajectory['beta_mean'], trajectory['beta_std'])],
                    alpha=0.3, color='b')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('β')
    ax.set_title('β Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # V_th (fixed)
    ax = axes[1, 0]
    ax.plot(epochs, trajectory['vth_mean'], 'm-', label='V_th (fixed)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('V_th')
    ax.set_title('V_th (Fixed - Gate Defaults)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral radius
    ax = axes[1, 1]
    ax.plot(epochs, trajectory['spectral_radius'], 'c-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ρ')
    ax.set_title('Spectral Radius')
    ax.grid(True, alpha=0.3)

    # SPSA diff
    ax = axes[1, 2]
    ax.plot(epochs, trajectory['diff'], 'orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L+ - L-')
    ax.set_title('SPSA Gradient Signal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Saved to {save_path}", flush=True)


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    trajectory = train_flood_autoregressive()
    print("\n[完成]", flush=True)
