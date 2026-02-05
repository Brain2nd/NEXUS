"""
实验 13：多尺度并发演化与量纲匹配 (Multi-Scale Simultaneous Evolution & Dimensional Matching)
============================================================================================

核心思想：不再人为用 Epoch 切分参数更新，而是让 W、β、V_th 在同一时间步内
各自按照自己的本征时间常数并发演化。

物理类比：
- 快变量 W：对应信号高频波动，大扰动+大学习率
- 慢变量 β：对应宏观序参量，微扰动+小学习率（大惯性）
- 中速变量 V_th：跟随信号强度调整

数学本质：各向异性 SPSA (Anisotropic SPSA) = 对 Hessian 特征值分布的预处理

对照实验：
- Group A (Isotropic): 所有参数使用相同的 c, a
- Group B (Anisotropic): W、β、V_th 各自使用匹配其量纲的 c, a
"""

print("[import] torch, numpy...", flush=True)
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime

print("[import] matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("[import] atomic_ops...", flush=True)
import sys
sys.path.insert(0, '/Users/tangzhengzheng/Desktop/NEXUS')

from atomic_ops import (
    SpikeMode,
    SimpleLIFNode,
    SpikeFP32Linear_MultiPrecision,
)
from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

print("[import] done", flush=True)

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] {DEVICE}", flush=True)

# 实验参数
N_EPOCHS = 100
T_USE = 15  # 时间步数

# ============================================================
# 各向异性 SPSA 超参数 (核心创新)
# ============================================================
# Group A: 各向同性 (Isotropic) - 所有参数同一尺度
ISO_c = 0.01      # 统一扰动幅度
ISO_a = 0.005     # 统一学习率

# Group B: 各向异性 (Anisotropic) - 量纲匹配
ANISO_c_W = 0.05      # W 需要较大探索范围
ANISO_c_beta = 0.001  # β 对误差极其敏感，必须微扰
ANISO_c_vth = 0.1     # V_th 数量级较大，需要较大扰动

ANISO_a_W = 0.01      # 快变量，允许快速弛豫
ANISO_a_beta = 0.0001 # 慢变量，保持大惯性
ANISO_a_vth = 0.001   # 中速变量

# 衰减参数
alpha = 0.602
gamma_spsa = 0.101

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
# 数据生成
# ============================================================
def generate_temporal_sequences(n_samples=8, seq_len=15, n_features=4):
    """生成时序分类数据"""
    torch.manual_seed(42)
    sequences = []
    labels = []
    for i in range(n_samples):
        label = i % 2
        seq = torch.randn(seq_len, n_features) * 0.5
        if label == 0:
            seq[:, 0] += 0.5
        else:
            seq[:, 0] -= 0.5
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.tensor(labels)


# ============================================================
# 参数工具函数 (与实验12对齐)
# ============================================================
def _get_weight_floats(model):
    w1_pulse = model.linear1.weight_pulse
    w2_pulse = model.linear2.weight_pulse
    w1_float = pulse_to_float32(w1_pulse.float())
    w2_float = pulse_to_float32(w2_pulse.float())
    return w1_float, w2_float


def _set_weight_floats(model, w1_float, w2_float):
    model.linear1.set_weight_from_float(w1_float)
    model.linear2.set_weight_from_float(w2_float)


def _get_lif_params(model):
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.flatten())
            vths.append(module._v_threshold.data.flatten())
    if not betas:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(betas), torch.cat(vths)


def _set_lif_params(model, all_beta, all_vth):
    idx_beta, idx_vth = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            n_beta = module._beta.numel()
            n_vth = module._v_threshold.numel()
            module._beta.data.copy_(all_beta[idx_beta:idx_beta+n_beta].view(module._beta.shape))
            module._v_threshold.data.copy_(all_vth[idx_vth:idx_vth+n_vth].view(module._v_threshold.shape))
            idx_beta += n_beta
            idx_vth += n_vth


def _bernoulli_delta(n, device):
    """Bernoulli +/-1 perturbation (standard SPSA)."""
    return torch.where(torch.rand(n, device=device) > 0.5,
                       torch.ones(n, device=device),
                       -torch.ones(n, device=device))


def clamp_physics(model):
    """物理约束"""
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module._beta.data.clamp_(0.01, 0.999)
            module._v_threshold.data.clamp_(0.1, 100.0)


# ============================================================
# 物理量提取
# ============================================================
def extract_physics(model):
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.cpu().numpy().flatten())
            vths.append(module._v_threshold.data.cpu().numpy().flatten())
    if not betas:
        return 0.5, 1.0, 1.0, 0.0, 0.0
    all_beta = np.concatenate(betas)
    all_vth = np.concatenate(vths)
    return (float(np.mean(all_beta)), float(np.mean(all_vth)),
            1.0/max(np.mean(all_vth), 1e-6),
            float(np.std(all_beta)), float(np.std(all_vth)))


def extract_physics_full(model):
    params_by_component = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            parts = name.split('.')
            component = parts[0] if parts else name
            beta_vals = module._beta.data.cpu().numpy().flatten().tolist()
            vth_vals = module._v_threshold.data.cpu().numpy().flatten().tolist()
            if component not in params_by_component:
                params_by_component[component] = {'beta': [], 'v_th': []}
            params_by_component[component]['beta'].extend(beta_vals)
            params_by_component[component]['v_th'].extend(vth_vals)
    result = {}
    for comp, data in params_by_component.items():
        result[comp] = {
            'beta': data['beta'],
            'v_th': data['v_th'],
            'n_params': len(data['beta'])
        }
    return result


def extract_spectral_radius(model):
    w1_float, w2_float = _get_weight_floats(model)
    try:
        s1 = torch.linalg.svdvals(w1_float)[0].item()
        s2 = torch.linalg.svdvals(w2_float)[0].item()
        return max(s1, s2)
    except:
        return 0.0


def extract_weight_stats(model):
    """提取权重统计：mean, std, norm"""
    w1_float, w2_float = _get_weight_floats(model)
    w_all = torch.cat([w1_float.flatten(), w2_float.flatten()])
    w_mean = float(w_all.mean().item())
    w_std = float(w_all.std().item())
    w_norm = float(torch.norm(w_all).item())
    return w_mean, w_std, w_norm


# ============================================================
# 前向评估
# ============================================================
def evaluate_model(model, sequences, labels, device, T_use):
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    total_loss = 0.0
    correct = 0
    n_samples = len(sequences)

    stats = {'nan_count': 0, 'inf_count': 0}

    anchor_0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    anchor_1 = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device)

    for i in range(n_samples):
        model.reset()
        seq = sequences[i].to(device)
        label = labels[i].item()

        for t in range(min(T_use, seq.shape[0])):
            x_t = seq[t:t+1]
            x_pulse = float32_to_pulse(x_t, device=device)
            out_pulse = model(x_pulse)

        out_float = pulse_to_float32(out_pulse).squeeze(0)

        if torch.isnan(out_float).any():
            stats['nan_count'] += 1
            loss_i = 1e12
        elif torch.isinf(out_float).any():
            stats['inf_count'] += 1
            loss_i = 1e12
        else:
            target = anchor_0 if label == 0 else anchor_1
            loss_i = torch.mean((out_float - target) ** 2).item()

        total_loss += min(loss_i, 1e12)

        if not (torch.isnan(out_float).any() or torch.isinf(out_float).any()):
            pred = 0 if out_float[0] > 0 else 1
            if pred == label:
                correct += 1

    avg_loss = total_loss / n_samples
    acc = correct / n_samples
    fr = 0.5

    return avg_loss, acc, fr, stats


# ============================================================
# 各向同性 SPSA (Isotropic) - Group A
# ============================================================
def spsa_step_isotropic(model, sequences, labels, device, c_k, a_k, T_use,
                        w1_shape, w2_shape, n_beta, n_vth, momentum_buf=None, mu=0.9):
    """
    各向同性 SPSA：所有参数使用相同的扰动幅度 c 和学习率 a
    (与实验12对齐的基础 SPSA)
    """
    # 获取当前参数
    w1_float, w2_float = _get_weight_floats(model)
    all_beta, all_vth = _get_lif_params(model)

    # 展平
    w_flat = torch.cat([w1_float.flatten(), w2_float.flatten()])
    orig_flat = torch.cat([w_flat, all_beta, all_vth])
    n_params = orig_flat.shape[0]

    # 生成扰动
    delta = _bernoulli_delta(n_params, device)

    # 正向扰动
    perturbed_plus = orig_flat + c_k * delta
    w_dim = w1_shape[0]*w1_shape[1] + w2_shape[0]*w2_shape[1]
    w1_p = perturbed_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_p = perturbed_plus[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_p = perturbed_plus[w_dim:w_dim+n_beta]
    vth_p = perturbed_plus[w_dim+n_beta:]
    _set_weight_floats(model, w1_p, w2_p)
    _set_lif_params(model, beta_p, vth_p)
    clamp_physics(model)
    loss_plus, _, _, _ = evaluate_model(model, sequences, labels, device, T_use)

    # 负向扰动
    perturbed_minus = orig_flat - c_k * delta
    w1_m = perturbed_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_m = perturbed_minus[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_m = perturbed_minus[w_dim:w_dim+n_beta]
    vth_m = perturbed_minus[w_dim+n_beta:]
    _set_weight_floats(model, w1_m, w2_m)
    _set_lif_params(model, beta_m, vth_m)
    clamp_physics(model)
    loss_minus, _, _, _ = evaluate_model(model, sequences, labels, device, T_use)

    diff = loss_plus - loss_minus

    # 阈值判断 (与实验12对齐: 1e-15)
    if abs(diff) < 1e-15:
        # 恢复原参数
        w1_orig = orig_flat[:w1_shape[0]*w1_shape[1]].view(w1_shape)
        w2_orig = orig_flat[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
        beta_orig = orig_flat[w_dim:w_dim+n_beta]
        vth_orig = orig_flat[w_dim+n_beta:]
        _set_weight_floats(model, w1_orig, w2_orig)
        _set_lif_params(model, beta_orig, vth_orig)
        loss, acc, fr, stats = evaluate_model(model, sequences, labels, device, T_use)
        return loss, acc, diff, fr, momentum_buf, stats

    # 计算梯度估计 (与实验12对齐)
    g_hat = (diff / (2 * c_k)) * (1.0 / delta)
    g_hat = g_hat.clamp(-10.0, 10.0)  # 梯度裁剪

    # 动量 (与实验12对齐)
    if momentum_buf is None:
        momentum_buf = g_hat.clone()
    else:
        momentum_buf = mu * momentum_buf + (1 - mu) * g_hat

    # 更新
    new_flat = orig_flat - a_k * momentum_buf
    w1_new = new_flat[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_flat[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_new = new_flat[w_dim:w_dim+n_beta]
    vth_new = new_flat[w_dim+n_beta:]
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params(model, beta_new, vth_new)
    clamp_physics(model)

    loss, acc, fr, stats = evaluate_model(model, sequences, labels, device, T_use)
    return loss, acc, diff, fr, momentum_buf, stats


# ============================================================
# 各向异性 SPSA (Anisotropic) - Group B (核心创新)
# ============================================================
def spsa_step_anisotropic(model, sequences, labels, device,
                          c_W, c_beta, c_vth, a_W, a_beta, a_vth, T_use,
                          w1_shape, w2_shape, n_beta, n_vth,
                          momentum_W=None, momentum_beta=None, momentum_vth=None, mu=0.9):
    """
    各向异性 SPSA：W、β、V_th 各自使用匹配其量纲的扰动幅度和学习率

    在同一时间步内，三类参数并发演化，但各自按自己的本征时间常数。
    """
    # 获取当前参数
    w1_float, w2_float = _get_weight_floats(model)
    all_beta, all_vth = _get_lif_params(model)

    w_flat = torch.cat([w1_float.flatten(), w2_float.flatten()])
    w_dim = w_flat.shape[0]

    # 为每类参数独立生成扰动
    delta_W = _bernoulli_delta(w_dim, device)
    delta_beta = _bernoulli_delta(n_beta, device)
    delta_vth = _bernoulli_delta(n_vth, device)

    # 构建尺度匹配的扰动向量
    # C ⊙ Δ: 每类参数使用自己的扰动幅度
    perturbation_plus = torch.cat([
        c_W * delta_W,
        c_beta * delta_beta,
        c_vth * delta_vth
    ])
    perturbation_minus = -perturbation_plus

    orig_flat = torch.cat([w_flat, all_beta, all_vth])

    # 正向扰动
    perturbed_plus = orig_flat + perturbation_plus
    w1_p = perturbed_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_p = perturbed_plus[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_p = perturbed_plus[w_dim:w_dim+n_beta]
    vth_p = perturbed_plus[w_dim+n_beta:]
    _set_weight_floats(model, w1_p, w2_p)
    _set_lif_params(model, beta_p, vth_p)
    clamp_physics(model)
    loss_plus, _, _, _ = evaluate_model(model, sequences, labels, device, T_use)

    # 负向扰动
    perturbed_minus = orig_flat + perturbation_minus
    w1_m = perturbed_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_m = perturbed_minus[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_m = perturbed_minus[w_dim:w_dim+n_beta]
    vth_m = perturbed_minus[w_dim+n_beta:]
    _set_weight_floats(model, w1_m, w2_m)
    _set_lif_params(model, beta_m, vth_m)
    clamp_physics(model)
    loss_minus, _, _, _ = evaluate_model(model, sequences, labels, device, T_use)

    diff = loss_plus - loss_minus

    # 阈值判断
    if abs(diff) < 1e-15:
        w1_orig = orig_flat[:w1_shape[0]*w1_shape[1]].view(w1_shape)
        w2_orig = orig_flat[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
        beta_orig = orig_flat[w_dim:w_dim+n_beta]
        vth_orig = orig_flat[w_dim+n_beta:]
        _set_weight_floats(model, w1_orig, w2_orig)
        _set_lif_params(model, beta_orig, vth_orig)
        loss, acc, fr, stats = evaluate_model(model, sequences, labels, device, T_use)
        return loss, acc, diff, fr, momentum_W, momentum_beta, momentum_vth, stats

    # 分别计算每类参数的梯度估计
    # g = (L+ - L-) / (2 * c * delta)
    g_W = (diff / (2 * c_W)) * (1.0 / delta_W)
    g_beta = (diff / (2 * c_beta)) * (1.0 / delta_beta)
    g_vth = (diff / (2 * c_vth)) * (1.0 / delta_vth)

    # 梯度裁剪
    g_W = g_W.clamp(-10.0, 10.0)
    g_beta = g_beta.clamp(-10.0, 10.0)
    g_vth = g_vth.clamp(-10.0, 10.0)

    # 分别使用动量
    if momentum_W is None:
        momentum_W = g_W.clone()
    else:
        momentum_W = mu * momentum_W + (1 - mu) * g_W

    if momentum_beta is None:
        momentum_beta = g_beta.clone()
    else:
        momentum_beta = mu * momentum_beta + (1 - mu) * g_beta

    if momentum_vth is None:
        momentum_vth = g_vth.clone()
    else:
        momentum_vth = mu * momentum_vth + (1 - mu) * g_vth

    # 多速率更新：每类参数使用自己的学习率
    new_W = w_flat - a_W * momentum_W
    new_beta = all_beta - a_beta * momentum_beta
    new_vth = all_vth - a_vth * momentum_vth

    w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params(model, new_beta, new_vth)
    clamp_physics(model)

    loss, acc, fr, stats = evaluate_model(model, sequences, labels, device, T_use)
    return loss, acc, diff, fr, momentum_W, momentum_beta, momentum_vth, stats


# ============================================================
# 初始化
# ============================================================
def orthogonal_init_with_spectral_radius(shape, target_rho=1.5, device='cpu'):
    rows, cols = shape
    A = torch.randn(rows, cols, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    w = U @ Vh
    svs = torch.linalg.svdvals(w)
    current_rho = svs[0].item()
    if current_rho > 0:
        w = w * (target_rho / current_rho)
    return w


def create_model(device):
    """创建模型：高泄漏启动 + 正交权重 + 门电路默认V_th"""
    torch.manual_seed(42)
    template = SimpleLIFNode(beta=0.01)
    model = SimpleSpikeMLP(4, 8, 4, neuron_template=template).to(device)

    # 正交初始化权重
    w1 = orthogonal_init_with_spectral_radius((8, 4), target_rho=1.5, device=device)
    w2 = orthogonal_init_with_spectral_radius((4, 8), target_rho=1.5, device=device)
    model.set_weights(w1, w2)

    # β 加小噪声
    rng = np.random.RandomState(42)
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.normal(0, 0.01, module._beta.shape),
                                 dtype=torch.float32, device=device)
            module._beta.data.add_(noise)
            module._beta.data.clamp_(0.01, 0.999)

    # V_th 保持门电路默认值
    return model


# ============================================================
# 训练函数
# ============================================================
def train_isotropic(model, sequences, labels, device, w1_shape, w2_shape, n_beta, n_vth):
    """Group A: 各向同性训练"""
    print("\n" + "="*60, flush=True)
    print("[Group A] 各向同性 SPSA (Isotropic)", flush=True)
    print(f"  c = {ISO_c}, a = {ISO_a} (所有参数相同)", flush=True)
    print("="*60, flush=True)

    # 初始评估
    beta0, vth0, gamma0, beta_std0, vth_std0 = extract_physics(model)
    rho0 = extract_spectral_radius(model)
    w_mean0, w_std0, w_norm0 = extract_weight_stats(model)
    loss0, acc0, fr0, stats0 = evaluate_model(model, sequences, labels, device, T_USE)
    params_full_0 = extract_physics_full(model)

    print(f"[A] Init: Loss={loss0:.4f}, W={w_mean0:.4f}±{w_std0:.4f}(‖{w_norm0:.2f}‖), "
          f"β={beta0:.4f}±{beta_std0:.4f}, V_th={vth0:.4f}±{vth_std0:.4f}, ρ={rho0:.4f}", flush=True)

    trajectory = {
        'epoch': [0], 'beta': [beta0], 'v_th': [vth0],
        'beta_std': [beta_std0], 'vth_std': [vth_std0],
        'w_mean': [w_mean0], 'w_std': [w_std0], 'w_norm': [w_norm0],
        'loss': [loss0], 'acc': [acc0], 'spectral_radius': [rho0],
        'diff': [0.0], 'firing_rate': [fr0],
        'params_full': [params_full_0],
    }

    momentum_buf = None
    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        a_k = ISO_a / (epoch + 1) ** alpha
        c_k = ISO_c / (epoch + 1) ** gamma_spsa

        loss, acc, diff, fr, momentum_buf, stats = spsa_step_isotropic(
            model, sequences, labels, device, c_k, a_k, T_USE,
            w1_shape, w2_shape, n_beta, n_vth, momentum_buf
        )

        beta_now, vth_now, gamma_now, beta_std_now, vth_std_now = extract_physics(model)
        rho_now = extract_spectral_radius(model)
        w_mean_now, w_std_now, w_norm_now = extract_weight_stats(model)
        params_full_now = extract_physics_full(model)

        trajectory['epoch'].append(epoch)
        trajectory['beta'].append(beta_now)
        trajectory['v_th'].append(vth_now)
        trajectory['beta_std'].append(beta_std_now)
        trajectory['vth_std'].append(vth_std_now)
        trajectory['w_mean'].append(w_mean_now)
        trajectory['w_std'].append(w_std_now)
        trajectory['w_norm'].append(w_norm_now)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)
        trajectory['spectral_radius'].append(rho_now)
        trajectory['diff'].append(diff)
        trajectory['firing_rate'].append(fr)
        trajectory['params_full'].append(params_full_now)

        if epoch % 10 == 0 or epoch <= 5:
            elapsed = time.time() - t_start
            print(f"[A] Ep {epoch:3d}: Loss={loss:.4f}, Acc={acc:.2f}, "
                  f"W={w_mean_now:.4f}±{w_std_now:.4f}(‖{w_norm_now:.2f}‖), "
                  f"β={beta_now:.4f}±{beta_std_now:.4f}, V_th={vth_now:.4f}±{vth_std_now:.4f}, "
                  f"ρ={rho_now:.4f}, fr={fr:.3f}, diff={diff:.2e} [{elapsed:.0f}s]", flush=True)

    return trajectory


def train_anisotropic(model, sequences, labels, device, w1_shape, w2_shape, n_beta, n_vth):
    """Group B: 各向异性训练 (多尺度并发演化)"""
    print("\n" + "="*60, flush=True)
    print("[Group B] 各向异性 SPSA (Anisotropic) - 多尺度并发演化", flush=True)
    print(f"  c_W={ANISO_c_W}, c_β={ANISO_c_beta}, c_Vth={ANISO_c_vth}", flush=True)
    print(f"  a_W={ANISO_a_W}, a_β={ANISO_a_beta}, a_Vth={ANISO_a_vth}", flush=True)
    print("="*60, flush=True)

    # 初始评估
    beta0, vth0, gamma0, beta_std0, vth_std0 = extract_physics(model)
    rho0 = extract_spectral_radius(model)
    w_mean0, w_std0, w_norm0 = extract_weight_stats(model)
    loss0, acc0, fr0, stats0 = evaluate_model(model, sequences, labels, device, T_USE)
    params_full_0 = extract_physics_full(model)

    print(f"[B] Init: Loss={loss0:.4f}, Acc={acc0:.2f}, "
          f"W={w_mean0:.4f}±{w_std0:.4f}(‖{w_norm0:.2f}‖), "
          f"β={beta0:.4f}±{beta_std0:.4f}, V_th={vth0:.4f}±{vth_std0:.4f}, ρ={rho0:.4f}", flush=True)

    trajectory = {
        'epoch': [0], 'beta': [beta0], 'v_th': [vth0],
        'beta_std': [beta_std0], 'vth_std': [vth_std0],
        'w_mean': [w_mean0], 'w_std': [w_std0], 'w_norm': [w_norm0],
        'loss': [loss0], 'acc': [acc0], 'spectral_radius': [rho0],
        'diff': [0.0], 'firing_rate': [fr0],
        'params_full': [params_full_0],
    }

    momentum_W, momentum_beta, momentum_vth = None, None, None
    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        # 各自衰减
        c_W_k = ANISO_c_W / (epoch + 1) ** gamma_spsa
        c_beta_k = ANISO_c_beta / (epoch + 1) ** gamma_spsa
        c_vth_k = ANISO_c_vth / (epoch + 1) ** gamma_spsa

        a_W_k = ANISO_a_W / (epoch + 1) ** alpha
        a_beta_k = ANISO_a_beta / (epoch + 1) ** alpha
        a_vth_k = ANISO_a_vth / (epoch + 1) ** alpha

        loss, acc, diff, fr, momentum_W, momentum_beta, momentum_vth, stats = spsa_step_anisotropic(
            model, sequences, labels, device,
            c_W_k, c_beta_k, c_vth_k, a_W_k, a_beta_k, a_vth_k, T_USE,
            w1_shape, w2_shape, n_beta, n_vth,
            momentum_W, momentum_beta, momentum_vth
        )

        beta_now, vth_now, gamma_now, beta_std_now, vth_std_now = extract_physics(model)
        rho_now = extract_spectral_radius(model)
        w_mean_now, w_std_now, w_norm_now = extract_weight_stats(model)
        params_full_now = extract_physics_full(model)

        trajectory['epoch'].append(epoch)
        trajectory['beta'].append(beta_now)
        trajectory['v_th'].append(vth_now)
        trajectory['beta_std'].append(beta_std_now)
        trajectory['vth_std'].append(vth_std_now)
        trajectory['w_mean'].append(w_mean_now)
        trajectory['w_std'].append(w_std_now)
        trajectory['w_norm'].append(w_norm_now)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)
        trajectory['spectral_radius'].append(rho_now)
        trajectory['diff'].append(diff)
        trajectory['firing_rate'].append(fr)
        trajectory['params_full'].append(params_full_now)

        if epoch % 10 == 0 or epoch <= 5:
            elapsed = time.time() - t_start
            print(f"[B] Ep {epoch:3d}: Loss={loss:.4f}, Acc={acc:.2f}, "
                  f"W={w_mean_now:.4f}±{w_std_now:.4f}(‖{w_norm_now:.2f}‖), "
                  f"β={beta_now:.4f}±{beta_std_now:.4f}, V_th={vth_now:.4f}±{vth_std_now:.4f}, "
                  f"ρ={rho_now:.4f}, fr={fr:.3f}, diff={diff:.2e} [{elapsed:.0f}s]", flush=True)

    return trajectory


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*70, flush=True)
    print("实验 13: 多尺度并发演化与量纲匹配", flush=True)
    print("(Multi-Scale Simultaneous Evolution & Dimensional Matching)", flush=True)
    print("="*70, flush=True)

    # 生成数据
    sequences, labels = generate_temporal_sequences(n_samples=8, seq_len=T_USE, n_features=4)
    sequences = sequences.to(DEVICE)
    labels = labels.to(DEVICE)
    print(f"[data] {len(sequences)} sequences, T_use={T_USE}", flush=True)

    w1_shape = (8, 4)
    w2_shape = (4, 8)

    # 创建模型 A
    model_A = create_model(DEVICE)
    all_beta, all_vth = _get_lif_params(model_A)
    n_beta = len(all_beta)
    n_vth = len(all_vth)
    print(f"[params] W: {w1_shape[0]*w1_shape[1] + w2_shape[0]*w2_shape[1]}, β: {n_beta}, V_th: {n_vth}", flush=True)

    # 保存初始状态
    init_w1, init_w2 = _get_weight_floats(model_A)
    init_beta = all_beta.clone()
    init_vth = all_vth.clone()

    # Group A: 各向同性
    trajectory_A = train_isotropic(model_A, sequences, labels, DEVICE,
                                    w1_shape, w2_shape, n_beta, n_vth)

    # 创建模型 B (相同初始状态)
    model_B = create_model(DEVICE)
    model_B.set_weights(init_w1.clone(), init_w2.clone())
    _set_lif_params(model_B, init_beta.clone(), init_vth.clone())

    # Group B: 各向异性
    trajectory_B = train_anisotropic(model_B, sequences, labels, DEVICE,
                                      w1_shape, w2_shape, n_beta, n_vth)

    # ============================================================
    # 保存结果
    # ============================================================
    results = {
        'config': {
            'N_EPOCHS': N_EPOCHS,
            'T_USE': T_USE,
            'isotropic': {'c': ISO_c, 'a': ISO_a},
            'anisotropic': {
                'c_W': ANISO_c_W, 'c_beta': ANISO_c_beta, 'c_vth': ANISO_c_vth,
                'a_W': ANISO_a_W, 'a_beta': ANISO_a_beta, 'a_vth': ANISO_a_vth,
            },
            'alpha': alpha,
            'gamma_spsa': gamma_spsa,
        },
        'Group_A_Isotropic': trajectory_A,
        'Group_B_Anisotropic': trajectory_B,
        'summary': {
            'Group_A': {
                'initial_beta': trajectory_A['beta'][0],
                'final_beta': trajectory_A['beta'][-1],
                'beta_change': trajectory_A['beta'][-1] - trajectory_A['beta'][0],
                'initial_loss': trajectory_A['loss'][0],
                'final_loss': trajectory_A['loss'][-1],
            },
            'Group_B': {
                'initial_beta': trajectory_B['beta'][0],
                'final_beta': trajectory_B['beta'][-1],
                'beta_change': trajectory_B['beta'][-1] - trajectory_B['beta'][0],
                'initial_loss': trajectory_B['loss'][0],
                'final_loss': trajectory_B['loss'][-1],
            }
        }
    }

    json_path = '/Users/tangzhengzheng/Desktop/NEXUS/experiments/exp13_orthogonal_subspace_data.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] JSON saved to {json_path}", flush=True)

    # ============================================================
    # 可视化
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs_A = trajectory_A['epoch']
    epochs_B = trajectory_B['epoch']

    # 1. β 演化对比
    ax1 = axes[0, 0]
    ax1.plot(epochs_A, trajectory_A['beta'], 'b-', label='Group A (Isotropic)', linewidth=2)
    ax1.plot(epochs_B, trajectory_B['beta'], 'r-', label='Group B (Anisotropic)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('β (mean)')
    ax1.set_title('β Evolution (Core Metric)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Loss 对比
    ax2 = axes[0, 1]
    ax2.plot(epochs_A, trajectory_A['loss'], 'b-', label='Group A', linewidth=2)
    ax2.plot(epochs_B, trajectory_B['loss'], 'r-', label='Group B', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 3. V_th 演化
    ax3 = axes[0, 2]
    ax3.plot(epochs_A, trajectory_A['v_th'], 'b-', label='Group A', linewidth=2)
    ax3.plot(epochs_B, trajectory_B['v_th'], 'r-', label='Group B', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('V_th (mean)')
    ax3.set_title('V_th Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. SPSA Diff
    ax4 = axes[1, 0]
    ax4.plot(epochs_A, np.abs(trajectory_A['diff']), 'b-', alpha=0.7, label='Group A |diff|')
    ax4.plot(epochs_B, np.abs(trajectory_B['diff']), 'r-', alpha=0.7, label='Group B |diff|')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('|L+ - L-|')
    ax4.set_title('SPSA Diff (Gradient Signal)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # 5. 谱半径
    ax5 = axes[1, 1]
    ax5.plot(epochs_A, trajectory_A['spectral_radius'], 'b-', label='Group A', linewidth=2)
    ax5.plot(epochs_B, trajectory_B['spectral_radius'], 'r-', label='Group B', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('ρ(W)')
    ax5.set_title('Spectral Radius')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. β 分化程度
    ax6 = axes[1, 2]
    ax6.plot(epochs_A, trajectory_A['beta_std'], 'b-', label='Group A β_std', linewidth=2)
    ax6.plot(epochs_B, trajectory_B['beta_std'], 'r-', label='Group B β_std', linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('β std')
    ax6.set_title('β Differentiation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Experiment 13: Multi-Scale Simultaneous Evolution\n'
                 'Isotropic (same c,a) vs Anisotropic (dimension-matched c,a)', fontsize=14)
    plt.tight_layout()

    fig_path = '/Users/tangzhengzheng/Desktop/NEXUS/experiments/exp13_orthogonal_subspace_results.png'
    plt.savefig(fig_path, dpi=150)
    print(f"[save] Figure saved to {fig_path}", flush=True)

    # ============================================================
    # 打印总结
    # ============================================================
    print("\n" + "="*70, flush=True)
    print("实验 13 总结: 多尺度并发演化", flush=True)
    print("="*70, flush=True)

    print(f"\n{'指标':<20} {'Group A (Isotropic)':<25} {'Group B (Anisotropic)':<25}")
    print("-"*70)
    print(f"{'Initial β':<20} {trajectory_A['beta'][0]:<25.6f} {trajectory_B['beta'][0]:<25.6f}")
    print(f"{'Final β':<20} {trajectory_A['beta'][-1]:<25.6f} {trajectory_B['beta'][-1]:<25.6f}")
    print(f"{'β Change':<20} {trajectory_A['beta'][-1] - trajectory_A['beta'][0]:<25.6f} {trajectory_B['beta'][-1] - trajectory_B['beta'][0]:<25.6f}")
    print(f"{'Initial V_th':<20} {trajectory_A['v_th'][0]:<25.4f} {trajectory_B['v_th'][0]:<25.4f}")
    print(f"{'Final V_th':<20} {trajectory_A['v_th'][-1]:<25.4f} {trajectory_B['v_th'][-1]:<25.4f}")
    print(f"{'Initial Loss':<20} {trajectory_A['loss'][0]:<25.4f} {trajectory_B['loss'][0]:<25.4f}")
    print(f"{'Final Loss':<20} {trajectory_A['loss'][-1]:<25.4f} {trajectory_B['loss'][-1]:<25.4f}")
    loss_improve_A = (1 - trajectory_A['loss'][-1]/trajectory_A['loss'][0]) * 100
    loss_improve_B = (1 - trajectory_B['loss'][-1]/trajectory_B['loss'][0]) * 100
    print(f"{'Loss Improve %':<20} {loss_improve_A:<25.2f} {loss_improve_B:<25.2f}")
    print(f"{'Final ρ(W)':<20} {trajectory_A['spectral_radius'][-1]:<25.4f} {trajectory_B['spectral_radius'][-1]:<25.4f}")

    print("\n[分析] β 逃逸速度:")
    beta_velocity_A = (trajectory_A['beta'][-1] - trajectory_A['beta'][0]) / N_EPOCHS
    beta_velocity_B = (trajectory_B['beta'][-1] - trajectory_B['beta'][0]) / N_EPOCHS
    print(f"  Group A (Isotropic):   {beta_velocity_A:.8f} /epoch")
    print(f"  Group B (Anisotropic): {beta_velocity_B:.8f} /epoch")
    if abs(beta_velocity_A) > 1e-10:
        print(f"  => Anisotropic / Isotropic 比值: {beta_velocity_B/beta_velocity_A:.2f}x")

    print("\n[完成]", flush=True)


if __name__ == "__main__":
    main()
