"""
实验 11: 全参数自组织演化与临界态涌现
(Self-Organization & Emergence of Criticality via Zero-Order Optimization)
================================================================================

物理动机：
- 将 NEXUS 从"平庸/混乱"初始态，通过纯任务 Loss（无物理正则项）驱动的零阶优化，
  观察系统是否自发演化到"临界同步区"（β≈0.9, γ≈0.1）
- 如果成功，证明临界计算态是任务最优解，不是手动调参的人工产物

KEY CORRECTION:
- OLD: Loss = intra_var / (inter_dist + eps) based on decoder output centroids (WRONG)
- NEW: Loss = MSE(decoded_output, anchor_target) with anchors [+1,0,0,0] / [-1,0,0,0]
- Evaluation uses TEMPORAL mode: feed T timesteps sequentially, decode LAST timestep
- Centroid separation is now an OBSERVATION metric, not the training loss

训练对象：SimpleSpikeMLP（2层 SpikeFP32Linear）
优化方式：SPSA 直接作用于 model.parameters()（所有 _beta, _v_threshold 向量）
评估方式：TEMPORAL mode → decoder MSE loss + anchor classification accuracy
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import copy

print("[import] torch, numpy done", flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
print("[import] matplotlib done", flush=True)

from atomic_ops import (
    SpikeMode,
    SpikeFP32Linear_MultiPrecision,
    SimpleLIFNode,
)
from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32
print("[import] atomic_ops done", flush=True)


# =============================================================================
# Model
# =============================================================================

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


# =============================================================================
# 全参数展平/写回工具（W + β + V_th 联合）
# =============================================================================
# 关键设计：
#   - β, V_th 是 nn.Parameter，可以直接读写 .data
#   - weight_pulse 是 bool buffer（非 Parameter），需要 pulse↔float 转换
#   - 展平顺序：[W1_float, W2_float, β_all, V_th_all]

def _get_weight_floats(model):
    """从模型提取权重的浮点表示（pulse→float 边界解码）"""
    w1_pulse = model.linear1.weight_pulse  # [out, in, 32] float from bool
    w2_pulse = model.linear2.weight_pulse
    w1_float = pulse_to_float32(w1_pulse.float())  # [out, in]
    w2_float = pulse_to_float32(w2_pulse.float())
    return w1_float, w2_float


def _set_weight_floats(model, w1_float, w2_float):
    """将浮点权重编码回脉冲写入模型（float→pulse 边界编码）"""
    model.linear1.set_weight_from_float(w1_float)
    model.linear2.set_weight_from_float(w2_float)


def _get_lif_params(model):
    """提取所有 LIF 的 β 和 V_th"""
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.flatten())
            vths.append(module._v_threshold.data.flatten())
    return torch.cat(betas), torch.cat(vths)


def _set_lif_params(model, beta_flat, vth_flat):
    """将扁平化的 β/V_th 写回 LIF 模块"""
    b_idx, v_idx = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            bn = module._beta.numel()
            module._beta.data.copy_(beta_flat[b_idx:b_idx+bn].reshape(module._beta.shape))
            b_idx += bn
            vn = module._v_threshold.numel()
            module._v_threshold.data.copy_(vth_flat[v_idx:v_idx+vn].reshape(module._v_threshold.shape))
            v_idx += vn


def params_to_flat(model):
    """全参数展平：W1_float + W2_float + β + V_th → 一维向量"""
    w1_float, w2_float = _get_weight_floats(model)
    beta_flat, vth_flat = _get_lif_params(model)
    return torch.cat([w1_float.flatten(), w2_float.flatten(),
                      beta_flat, vth_flat])


def flat_to_params(model, flat):
    """全参数写回：一维向量 → 模型（权重经过 float→pulse 编码）"""
    w1_float, w2_float = _get_weight_floats(model)
    w1_shape = w1_float.shape
    w2_shape = w2_float.shape
    beta_flat, vth_flat = _get_lif_params(model)
    n_beta = beta_flat.numel()
    n_vth = vth_flat.numel()

    idx = 0
    n_w1 = w1_shape[0] * w1_shape[1]
    w1_new = flat[idx:idx+n_w1].reshape(w1_shape)
    idx += n_w1
    n_w2 = w2_shape[0] * w2_shape[1]
    w2_new = flat[idx:idx+n_w2].reshape(w2_shape)
    idx += n_w2
    beta_new = flat[idx:idx+n_beta]
    idx += n_beta
    vth_new = flat[idx:idx+n_vth]

    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params(model, beta_new, vth_new)


def count_params(model):
    """统计全部可演化参数数量（W + β + V_th）"""
    w1_float, w2_float = _get_weight_floats(model)
    beta_flat, vth_flat = _get_lif_params(model)
    return w1_float.numel() + w2_float.numel() + beta_flat.numel() + vth_flat.numel()


# =============================================================================
# 物理量提取（观测用，不参与训练）
# =============================================================================

def extract_physics(model):
    """从模型中提取物理量均值：beta、v_threshold"""
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.mean().item())
            vths.append(module._v_threshold.data.mean().item())
    avg_beta = np.mean(betas) if betas else 0.5
    avg_vth = np.mean(vths) if vths else 1.0
    gamma = 1.0 / avg_vth if avg_vth > 0 else float('inf')
    return avg_beta, avg_vth, gamma


def extract_spectral_radius(model):
    """提取 linear1 权重的谱半径（从脉冲权重解码回浮点后计算）"""
    try:
        # linear1 的权重脉冲矩阵: [out, in, 32]
        w_pulse = None
        if hasattr(model.linear1, 'weight_pulse'):
            w_pulse = model.linear1.weight_pulse
        elif hasattr(model.linear1, '_weight_pulse'):
            w_pulse = model.linear1._weight_pulse

        if w_pulse is not None:
            w_float = pulse_to_float32(w_pulse.float())
            svs = torch.linalg.svdvals(w_float)
            return svs[0].item()
    except:
        pass
    return 0.0


# =============================================================================
# Anchor Targets (NEW)
# =============================================================================
# Class 0 → [+1, 0, 0, 0]
# Class 1 → [-1, 0, 0, 0]

ANCHOR_0 = torch.tensor([+1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
ANCHOR_1 = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


# =============================================================================
# 螺旋数据生成（同实验 9）
# =============================================================================

def generate_spiral_sequences(n_samples_per_class=20, T=100, noise_std=0.15, seed=42):
    rng = np.random.RandomState(seed)
    sequences, labels = [], []
    for cls in range(2):
        for i in range(n_samples_per_class):
            phase_offset = rng.uniform(0, 0.3)
            t = np.linspace(0, 4 * np.pi, T) + phase_offset
            r = 0.5 + t / (4 * np.pi) * 1.5
            if cls == 0:
                x = r * np.cos(t)
                y = r * np.sin(t)
            else:
                x = r * np.cos(t + np.pi)
                y = r * np.sin(t + np.pi)
            z = t / (4 * np.pi) * 2.0
            x += rng.randn(T) * noise_std
            y += rng.randn(T) * noise_std
            z += rng.randn(T) * noise_std
            w = rng.randn(T) * 0.5
            seq = np.stack([x, y, z, w], axis=-1).astype(np.float32)
            sequences.append(seq)
            labels.append(cls)
    return sequences, labels


# =============================================================================
# 评估函数 (NEW) - TEMPORAL mode + Decoder-based MSE loss
# =============================================================================

def evaluate_model_decoder(model, sequences, labels, device, T_use=80):
    """
    在 TEMPORAL 模式下评估分类（正确架构）：
    - 每个序列：reset model → 逐步 feed T timesteps → decode LAST timestep output
    - Loss = MSE(decoded_output, anchor_target)
    - Accuracy = nearest-anchor classification

    返回：(loss, accuracy, firing_rate, decoder_stats)
    """
    model.eval()

    anchor_0 = ANCHOR_0.to(device)
    anchor_1 = ANCHOR_1.to(device)

    total_loss = 0.0
    correct = 0
    total_spikes = 0
    total_neurons = 0
    decoder_nans = 0
    decoder_infs = 0

    with torch.no_grad():
        with SpikeMode.temporal():  # TEMPORAL mode for sequence processing
            for seq_np, label in zip(sequences, labels):
                # Reset model for new sequence
                model.reset()

                seq = torch.from_numpy(seq_np[:T_use]).to(device)

                # Feed T timesteps one-by-one
                for t in range(T_use):
                    x_t = seq[t:t+1, :]  # [1, 4]
                    x_pulse = float32_to_pulse(x_t)  # [1, 4, 32]
                    y_pulse = model(x_pulse)  # [1, out_f, 32]

                    # Track firing rate
                    total_spikes += y_pulse.sum().item()
                    total_neurons += y_pulse.numel()

                # Decode LAST timestep output
                y_decoded = pulse_to_float32(y_pulse)  # [1, out_f]
                y_decoded = y_decoded.squeeze(0)  # [out_f]

                # NaN handling
                has_nan = torch.isnan(y_decoded).any().item()
                has_inf = torch.isinf(y_decoded).any().item()
                if has_nan:
                    decoder_nans += 1
                if has_inf:
                    decoder_infs += 1

                y_decoded = torch.nan_to_num(y_decoded, nan=1e6, posinf=1e6, neginf=-1e6)

                # Get target anchor
                target = anchor_0 if label == 0 else anchor_1

                # Compute MSE loss
                seq_loss = torch.mean((y_decoded - target) ** 2).item()
                seq_loss = min(seq_loss, 1e12)  # Clamp to avoid overflow
                total_loss += seq_loss

                # Compute accuracy (nearest anchor)
                dist_0 = torch.sum((y_decoded - anchor_0) ** 2).item()
                dist_1 = torch.sum((y_decoded - anchor_1) ** 2).item()
                pred = 0 if dist_0 < dist_1 else 1
                if pred == label:
                    correct += 1

    n_sequences = len(sequences)
    avg_loss = total_loss / n_sequences
    accuracy = correct / n_sequences
    firing_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0

    decoder_stats = {
        'nans': decoder_nans,
        'infs': decoder_infs,
        'total': n_sequences,
    }

    return avg_loss, accuracy, firing_rate, decoder_stats


def observe_centroid_separation(model, sequences, labels, device, T_use=80):
    """
    OBSERVATION ONLY: 收集 LIF 膜电位，计算类间质心距离
    不用于训练，仅用于监控物理动力学

    返回：centroid_separation (inter-class distance in V space)
    """
    model.eval()

    # Collect all LIF V states after running sequences
    class_0_states = []
    class_1_states = []

    with torch.no_grad():
        with SpikeMode.temporal():  # TEMPORAL mode to accumulate V
            for seq_np, label in zip(sequences, labels):
                # Reset model for new sequence
                model.reset()

                seq = torch.from_numpy(seq_np[:T_use]).to(device)

                # Feed T timesteps
                for t in range(T_use):
                    x_t = seq[t:t+1, :]
                    x_pulse = float32_to_pulse(x_t)
                    _ = model(x_pulse)

                # Collect all LIF V potentials
                v_states = []
                for name, module in model.named_modules():
                    if isinstance(module, SimpleLIFNode):
                        if module.v is not None:
                            v_states.append(module.v.detach().flatten().cpu().numpy())

                if v_states:
                    v_concat = np.concatenate(v_states)
                    v_concat = np.nan_to_num(v_concat, nan=0.0, posinf=1e6, neginf=-1e6)
                    if label == 0:
                        class_0_states.append(v_concat)
                    else:
                        class_1_states.append(v_concat)

    # Compute centroids
    if len(class_0_states) == 0 or len(class_1_states) == 0:
        return 0.0

    centroid_0 = np.mean(class_0_states, axis=0)
    centroid_1 = np.mean(class_1_states, axis=0)

    # Euclidean distance between centroids
    separation = np.linalg.norm(centroid_0 - centroid_1)

    return float(separation)


def evaluate_sync_error(model, device, in_f=4, n_trials=3, T=10):
    """双胞胎一致性测试：用略有不同的输入驱动同一模型，比较输出差异。BIT_EXACT模式。"""
    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)
    errors = []

    with torch.no_grad():
        for trial in range(n_trials):
            torch.manual_seed(200 + trial)
            x = torch.randn(1, in_f, device=device) * 0.5
            x_pulse = float32_to_pulse(x)
            y1 = model(x_pulse)
            out1 = pulse_to_float32(y1).detach().cpu().flatten().float()

            noise = torch.randn(1, in_f, device=device) * 0.01
            x2 = x + noise
            x2_pulse = float32_to_pulse(x2)
            y2 = model(x2_pulse)
            out2 = pulse_to_float32(y2).detach().cpu().flatten().float()

            out1 = torch.nan_to_num(out1, nan=0.0, posinf=1e6, neginf=-1e6)
            out2 = torch.nan_to_num(out2, nan=0.0, posinf=1e6, neginf=-1e6)
            err = torch.mean((out1 - out2)**2).item()
            errors.append(err)

    return np.mean(errors) if errors else 0.0


# =============================================================================
# SPSA 零阶优化器（直接操作 model.parameters()）
# =============================================================================

def spsa_step(model, sequences, labels, device, c_k, a_k, T_use=80):
    """
    SPSA 单步：直接对 model.parameters() 展平向量做扰动。
    使用 evaluate_model_decoder 返回 4 个值：(loss, acc, firing_rate, decoder_stats)
    """
    flat = params_to_flat(model)
    n_params = flat.shape[0]

    # 伯努利随机扰动
    delta = (torch.bernoulli(torch.ones(n_params) * 0.5) * 2 - 1).to(device)

    # 保存原始参数
    orig_flat = flat.clone()

    # 正向扰动评估
    flat_to_params(model, orig_flat + c_k * delta)
    loss_plus, acc_plus, fr_plus, stats_plus = evaluate_model_decoder(model, sequences, labels, device, T_use)

    # 负向扰动评估
    flat_to_params(model, orig_flat - c_k * delta)
    loss_minus, acc_minus, fr_minus, stats_minus = evaluate_model_decoder(model, sequences, labels, device, T_use)

    # 梯度估计
    diff = loss_plus - loss_minus
    if abs(diff) < 1e-12:
        flat_to_params(model, orig_flat)
        loss, acc = loss_plus, acc_plus
        print(f"    [SPSA] diff=0, L+={loss_plus:.6f}, L-={loss_minus:.6f}, skip", flush=True)
        return loss, acc

    g_hat = diff / (2 * c_k) * delta
    # 逐元素 clamp（避免全局 norm 在高维下稀释更新）
    g_hat = g_hat.clamp(-10.0, 10.0)
    g_norm = torch.norm(g_hat).item()
    update = a_k * g_hat
    update_abs = update.abs()
    print(f"    [SPSA] diff={diff:.6f}, |g|={g_norm:.4f}, c={c_k:.4f}, a={a_k:.6f}", flush=True)
    print(f"    [SPSA] update: mean={update_abs.mean():.8f}, max={update_abs.max():.8f}", flush=True)

    # 更新
    new_flat = orig_flat - update
    flat_to_params(model, new_flat)

    # 评估当前
    loss, acc, fr, stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
    return loss, acc


# =============================================================================
# 主实验
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}", flush=True)

    # 生成数据
    sequences, labels = generate_spiral_sequences(n_samples_per_class=4, T=100, seed=42)
    T_use = 10  # 极短序列加速
    print(f"[data] {len(sequences)} sequences, T_use={T_use}", flush=True)

    # 构建模型（初始 β=0.5, V_th=1.0）
    torch.manual_seed(42)
    template = SimpleLIFNode(beta=0.5)
    model = SimpleSpikeMLP(4, 8, 4, neuron_template=template).to(device)

    # 初始化权重
    w1 = torch.randn(8, 4, device=device) * 0.5
    w2 = torch.randn(4, 8, device=device) * 0.5
    model.set_weights(w1, w2)

    n_total = count_params(model)
    beta0, vth0, gamma0 = extract_physics(model)
    rho0 = extract_spectral_radius(model)
    print(f"[init] {n_total} params, beta={beta0:.4f}, V_th={vth0:.4f}, "
          f"gamma={gamma0:.4f}, rho={rho0:.4f}", flush=True)

    # =========================================================================
    # Phase 1: 初始评估
    # =========================================================================
    print("[eval] Initial evaluation...", flush=True)
    loss0, acc0, fr0, stats0 = evaluate_model_decoder(model, sequences, labels, device, T_use)
    sync0 = evaluate_sync_error(model, device)
    centroid0 = observe_centroid_separation(model, sequences, labels, device, T_use)
    print(f"[initial] loss={loss0:.6f}, acc={acc0:.4f}, sync_err={sync0:.6f}, "
          f"centroid_sep={centroid0:.4f}", flush=True)

    # =========================================================================
    # Phase 2: 鲁棒性平台扫描 — β/V_th 扰动对输出的影响
    # =========================================================================
    print("\n[Phase 2] Robustness plateau scan...", flush=True)

    # 保存原始参数
    orig_flat = params_to_flat(model)

    # 扫描不同扰动幅度
    perturbation_scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    n_trials_per_scale = 5
    plateau_results = []

    t_start = time.time()

    for c_val in perturbation_scales:
        losses_at_scale = []
        accs_at_scale = []
        for trial in range(n_trials_per_scale):
            # 随机 Bernoulli 扰动
            delta = (torch.bernoulli(torch.ones(orig_flat.shape[0]) * 0.5) * 2 - 1).to(device)
            flat_to_params(model, orig_flat + c_val * delta)
            loss_p, acc_p, fr_p, stats_p = evaluate_model_decoder(model, sequences, labels, device, T_use)
            losses_at_scale.append(loss_p)
            accs_at_scale.append(acc_p)

        # 恢复原始参数
        flat_to_params(model, orig_flat)

        mean_loss = np.mean(losses_at_scale)
        std_loss = np.std(losses_at_scale)
        mean_acc = np.mean(accs_at_scale)
        elapsed = time.time() - t_start
        plateau_results.append({
            'scale': c_val, 'mean_loss': mean_loss, 'std_loss': std_loss,
            'mean_acc': mean_acc, 'losses': losses_at_scale, 'accs': accs_at_scale,
        })
        print(f"  c={c_val:.2f}: loss={mean_loss:.6f}±{std_loss:.6f}, "
              f"acc={mean_acc:.4f}, time={elapsed:.0f}s", flush=True)

    # =========================================================================
    # Phase 3: SPSA 优化（少量 epoch 作为对照）
    # =========================================================================
    print("\n[Phase 3] SPSA optimization (10 epochs)...", flush=True)
    n_epochs = 10
    a0, c0, alpha, gamma_spsa = 0.5, 0.1, 0.602, 0.101

    trajectory = {
        'epoch': [], 'beta': [], 'gamma': [], 'v_th': [],
        'acc': [], 'loss': [], 'sync_error': [], 'spectral_radius': [],
        'centroid_separation': [],  # NEW: observation metric
    }
    trajectory['epoch'].append(0)
    trajectory['beta'].append(beta0)
    trajectory['gamma'].append(gamma0)
    trajectory['v_th'].append(vth0)
    trajectory['acc'].append(acc0)
    trajectory['loss'].append(loss0)
    trajectory['sync_error'].append(sync0)
    trajectory['spectral_radius'].append(rho0)
    trajectory['centroid_separation'].append(centroid0)

    for epoch in range(1, n_epochs + 1):
        a_k = a0 / (epoch + 1) ** alpha
        c_k = c0 / (epoch + 1) ** gamma_spsa
        loss, acc = spsa_step(model, sequences, labels, device, c_k, a_k, T_use)

        beta_now, vth_now, gamma_now = extract_physics(model)
        rho_now = extract_spectral_radius(model)
        sync_err = evaluate_sync_error(model, device) if epoch % 5 == 0 else trajectory['sync_error'][-1]
        centroid_sep = observe_centroid_separation(model, sequences, labels, device, T_use)

        trajectory['epoch'].append(epoch)
        trajectory['beta'].append(beta_now)
        trajectory['gamma'].append(gamma_now)
        trajectory['v_th'].append(vth_now)
        trajectory['acc'].append(acc)
        trajectory['loss'].append(loss)
        trajectory['sync_error'].append(sync_err)
        trajectory['spectral_radius'].append(rho_now)
        trajectory['centroid_separation'].append(centroid_sep)

        elapsed = time.time() - t_start
        print(f"[epoch {epoch:3d}] loss={loss:.6f}, acc={acc:.4f}, "
              f"beta={beta_now:.4f}, V_th={vth_now:.4f}, centroid_sep={centroid_sep:.4f}, "
              f"time={elapsed:.0f}s", flush=True)

    final_loss, final_acc, final_fr, final_stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
    final_sync = evaluate_sync_error(model, device)
    final_beta, final_vth, final_gamma = extract_physics(model)
    final_rho = extract_spectral_radius(model)
    final_centroid = observe_centroid_separation(model, sequences, labels, device, T_use)
    print(f"\n[FINAL] loss={final_loss:.6f}, acc={final_acc:.4f}, "
          f"beta={final_beta:.4f}, V_th={final_vth:.4f}, centroid_sep={final_centroid:.4f}", flush=True)

    # =============================================================================
    # 可视化
    # =============================================================================
    print("\n[Plot] Generating 6-panel figure...", flush=True)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    epochs = trajectory['epoch']
    betas = trajectory['beta']
    gammas = trajectory['gamma']
    accs = trajectory['acc']
    losses = trajectory['loss']
    syncs = trajectory['sync_error']
    rhos = trajectory['spectral_radius']
    vths = trajectory['v_th']
    centroid_seps = trajectory['centroid_separation']

    # --- Panel 1: 鲁棒性平台图 ---
    ax1 = fig.add_subplot(gs[0, 0])
    scales = [r['scale'] for r in plateau_results]
    mean_losses = [r['mean_loss'] for r in plateau_results]
    std_losses = [r['std_loss'] for r in plateau_results]
    mean_accs_p = [r['mean_acc'] for r in plateau_results]
    ax1.errorbar(scales, mean_losses, yerr=std_losses, fmt='bo-', capsize=3, linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(scales, mean_accs_p, 'rs--', linewidth=2, label='Accuracy')
    ax1.set_xscale('log')
    ax1.set_xlabel('Perturbation scale (c)')
    ax1.set_ylabel('Loss (Decoder MSE)', color='blue')
    ax1_twin.set_ylabel('Accuracy', color='red')
    ax1.set_title('Robustness Plateau\n(Loss vs perturbation scale)')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: beta / V_th 演化 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    l1, = ax2.plot(epochs, betas, 'b-o', markersize=3, label='beta', linewidth=2)
    l2, = ax2_twin.plot(epochs, vths, 'r-s', markersize=3, label='V_th', linewidth=2)
    ax2.axhline(y=0.9, color='blue', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('beta (mean)', color='blue')
    ax2_twin.set_ylabel('V_th (mean)', color='red')
    ax2.set_title('Physics Parameters Evolution')
    ax2.legend([l1, l2], ['beta', 'V_th'], fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Accuracy / Loss ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, accs, 'g-o', markersize=3, label='Accuracy', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, losses, 'r-', alpha=0.5, label='Loss (Decoder MSE)', linewidth=1)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy', color='green')
    ax3_twin.set_ylabel('Loss (Decoder MSE)', color='red')
    ax3.set_title('Task Performance')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='lower right')

    # --- Panel 4: gamma 演化 + centroid separation (observation) ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, gammas, 'm-o', markersize=3, linewidth=2, label='gamma (drive ratio)')
    ax4.axhline(y=0.1, color='gold', linestyle='--', linewidth=2, label='target gamma=0.1')
    ax4_twin2 = ax4.twinx()
    ax4_twin2.plot(epochs, centroid_seps, 'c--', markersize=2, linewidth=1.5,
                   label='Centroid Sep (obs)', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('gamma (drive ratio)', color='m')
    ax4_twin2.set_ylabel('Centroid Separation (obs)', color='c')
    ax4.set_title('Drive Ratio + Centroid Separation (Observation)')
    ax4.legend(fontsize=8, loc='upper left')
    ax4_twin2.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: 同步误差 ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, syncs, 'c-o', markersize=3, linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Sync Error (output MSE)')
    ax5.set_title('Twin Synchronization Error\n(decoded output difference)')
    ax5.set_yscale('symlog', linthresh=1e-6)
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: 谱半径 ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, rhos, 'orange', marker='o', markersize=3, linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Spectral Radius (W1)')
    ax6.set_title('Weight Structure Evolution')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f'Experiment 11: Self-Organization with Decoder-based Loss ({n_total} params)\n'
                 f'NEXUS — Loss = MSE(decoded, anchor) | Centroid separation as observation | '
                 f'Initial: acc={acc0:.0%}, Final: acc={final_acc:.0%}',
                 fontsize=13, fontweight='bold', y=0.98)

    out_path = os.path.join(os.path.dirname(__file__), 'self_organization_results.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] Saved to {out_path}", flush=True)

    # =============================================================================
    # 摘要
    # =============================================================================
    print("\n" + "="*70, flush=True)
    print("SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"Total trainable params: {n_total}", flush=True)
    print(f"Initial: beta={beta0:.4f}, V_th={vth0:.4f}, acc={acc0:.4f}", flush=True)
    print(f"Final:   beta={final_beta:.4f}, V_th={final_vth:.4f}, acc={final_acc:.4f}", flush=True)

    print(f"\nKEY CORRECTION: Decoder-based Loss Architecture", flush=True)
    print(f"  OLD (WRONG): Loss = intra_var / (inter_dist + eps) based on decoder centroids", flush=True)
    print(f"  NEW (CORRECT): Loss = MSE(decoded_output, anchor_target)", flush=True)
    print(f"    - Anchors: Class 0 → [+1,0,0,0], Class 1 → [-1,0,0,0]", flush=True)
    print(f"    - Evaluation: TEMPORAL mode, feed T timesteps, decode LAST timestep", flush=True)
    print(f"    - Centroid separation: OBSERVATION metric only, not training loss", flush=True)

    print(f"\nRobustness Plateau Results:", flush=True)
    for r in plateau_results:
        print(f"  c={r['scale']:.2f}: loss={r['mean_loss']:.6f}±{r['std_loss']:.6f}, "
              f"acc={r['mean_acc']:.4f}", flush=True)

    print(f"\nCentroid Separation Observation:", flush=True)
    print(f"  Initial: {centroid0:.4f}", flush=True)
    print(f"  Final:   {final_centroid:.4f}", flush=True)
    print(f"  (This is NOT the training loss, just a physical dynamic observation)", flush=True)

    print(f"\n[DONE] Experiment 11 complete.", flush=True)


if __name__ == '__main__':
    main()
