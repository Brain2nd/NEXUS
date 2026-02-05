"""
实验 9: 时序流形解缠与核质量测定 (Temporal Manifold Untangling & Kernel Quality)
================================================================================

物理动机：
- 未经训练的深层网络可视为核函数，将低维线性不可分输入投射到高维状态空间
- 好的投影：把纠缠的不同类别轨迹"拉开"（解缠），使其线性可分
- 坏的投影：轨迹依然纠缠，或完全坍缩到一起

实验设计：
- 输入：多维螺旋分类的时序版（两类三维螺旋线缠绕+噪声）
- Group A: β ≈ 1, V_th=1.0 (线性积分器)
- Group B: β = 0.90, V_th=10.0 (临界同步区/黄金工作区)
- Group C: β = 0.90, V_th=1.0 (湍流区)
- 额外: β = 0.50, V_th=10.0 (强折叠区)
- 额外: β = 0.99, V_th=10.0 (弱折叠区)

关键改进：
- PRIMARY 指标：Decoder 端到端性能（TEMPORAL 模式下的输出脉冲解码 → MSE vs anchor）
- OBSERVATION 指标：V-space 核质量、流形分离度（用于理解动力学）
- Anchor targets: ANCHOR_0 = [+1, 0, 0, 0], ANCHOR_1 = [-1, 0, 0, 0]
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np

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
# Model (与实验 7/8 完全一致)
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


def create_model(device, beta, v_threshold=1.0, seed=42):
    torch.manual_seed(seed)
    in_f, hid_f, out_f = 4, 8, 4
    template = SimpleLIFNode(beta=beta)
    model = SimpleSpikeMLP(in_f, hid_f, out_f, neuron_template=template).to(device)
    w1 = torch.randn(hid_f, in_f, device=device) * 0.5
    w2 = torch.randn(out_f, hid_f, device=device) * 0.5
    model.set_weights(w1, w2)
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module.v_threshold = v_threshold
    return model, in_f


# =============================================================================
# 螺旋数据生成
# =============================================================================

def generate_spiral_sequences(n_samples_per_class=20, T=100, noise_std=0.15, seed=0):
    """
    生成两类三维螺旋线的时序版本。
    每个样本是一段 T 步的 4 维时序信号（3D 螺旋坐标 + 1D 噪声通道）。
    两类螺旋在空间中紧密缠绕但方向相反。

    Returns:
        sequences: list of (T, 4) tensors
        labels: list of int (0 or 1)
    """
    rng = np.random.RandomState(seed)
    sequences = []
    labels = []

    for cls in range(2):
        for i in range(n_samples_per_class):
            # 螺旋参数：两类方向相反，每个样本有微小相位偏移
            phase_offset = rng.uniform(0, 0.3)
            t = np.linspace(0, 4 * np.pi, T) + phase_offset

            # 螺旋半径随时间增大（使两类更加纠缠）
            r = 0.5 + t / (4 * np.pi) * 1.5

            if cls == 0:
                x = r * np.cos(t)
                y = r * np.sin(t)
                z = t / (4 * np.pi) * 2.0
            else:
                x = r * np.cos(t + np.pi)  # 180度旋转
                y = r * np.sin(t + np.pi)
                z = t / (4 * np.pi) * 2.0

            # 添加噪声
            x += rng.randn(T) * noise_std
            y += rng.randn(T) * noise_std
            z += rng.randn(T) * noise_std

            # 第4通道：噪声（干扰通道）
            w = rng.randn(T) * 0.5

            seq = np.stack([x, y, z, w], axis=-1).astype(np.float32)  # (T, 4)
            sequences.append(seq)
            labels.append(cls)

    return sequences, labels


# =============================================================================
# Anchor targets
# =============================================================================

# Class 0: [+1, 0, 0, 0]
# Class 1: [-1, 0, 0, 0]
ANCHOR_0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
ANCHOR_1 = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


# =============================================================================
# PRIMARY 指标：Decoder 端到端性能
# =============================================================================

def evaluate_decoder_performance(model, sequences, labels, device, T_use=80):
    """
    PRIMARY 评估指标：Decoder 端到端性能（TEMPORAL 模式）。

    对每个序列：
    1. reset() 清空状态
    2. TEMPORAL 模式运行 T_use 步
    3. 解码最后一步的输出脉冲 → 浮点向量
    4. 与 anchor target 比较 MSE
    5. 计算分类准确率（距离最近的 anchor）

    Returns:
        dict: {
            'loss': float,         # 平均 MSE
            'accuracy': float,     # 分类准确率
            'firing_rate': float,  # 平均发放率
            'nan_ratio': float,    # NaN 比例
        }
    """
    ANCHOR_0_dev = ANCHOR_0.to(device)
    ANCHOR_1_dev = ANCHOR_1.to(device)

    total_loss = 0.0
    correct = 0
    total_samples = len(sequences)
    total_spikes = 0
    total_nan = 0
    total_elements = 0

    for seq_np, label in zip(sequences, labels):
        seq = torch.from_numpy(seq_np).to(device)  # (T_total, 4)
        T_total = seq.shape[0]
        T_run = min(T_use, T_total)

        # 选择 anchor
        target = ANCHOR_0_dev if label == 0 else ANCHOR_1_dev

        model.reset()

        with torch.no_grad():
            # TEMPORAL 模式运行 T_run 步
            output_pulse = None
            for t in range(T_run):
                x_t = seq[t:t+1, :]  # (1, 4)
                x_pulse = float32_to_pulse(x_t)
                output_pulse = model(x_pulse)  # (1, 32*4)

            # 解码最后一步的输出脉冲
            if output_pulse is not None:
                output_float = pulse_to_float32(output_pulse)  # (1, 4)
                output_float = output_float.squeeze(0)  # (4,)
            else:
                output_float = torch.zeros(4, device=device)

            # 计算 MSE
            mse = torch.mean((output_float - target) ** 2).item()
            total_loss += mse

            # 计算分类准确率（距离最近的 anchor）
            dist_0 = torch.sum((output_float - ANCHOR_0_dev) ** 2).item()
            dist_1 = torch.sum((output_float - ANCHOR_1_dev) ** 2).item()
            pred = 0 if dist_0 < dist_1 else 1
            if pred == label:
                correct += 1

            # 统计发放率和 NaN
            if output_pulse is not None:
                total_spikes += torch.sum(output_pulse > 0.5).item()
                total_elements += output_pulse.numel()
                total_nan += torch.isnan(output_pulse).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    firing_rate = total_spikes / total_elements if total_elements > 0 else 0.0
    nan_ratio = total_nan / total_elements if total_elements > 0 else 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'firing_rate': firing_rate,
        'nan_ratio': nan_ratio,
    }


# =============================================================================
# OBSERVATION 指标：V-space 状态收集 & 核质量计算
# =============================================================================

def collect_state_trajectories(model, sequences, device, warmup=10):
    """
    OBSERVATION 指标：收集 V-space 状态轨迹（读取 module.v）。

    对每个输入序列，在 TEMPORAL 模式下前向传播，收集所有 SimpleLIFNode 的膜电位。
    返回每个序列的平均状态向量（时间平均后的表示）。
    """
    all_states = []

    for seq_np in sequences:
        seq = torch.from_numpy(seq_np).to(device)  # (T, 4)
        T_total = seq.shape[0]

        model.reset()

        # 收集膜电位
        v_history = []

        with torch.no_grad():
            for t in range(T_total):
                x_t = seq[t:t+1, :]  # (1, 4)
                x_pulse = float32_to_pulse(x_t)
                _ = model(x_pulse)

                # 收集膜电位
                if t >= warmup:
                    v_snap = []
                    for name, module in model.named_modules():
                        if isinstance(module, SimpleLIFNode) and module.v is not None:
                            v_flat = module.v.detach().cpu().flatten().float()
                            # 处理可能的 inf/nan
                            v_flat = torch.nan_to_num(v_flat, nan=0.0, posinf=1e6, neginf=-1e6)
                            v_snap.append(v_flat)
                    if v_snap:
                        v_history.append(torch.cat(v_snap))

        if v_history:
            # 时间平均 + 最后时刻的状态
            V = torch.stack(v_history)  # (T-warmup, N)
            # 使用时间平均作为该序列的表示
            state_mean = V.mean(dim=0)
            # 也记录最后时刻状态
            state_last = V[-1]
            # 综合表示：拼接均值和末态
            state_repr = torch.cat([state_mean, state_last])
            all_states.append(state_repr.numpy())
        else:
            all_states.append(np.zeros(1))

    return all_states


def compute_kernel_quality(states, labels):
    """
    OBSERVATION 指标：计算 Fisher Ratio (Kernel Quality)。

    KQ = 类间距离 / 类内离散度

    类间距离 = ||μ_0 - μ_1||²
    类内离散度 = (σ²_0 + σ²_1) / 2  (各类平均方差)
    """
    states = np.array(states)
    labels = np.array(labels)

    # 过滤掉全零或维度不一致的
    valid_dim = max(s.shape[0] for s in states)
    states_padded = []
    for s in states:
        if s.shape[0] < valid_dim:
            s = np.pad(s, (0, valid_dim - s.shape[0]))
        states_padded.append(s)
    states = np.array(states_padded)

    idx_0 = labels == 0
    idx_1 = labels == 1

    states_0 = states[idx_0]
    states_1 = states[idx_1]

    # 类质心
    mu_0 = states_0.mean(axis=0)
    mu_1 = states_1.mean(axis=0)

    # 类间距离（L2²）
    inter_dist = np.sum((mu_0 - mu_1) ** 2)

    # 类内方差（平均 trace of covariance）
    var_0 = np.mean(np.var(states_0, axis=0))
    var_1 = np.mean(np.var(states_1, axis=0))
    intra_var = (var_0 + var_1) / 2.0

    # Fisher Ratio
    if intra_var < 1e-30:
        kq = 0.0 if inter_dist < 1e-30 else float('inf')
    else:
        kq = inter_dist / intra_var

    # 额外指标：线性可分性（用最简单的质心分类器）
    # 每个样本到两个质心的距离，选近的
    dist_to_0 = np.sum((states - mu_0[None, :]) ** 2, axis=1)
    dist_to_1 = np.sum((states - mu_1[None, :]) ** 2, axis=1)
    pred = (dist_to_1 < dist_to_0).astype(int)
    accuracy = np.mean(pred == labels)

    return {
        'kq': kq,
        'inter_dist': inter_dist,
        'intra_var_0': var_0,
        'intra_var_1': var_1,
        'intra_var': intra_var,
        'centroid_accuracy': accuracy,
        'mu_0': mu_0,
        'mu_1': mu_1,
        'states_0': states_0,
        'states_1': states_1,
    }


def compute_manifold_separability(states_0, states_1):
    """
    OBSERVATION 指标：计算流形分离度的更细粒度指标。

    1. 最小类间距离（最近点对）
    2. 类间/类内距离比的分布
    """
    from scipy.spatial.distance import cdist

    # 类间距离矩阵
    D_inter = cdist(states_0, states_1)
    min_inter = D_inter.min()
    mean_inter = D_inter.mean()

    # 类内距离矩阵
    D_intra_0 = cdist(states_0, states_0)
    D_intra_1 = cdist(states_1, states_1)

    # 去除对角线
    np.fill_diagonal(D_intra_0, np.nan)
    np.fill_diagonal(D_intra_1, np.nan)

    mean_intra_0 = np.nanmean(D_intra_0)
    mean_intra_1 = np.nanmean(D_intra_1)
    mean_intra = (mean_intra_0 + mean_intra_1) / 2.0

    separability = mean_inter / (mean_intra + 1e-30)

    return {
        'min_inter_dist': min_inter,
        'mean_inter_dist': mean_inter,
        'mean_intra_dist': mean_intra,
        'separability_ratio': separability,
    }


# =============================================================================
# 主实验
# =============================================================================

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    print("\n" + "=" * 70)
    print("实验 9: 时序流形解缠与核质量测定 (Temporal Manifold Untangling)")
    print("对比线性积分器 vs 临界同步区 vs 湍流区的螺旋分类能力")
    print("PRIMARY: Decoder MSE, OBSERVATION: V-space Kernel Quality")
    print("=" * 70)

    # 生成螺旋数据
    n_samples = 20  # 每类20个样本
    T = 80  # 80步（+10 warmup = 有效70步）
    warmup = 10
    sequences, labels = generate_spiral_sequences(
        n_samples_per_class=n_samples, T=T, noise_std=0.15, seed=42)

    print(f"  螺旋数据: {len(sequences)} 样本, 每样本 {T} 步, 4 维")
    print(f"  Class 0: {sum(1 for l in labels if l==0)}, Class 1: {sum(1 for l in labels if l==1)}")
    print(f"  Anchor 0: {ANCHOR_0.tolist()}, Anchor 1: {ANCHOR_1.tolist()}")

    # 实验条件
    conditions = [
        ("Linear Integrator (beta~1, Vth=1)", 1.0 - 1e-7, 1.0),
        ("Critical Sync (beta=0.90, Vth=10)", 0.90, 10.0),
        ("Strong Folding (beta=0.50, Vth=10)", 0.50, 10.0),
        ("Turbulent (beta=0.90, Vth=1)", 0.90, 1.0),
        ("Weak Folding (beta=0.99, Vth=10)", 0.99, 10.0),
    ]

    results = {}

    for i, (name, beta, vth) in enumerate(conditions):
        t0 = time.time()

        # 创建模型
        model, in_f = create_model(device, beta, v_threshold=vth)

        # TEMPORAL 模式
        SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

        # PRIMARY 指标：Decoder 端到端性能
        decoder_perf = evaluate_decoder_performance(model, sequences, labels, device, T_use=T)

        # OBSERVATION 指标：收集 V-space 状态
        states = collect_state_trajectories(model, sequences, device, warmup=warmup)

        # OBSERVATION 指标：计算核质量
        kq_result = compute_kernel_quality(states, labels)

        # OBSERVATION 指标：计算流形分离度
        try:
            sep_result = compute_manifold_separability(
                kq_result['states_0'], kq_result['states_1'])
        except Exception as e:
            sep_result = {
                'min_inter_dist': 0, 'mean_inter_dist': 0,
                'mean_intra_dist': 0, 'separability_ratio': 0,
            }

        # 恢复 BIT_EXACT
        SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

        elapsed = time.time() - t0

        results[name] = {
            # PRIMARY metrics
            'decoder_loss': decoder_perf['loss'],
            'decoder_accuracy': decoder_perf['accuracy'],
            'firing_rate': decoder_perf['firing_rate'],
            'nan_ratio': decoder_perf['nan_ratio'],
            # OBSERVATION metrics
            **kq_result,
            **sep_result,
            'beta': beta,
            'vth': vth,
            'states': states,
        }

        dec_loss = decoder_perf['loss']
        dec_acc = decoder_perf['accuracy']
        kq = kq_result['kq']
        kq_acc = kq_result['centroid_accuracy']
        sep = sep_result['separability_ratio']

        print(f"  [{i+1}/{len(conditions)}] {name}:", flush=True)
        print(f"    PRIMARY: Decoder Loss={dec_loss:.4f}, Acc={dec_acc:.1%}", flush=True)
        print(f"    OBSERVE: KQ={kq:.4f}, KQ-Acc={kq_acc:.1%}, Sep={sep:.4f} ({elapsed:.1f}s)", flush=True)

    # =========================================================================
    # 可视化 (6 panels)
    # =========================================================================

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    cond_names = list(results.keys())
    short_names = [
        "Linear\nInteg.",
        "Critical\nSync",
        "Strong\nFold",
        "Turbulent",
        "Weak\nFold",
    ]
    colors = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50', '#9C27B0']

    # Panel 1: Decoder Accuracy comparison (PRIMARY metric)
    ax1 = fig.add_subplot(gs[0, 0])
    dec_accs = [results[n]['decoder_accuracy'] for n in cond_names]
    bars = ax1.bar(range(len(cond_names)), dec_accs, color=colors, alpha=0.8, edgecolor='black')
    for bar, acc in zip(bars, dec_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{acc:.1%}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(len(cond_names)))
    ax1.set_xticklabels(short_names, fontsize=8)
    ax1.set_ylabel('Decoder Accuracy')
    ax1.set_title('PRIMARY: Decoder Accuracy (Anchor Classification)')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Decoder Loss comparison
    ax2 = fig.add_subplot(gs[0, 1])
    dec_losses = [results[n]['decoder_loss'] for n in cond_names]
    bars = ax2.bar(range(len(cond_names)), dec_losses, color=colors, alpha=0.8, edgecolor='black')
    for bar, loss in zip(bars, dec_losses):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{loss:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(cond_names)))
    ax2.set_xticklabels(short_names, fontsize=8)
    ax2.set_ylabel('Decoder MSE Loss')
    ax2.set_title('PRIMARY: Decoder MSE vs Anchor Targets')
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Kernel Quality (OBSERVATION)
    ax3 = fig.add_subplot(gs[0, 2])
    kqs = [results[n]['kq'] for n in cond_names]
    # Clip inf for display
    kqs_display = [min(k, max(k for k in kqs if k != float('inf')) * 2) if k == float('inf') else k for k in kqs]
    if all(k == float('inf') or k == 0 for k in kqs):
        kqs_display = [1.0 if k == float('inf') else 0.0 for k in kqs]
    bars = ax3.bar(range(len(cond_names)), kqs_display, color=colors, alpha=0.8, edgecolor='black')
    for j, (bar, kq) in enumerate(zip(bars, kqs)):
        label = f"{kq:.2f}" if kq != float('inf') else "inf"
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_xticks(range(len(cond_names)))
    ax3.set_xticklabels(short_names, fontsize=8)
    ax3.set_ylabel('Kernel Quality (Fisher Ratio)')
    ax3.set_title('OBSERVE: Kernel Quality (V-space)')
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Separability Ratio (OBSERVATION)
    ax4 = fig.add_subplot(gs[1, 0])
    seps = [results[n]['separability_ratio'] for n in cond_names]
    bars = ax4.bar(range(len(cond_names)), seps, color=colors, alpha=0.8, edgecolor='black')
    for bar, s in zip(bars, seps):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{s:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_xticks(range(len(cond_names)))
    ax4.set_xticklabels(short_names, fontsize=8)
    ax4.set_ylabel('Mean Inter / Mean Intra Distance')
    ax4.set_title('OBSERVE: Manifold Separability Ratio (V-space)')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Panel 5: PCA of Critical Sync V-space (OBSERVATION)
    ax5 = fig.add_subplot(gs[1, 1])
    target_idx = 1  # Critical Sync
    cname = cond_names[target_idx]
    r = results[cname]

    all_states = np.vstack([r['states_0'], r['states_1']])
    all_labels = np.array([0]*len(r['states_0']) + [1]*len(r['states_1']))

    # PCA to 2D
    mean = all_states.mean(axis=0)
    centered = all_states - mean
    # Handle inf/nan
    centered = np.nan_to_num(centered, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2, :].T
        proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        proj = centered[:, :2] if centered.shape[1] >= 2 else np.zeros((len(centered), 2))

    for cls, color, marker, label in [(0, '#E91E63', 'o', 'Spiral A'),
                                       (1, '#2196F3', 's', 'Spiral B')]:
        mask = all_labels == cls
        ax5.scatter(proj[mask, 0], proj[mask, 1], c=color, marker=marker,
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=label)

    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title('OBSERVE: Critical Sync V-space (PC1-PC2)')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # Panel 6: PCA of Linear Integrator V-space (OBSERVATION)
    ax6 = fig.add_subplot(gs[1, 2])
    target_idx = 0  # Linear Integrator
    cname = cond_names[target_idx]
    r = results[cname]

    all_states = np.vstack([r['states_0'], r['states_1']])
    all_labels = np.array([0]*len(r['states_0']) + [1]*len(r['states_1']))

    # PCA to 2D
    mean = all_states.mean(axis=0)
    centered = all_states - mean
    # Handle inf/nan
    centered = np.nan_to_num(centered, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2, :].T
        proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        proj = centered[:, :2] if centered.shape[1] >= 2 else np.zeros((len(centered), 2))

    for cls, color, marker, label in [(0, '#E91E63', 'o', 'Spiral A'),
                                       (1, '#2196F3', 's', 'Spiral B')]:
        mask = all_labels == cls
        ax6.scatter(proj[mask, 0], proj[mask, 1], c=color, marker=marker,
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=label)

    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.set_title('OBSERVE: Linear Integrator V-space (PC1-PC2)')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    fig.suptitle('Experiment 9: Temporal Manifold Untangling & Kernel Quality\n'
                 'Primary: Decoder MSE, Observation: V-space Kernel Quality',
                 fontsize=14, fontweight='bold', y=1.02)

    save_path = os.path.join(os.path.dirname(__file__), 'temporal_manifold_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    # =========================================================================
    # 汇总
    # =========================================================================
    print(f"\n实验 9 完成")

    print("\n" + "=" * 70)
    print("[汇总] PRIMARY (Decoder) vs OBSERVATION (V-space) 指标对比")
    print("=" * 70)

    header = f"  {'Condition':<42} | {'Dec-Acc':>8} | {'Dec-Loss':>9} | {'KQ':>10} | {'KQ-Acc':>7} | {'Sep':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in cond_names:
        r = results[name]
        kq_str = f"{r['kq']:.4f}" if r['kq'] != float('inf') else "inf"
        print(f"  {name:<42} | {r['decoder_accuracy']:>7.1%} | {r['decoder_loss']:>9.4f} | "
              f"{kq_str:>10} | {r['centroid_accuracy']:>6.1%} | {r['separability_ratio']:>8.4f}")

    # 关键比较
    print("\n  [PRIMARY] Decoder Accuracy:")
    best_dec_name = max(cond_names, key=lambda n: results[n]['decoder_accuracy'])
    print(f"    Best: {best_dec_name} ({results[best_dec_name]['decoder_accuracy']:.1%})")

    print("\n  [OBSERVATION] Kernel Quality (V-space):")
    kq_linear = results[cond_names[0]]['kq']
    kq_critical = results[cond_names[1]]['kq']
    kq_strong = results[cond_names[2]]['kq']

    print(f"    KQ ratio (Critical Sync / Linear) = ", end="")
    if kq_linear > 0 and kq_linear != float('inf'):
        print(f"{kq_critical/kq_linear:.2f}x")
    else:
        print(f"N/A (linear KQ = {kq_linear})")

    print(f"    KQ ratio (Strong Fold / Linear) = ", end="")
    if kq_linear > 0 and kq_linear != float('inf'):
        print(f"{kq_strong/kq_linear:.2f}x")
    else:
        print(f"N/A (linear KQ = {kq_linear})")

    best_kq_name = max(cond_names, key=lambda n: results[n]['kq'] if results[n]['kq'] != float('inf') else -1)
    print(f"    Best KQ: {best_kq_name}")


if __name__ == '__main__':
    run_experiment()
