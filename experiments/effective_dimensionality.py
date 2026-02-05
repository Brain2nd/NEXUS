"""
实验 8: 有效维度与状态空间膨胀 (Effective Dimensionality & State Space Expansion)
================================================================================

物理动机：
- 混沌边缘的折叠操作能在有限物理节点内"虚拟"出高有效维度
- 这是动力学版本的 Kernel Trick
- 若 ED(NEXUS_temporal) >> ED(linear_integrator)，证明软复位折叠提供了额外计算维度

实验设计：
- 对照组 A: β ≈ 1 (1-1e-7)，V_th=1.0 (线性积分器，无折叠)
- 实验组 B: β = 0.90, V_th = 10.0 (γ=0.1, 临界同步区)
- 额外对照: β = 0.50, V_th = 10.0 (强折叠区)
- 额外对照: β = 0.90, V_th = 1.0 (γ=1.0, 非同步湍流)
- 额外对照: β = 0.99, V_th = 10.0 (近积分器)

输入：混合正弦波 + 少量噪声（低维复杂序列）
指标：Effective Dimensionality = (Σλ_i)² / Σ(λ_i²)
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
# Model & Utils (与实验7完全一致)
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
    """创建模型，创建后直接遍历所有 SimpleLIFNode 设置 v_threshold"""
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


def get_state_vector(model):
    """收集所有 LIF 神经元的膜电位拼接成一个向量"""
    parts = []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            parts.append(module.v.detach().cpu().numpy().flatten())
    if len(parts) == 0:
        return None
    return np.concatenate(parts)


def generate_input_sequence(T, dim=4, seed=123):
    """生成混合正弦波 + 噪声的低维复杂输入序列"""
    np.random.seed(seed)
    t = np.arange(T).astype(np.float32)
    x = np.zeros((T, dim), dtype=np.float32)
    # 不同频率的正弦波叠加
    freqs = [0.05, 0.13, 0.31, 0.07]
    phases = [0, np.pi/3, np.pi/6, np.pi/4]
    for d in range(dim):
        x[:, d] = (0.3 * np.sin(2 * np.pi * freqs[d] * t + phases[d])
                    + 0.15 * np.sin(2 * np.pi * freqs[(d+1) % dim] * 2.3 * t)
                    + 0.05 * np.random.randn(T))
    return x


def compute_effective_dimensionality(state_matrix):
    """
    计算有效维度 ED = (Σλ_i)² / Σ(λ_i²)
    state_matrix: (T, N) — T个时间步, N个神经元状态
    返回: ED, eigenvalues, explained_variance_ratio
    """
    # 去掉常量列
    stds = np.std(state_matrix, axis=0)
    active_cols = stds > 1e-12
    X = state_matrix[:, active_cols]
    n_active = X.shape[1]

    if n_active == 0:
        return 0.0, np.array([]), np.array([]), 0

    # 中心化
    X = X - X.mean(axis=0)

    # SVD (比协方差矩阵更数值稳定)
    # X = U S V^T, 协方差特征值 λ_i = s_i² / (T-1)
    T_steps = X.shape[0]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (S ** 2) / (T_steps - 1)

    # 去掉数值噪声
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    if len(eigenvalues) == 0:
        return 0.0, np.array([]), np.array([]), n_active

    # Effective Dimensionality
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    ED = (sum_lambda ** 2) / sum_lambda_sq

    # Explained variance ratio
    evr = eigenvalues / sum_lambda

    return ED, eigenvalues, evr, n_active


def compute_participation_ratio(eigenvalues):
    """参与比 PR = (Σλ_i)² / Σ(λ_i²)，与ED等价但名称更标准"""
    if len(eigenvalues) == 0:
        return 0.0
    s = np.sum(eigenvalues)
    s2 = np.sum(eigenvalues ** 2)
    return (s ** 2) / s2 if s2 > 0 else 0.0


# =============================================================================
# 核心实验
# =============================================================================

def run_effective_dimensionality(device):
    """运行有效维度对比实验"""
    print("\n" + "=" * 70, flush=True)
    print("实验 8: 有效维度与状态空间膨胀 (Effective Dimensionality)", flush=True)
    print("对比线性积分器 vs 临界同步区 vs 强折叠区的状态空间结构", flush=True)
    print("=" * 70, flush=True)

    T = 100  # 时间步数
    warmup = 10  # 预热步数（丢弃）

    # 实验条件
    conditions = [
        # (label, beta, v_threshold, description)
        ("A: β≈1, V_th=1.0\n(线性积分器)", 1.0 - 1e-7, 1.0, "线性积分器 (BIT_EXACT参数区)"),
        ("B: β=0.90, V_th=10.0\n(临界同步)", 0.90, 10.0, "临界同步区 (黄金工作区)"),
        ("C: β=0.50, V_th=10.0\n(强折叠)", 0.50, 10.0, "强衰减+高阈值折叠"),
        ("D: β=0.90, V_th=1.0\n(非同步湍流)", 0.90, 1.0, "湍流区"),
        ("E: β=0.99, V_th=10.0\n(弱折叠)", 0.99, 10.0, "近积分器+高阈值"),
    ]

    # 生成共享输入
    input_seq = generate_input_sequence(T + warmup, dim=4, seed=123)
    print(f"  输入序列: {T+warmup}步, 4维混合正弦波+噪声", flush=True)

    results = []

    for idx, (label, beta, v_th, desc) in enumerate(conditions):
        t_start = time.time()

        # 创建模型
        model, in_f = create_model(device, beta, v_threshold=v_th, seed=42)
        model.reset()

        # 切换到 TEMPORAL 模式
        SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

        # 运行并收集状态
        state_list = []
        for t in range(T + warmup):
            x_float = torch.tensor(input_seq[t], dtype=torch.float32, device=device)
            x_pulse = float32_to_pulse(x_float, device=device)
            _ = model(x_pulse.unsqueeze(0))

            if t >= warmup:
                sv = get_state_vector(model)
                if sv is not None:
                    state_list.append(sv)

        # 恢复模式
        SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

        if len(state_list) == 0:
            print(f"  [{idx+1}/{len(conditions)}] {desc}: 无状态数据", flush=True)
            results.append({
                'label': label, 'desc': desc, 'beta': beta, 'v_th': v_th,
                'ED': 0, 'eigenvalues': np.array([]), 'evr': np.array([]),
                'n_active': 0, 'n_total': 0, 'state_matrix': np.zeros((1, 1)),
            })
            continue

        state_matrix = np.array(state_list)  # (T, N)
        n_total = state_matrix.shape[1]

        # 计算有效维度
        ED, eigenvalues, evr, n_active = compute_effective_dimensionality(state_matrix)

        # 额外指标：90% 方差需要的主成分数
        if len(evr) > 0:
            cum_var = np.cumsum(evr)
            n_90 = np.searchsorted(cum_var, 0.90) + 1
            n_95 = np.searchsorted(cum_var, 0.95) + 1
            n_99 = np.searchsorted(cum_var, 0.99) + 1
        else:
            n_90 = n_95 = n_99 = 0

        elapsed = time.time() - t_start
        print(f"  [{idx+1}/{len(conditions)}] {desc}: "
              f"ED={ED:.2f}, n_active={n_active}/{n_total}, "
              f"PC90={n_90}, PC95={n_95}, PC99={n_99} ({elapsed:.1f}s)", flush=True)

        results.append({
            'label': label, 'desc': desc, 'beta': beta, 'v_th': v_th,
            'ED': ED, 'eigenvalues': eigenvalues, 'evr': evr,
            'n_active': n_active, 'n_total': n_total,
            'n_90': n_90, 'n_95': n_95, 'n_99': n_99,
            'state_matrix': state_matrix,
        })

    return results


def plot_results(results, save_path):
    """6面板可视化"""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Experiment 8: Effective Dimensionality & State Space Expansion\n"
                 "Comparing Linear Integrator vs Critical Sync vs Strong Folding",
                 fontsize=14, fontweight='bold')

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.96, top=0.90, bottom=0.08)

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']

    # --- Panel 1: ED Bar Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    labels_short = ['A: β≈1\nIntegrator', 'B: β=0.9\nCritical', 'C: β=0.5\nStrong Fold',
                     'D: β=0.9\nTurbulent', 'E: β=0.99\nWeak Fold']
    eds = [r['ED'] for r in results]
    bars = ax1.bar(range(len(eds)), eds, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(eds):
        ax1.text(i, v + max(eds) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(labels_short)))
    ax1.set_xticklabels(labels_short, fontsize=8)
    ax1.set_ylabel('Effective Dimensionality (ED)')
    ax1.set_title('Effective Dimensionality Comparison')
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel 2: Eigenvalue Spectrum (log scale) ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(results):
        if len(r['eigenvalues']) > 0:
            ax2.semilogy(range(1, len(r['eigenvalues']) + 1), r['eigenvalues'],
                        color=colors[i], label=r['desc'].split('(')[0].strip(), linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Principal Component Index')
    ax2.set_ylabel('Eigenvalue (log scale)')
    ax2.set_title('Eigenvalue Spectrum')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 50)

    # --- Panel 3: Cumulative Explained Variance ---
    ax3 = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(results):
        if len(r['evr']) > 0:
            cum_var = np.cumsum(r['evr'])
            ax3.plot(range(1, len(cum_var) + 1), cum_var,
                    color=colors[i], label=r['desc'].split('(')[0].strip(), linewidth=1.5, alpha=0.8)
    ax3.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax3.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Cumulative Explained Variance')
    ax3.set_title('Cumulative Variance (How Many PCs Needed?)')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, 1.05)

    # --- Panel 4: PC Components Needed (bar chart) ---
    ax4 = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(results))
    width = 0.25
    n90s = [r.get('n_90', 0) for r in results]
    n95s = [r.get('n_95', 0) for r in results]
    n99s = [r.get('n_99', 0) for r in results]
    ax4.bar(x_pos - width, n90s, width, label='90% var', color='#66c2a5', edgecolor='black', linewidth=0.5)
    ax4.bar(x_pos, n95s, width, label='95% var', color='#fc8d62', edgecolor='black', linewidth=0.5)
    ax4.bar(x_pos + width, n99s, width, label='99% var', color='#8da0cb', edgecolor='black', linewidth=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels_short, fontsize=8)
    ax4.set_ylabel('Number of PCs Required')
    ax4.set_title('PCs Needed for 90%/95%/99% Variance')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # --- Panel 5: Top-2 PC Projection (2D scatter) ---
    ax5 = fig.add_subplot(gs[1, 1])
    for i, r in enumerate(results):
        sm = r['state_matrix']
        if sm.shape[0] < 3 or sm.shape[1] < 2:
            continue
        # 去常量列, 中心化
        stds = np.std(sm, axis=0)
        active = stds > 1e-12
        X = sm[:, active]
        if X.shape[1] < 2:
            continue
        X = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # 投影到前2个PC
        proj = X @ Vt[:2, :].T
        ax5.plot(proj[:, 0], proj[:, 1], color=colors[i], alpha=0.6, linewidth=0.8,
                label=r['desc'].split('(')[0].strip())
        ax5.scatter(proj[0, 0], proj[0, 1], color=colors[i], s=40, zorder=5, marker='o')
        ax5.scatter(proj[-1, 0], proj[-1, 1], color=colors[i], s=40, zorder=5, marker='s')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title('State Trajectory in PC1-PC2 Space')
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(alpha=0.3)

    # --- Panel 6: Summary Table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    table_data = []
    for r in results:
        table_data.append([
            r['desc'].split('(')[0].strip()[:20],
            f"β={r['beta']:.2f}" if r['beta'] < 0.999 else "β≈1",
            f"V={r['v_th']:.1f}",
            f"{r['ED']:.1f}",
            f"{r.get('n_90', 0)}",
            f"{r.get('n_95', 0)}",
            f"{r.get('n_99', 0)}",
            f"{r['n_active']}",
        ])
    table = ax6.table(cellText=table_data,
                      colLabels=['Condition', 'β', 'V_th', 'ED', 'PC90', 'PC95', 'PC99', 'Active'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    # Highlight the critical sync row
    for j in range(8):
        table[2, j].set_facecolor('#d4edda')  # row index 2 = condition B (1-indexed with header)
    ax6.set_title('Summary Table', fontsize=12, fontweight='bold', pad=20)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}", flush=True)
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)

    t0 = time.time()
    results = run_effective_dimensionality(device)

    # 可视化
    save_path = os.path.join(os.path.dirname(__file__), 'effective_dimensionality_results.png')
    plot_results(results, save_path)

    elapsed = time.time() - t0
    print(f"\n实验 8 完成 (总耗时: {elapsed:.1f}s)", flush=True)

    # 打印汇总
    print("\n" + "=" * 70, flush=True)
    print("[汇总] 有效维度对比", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'条件':<35s} | {'ED':>8s} | {'PC90':>5s} | {'PC95':>5s} | {'PC99':>5s} | {'Active':>7s}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"  {r['desc']:<35s} | {r['ED']:8.2f} | {r.get('n_90',0):5d} | "
              f"{r.get('n_95',0):5d} | {r.get('n_99',0):5d} | {r['n_active']:7d}", flush=True)

    # ED 对比分析
    if len(results) >= 2:
        ed_integrator = results[0]['ED']
        ed_critical = results[1]['ED']
        if ed_integrator > 0:
            ratio = ed_critical / ed_integrator
            print(f"\n  ED 比值 (临界同步 / 线性积分器) = {ratio:.2f}x", flush=True)
            if ratio > 1.5:
                print(f"  *** 临界同步区的有效维度显著高于线性积分器 ***", flush=True)
                print(f"  *** 软复位折叠提供了额外的计算维度 (隐式升维) ***", flush=True)
            elif ratio > 1.0:
                print(f"  临界同步区有效维度略高于线性积分器", flush=True)
            else:
                print(f"  线性积分器有效维度不低于临界同步区", flush=True)
