"""
时间-空间编码可分性实验 & 熵探针
================================

验证 TEMPORAL 模式下 LIF 神经元膜电位残余的三个核心假设:

1. 熵探针 (Entropy Probe):
   - 理论容量基准: Σ ln(1 - β²)
   - 实测微分熵 (k-NN 估计器)
   - 对比 BIT_EXACT (熵=0) vs TEMPORAL (熵>0)

2. 时间-空间编码可分性:
   - 序列 A/B 共享末尾 token, 不同上下文
   - BIT_EXACT: V=None (无记忆)  vs  TEMPORAL: V 编码上下文

3. 多尺度 β 初始化:
   - 对比不同 β 的状态熵差异

运行: conda activate dapo && python experiments/temporal_entropy_probe.py
作者: NEXUS Project (exp branch)
"""

import sys
import os
import time
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

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.special import digamma
print("[import] sklearn, scipy done", flush=True)

from atomic_ops import (
    SpikeMode,
    SpikeFP32Linear_MultiPrecision,
    SimpleLIFNode,
)
from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32
print("[import] atomic_ops done", flush=True)


# =============================================================================
# 计时工具 (借鉴 test_qwen3_minimal.py)
# =============================================================================

class Timer:
    """细粒度计时上下文管理器"""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print(f"  → {self.name}...", end='', flush=True)
        self.t0 = time.time()
        return self
    def __exit__(self, *args):
        elapsed = time.time() - self.t0
        print(f" done ({elapsed:.3f}s)", flush=True)


def progress(i, total, prefix='', every=10):
    """每 every 步打印一次进度"""
    if i == 0 or (i + 1) % every == 0 or i + 1 == total:
        print(f"    {prefix}[{i+1}/{total}]", flush=True)


# =============================================================================
# 工具函数
# =============================================================================

def collect_lif_membrane_potentials(model):
    """递归收集模型中所有 LIF 神经元的膜电位 V"""
    potentials = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            potentials[name] = module.v.detach().clone()
    return potentials


def collect_lif_betas(model):
    """递归收集模型中所有 LIF 神经元的 β 参数"""
    betas = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas[name] = module._beta.detach().clone()
    return betas


def knn_entropy_estimate(samples, k=5):
    """k-NN 微分熵估计器 (Kozachenko-Leonenko)"""
    N, D = samples.shape
    if N < k + 1:
        return float('nan')
    std = samples.std(axis=0)
    std[std < 1e-12] = 1e-12
    samples_normed = samples / std
    tree = KDTree(samples_normed)
    dists, _ = tree.query(samples_normed, k=k+1)
    rho_k = dists[:, -1]
    rho_k[rho_k < 1e-30] = 1e-30
    h = D * np.mean(np.log(rho_k)) + np.log(N - 1) - digamma(k) + np.log(2) * D
    h += np.sum(np.log(std))
    return h


def theoretical_state_entropy(betas, sigma_I=1.0):
    """h(V) = N/2 * ln(2πe * σ²) - 1/2 * Σ ln(1 - β²)"""
    N_total = 0
    log_sum = 0.0
    for name, beta in betas.items():
        beta_np = beta.cpu().numpy().flatten()
        N_total += len(beta_np)
        one_minus_beta_sq = np.clip(1.0 - beta_np ** 2, 1e-30, None)
        log_sum += np.sum(np.log(one_minus_beta_sq))
    h = (N_total / 2.0) * np.log(2 * np.pi * np.e * sigma_I ** 2) - 0.5 * log_sum
    return h, N_total, log_sum


# =============================================================================
# 简单 SNN MLP
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
        h = self.linear1(x_pulse)
        return self.linear2(h)

    def reset(self):
        self.linear1.reset()
        self.linear2.reset()


# =============================================================================
# 实验 1: 时间-空间编码可分性
# =============================================================================

def run_separability_experiment(device='cpu'):
    print("=" * 70)
    print("实验 1: 时间-空间编码可分性")
    print("=" * 70)

    torch.manual_seed(42)
    in_f, hid_f, out_f = 4, 8, 4

    with Timer("创建模型"):
        model = SimpleSpikeMLP(in_f, hid_f, out_f).to(device)
        w1 = torch.randn(hid_f, in_f, device=device) * 0.5
        w2 = torch.randn(out_f, hid_f, device=device) * 0.5
        model.set_weights(w1, w2)

    n_trials = 20
    seq_len_a, seq_len_b = 5, 3
    target_token = torch.randn(in_f, device=device) * 0.3
    context_a = torch.randn(n_trials, seq_len_a - 1, in_f, device=device) * 0.5
    context_b = torch.randn(n_trials, seq_len_b - 1, in_f, device=device) * 0.5

    # --- BIT_EXACT ---
    print("\n  [BIT_EXACT 模式]")
    n_with_v = 0
    with SpikeMode.bit_exact():
        for trial in range(n_trials):
            model.reset()
            for t in range(seq_len_a - 1):
                x_pulse = float32_to_pulse(context_a[trial, t], device=device)
                _ = model(x_pulse.unsqueeze(0))
            x_target = float32_to_pulse(target_token, device=device)
            _ = model(x_target.unsqueeze(0))
            v = collect_lif_membrane_potentials(model)
            if v:
                n_with_v += 1
            progress(trial, n_trials, prefix='BIT_EXACT ')
    print(f"  BIT_EXACT: {n_with_v}/{n_trials} 有 V 残余")
    if n_with_v == 0:
        print("  ✓ 确认: BIT_EXACT 所有 V 被清除")

    # --- TEMPORAL ---
    print("\n  [TEMPORAL 模式]")
    v_temporal_a, v_temporal_b = [], []
    with SpikeMode.temporal():
        for trial in range(n_trials):
            # 序列 A
            model.reset()
            for t in range(seq_len_a - 1):
                x_pulse = float32_to_pulse(context_a[trial, t], device=device)
                _ = model(x_pulse.unsqueeze(0))
            x_target = float32_to_pulse(target_token, device=device)
            _ = model(x_target.unsqueeze(0))
            v_a = collect_lif_membrane_potentials(model)
            if v_a:
                v_temporal_a.append(np.concatenate([v.cpu().numpy().flatten() for v in v_a.values()]))

            # 序列 B
            model.reset()
            for t in range(seq_len_b - 1):
                x_pulse = float32_to_pulse(context_b[trial, t], device=device)
                _ = model(x_pulse.unsqueeze(0))
            x_target = float32_to_pulse(target_token, device=device)
            _ = model(x_target.unsqueeze(0))
            v_b = collect_lif_membrane_potentials(model)
            if v_b:
                v_temporal_b.append(np.concatenate([v.cpu().numpy().flatten() for v in v_b.values()]))

            progress(trial, n_trials, prefix='TEMPORAL ')

    print(f"  TEMPORAL A: {len(v_temporal_a)} 试验有 V")
    print(f"  TEMPORAL B: {len(v_temporal_b)} 试验有 V")

    if not v_temporal_a or not v_temporal_b:
        print("  ✗ 未收集到膜电位残余!")
        return None, None, None

    v_temporal_a = np.nan_to_num(np.array(v_temporal_a), nan=0.0, posinf=0.0, neginf=0.0)
    v_temporal_b = np.nan_to_num(np.array(v_temporal_b), nan=0.0, posinf=0.0, neginf=0.0)
    # 合并后去掉常量列，再拆分
    v_all = np.concatenate([v_temporal_a, v_temporal_b], axis=0)
    col_var = np.var(v_all, axis=0)
    nonzero_cols = col_var > 1e-30
    v_temporal_a = v_temporal_a[:, nonzero_cols]
    v_temporal_b = v_temporal_b[:, nonzero_cols]
    print(f"  有效维度: {v_temporal_a.shape[1]} (去掉 {(~nonzero_cols).sum()} 常量列)")

    with Timer("计算可分性指标"):
        centroid_a = v_temporal_a.mean(axis=0)
        centroid_b = v_temporal_b.mean(axis=0)
        euclidean_dist = np.linalg.norm(centroid_a - centroid_b)
        var_a = np.mean(np.linalg.norm(v_temporal_a - centroid_a, axis=1) ** 2)
        var_b = np.mean(np.linalg.norm(v_temporal_b - centroid_b, axis=1) ** 2)
        fisher_ratio = euclidean_dist ** 2 / (var_a + var_b + 1e-30)

    print(f"\n  质心欧氏距离: {euclidean_dist:.6f}")
    print(f"  类内方差 A: {var_a:.6f}, B: {var_b:.6f}")
    print(f"  Fisher 判别比: {fisher_ratio:.4f}")
    if fisher_ratio > 1.0:
        print("  ✓ 上下文编码具有显著可分性 (Fisher > 1)")

    return v_temporal_a, v_temporal_b, fisher_ratio


# =============================================================================
# 实验 2: 熵探针
# =============================================================================

def run_entropy_probe(device='cpu'):
    print("\n" + "=" * 70)
    print("实验 2: 熵探针 (Entropy Probe)")
    print("=" * 70)

    torch.manual_seed(123)
    in_f, hid_f, out_f = 4, 8, 4

    with Timer("创建模型"):
        model = SimpleSpikeMLP(in_f, hid_f, out_f).to(device)
        w1 = torch.randn(hid_f, in_f, device=device) * 0.5
        w2 = torch.randn(out_f, hid_f, device=device) * 0.5
        model.set_weights(w1, w2)

    with Timer("计算理论基准"):
        betas = collect_lif_betas(model)
        h_theory, N_total, log_sum = theoretical_state_entropy(betas, sigma_I=0.5)
    print(f"  LIF 神经元总数: {N_total}")
    print(f"  理论状态熵: {h_theory:.2f} nats")
    print(f"  Σ ln(1-β²) = {log_sum:.4f}")
    for name, beta in betas.items():
        b = beta.cpu().numpy().flatten()
        print(f"    {name}: β mean={b.mean():.9f}, len={len(b)}")

    # TEMPORAL 采样
    n_samples, seq_len = 30, 3
    v_samples = []
    print(f"\n  [TEMPORAL 采样: {n_samples} samples × {seq_len} steps]")
    with SpikeMode.temporal():
        for i in range(n_samples):
            model.reset()
            for t in range(seq_len):
                x = torch.randn(in_f, device=device) * 0.5
                x_pulse = float32_to_pulse(x, device=device)
                _ = model(x_pulse.unsqueeze(0))
            v_all = collect_lif_membrane_potentials(model)
            if v_all:
                v_samples.append(np.concatenate([v.cpu().numpy().flatten() for v in v_all.values()]))
            progress(i, n_samples, prefix='TEMPORAL采样 ', every=5)

    if not v_samples:
        print("  ✗ 未收集到膜电位样本!")
        return None, None, None

    v_samples = np.array(v_samples)
    # 清洗 inf/nan
    v_samples = np.nan_to_num(v_samples, nan=0.0, posinf=0.0, neginf=0.0)
    # 去掉方差为 0 的列（预分配未使用的位全为0）
    col_var = np.var(v_samples, axis=0)
    nonzero_cols = col_var > 1e-30
    v_samples = v_samples[:, nonzero_cols]
    print(f"  收集: {v_samples.shape[0]} 样本, 有效维度 {v_samples.shape[1]} (去掉 {(~nonzero_cols).sum()} 常量列)")
    # 标准化防止 matmul 溢出
    col_std = np.std(v_samples, axis=0)
    col_std[col_std < 1e-30] = 1.0
    v_samples_normed = (v_samples - v_samples.mean(axis=0)) / col_std

    with Timer("k-NN 熵估计"):
        D = v_samples_normed.shape[1]
        if D > 50:
            pca = PCA(n_components=min(50, D, n_samples - 1))
            v_reduced = pca.fit_transform(v_samples_normed)
            h_empirical = knn_entropy_estimate(v_reduced, k=5)
            print(f"  (PCA 降至 {v_reduced.shape[1]} 维)")
        else:
            h_empirical = knn_entropy_estimate(v_samples_normed, k=5)
    print(f"  实测微分熵: {h_empirical:.2f} nats")

    # BIT_EXACT 对照
    print("\n  [BIT_EXACT 对照]")
    n_be_with_v = 0
    with SpikeMode.bit_exact():
        for i in range(20):
            model.reset()
            for t in range(seq_len):
                x = torch.randn(in_f, device=device) * 0.5
                x_pulse = float32_to_pulse(x, device=device)
                _ = model(x_pulse.unsqueeze(0))
            v_all = collect_lif_membrane_potentials(model)
            if v_all:
                n_be_with_v += 1
            progress(i, 20, prefix='BIT_EXACT对照 ', every=10)
    print(f"  BIT_EXACT: {n_be_with_v}/20 有 V")
    if n_be_with_v == 0:
        print("  ✓ BIT_EXACT 状态熵 = 0")

    return v_samples, h_theory, h_empirical


# =============================================================================
# 实验 3: 多尺度 β 对比
# =============================================================================

def run_multiscale_beta_experiment(device='cpu'):
    print("\n" + "=" * 70)
    print("实验 3: 多尺度 β 初始化对比")
    print("=" * 70)

    torch.manual_seed(456)
    in_f, hid_f, out_f = 4, 8, 4
    results = {}

    beta_configs = [
        ("default(1-1e-7)", 1.0 - 1e-7),
        ("0.999", 0.999),
        ("0.99", 0.99),
        ("0.9", 0.9),
    ]

    for beta_name, beta_value in beta_configs:
        print(f"\n  --- β = {beta_value} (ε = {1-beta_value:.1e}) ---")

        with Timer(f"创建模型 β={beta_name}"):
            nt = SimpleLIFNode(beta=beta_value)
            model = SimpleSpikeMLP(in_f, hid_f, out_f, neuron_template=nt).to(device)
            w1 = torch.randn(hid_f, in_f, device=device) * 0.5
            w2 = torch.randn(out_f, hid_f, device=device) * 0.5
            model.set_weights(w1, w2)

        betas = collect_lif_betas(model)
        h_theory, N, _ = theoretical_state_entropy(betas, sigma_I=0.5)

        n_samples, seq_len = 20, 3
        v_samples = []
        with SpikeMode.temporal():
            for i in range(n_samples):
                model.reset()
                for t in range(seq_len):
                    x = torch.randn(in_f, device=device) * 0.5
                    x_pulse = float32_to_pulse(x, device=device)
                    _ = model(x_pulse.unsqueeze(0))
                v_all = collect_lif_membrane_potentials(model)
                if v_all:
                    v_samples.append(np.concatenate([v.cpu().numpy().flatten() for v in v_all.values()]))
                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{n_samples}]", flush=True)

        h_emp = knn_entropy_estimate(np.array(v_samples), k=5) if v_samples else float('nan')
        results[beta_name] = {
            'beta': beta_value, 'h_theory': h_theory,
            'h_empirical': h_emp, 'n_neurons': N, 'epsilon': 1.0 - beta_value,
        }
        print(f"  理论熵: {h_theory:.2f} nats, 实测熵: {h_emp:.2f} nats")

    return results


# =============================================================================
# 可视化
# =============================================================================

def visualize_results(v_temporal_a, v_temporal_b, fisher_ratio,
                      v_entropy_samples, h_theory, h_empirical,
                      beta_results, save_path):
    print("\n" + "=" * 70)
    print("生成可视化")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('NEXUS: Temporal Mode — Entropy & Separability Analysis',
                 fontsize=14, fontweight='bold', y=0.98)

    # (1) t-SNE
    if v_temporal_a is not None and v_temporal_b is not None:
        with Timer("t-SNE"):
            ax1 = fig.add_subplot(gs[0, 0])
            all_v = np.vstack([v_temporal_a, v_temporal_b])
            # 标准化防止溢出
            std = np.std(all_v, axis=0)
            std[std < 1e-30] = 1.0
            all_v_normed = (all_v - all_v.mean(axis=0)) / std
            labels = np.array([0] * len(v_temporal_a) + [1] * len(v_temporal_b))
            if all_v_normed.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_v_normed)//2))
                coords = tsne.fit_transform(all_v_normed)
            else:
                coords = all_v
            ax1.scatter(coords[labels==0, 0], coords[labels==0, 1],
                        c='#2196F3', alpha=0.7, s=30, label='Context A (len=5)')
            ax1.scatter(coords[labels==1, 0], coords[labels==1, 1],
                        c='#FF5722', alpha=0.7, s=30, label='Context B (len=3)')
            ax1.set_title(f't-SNE of V_residual (TEMPORAL)\nFisher Ratio = {fisher_ratio:.3f}')
            ax1.set_xlabel('t-SNE dim 1'); ax1.set_ylabel('t-SNE dim 2')
            ax1.legend(fontsize=8)

    # (2) PCA
    if v_temporal_a is not None and v_temporal_b is not None:
        with Timer("PCA"):
            ax2 = fig.add_subplot(gs[0, 1])
            all_v = np.vstack([v_temporal_a, v_temporal_b])
            std = np.std(all_v, axis=0)
            std[std < 1e-30] = 1.0
            all_v_normed = (all_v - all_v.mean(axis=0)) / std
            labels = np.array([0] * len(v_temporal_a) + [1] * len(v_temporal_b))
            pca = PCA(n_components=2)
            coords_pca = pca.fit_transform(all_v_normed)
            var_exp = pca.explained_variance_ratio_
            ax2.scatter(coords_pca[labels==0, 0], coords_pca[labels==0, 1],
                        c='#2196F3', alpha=0.7, s=30, label='Context A')
            ax2.scatter(coords_pca[labels==1, 0], coords_pca[labels==1, 1],
                        c='#FF5722', alpha=0.7, s=30, label='Context B')
            ax2.set_title(f'PCA of V_residual\nPC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}')
            ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
            ax2.legend(fontsize=8)

    # (3) V 分布直方图
    if v_entropy_samples is not None:
        with Timer("直方图"):
            ax3 = fig.add_subplot(gs[0, 2])
            n_dims = min(5, v_entropy_samples.shape[1])
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_dims))
            for d in range(n_dims):
                ax3.hist(v_entropy_samples[:, d], bins=30, alpha=0.5, color=colors[d], label=f'dim {d}')
            ax3.set_title('V_residual Distribution')
            ax3.set_xlabel('Membrane Potential'); ax3.set_ylabel('Count')
            ax3.legend(fontsize=7)

    # (4) 熵对比柱状图
    if h_theory is not None:
        with Timer("熵对比图"):
            ax4 = fig.add_subplot(gs[1, 0])
            vals = [0, h_theory, h_empirical if h_empirical == h_empirical else 0]
            bars = ax4.bar(['BIT_EXACT', 'TEMPORAL\n(Theory)', 'TEMPORAL\n(Empirical)'],
                           vals, color=['#9E9E9E', '#2196F3', '#FF9800'], edgecolor='black', linewidth=0.5)
            ax4.set_ylabel('Differential Entropy (nats)')
            ax4.set_title('State Entropy Comparison')
            for bar, val in zip(bars, vals):
                if val == val:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             f'{val:.1f}', ha='center', fontsize=9)

    # (5) 多尺度 β 对比
    if beta_results:
        with Timer("β 对比图"):
            ax5 = fig.add_subplot(gs[1, 1])
            names = list(beta_results.keys())
            eps = [beta_results[n]['epsilon'] for n in names]
            ht = [beta_results[n]['h_theory'] for n in names]
            he = [beta_results[n]['h_empirical'] for n in names]
            x_pos = np.arange(len(names))
            w = 0.35
            ax5.bar(x_pos - w/2, ht, w, label='Theoretical', color='#2196F3', edgecolor='black', linewidth=0.5)
            ax5.bar(x_pos + w/2, he, w, label='Empirical', color='#FF9800', edgecolor='black', linewidth=0.5)
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels([f'ε={e:.0e}' for e in eps], fontsize=8)
            ax5.set_ylabel('Entropy (nats)')
            ax5.set_title('State Entropy vs ε')
            ax5.legend(fontsize=8)

    # (6) 理论曲线
    with Timer("理论曲线"):
        ax6 = fig.add_subplot(gs[1, 2])
        beta_range = np.linspace(0.5, 1.0 - 1e-9, 1000)
        ent = -0.5 * np.log(1 - beta_range ** 2)
        ax6.plot(beta_range, ent, color='#2196F3', linewidth=2)
        ax6.set_xlabel('β'); ax6.set_ylabel('Entropy per neuron (nats)')
        ax6.set_title('Single Neuron State Entropy vs β')
        ax6.set_yscale('log')
        ax6.axvline(x=1-1e-7, color='red', ls='--', alpha=0.7, label='default β')
        ax6.axvline(x=0.999, color='orange', ls='--', alpha=0.7, label='β=0.999')
        ax6.axvline(x=0.99, color='green', ls='--', alpha=0.7, label='β=0.99')
        ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

    with Timer("保存图像"):
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # 实验 1
    v_a, v_b, fisher = run_separability_experiment(device=device)

    # 实验 2
    v_samples, h_theory, h_empirical = run_entropy_probe(device=device)

    # 实验 3
    beta_results = run_multiscale_beta_experiment(device=device)

    # 可视化
    save_path = os.path.join(os.path.dirname(__file__), 'temporal_entropy_results.png')
    visualize_results(v_a, v_b, fisher, v_samples, h_theory, h_empirical,
                      beta_results, save_path=save_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"所有实验完成 (总耗时: {elapsed:.1f}s)")
    print(f"{'='*70}")
