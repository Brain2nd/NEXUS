"""
Deep Chaos Analysis: 门电路拓扑隐式耦合的理论验证
==================================================

核心命题：NEXUS 的门电路级联在 TEMPORAL 模式下构成隐式耦合非线性动力系统。
本实验从四个理论角度验证这一命题：

实验 A: 关联维数 (Correlation Dimension, D₂)
    - Grassberger-Procaccia 算法
    - D₂ 为非整数 → 分形吸引子（混沌的标志）
    - D₂ 为整数 → 极限环或不动点
    - 对比不同 β 下的 D₂

实验 B: 递归量化分析 (Recurrence Quantification Analysis, RQA)
    - 递归图 (Recurrence Plot) 区分确定性 vs 随机
    - DET (确定性): 确定性系统 > 随机系统
    - ENTR (Shannon 熵): 混沌系统 > 周期系统
    - LAM (层流性): 间歇混沌的标志

实验 C: 门电路层间互信息 (Mutual Information)
    - 如果门电路拓扑构成耦合，同一 FP32 乘法器内的神经元应有更高 MI
    - 不同组件间 MI 应低于同组件内 MI
    - MI 随 β 变化的模式揭示耦合强度

实验 D: 相变序参量 (Order Parameter)
    - 高分辨率 β 扫描 (0.40 ~ 0.99, 30 个点)
    - 序参量: Reset 触发率 (spike rate)、V(t) 自相关时间、最大 Lyapunov 指数
    - 寻找相变临界点 β_c
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from scipy import spatial

print("[import] torch, numpy, scipy done", flush=True)

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
# Model & Utils (same as beta_sweep)
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


def create_model(device, beta, seed=42):
    torch.manual_seed(seed)
    in_f, hid_f, out_f = 4, 8, 4
    template = SimpleLIFNode(beta=beta)
    model = SimpleSpikeMLP(in_f, hid_f, out_f, neuron_template=template).to(device)
    w1 = torch.randn(hid_f, in_f, device=device) * 0.5
    w2 = torch.randn(out_f, hid_f, device=device) * 0.5
    model.set_weights(w1, w2)
    return model, in_f


def forward_one_step(model, x_float, device):
    x_pulse = float32_to_pulse(x_float, device=device)
    _ = model(x_pulse.unsqueeze(0))


def collect_all_v(model):
    """收集所有 LIF 的 V，按组件路径分组"""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            result[name] = module.v.detach().cpu().numpy().flatten()
    return result


def collect_neuron_traces(model, device, in_f, beta, T=500, seed=777):
    """收集长时间序列的所有神经元 V(t)"""
    torch.manual_seed(seed)
    inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

    model.reset()
    all_traces = {}  # name -> list of scalar

    with SpikeMode.temporal():
        for t in range(T):
            forward_one_step(model, inputs[t], device)
            v_dict = collect_all_v(model)
            for name, v_flat in v_dict.items():
                nz = np.nonzero(np.abs(v_flat) > 1e-30)[0]
                scalar = float(v_flat[nz[0]]) if len(nz) > 0 else 0.0
                if name not in all_traces:
                    all_traces[name] = []
                all_traces[name].append(scalar)

            if (t+1) % 100 == 0:
                print(f"    [{t+1}/{T}]", flush=True)

    # Filter non-trivial traces
    valid = {}
    for name, trace in all_traces.items():
        arr = np.array(trace)
        if np.std(arr) > 1e-10:
            valid[name] = arr
    return valid


# =============================================================================
# 实验 A: 关联维数 (Grassberger-Procaccia)
# =============================================================================

def correlation_dimension(trace, tau=1, d_max=8, n_points=2000):
    """
    Grassberger-Procaccia 算法估计关联维数 D₂

    C(r) = (2 / N(N-1)) Σ_{i<j} H(r - ||X_i - X_j||)
    D₂ = lim_{r→0} d(ln C(r)) / d(ln r)

    对不同嵌入维度 d=1..d_max 计算 D₂，D₂ 在 d 充分大时应饱和。
    """
    results = {}
    trace = np.array(trace, dtype=np.float64)

    for d in range(2, min(d_max + 1, len(trace))):
        N = len(trace) - (d - 1) * tau
        if N < 50:
            break

        # Construct delay embedding
        embedded = np.zeros((N, d))
        for dim in range(d):
            embedded[:, dim] = trace[dim * tau: dim * tau + N]

        # Subsample if too many points
        if N > n_points:
            idx = np.random.choice(N, n_points, replace=False)
            embedded = embedded[idx]
            N = n_points

        # Compute pairwise distances
        dists = spatial.distance.pdist(embedded)
        dists = dists[dists > 0]  # remove zeros

        if len(dists) < 100:
            continue

        # Correlation integral for various r
        r_min = np.percentile(dists, 1)
        r_max = np.percentile(dists, 99)
        if r_min <= 0 or r_max <= r_min:
            continue

        rs = np.logspace(np.log10(r_min), np.log10(r_max), 30)
        Cr = np.array([np.mean(dists < r) for r in rs])

        # Filter valid points
        valid = (Cr > 1e-10) & (Cr < 1 - 1e-10)
        if valid.sum() < 5:
            continue

        log_r = np.log(rs[valid])
        log_C = np.log(Cr[valid])

        # Linear regression on middle portion (avoid edge effects)
        n_valid = len(log_r)
        start = n_valid // 4
        end = 3 * n_valid // 4
        if end - start < 3:
            start = 0
            end = n_valid

        coeffs = np.polyfit(log_r[start:end], log_C[start:end], 1)
        D2 = coeffs[0]  # slope = correlation dimension

        results[d] = {
            'D2': D2,
            'log_r': log_r,
            'log_C': log_C,
            'n_points': N,
        }

    return results


def run_correlation_dimension(device):
    """对不同 β 计算关联维数"""
    print("\n" + "=" * 70)
    print("实验 A: 关联维数 (Grassberger-Procaccia)")
    print("=" * 70)

    betas = [0.50, 0.70, 0.90, 0.99, 1.0 - 1e-7]
    T = 500
    all_results = {}

    for beta in betas:
        print(f"\n  β = {beta:.7f}")
        model, in_f = create_model(device, beta)
        traces = collect_neuron_traces(model, device, in_f, beta, T=T)

        # Pick top-3 variance traces
        sorted_names = sorted(traces.keys(), key=lambda n: np.std(traces[n]), reverse=True)
        top3 = sorted_names[:3]

        beta_results = {}
        for name in top3:
            trace = traces[name]
            short = name.split('.')[-1][:25]
            d2_results = correlation_dimension(trace, tau=1, d_max=8)
            if d2_results:
                # Take D2 at highest embedding dimension
                max_d = max(d2_results.keys())
                D2 = d2_results[max_d]['D2']
                print(f"    {short}: D₂ = {D2:.3f} (d_embed={max_d})")
                beta_results[name] = d2_results
            else:
                print(f"    {short}: 数据不足")

        all_results[beta] = beta_results

    return all_results


# =============================================================================
# 实验 B: 递归量化分析 (RQA)
# =============================================================================

def recurrence_plot_metrics(trace, epsilon=None, tau=1, d=3):
    """
    计算递归图 (Recurrence Plot) 及 RQA 指标

    R(i,j) = H(ε - ||X_i - X_j||)

    指标:
    - RR (Recurrence Rate): 递归点占比
    - DET (Determinism): 对角线结构占比 → 确定性系统高
    - ENTR (Diagonal line entropy): 对角线长度分布的 Shannon 熵 → 混沌高
    - LAM (Laminarity): 垂直线结构占比 → 间歇性/层流高
    """
    trace = np.array(trace, dtype=np.float64)
    N = len(trace) - (d - 1) * tau
    if N < 50:
        return None

    # Delay embedding
    embedded = np.zeros((N, d))
    for dim in range(d):
        embedded[:, dim] = trace[dim * tau: dim * tau + N]

    # Subsample for speed
    max_n = 800
    if N > max_n:
        idx = np.linspace(0, N-1, max_n, dtype=int)
        embedded = embedded[idx]
        N = max_n

    # Pairwise distances
    dist_matrix = spatial.distance.cdist(embedded, embedded)

    # Auto-select epsilon if not given (10% of mean distance)
    if epsilon is None:
        epsilon = 0.1 * np.mean(dist_matrix)

    # Recurrence matrix
    R = (dist_matrix < epsilon).astype(int)
    np.fill_diagonal(R, 0)  # exclude self-recurrence

    N_pairs = N * (N - 1)
    RR = np.sum(R) / N_pairs if N_pairs > 0 else 0

    # DET: fraction of recurrence points forming diagonal lines (length >= 2)
    diag_lengths = []
    for k in range(-N + 2, N - 1):
        diag = np.diag(R, k)
        # Count runs of 1s
        in_run = False
        run_len = 0
        for val in diag:
            if val == 1:
                run_len += 1
                in_run = True
            else:
                if in_run and run_len >= 2:
                    diag_lengths.append(run_len)
                run_len = 0
                in_run = False
        if in_run and run_len >= 2:
            diag_lengths.append(run_len)

    total_diag_points = sum(diag_lengths) if diag_lengths else 0
    total_recurrence = np.sum(R)
    DET = total_diag_points / total_recurrence if total_recurrence > 0 else 0

    # ENTR: Shannon entropy of diagonal line length distribution
    if diag_lengths:
        lengths, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / counts.sum()
        ENTR = -np.sum(probs * np.log(probs + 1e-30))
    else:
        ENTR = 0.0

    # LAM: fraction of recurrence points forming vertical lines (length >= 2)
    vert_lengths = []
    for col in range(N):
        in_run = False
        run_len = 0
        for row in range(N):
            if R[row, col] == 1:
                run_len += 1
                in_run = True
            else:
                if in_run and run_len >= 2:
                    vert_lengths.append(run_len)
                run_len = 0
                in_run = False
        if in_run and run_len >= 2:
            vert_lengths.append(run_len)

    total_vert_points = sum(vert_lengths) if vert_lengths else 0
    LAM = total_vert_points / total_recurrence if total_recurrence > 0 else 0

    return {
        'RR': RR,
        'DET': DET,
        'ENTR': ENTR,
        'LAM': LAM,
        'R_matrix': R,
        'epsilon': epsilon,
        'N': N,
    }


def run_rqa(device):
    """对不同 β 计算 RQA 指标"""
    print("\n" + "=" * 70)
    print("实验 B: 递归量化分析 (RQA)")
    print("=" * 70)

    betas = [0.50, 0.70, 0.90, 0.99, 1.0 - 1e-7]
    T = 500
    all_results = {}

    for beta in betas:
        print(f"\n  β = {beta:.7f}")
        model, in_f = create_model(device, beta)
        traces = collect_neuron_traces(model, device, in_f, beta, T=T)

        sorted_names = sorted(traces.keys(), key=lambda n: np.std(traces[n]), reverse=True)
        top_name = sorted_names[0] if sorted_names else None

        if top_name is None:
            print("    无有效 trace")
            continue

        trace = traces[top_name]
        short = top_name.split('.')[-1][:25]

        rqa = recurrence_plot_metrics(trace, tau=1, d=3)
        if rqa:
            print(f"    {short}:")
            print(f"      RR  = {rqa['RR']:.4f} (递归率)")
            print(f"      DET = {rqa['DET']:.4f} (确定性)")
            print(f"      ENTR= {rqa['ENTR']:.4f} (对角线熵)")
            print(f"      LAM = {rqa['LAM']:.4f} (层流性)")
            print(f"      ε   = {rqa['epsilon']:.4f}")
            all_results[beta] = rqa
            all_results[beta]['trace_name'] = short

    return all_results


# =============================================================================
# 实验 C: 门电路层间互信息
# =============================================================================

def mutual_information_discrete(x, y, n_bins=20):
    """离散化互信息 I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Discretize
    x_bins = np.digitize(x, np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1))
    y_bins = np.digitize(y, np.linspace(y.min() - 1e-10, y.max() + 1e-10, n_bins + 1))

    # Joint histogram
    joint = np.zeros((n_bins + 2, n_bins + 2))
    for xi, yi in zip(x_bins, y_bins):
        joint[xi, yi] += 1
    joint /= joint.sum()

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # H(X), H(Y), H(X,Y)
    Hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    Hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    Hxy = -np.sum(joint[joint > 0] * np.log(joint[joint > 0]))

    MI = Hx + Hy - Hxy
    # Normalized MI
    NMI = MI / min(Hx, Hy) if min(Hx, Hy) > 1e-10 else 0.0

    return MI, NMI


def run_mutual_information(device):
    """测量门电路层间互信息"""
    print("\n" + "=" * 70)
    print("实验 C: 门电路层间互信息")
    print("=" * 70)

    beta = 0.90  # 使用最有趣的 β 值
    T = 500
    print(f"  β = {beta}, T = {T}")

    model, in_f = create_model(device, beta)
    traces = collect_neuron_traces(model, device, in_f, beta, T=T)

    # Classify neurons by component
    components = {}
    for name in traces:
        # Extract component path (e.g., linear1.mul.lzd vs linear2.acc)
        parts = name.split('.')
        if len(parts) >= 3:
            comp = '.'.join(parts[:3])  # e.g., linear1.mul.lzd
        elif len(parts) >= 2:
            comp = '.'.join(parts[:2])
        else:
            comp = parts[0]

        if comp not in components:
            components[comp] = []
        components[comp].append(name)

    print(f"  组件数: {len(components)}")
    for comp, names in sorted(components.items()):
        print(f"    {comp}: {len(names)} 个神经元")

    # Compute intra-component MI vs inter-component MI
    comp_list = sorted(components.keys())
    n_comp = len(comp_list)

    # Pick one representative neuron per component (highest variance)
    reps = {}
    for comp in comp_list:
        names = components[comp]
        best = max(names, key=lambda n: np.std(traces[n]))
        reps[comp] = best

    # MI matrix
    mi_matrix = np.zeros((n_comp, n_comp))
    nmi_matrix = np.zeros((n_comp, n_comp))

    for i, ci in enumerate(comp_list):
        for j, cj in enumerate(comp_list):
            if i <= j:
                ti = traces[reps[ci]]
                tj = traces[reps[cj]]
                mi, nmi = mutual_information_discrete(ti, tj)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                nmi_matrix[i, j] = nmi
                nmi_matrix[j, i] = nmi

    # Compute average intra vs inter MI
    # "Intra" = same top-level component (linear1 vs linear2)
    intra_mis = []
    inter_mis = []
    for i, ci in enumerate(comp_list):
        for j, cj in enumerate(comp_list):
            if i < j:
                top_i = ci.split('.')[0]  # linear1 or linear2
                top_j = cj.split('.')[0]
                if top_i == top_j:
                    intra_mis.append(mi_matrix[i, j])
                else:
                    inter_mis.append(mi_matrix[i, j])

    print(f"\n  层内平均 MI: {np.mean(intra_mis):.4f} ± {np.std(intra_mis):.4f}" if intra_mis else "  层内: N/A")
    print(f"  层间平均 MI: {np.mean(inter_mis):.4f} ± {np.std(inter_mis):.4f}" if inter_mis else "  层间: N/A")

    if intra_mis and inter_mis:
        ratio = np.mean(intra_mis) / (np.mean(inter_mis) + 1e-10)
        print(f"  层内/层间 MI 比值: {ratio:.2f}")
        print(f"  → {'层内耦合显著强于层间' if ratio > 1.5 else '层内层间耦合相当' if ratio > 0.7 else '层间耦合更强'}")

    return {
        'mi_matrix': mi_matrix,
        'nmi_matrix': nmi_matrix,
        'comp_list': comp_list,
        'reps': reps,
        'intra_mis': intra_mis,
        'inter_mis': inter_mis,
    }


# =============================================================================
# 实验 D: 高分辨率相变序参量
# =============================================================================

def run_phase_transition(device):
    """高分辨率 β 扫描，寻找相变临界点"""
    print("\n" + "=" * 70)
    print("实验 D: 高分辨率相变序参量")
    print("=" * 70)

    betas = np.concatenate([
        np.arange(0.40, 0.70, 0.05),   # 强衰减区
        np.arange(0.70, 0.92, 0.02),    # 过渡区（密采样）
        np.arange(0.92, 1.00, 0.01),    # 近积分器区
    ])
    betas = np.unique(np.round(betas, 4))

    T = 200
    in_f = 4
    results = []

    for i, beta in enumerate(betas):
        print(f"  [{i+1}/{len(betas)}] β={beta:.4f}...", end='', flush=True)
        t0 = time.time()

        model, _ = create_model(device, beta)
        torch.manual_seed(777)
        inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

        model.reset()
        v_trace = []
        spike_counts = []

        with SpikeMode.temporal():
            for t in range(T):
                forward_one_step(model, inputs[t], device)

                # Collect V and spike count
                v_vals = []
                n_spikes = 0
                for name, module in model.named_modules():
                    if isinstance(module, SimpleLIFNode) and module.v is not None:
                        v_flat = module.v.detach().cpu().numpy().flatten()
                        nz = np.nonzero(np.abs(v_flat) > 1e-30)[0]
                        if len(nz) > 0:
                            v_vals.append(float(v_flat[nz[0]]))
                        # Count spikes (V that got reset = V < threshold and was recently above)
                        vth = module.v_threshold.flatten()[0].item() if torch.is_tensor(module.v_threshold) else float(module.v_threshold)
                        n_spikes += np.sum(v_flat < 0.1 * vth)

                if v_vals:
                    v_trace.append(np.mean(v_vals[:5]))  # average of top neurons
                spike_counts.append(n_spikes)

        trace = np.array(v_trace) if v_trace else np.zeros(T)
        spikes = np.array(spike_counts, dtype=float)

        # Order parameters
        # 1. V range (bounded vs unbounded)
        v_range = trace.max() - trace.min() if len(trace) > 0 else 0

        # 2. Spike rate (reset frequency)
        spike_rate = np.mean(spikes) if len(spikes) > 0 else 0

        # 3. Autocorrelation time (how fast V decorrelates)
        if len(trace) > 20 and np.std(trace) > 1e-10:
            trace_centered = trace - np.mean(trace)
            autocorr = np.correlate(trace_centered, trace_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr /= autocorr[0] if autocorr[0] > 0 else 1
            # Find first crossing below 1/e
            tau_corr = len(autocorr)
            for k in range(1, len(autocorr)):
                if autocorr[k] < 1.0 / np.e:
                    tau_corr = k
                    break
        else:
            tau_corr = T  # no decorrelation

        # 4. V(t) std (fluctuation amplitude)
        v_std = np.std(trace) if len(trace) > 0 else 0

        # 5. Quick MLE estimate
        mle = float('nan')
        # (reuse Lyapunov from beta_sweep logic, simplified)

        dt = time.time() - t0
        print(f" range={v_range:.2f}, τ_corr={tau_corr}, spike_rate={spike_rate:.0f} ({dt:.1f}s)")

        results.append({
            'beta': beta,
            'v_range': v_range,
            'v_std': v_std,
            'spike_rate': spike_rate,
            'tau_corr': tau_corr,
            'trace': trace,
        })

    return results


# =============================================================================
# Visualization
# =============================================================================

def visualize_deep(corr_dim_results, rqa_results, mi_results, phase_results, save_path):
    print("\n" + "=" * 70)
    print("生成可视化")
    print("=" * 70)

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('NEXUS Deep Chaos Analysis\n'
                 'Gate Circuit Topology as Implicit Coupling Network',
                 fontsize=14, fontweight='bold', y=0.99)

    # --- (1) Correlation Dimension vs embedding dim, per β ---
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
    for i, (beta, beta_res) in enumerate(corr_dim_results.items()):
        if not beta_res:
            continue
        # Take first trace
        first_name = list(beta_res.keys())[0]
        d2_data = beta_res[first_name]
        dims = sorted(d2_data.keys())
        d2_vals = [d2_data[d]['D2'] for d in dims]
        label = f'β={beta:.2f}' if beta < 0.999 else 'β≈1'
        ax1.plot(dims, d2_vals, 'o-', color=colors[i % len(colors)], label=label, linewidth=2)
    ax1.set_xlabel('Embedding Dimension d')
    ax1.set_ylabel('Correlation Dimension D₂')
    ax1.set_title('D₂ vs Embedding Dim\n(saturation → true D₂)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # --- (2) log C(r) vs log r scaling (one β) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Pick β=0.90
    if 0.90 in corr_dim_results and corr_dim_results[0.90]:
        first_name = list(corr_dim_results[0.90].keys())[0]
        d2_data = corr_dim_results[0.90][first_name]
        for d in sorted(d2_data.keys()):
            lr = d2_data[d]['log_r']
            lc = d2_data[d]['log_C']
            ax2.plot(lr, lc, '-', linewidth=1.5, label=f'd={d}, D₂={d2_data[d]["D2"]:.2f}')
        ax2.set_xlabel('ln(r)')
        ax2.set_ylabel('ln(C(r))')
        ax2.set_title('Correlation Integral (β=0.90)\nSlope = D₂')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    # --- (3) RQA metrics bar chart ---
    ax3 = fig.add_subplot(gs[0, 2])
    if rqa_results:
        rqa_betas = sorted(rqa_results.keys())
        x = range(len(rqa_betas))
        det_vals = [rqa_results[b]['DET'] for b in rqa_betas]
        entr_vals = [rqa_results[b]['ENTR'] for b in rqa_betas]
        lam_vals = [rqa_results[b]['LAM'] for b in rqa_betas]

        width = 0.25
        ax3.bar([xi - width for xi in x], det_vals, width, label='DET', color='#4CAF50')
        ax3.bar(list(x), entr_vals, width, label='ENTR', color='#FF9800')
        ax3.bar([xi + width for xi in x], lam_vals, width, label='LAM', color='#2196F3')
        ax3.set_xticks(list(x))
        ax3.set_xticklabels([f'{b:.2f}' if b < 0.999 else '≈1' for b in rqa_betas], fontsize=8)
        ax3.set_xlabel('β')
        ax3.set_title('RQA Metrics vs β\n(DET=确定性, ENTR=复杂度, LAM=层流)')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

    # --- (4) Recurrence Plot (β=0.90) ---
    ax4 = fig.add_subplot(gs[0, 3])
    if 0.90 in rqa_results:
        R = rqa_results[0.90]['R_matrix']
        ax4.imshow(R[:200, :200], cmap='binary', origin='lower', aspect='equal')
        ax4.set_xlabel('i')
        ax4.set_ylabel('j')
        rr = rqa_results[0.90]['RR']
        det = rqa_results[0.90]['DET']
        ax4.set_title(f'Recurrence Plot (β=0.90)\nRR={rr:.3f}, DET={det:.3f}')

    # --- (5) Recurrence Plot (β=0.50) ---
    ax5 = fig.add_subplot(gs[1, 0])
    if 0.50 in rqa_results:
        R = rqa_results[0.50]['R_matrix']
        ax5.imshow(R[:200, :200], cmap='binary', origin='lower', aspect='equal')
        ax5.set_xlabel('i')
        ax5.set_ylabel('j')
        rr = rqa_results[0.50]['RR']
        det = rqa_results[0.50]['DET']
        ax5.set_title(f'Recurrence Plot (β=0.50)\nRR={rr:.3f}, DET={det:.3f}')

    # --- (6) Recurrence Plot (β≈1) ---
    ax6 = fig.add_subplot(gs[1, 1])
    b_approx1 = 1.0 - 1e-7
    if b_approx1 in rqa_results:
        R = rqa_results[b_approx1]['R_matrix']
        ax6.imshow(R[:200, :200], cmap='binary', origin='lower', aspect='equal')
        ax6.set_xlabel('i')
        ax6.set_ylabel('j')
        rr = rqa_results[b_approx1]['RR']
        det = rqa_results[b_approx1]['DET']
        ax6.set_title(f'Recurrence Plot (β≈1)\nRR={rr:.3f}, DET={det:.3f}')

    # --- (7) MI matrix ---
    ax7 = fig.add_subplot(gs[1, 2])
    if mi_results and mi_results['mi_matrix'] is not None:
        mi_mat = mi_results['mi_matrix']
        im = ax7.imshow(mi_mat, cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax7, fraction=0.046)
        n_comp = len(mi_results['comp_list'])
        short_labels = [c.split('.')[-1][:8] for c in mi_results['comp_list']]
        if n_comp <= 15:
            ax7.set_xticks(range(n_comp))
            ax7.set_xticklabels(short_labels, rotation=90, fontsize=5)
            ax7.set_yticks(range(n_comp))
            ax7.set_yticklabels(short_labels, fontsize=5)
        ax7.set_title('Mutual Information Matrix (β=0.90)\nBright = strong coupling')

    # --- (8) Intra vs Inter MI ---
    ax8 = fig.add_subplot(gs[1, 3])
    if mi_results and mi_results['intra_mis'] and mi_results['inter_mis']:
        ax8.boxplot([mi_results['intra_mis'], mi_results['inter_mis']],
                    labels=['Intra-layer', 'Inter-layer'])
        ax8.set_ylabel('Mutual Information (nats)')
        ax8.set_title('Layer Coupling Structure\nIntra vs Inter MI')
        ax8.grid(True, alpha=0.3)

        # Add means
        ax8.scatter([1], [np.mean(mi_results['intra_mis'])], color='red', s=100, zorder=5, marker='D', label='mean')
        ax8.scatter([2], [np.mean(mi_results['inter_mis'])], color='red', s=100, zorder=5, marker='D')
        ax8.legend()

    # --- (9-12) Phase transition order parameters ---
    if phase_results:
        betas_pt = [r['beta'] for r in phase_results]

        # V range
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.plot(betas_pt, [r['v_range'] for r in phase_results], 'o-', color='#FF5722', linewidth=2)
        ax9.set_xlabel('β')
        ax9.set_ylabel('V(t) Range')
        ax9.set_title('Order Param 1: V Range\n(integrator→oscillator transition)')
        ax9.grid(True, alpha=0.3)

        # Autocorrelation time
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.plot(betas_pt, [r['tau_corr'] for r in phase_results], 's-', color='#9C27B0', linewidth=2)
        ax10.set_xlabel('β')
        ax10.set_ylabel('τ_corr (steps)')
        ax10.set_title('Order Param 2: Autocorrelation Time\n(divergent at phase transition)')
        ax10.grid(True, alpha=0.3)

        # V std
        ax11 = fig.add_subplot(gs[2, 2])
        ax11.plot(betas_pt, [r['v_std'] for r in phase_results], 'D-', color='#4CAF50', linewidth=2)
        ax11.set_xlabel('β')
        ax11.set_ylabel('V(t) Std')
        ax11.set_title('Order Param 3: Fluctuation Amplitude')
        ax11.grid(True, alpha=0.3)

        # Spike rate
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.plot(betas_pt, [r['spike_rate'] for r in phase_results], '^-', color='#2196F3', linewidth=2)
        ax12.set_xlabel('β')
        ax12.set_ylabel('Spike Rate (per step)')
        ax12.set_title('Order Param 4: Reset Frequency\n(nonlinearity activation)')
        ax12.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # A: Correlation Dimension
    corr_results = run_correlation_dimension(device) if only in (None, 'A') else {}

    # B: RQA
    rqa_results = run_rqa(device) if only in (None, 'B') else {}

    # C: Mutual Information
    mi_results = run_mutual_information(device) if only in (None, 'C') else {}

    # D: Phase Transition
    phase_results = run_phase_transition(device) if only in (None, 'D') else {}

    # Visualize
    save_path = os.path.join(os.path.dirname(__file__), 'deep_chaos_analysis.png')
    visualize_deep(corr_results, rqa_results, mi_results, phase_results, save_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Deep Chaos Analysis 完成 (总耗时: {elapsed:.1f}s)")
    print(f"{'='*70}")

    # Summary
    print("\n[A] 关联维数 D₂:")
    for beta, beta_res in corr_results.items():
        bl = f'{beta:.2f}' if beta < 0.999 else '≈1'
        for name, d2_data in beta_res.items():
            max_d = max(d2_data.keys())
            D2 = d2_data[max_d]['D2']
            short = name.split('.')[-1][:20]
            print(f"  β={bl}: D₂ = {D2:.3f} (d={max_d}, {short})")

    print("\n[B] RQA 指标:")
    for beta, rqa in rqa_results.items():
        bl = f'{beta:.2f}' if beta < 0.999 else '≈1'
        if isinstance(rqa, dict) and 'DET' in rqa:
            print(f"  β={bl}: DET={rqa['DET']:.3f}, ENTR={rqa['ENTR']:.3f}, LAM={rqa['LAM']:.3f}")

    print("\n[C] 互信息:")
    if mi_results and mi_results['intra_mis'] and mi_results['inter_mis']:
        print(f"  层内 MI: {np.mean(mi_results['intra_mis']):.4f}")
        print(f"  层间 MI: {np.mean(mi_results['inter_mis']):.4f}")
        ratio = np.mean(mi_results['intra_mis']) / (np.mean(mi_results['inter_mis']) + 1e-10)
        print(f"  比值: {ratio:.2f}")

    print("\n[D] 相变序参量 (关键转变点):")
    if phase_results:
        # Find beta where tau_corr is maximal
        max_tau = max(phase_results, key=lambda r: r['tau_corr'])
        print(f"  最大自相关时间: β={max_tau['beta']:.4f}, τ={max_tau['tau_corr']}")
        # Find beta where v_range derivative is maximal
        v_ranges = [r['v_range'] for r in phase_results]
        if len(v_ranges) > 2:
            dv = np.diff(v_ranges)
            db = np.diff([r['beta'] for r in phase_results])
            deriv = dv / (db + 1e-10)
            max_idx = np.argmax(np.abs(deriv))
            print(f"  V_range 最大变化率: β ≈ {phase_results[max_idx]['beta']:.4f} → {phase_results[max_idx+1]['beta']:.4f}")
