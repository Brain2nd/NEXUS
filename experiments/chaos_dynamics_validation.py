"""
混沌动力学验证实验
==================

三个基础验证实验，测试 NEXUS TEMPORAL 模式下纯 SNN 门电路系统的动力学稳定性。

实验 1: Lyapunov 指数测定
    - 微扰初始条件，观测轨道发散/收敛速率
    - λ > 0 => 混沌（蝴蝶效应，梯度下降可能无效）
    - λ < 0 => 稳定（耗散系统，适合训练）
    - λ ≈ 0 => 边缘（临界态）

实验 2: Takens 吸引子延迟嵌入
    - 单个神经元膜电位残差的相空间重构
    - 揭示底层动力学结构：固定点/极限环/奇异吸引子

实验 3: Reservoir Memory Capacity (MC)
    - 线性回归测算蓄水池对过去输入的记忆能力
    - MC = Σ_k r²(y_k, u_{t-k})，理论上界 = 神经元数

被测系统: 与 temporal_entropy_probe.py 完全一致的 2 层纯 SNN MLP
    - SimpleSpikeMLP(4, 8, 4)
    - SpikeFP32Linear_MultiPrecision，内部为纯 SNN 门电路
    - 约 12,554 个 LIF 神经元

运行: conda activate dapo && python experiments/chaos_dynamics_validation.py
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

from atomic_ops import (
    SpikeMode,
    SpikeFP32Linear_MultiPrecision,
    SimpleLIFNode,
)
from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32
print("[import] atomic_ops done", flush=True)


# =============================================================================
# 计时工具 & 模型（与 temporal_entropy_probe.py 完全一致）
# =============================================================================

class Timer:
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
    if i == 0 or (i + 1) % every == 0 or i + 1 == total:
        print(f"    {prefix}[{i+1}/{total}]", flush=True)


class SimpleSpikeMLP(nn.Module):
    """与 temporal_entropy_probe.py 完全一致的 2 层纯 SNN MLP"""
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


def collect_lif_membrane_potentials(model):
    """递归收集模型中所有 LIF 神经元的膜电位 V"""
    potentials = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            potentials[name] = module.v.detach().clone()
    return potentials


def create_model(device, seed=42):
    """创建与 temporal_entropy_probe.py 一致的模型"""
    torch.manual_seed(seed)
    in_f, hid_f, out_f = 4, 8, 4
    model = SimpleSpikeMLP(in_f, hid_f, out_f).to(device)
    w1 = torch.randn(hid_f, in_f, device=device) * 0.5
    w2 = torch.randn(out_f, hid_f, device=device) * 0.5
    model.set_weights(w1, w2)
    return model, in_f


def get_state_vector(model):
    """将所有 LIF 膜电位展平为单一状态向量"""
    v_dict = collect_lif_membrane_potentials(model)
    if not v_dict:
        return None
    parts = [v.cpu().numpy().flatten() for v in v_dict.values()]
    vec = np.concatenate(parts)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def forward_one_step(model, x_float, device):
    """执行一步 forward（输入为 float tensor，自动编码）"""
    x_pulse = float32_to_pulse(x_float, device=device)
    _ = model(x_pulse.unsqueeze(0))


# =============================================================================
# 实验 1: Lyapunov 指数测定
# =============================================================================

def run_lyapunov_experiment(device='cpu'):
    """
    最大 Lyapunov 指数 (MLE) 测定

    方法: 双轨道法 (twin trajectory)
    1. 初始化两条轨道，初始条件相差微扰 δ₀
    2. 每步后计算距离 δ(t) = ||V_1(t) - V_2(t)||
    3. MLE ≈ (1/T) * Σ ln(δ(t)/δ(t-1))

    但由于我们的系统是通过输入驱动的（非自治），
    采用更适合的方法: 固定输入序列，微扰初始膜电位，
    观测轨道发散率。
    """
    print("=" * 70)
    print("实验 1: Lyapunov 指数测定 (最大 Lyapunov 指数)")
    print("=" * 70)

    with Timer("创建参考模型"):
        model_ref, in_f = create_model(device, seed=42)

    # 生成固定输入序列
    torch.manual_seed(999)
    T = 50  # 时间步数
    inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

    # 微扰幅度
    perturbation_scales = [1e-8, 1e-6, 1e-4, 1e-2]
    results = {}

    for eps in perturbation_scales:
        print(f"\n  --- 微扰幅度 ε = {eps:.0e} ---")

        # 参考轨道
        model_ref.reset()
        with SpikeMode.temporal():
            # 先跑一步让系统产生膜电位
            forward_one_step(model_ref, inputs[0], device)
            v_ref_init = get_state_vector(model_ref)

        if v_ref_init is None:
            print("  ✗ 无法获取膜电位，跳过")
            continue

        # 微扰轨道：在膜电位上加微扰
        model_pert, _ = create_model(device, seed=42)
        model_pert.reset()
        with SpikeMode.temporal():
            forward_one_step(model_pert, inputs[0], device)
            # 对所有 LIF 神经元的 V 加微扰
            for name, module in model_pert.named_modules():
                if isinstance(module, SimpleLIFNode) and module.v is not None:
                    perturbation = torch.randn_like(module.v) * eps
                    module.v.add_(perturbation)

        v_pert_init = get_state_vector(model_pert)
        delta_0 = np.linalg.norm(v_ref_init - v_pert_init)
        print(f"  初始距离 δ₀ = {delta_0:.6e}")

        # 跟踪两条轨道
        distances = [delta_0]
        lyap_increments = []

        with SpikeMode.temporal():
            for t in range(1, T):
                forward_one_step(model_ref, inputs[t], device)
                forward_one_step(model_pert, inputs[t], device)

                v_ref = get_state_vector(model_ref)
                v_pert = get_state_vector(model_pert)

                if v_ref is None or v_pert is None:
                    break

                delta_t = np.linalg.norm(v_ref - v_pert)
                distances.append(delta_t)

                if delta_t > 1e-30 and distances[-2] > 1e-30:
                    lyap_increments.append(np.log(delta_t / distances[-2]))

                if (t + 1) % 10 == 0:
                    print(f"    t={t+1}: δ(t) = {delta_t:.6e}", flush=True)

        distances = np.array(distances)
        if lyap_increments:
            mle = np.mean(lyap_increments)
            mle_std = np.std(lyap_increments) / np.sqrt(len(lyap_increments))
        else:
            mle = float('nan')
            mle_std = float('nan')

        results[eps] = {
            'distances': distances,
            'mle': mle,
            'mle_std': mle_std,
            'delta_0': delta_0,
        }
        print(f"  MLE = {mle:.6f} ± {mle_std:.6f} (nats/step)")
        if mle > 0:
            print(f"  ⚠ λ > 0: 系统可能混沌（轨道发散）")
        elif mle < -0.01:
            print(f"  ✓ λ < 0: 系统稳定（耗散，轨道收敛）")
        else:
            print(f"  ~ λ ≈ 0: 边缘稳定性")

    return results


# =============================================================================
# 实验 2: Takens 吸引子延迟嵌入
# =============================================================================

def run_takens_embedding(device='cpu'):
    """
    Takens 延迟嵌入定理:
    从单个标量时间序列 {x(t)} 重构相空间:
    X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

    选取几个代表性 LIF 神经元，记录膜电位随时间的演化，
    用延迟嵌入重构相空间。

    吸引子类型判断:
    - 固定点: 所有点聚集在一点
    - 极限环: 闭合曲线
    - 奇异吸引子: 分形结构（混沌）
    - 无结构: 随机游走
    """
    print("\n" + "=" * 70)
    print("实验 2: Takens 吸引子延迟嵌入可视化")
    print("=" * 70)

    with Timer("创建模型"):
        model, in_f = create_model(device, seed=42)

    # 长时间序列记录
    T = 200  # 时间步
    torch.manual_seed(777)
    inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

    # 选取要追踪的神经元（按名称选前几个有 V 的）
    print(f"  记录 {T} 步膜电位时间序列...")
    neuron_traces = {}  # name -> list of scalar values

    model.reset()
    with SpikeMode.temporal():
        for t in range(T):
            forward_one_step(model, inputs[t], device)

            v_dict = collect_lif_membrane_potentials(model)
            for name, v_tensor in v_dict.items():
                v_flat = v_tensor.cpu().numpy().flatten()
                # 取每个神经元模块的第一个非零值作为标量
                nonzero_idx = np.nonzero(np.abs(v_flat) > 1e-30)[0]
                if len(nonzero_idx) > 0:
                    scalar_val = float(v_flat[nonzero_idx[0]])
                else:
                    scalar_val = 0.0

                if name not in neuron_traces:
                    neuron_traces[name] = []
                neuron_traces[name].append(scalar_val)

            progress(t, T, prefix='记录 ', every=50)

    # 过滤掉全零或近常数的 trace
    valid_traces = {}
    for name, trace in neuron_traces.items():
        arr = np.array(trace)
        if np.std(arr) > 1e-10:
            valid_traces[name] = arr

    print(f"  有效神经元 trace: {len(valid_traces)}/{len(neuron_traces)}")

    # 选取方差最大的前 6 个神经元做延迟嵌入
    sorted_names = sorted(valid_traces.keys(),
                          key=lambda n: np.std(valid_traces[n]), reverse=True)
    selected = sorted_names[:6]

    # 延迟嵌入参数
    tau = 1   # 延迟步数
    d = 3     # 嵌入维度

    embeddings = {}
    for name in selected:
        trace = valid_traces[name]
        N = len(trace) - (d - 1) * tau
        if N < 10:
            continue
        embedded = np.zeros((N, d))
        for dim in range(d):
            embedded[:, dim] = trace[dim * tau: dim * tau + N]
        embeddings[name] = embedded
        print(f"  {name}: std={np.std(trace):.6f}, range=[{trace.min():.4f}, {trace.max():.4f}]")

    return embeddings, valid_traces, selected


# =============================================================================
# 实验 3: Reservoir Memory Capacity
# =============================================================================

def run_memory_capacity(device='cpu'):
    """
    Reservoir Memory Capacity (Jaeger 2001):

    MC = Σ_{k=1}^{K} r²(y_k, u_{t-k})

    1. 输入独立随机序列 u(t) ~ U[-1, 1]
    2. 收集 reservoir 状态 x(t)（所有神经元膜电位）
    3. 对每个延迟 k，用线性回归从 x(t) 预测 u(t-k)
    4. MC_k = r²(ŷ_k, u_{t-k})
    5. MC = Σ MC_k

    理论上界: MC ≤ N（神经元数）
    """
    print("\n" + "=" * 70)
    print("实验 3: Reservoir Memory Capacity (MC)")
    print("=" * 70)

    with Timer("创建模型"):
        model, in_f = create_model(device, seed=42)

    # 参数
    T_washout = 20    # 预热步（丢弃）
    T_train = 100     # 训练步
    T_test = 50       # 测试步
    T_total = T_washout + T_train + T_test
    K_max = 30        # 最大回忆延迟

    # 生成随机输入序列 u(t) ∈ R^in_f，各维独立 U[-1,1]
    torch.manual_seed(314)
    u_sequence = torch.rand(T_total, in_f, device=device) * 2 - 1  # U[-1, 1]

    # 收集 reservoir 状态
    print(f"  运行 {T_total} 步 (washout={T_washout}, train={T_train}, test={T_test})...")
    states = []
    state_dim = None
    model.reset()
    with SpikeMode.temporal():
        for t in range(T_total):
            forward_one_step(model, u_sequence[t], device)
            v_vec = get_state_vector(model)
            if v_vec is not None:
                if state_dim is None:
                    state_dim = len(v_vec)
                states.append(v_vec.copy())
            else:
                states.append(np.zeros(state_dim or 1))
            progress(t, T_total, prefix='reservoir ', every=20)

    if state_dim is None:
        print("  ⚠ 未收集到任何有效状态！")
        return np.zeros(K_max), 0.0, 0
    states = np.array(states, dtype=np.float64)  # (T_total, N_state)
    print(f"  状态维度: {states.shape}")

    # 清除 inf/nan
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    # 去掉常量列和全零列
    col_var = np.var(states, axis=0)
    active_cols = col_var > 1e-20
    states_active = states[:, active_cols]
    N_active_full = states_active.shape[1]
    print(f"  有效状态维度: {N_active_full} (去掉 {(~active_cols).sum()} 常量列)")

    # z-score 标准化（防止膜电位值过大导致数值溢出）—— 先标准化再降维
    col_mean = np.mean(states_active, axis=0)
    col_std = np.std(states_active, axis=0)
    col_std[col_std < 1e-20] = 1.0
    states_active = (states_active - col_mean) / col_std
    print(f"  z-score 标准化完成 (原始均值范围: [{col_mean.min():.2f}, {col_mean.max():.2f}])")

    # 再次检查是否有 inf/nan（标准化后可能产生）
    finite_cols = np.all(np.isfinite(states_active), axis=0)
    if not np.all(finite_cols):
        n_bad = (~finite_cols).sum()
        states_active = states_active[:, finite_cols]
        print(f"  去掉 {n_bad} 个含 inf/nan 的列")

    # 如果维度过高，取方差最大的前 MAX_DIM 个特征
    MAX_DIM = 200
    if states_active.shape[1] > MAX_DIM:
        var_after = np.var(states_active, axis=0)
        top_idx = np.argsort(var_after)[-MAX_DIM:]
        states_active = states_active[:, top_idx]
        print(f"  降维至 top-{MAX_DIM} 方差特征 (原始 {states_active.shape[1] + MAX_DIM - MAX_DIM})")
    N_active = states_active.shape[1]

    # 最终安全检查：确保无 inf/nan 且范围合理
    states_active = np.nan_to_num(states_active, nan=0.0, posinf=0.0, neginf=0.0)
    safe_cols = np.all(np.isfinite(states_active), axis=0) & (np.max(np.abs(states_active), axis=0) < 1e10)
    if not np.all(safe_cols):
        n_unsafe = (~safe_cols).sum()
        states_active = states_active[:, safe_cols]
        N_active = states_active.shape[1]
        print(f"  去掉 {n_unsafe} 个数值不安全列，最终维度: {N_active}")
    else:
        print(f"  最终回归维度: {N_active}")

    print(f"  数据范围: [{states_active.min():.4f}, {states_active.max():.4f}]")

    # 划分 train / test
    X_train = states_active[T_washout:T_washout + T_train]
    X_test = states_active[T_washout + T_train:]

    u_np = u_sequence.cpu().numpy()  # (T_total, in_f)

    # 对每个延迟 k 和每个输入维度，用 Ridge 回归计算 MC_k
    from numpy.linalg import lstsq

    mc_per_delay = []
    mc_per_delay_per_dim = []

    for k in range(1, K_max + 1):
        r2_dims = []
        for dim in range(in_f):
            # 目标: u(t-k) 的第 dim 维
            y_train = u_np[T_washout - k:T_washout + T_train - k, dim]
            y_test = u_np[T_washout + T_train - k:T_washout + T_train + T_test - k, dim]

            if len(y_train) != X_train.shape[0] or len(y_test) != X_test.shape[0]:
                r2_dims.append(0.0)
                continue

            # Ridge 回归 (正则化防奇异)
            ridge_alpha = 1.0
            XtX = X_train.T @ X_train + ridge_alpha * np.eye(N_active)
            Xty = X_train.T @ y_train
            try:
                w = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                r2_dims.append(0.0)
                continue

            y_pred = X_test @ w
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-30) if ss_tot > 1e-30 else 0.0
            r2 = max(0.0, r2)  # clip 负值
            r2_dims.append(r2)

        mc_k = np.mean(r2_dims)
        mc_per_delay.append(mc_k)
        mc_per_delay_per_dim.append(r2_dims)

        if k <= 10 or k % 5 == 0:
            print(f"    k={k:2d}: MC_k = {mc_k:.4f} (per-dim: {['%.3f' % r for r in r2_dims]})")

    mc_total = np.sum(mc_per_delay)
    mc_per_delay = np.array(mc_per_delay)

    print(f"\n  总 Memory Capacity: MC = {mc_total:.4f}")
    print(f"  理论上界: MC ≤ {N_active} (有效神经元状态维度)")
    print(f"  MC/N_active = {mc_total / N_active:.6f}" if N_active > 0 else "  N/A")
    print(f"  有效记忆深度 (MC_k > 0.1): {np.sum(mc_per_delay > 0.1)} 步")
    print(f"  有效记忆深度 (MC_k > 0.01): {np.sum(mc_per_delay > 0.01)} 步")

    return mc_per_delay, mc_total, N_active


# =============================================================================
# 可视化
# =============================================================================

def visualize_all(lyap_results, takens_embeddings, takens_traces, takens_selected,
                  mc_per_delay, mc_total, n_active, save_path):
    print("\n" + "=" * 70)
    print("生成可视化")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('NEXUS: Chaos Dynamics Validation\n'
                 '(2-layer SNN MLP, ~12,554 LIF neurons, FP32 gate circuits)',
                 fontsize=13, fontweight='bold', y=0.99)

    # --- (1) Lyapunov: 轨道距离演化 ---
    ax1 = fig.add_subplot(gs[0, 0])
    if lyap_results:
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
        for i, (eps, res) in enumerate(lyap_results.items()):
            dists = res['distances']
            t_axis = np.arange(len(dists))
            ax1.semilogy(t_axis, dists, color=colors[i % len(colors)],
                         linewidth=1.5, label=f'ε={eps:.0e}, λ={res["mle"]:.3f}')
        ax1.set_xlabel('Time step t')
        ax1.set_ylabel('||δ(t)|| (log scale)')
        ax1.set_title('Lyapunov: Trajectory Divergence')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

    # --- (2) Lyapunov: MLE 汇总柱状图 ---
    ax2 = fig.add_subplot(gs[0, 1])
    if lyap_results:
        eps_labels = [f'ε={eps:.0e}' for eps in lyap_results.keys()]
        mle_vals = [res['mle'] for res in lyap_results.values()]
        mle_errs = [res['mle_std'] for res in lyap_results.values()]
        bar_colors = ['#4CAF50' if m < 0 else '#F44336' if m > 0.01 else '#FF9800' for m in mle_vals]
        bars = ax2.bar(eps_labels, mle_vals, yerr=mle_errs, color=bar_colors,
                       edgecolor='black', linewidth=0.5, capsize=4)
        ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
        ax2.set_ylabel('Max Lyapunov Exponent (nats/step)')
        ax2.set_title('MLE Summary\n(green=stable, red=chaotic)')
        for bar, val in zip(bars, mle_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # --- (3) Takens 延迟嵌入 (3D -> 取前2个) ---
    ax3 = fig.add_subplot(gs[0, 2])
    if takens_embeddings:
        colors_takens = plt.cm.Set1(np.linspace(0, 1, min(6, len(takens_embeddings))))
        for i, (name, emb) in enumerate(takens_embeddings.items()):
            if i >= 6:
                break
            short_name = name.split('.')[-1][:20]
            ax3.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.5,
                        color=colors_takens[i], label=short_name)
            # 连线显示时间顺序
            ax3.plot(emb[:, 0], emb[:, 1], alpha=0.2, linewidth=0.5,
                     color=colors_takens[i])
        ax3.set_xlabel('V(t)')
        ax3.set_ylabel(f'V(t-τ)')
        ax3.set_title('Takens Delay Embedding (τ=1, d=3)\n2D projection')
        ax3.legend(fontsize=6, loc='upper right')
        ax3.grid(True, alpha=0.3)

    # --- (4) 膜电位时间序列 (前几个神经元) ---
    ax4 = fig.add_subplot(gs[1, 0])
    if takens_traces and takens_selected:
        n_show = min(4, len(takens_selected))
        colors_ts = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        for i in range(n_show):
            name = takens_selected[i]
            trace = takens_traces[name]
            short_name = name.split('.')[-1][:25]
            ax4.plot(trace, linewidth=0.7, alpha=0.8,
                     color=colors_ts[i], label=short_name)
        ax4.set_xlabel('Time step t')
        ax4.set_ylabel('Membrane Potential V')
        ax4.set_title('LIF Neuron V(t) Time Series')
        ax4.legend(fontsize=6)
        ax4.grid(True, alpha=0.3)

    # --- (5) Memory Capacity per delay ---
    ax5 = fig.add_subplot(gs[1, 1])
    if mc_per_delay is not None:
        delays = np.arange(1, len(mc_per_delay) + 1)
        ax5.bar(delays, mc_per_delay, color='#2196F3', edgecolor='black', linewidth=0.3)
        ax5.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='threshold=0.1')
        ax5.set_xlabel('Delay k (steps)')
        ax5.set_ylabel('MC_k (R²)')
        ax5.set_title(f'Memory Capacity per Delay\nMC_total = {mc_total:.3f}')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

    # --- (6) MC 累积曲线 ---
    ax6 = fig.add_subplot(gs[1, 2])
    if mc_per_delay is not None:
        cumulative = np.cumsum(mc_per_delay)
        delays = np.arange(1, len(mc_per_delay) + 1)
        ax6.plot(delays, cumulative, 'o-', color='#FF5722', linewidth=2, markersize=3)
        ax6.axhline(y=mc_total, color='gray', linestyle=':', alpha=0.5,
                     label=f'MC_total = {mc_total:.3f}')
        ax6.axhline(y=n_active, color='blue', linestyle='--', alpha=0.3,
                     label=f'N_active = {n_active}')
        ax6.set_xlabel('Max delay K')
        ax6.set_ylabel('Cumulative MC')
        ax6.set_title('Cumulative Memory Capacity')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

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

    # 实验 1: Lyapunov 指数
    lyap_results = run_lyapunov_experiment(device=device)

    # 实验 2: Takens 延迟嵌入
    takens_emb, takens_traces, takens_selected = run_takens_embedding(device=device)

    # 实验 3: Memory Capacity
    mc_per_delay, mc_total, n_active = run_memory_capacity(device=device)

    # 可视化
    save_path = os.path.join(os.path.dirname(__file__), 'chaos_dynamics_results.png')
    visualize_all(lyap_results, takens_emb, takens_traces, takens_selected,
                  mc_per_delay, mc_total, n_active, save_path=save_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"所有实验完成 (总耗时: {elapsed:.1f}s)")
    print(f"{'='*70}")

    # 汇总
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    if lyap_results:
        print("\n[Lyapunov 指数]")
        for eps, res in lyap_results.items():
            status = "混沌⚠" if res['mle'] > 0.01 else "稳定✓" if res['mle'] < -0.01 else "边缘~"
            print(f"  ε={eps:.0e}: λ = {res['mle']:.4f} ± {res['mle_std']:.4f} ({status})")

    print(f"\n[Memory Capacity]")
    print(f"  MC_total = {mc_total:.4f}")
    print(f"  N_active = {n_active}")
    print(f"  有效记忆深度 (MC_k > 0.1): {np.sum(mc_per_delay > 0.1)} 步")

    print(f"\n[Takens 嵌入]")
    print(f"  追踪神经元数: {len(takens_emb)}")
    for name, emb in takens_emb.items():
        short = name.split('.')[-1][:30]
        print(f"  {short}: {emb.shape[0]} 嵌入点")
