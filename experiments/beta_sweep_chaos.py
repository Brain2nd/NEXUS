"""
β-Sweep Chaos Dynamics Experiment
==================================

核心假设：当前实验测出 MC=0 和"纯积分器"，是因为 β ≈ 1（拟合 ANN 的参数区）。
LIF 的 Reset 机制是非线性"折叠"（模运算），但 β 太高时衰减不足，V 单调增长。

本实验扫描 β 参数空间（0.5 ~ 0.99），寻找：
1. V(t) 从"单调增长"变为"有界振荡"的转变点
2. Lyapunov 指数 λ 从 ≈0 变为正数的区域
3. Takens 嵌入出现吸引子结构
4. Memory Capacity 从 0 变为正值

被测系统: SimpleSpikeMLP(4→8→4), ~12,554 LIF 神经元
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
# Model & Utils
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


def create_model(device, beta, seed=42):
    """创建指定 β 的模型"""
    torch.manual_seed(seed)
    in_f, hid_f, out_f = 4, 8, 4
    template = SimpleLIFNode(beta=beta)
    model = SimpleSpikeMLP(in_f, hid_f, out_f, neuron_template=template).to(device)
    w1 = torch.randn(hid_f, in_f, device=device) * 0.5
    w2 = torch.randn(out_f, hid_f, device=device) * 0.5
    model.set_weights(w1, w2)
    return model, in_f


def collect_lif_membrane_potentials(model):
    potentials = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            potentials[name] = module.v.detach().clone()
    return potentials


def get_state_vector(model):
    v_dict = collect_lif_membrane_potentials(model)
    if not v_dict:
        return None
    parts = [v.cpu().numpy().flatten() for v in v_dict.values()]
    vec = np.concatenate(parts)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def forward_one_step(model, x_float, device):
    x_pulse = float32_to_pulse(x_float, device=device)
    _ = model(x_pulse.unsqueeze(0))


# =============================================================================
# Per-β Experiments
# =============================================================================

def measure_lyapunov(model_factory, device, in_f, T=50, eps=1e-6):
    """Lyapunov exponent for a given β"""
    torch.manual_seed(999)
    inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

    model_ref = model_factory()
    model_pert = model_factory()

    model_ref.reset()
    model_pert.reset()

    with SpikeMode.temporal():
        forward_one_step(model_ref, inputs[0], device)
        forward_one_step(model_pert, inputs[0], device)

        # Add perturbation
        for name, module in model_pert.named_modules():
            if isinstance(module, SimpleLIFNode) and module.v is not None:
                module.v.add_(torch.randn_like(module.v) * eps)

        v_ref = get_state_vector(model_ref)
        v_pert = get_state_vector(model_pert)
        if v_ref is None or v_pert is None:
            return float('nan'), []

        d0 = np.linalg.norm(v_ref - v_pert)
        distances = [d0]
        lyap_inc = []

        for t in range(1, T):
            forward_one_step(model_ref, inputs[t], device)
            forward_one_step(model_pert, inputs[t], device)
            vr = get_state_vector(model_ref)
            vp = get_state_vector(model_pert)
            if vr is None or vp is None:
                break
            dt = np.linalg.norm(vr - vp)
            distances.append(dt)
            if dt > 1e-30 and distances[-2] > 1e-30:
                lyap_inc.append(np.log(dt / distances[-2]))

    mle = np.mean(lyap_inc) if lyap_inc else float('nan')
    return mle, distances


def measure_v_dynamics(model_factory, device, in_f, T=200):
    """Record V(t) trace for a single representative neuron"""
    torch.manual_seed(777)
    inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

    model = model_factory()
    model.reset()
    traces = []

    with SpikeMode.temporal():
        for t in range(T):
            forward_one_step(model, inputs[t], device)
            v_dict = collect_lif_membrane_potentials(model)
            # Pick the first neuron with non-trivial V
            best_v = 0.0
            for name, v_tensor in v_dict.items():
                v_flat = v_tensor.cpu().numpy().flatten()
                nz = np.nonzero(np.abs(v_flat) > 1e-30)[0]
                if len(nz) > 0:
                    best_v = float(v_flat[nz[0]])
                    break
            traces.append(best_v)

    trace = np.array(traces)
    v_range = trace.max() - trace.min()
    v_std = np.std(trace)
    v_bounded = v_range < 100 * v_std if v_std > 1e-10 else True
    return trace, v_range, v_std, v_bounded


def measure_memory_capacity(model_factory, device, in_f, T_washout=20, T_train=100, T_test=50, K_max=20):
    """Quick MC measurement"""
    T_total = T_washout + T_train + T_test
    torch.manual_seed(314)
    u_seq = torch.rand(T_total, in_f, device=device) * 2 - 1

    model = model_factory()
    model.reset()
    states = []
    state_dim = None

    with SpikeMode.temporal():
        for t in range(T_total):
            forward_one_step(model, u_seq[t], device)
            v_vec = get_state_vector(model)
            if v_vec is not None:
                if state_dim is None:
                    state_dim = len(v_vec)
                states.append(v_vec.copy())
            else:
                states.append(np.zeros(state_dim or 1))

    if state_dim is None:
        return 0.0, np.zeros(K_max)

    states = np.array(states, dtype=np.float64)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    # z-score
    col_var = np.var(states, axis=0)
    active = col_var > 1e-20
    sa = states[:, active]
    if sa.shape[1] == 0:
        return 0.0, np.zeros(K_max)

    cm = np.mean(sa, axis=0)
    cs = np.std(sa, axis=0)
    cs[cs < 1e-20] = 1.0
    sa = (sa - cm) / cs

    # dim reduction
    if sa.shape[1] > 200:
        v2 = np.var(sa, axis=0)
        top = np.argsort(v2)[-200:]
        sa = sa[:, top]

    sa = np.nan_to_num(sa, nan=0.0, posinf=0.0, neginf=0.0)
    N_act = sa.shape[1]

    X_train = sa[T_washout:T_washout + T_train]
    X_test = sa[T_washout + T_train:]
    u_np = u_seq.cpu().numpy()

    XtX = X_train.T @ X_train + 1.0 * np.eye(N_act)

    mc_per_delay = []
    for k in range(1, K_max + 1):
        r2_dims = []
        for dim in range(in_f):
            y_tr = u_np[T_washout - k:T_washout + T_train - k, dim]
            y_te = u_np[T_washout + T_train - k:T_washout + T_train + T_test - k, dim]
            if len(y_tr) != X_train.shape[0] or len(y_te) != X_test.shape[0]:
                r2_dims.append(0.0)
                continue
            try:
                w = np.linalg.solve(XtX, X_train.T @ y_tr)
            except np.linalg.LinAlgError:
                r2_dims.append(0.0)
                continue
            yp = X_test @ w
            ss_res = np.sum((y_te - yp) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-30)) if ss_tot > 1e-30 else 0.0
            r2_dims.append(r2)
        mc_per_delay.append(np.mean(r2_dims))

    mc_total = np.sum(mc_per_delay)
    return mc_total, np.array(mc_per_delay)


# =============================================================================
# Main β-sweep
# =============================================================================

def run_beta_sweep(device):
    betas = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.0 - 1e-7]
    in_f = 4

    results = {}
    for i, beta in enumerate(betas):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(betas)}] β = {beta:.7f}")
        print(f"{'='*60}")

        factory = lambda b=beta: create_model(device, b, seed=42)[0]

        # 1. Lyapunov
        print("  Lyapunov...", end='', flush=True)
        t0 = time.time()
        mle, dists = measure_lyapunov(factory, device, in_f, T=50, eps=1e-6)
        print(f" λ={mle:.4f} ({time.time()-t0:.1f}s)")

        # 2. V dynamics
        print("  V dynamics...", end='', flush=True)
        t0 = time.time()
        trace, v_range, v_std, v_bounded = measure_v_dynamics(factory, device, in_f, T=200)
        print(f" range={v_range:.2f}, std={v_std:.2f}, bounded={v_bounded} ({time.time()-t0:.1f}s)")

        # 3. MC
        print("  Memory Capacity...", end='', flush=True)
        t0 = time.time()
        mc_total, mc_delays = measure_memory_capacity(factory, device, in_f)
        print(f" MC={mc_total:.4f} ({time.time()-t0:.1f}s)")

        results[beta] = {
            'mle': mle,
            'distances': dists,
            'trace': trace,
            'v_range': v_range,
            'v_std': v_std,
            'v_bounded': v_bounded,
            'mc_total': mc_total,
            'mc_delays': mc_delays,
        }

    return results, betas


# =============================================================================
# Visualization
# =============================================================================

def visualize_sweep(results, betas, save_path):
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('NEXUS β-Sweep: Searching for Edge of Chaos\n'
                 'SimpleSpikeMLP(4→8→4), ~12,554 LIF neurons',
                 fontsize=14, fontweight='bold', y=0.99)

    beta_arr = np.array(betas)
    mle_arr = np.array([results[b]['mle'] for b in betas])
    mc_arr = np.array([results[b]['mc_total'] for b in betas])
    vrange_arr = np.array([results[b]['v_range'] for b in betas])
    vstd_arr = np.array([results[b]['v_std'] for b in betas])

    # --- (1) MLE vs β ---
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#F44336' if m > 0.01 else '#4CAF50' if m < -0.01 else '#FF9800' for m in mle_arr]
    ax1.bar(range(len(betas)), mle_arr, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(betas)))
    ax1.set_xticklabels([f'{b:.2f}' if b < 0.999 else '≈1' for b in betas], rotation=45, fontsize=7)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_xlabel('β (decay factor)')
    ax1.set_ylabel('Max Lyapunov Exponent (nats/step)')
    ax1.set_title('MLE vs β\n(red=chaotic, green=stable)')
    ax1.grid(True, alpha=0.3)

    # --- (2) MC vs β ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(betas)), mc_arr, color='#2196F3', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(betas)))
    ax2.set_xticklabels([f'{b:.2f}' if b < 0.999 else '≈1' for b in betas], rotation=45, fontsize=7)
    ax2.set_xlabel('β')
    ax2.set_ylabel('Total Memory Capacity')
    ax2.set_title('Memory Capacity vs β')
    ax2.grid(True, alpha=0.3)

    # --- (3) V range & std vs β ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(range(len(betas)), vrange_arr, 'o-', color='#FF5722', label='V range', linewidth=2)
    ax3.semilogy(range(len(betas)), vstd_arr, 's-', color='#9C27B0', label='V std', linewidth=2)
    ax3.set_xticks(range(len(betas)))
    ax3.set_xticklabels([f'{b:.2f}' if b < 0.999 else '≈1' for b in betas], rotation=45, fontsize=7)
    ax3.set_xlabel('β')
    ax3.set_ylabel('V statistics (log scale)')
    ax3.set_title('V(t) Range & Std vs β')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- (4-6) V(t) traces for low, mid, high β ---
    trace_indices = [0, len(betas)//2, -1]  # low, mid, high
    trace_labels = ['Low β', 'Mid β', 'High β']
    for col, (idx, label) in enumerate(zip(trace_indices, trace_labels)):
        ax = fig.add_subplot(gs[1, col])
        b = betas[idx]
        trace = results[b]['trace']
        ax.plot(trace, linewidth=0.7, color='#2196F3')
        ax.set_xlabel('Time step t')
        ax.set_ylabel('V(t)')
        bounded_str = "BOUNDED" if results[b]['v_bounded'] else "UNBOUNDED"
        ax.set_title(f'V(t) Trace: β={b:.2f}\n{bounded_str}, range={results[b]["v_range"]:.1f}')
        ax.grid(True, alpha=0.3)

    # --- (7-9) Takens embedding for low, mid, high β ---
    for col, (idx, label) in enumerate(zip(trace_indices, trace_labels)):
        ax = fig.add_subplot(gs[2, col])
        b = betas[idx]
        trace = results[b]['trace']
        tau = 1
        N = len(trace) - 2 * tau
        if N > 10:
            x = trace[:N]
            y = trace[tau:tau+N]
            ax.scatter(x, y, s=3, alpha=0.5, color='#E91E63')
            ax.plot(x, y, alpha=0.15, linewidth=0.5, color='#E91E63')
        ax.set_xlabel('V(t)')
        ax.set_ylabel('V(t-1)')
        ax.set_title(f'Takens Embedding: β={b:.2f}')
        ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    results, betas = run_beta_sweep(device)

    save_path = os.path.join(os.path.dirname(__file__), 'beta_sweep_chaos_results.png')
    visualize_sweep(results, betas, save_path)

    # Summary table
    print(f"\n{'='*70}")
    print("β-Sweep Summary")
    print(f"{'='*70}")
    print(f"{'β':>10s} | {'MLE':>8s} | {'MC':>8s} | {'V_range':>10s} | {'V_std':>10s} | {'Bounded':>8s}")
    print('-' * 70)
    for b in betas:
        r = results[b]
        bl = '≈1' if b > 0.999 else f'{b:.2f}'
        print(f"{bl:>10s} | {r['mle']:>8.4f} | {r['mc_total']:>8.4f} | {r['v_range']:>10.2f} | {r['v_std']:>10.2f} | {'YES' if r['v_bounded'] else 'NO':>8s}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
