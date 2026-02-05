"""
实验 10: 时空脉冲响应与孤子探测 (Spatiotemporal Impulse Response & Soliton Probing)
================================================================================

物理动机：
- 向 NEXUS 网络注入脉冲，观察格林函数响应
- 线性介质：脉冲扩散衰减（色散）
- 非线性介质（临界同步区）：可能产生类孤子结构——保持形状的波包
- 双脉冲碰撞测试：验证非线性叠加性（孤子判据）

实验设计：
- 对照组: β ≈ 1 (1-1e-7), V_th=100.0 (线性积分器)
- 实验组: β = 0.90, V_th = 10.0 (临界同步区)

步骤 A: 单脉冲传播 → 时空光锥图（基于膜电位）
步骤 B: 双脉冲碰撞 → 非线性残差分析
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
# Model (与实验 7-9 一致)
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
# 核心函数：基于膜电位的状态收集（与实验 9 一致）
# =============================================================================

def collect_membrane_snapshots(model):
    """收集所有 SimpleLIFNode 的膜电位快照，返回一个扁平向量。"""
    v_snap = []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            v_flat = module.v.detach().cpu().flatten().float()
            v_flat = torch.nan_to_num(v_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            v_snap.append(v_flat)
    if v_snap:
        return torch.cat(v_snap).numpy()
    return None


def run_temporal_sequence(model, input_sequence, device):
    """
    在 TEMPORAL 模式下逐步输入序列，收集每步的膜电位状态。
    input_sequence: list of float tensors, each shape (1, in_features)
    Returns: states (T, N_neurons) — 膜电位快照时间序列
    """
    model.reset()
    T = len(input_sequence)
    states = []

    with torch.no_grad():
        for t in range(T):
            x_float = input_sequence[t].to(device)
            x_pulse = float32_to_pulse(x_float)
            _ = model(x_pulse)

            snap = collect_membrane_snapshots(model)
            if snap is not None:
                states.append(snap)
            else:
                states.append(np.zeros(1))

    return np.array(states)


def create_impulse_sequence(T, in_features, pulse_time, pulse_dim, amplitude, noise_std=0.001, seed=None):
    """
    创建脉冲输入序列：背景噪声 + 在 pulse_time 时刻的 pulse_dim 维度注入振幅 amplitude 的脉冲。
    如果 pulse_time 是 list，则在多个时刻注入。
    """
    if seed is not None:
        torch.manual_seed(seed)
    seq = []
    if isinstance(pulse_time, int):
        pulse_time = [pulse_time]
    for t in range(T):
        x = torch.randn(1, in_features) * noise_std
        if t in pulse_time:
            x[0, pulse_dim] += amplitude
        seq.append(x)
    return seq


def compute_light_cone(states):
    """时空光锥：每个时间步各神经元膜电位的绝对值。"""
    return np.abs(states)


def compute_localization(light_cone, pulse_time):
    """
    局域化度量 (Participation Ratio)。
    PR = (Σ|a_i|)² / (N * Σ|a_i|²)
    """
    post_cone = light_cone[pulse_time+1:]
    T_post, N = post_cone.shape
    pr_list = []
    for t in range(T_post):
        a = post_cone[t]
        sum_a = np.sum(a)
        sum_a2 = np.sum(a**2)
        if sum_a2 < 1e-20:
            pr_list.append(0.0)
        else:
            pr = (sum_a**2) / (N * sum_a2)
            pr_list.append(pr)
    return np.array(pr_list)


def compute_persistence(light_cone, pulse_time, threshold_frac=0.1):
    """信号持续性：脉冲后能量保持在峰值 threshold_frac 以上的步数。"""
    post_cone = light_cone[pulse_time+1:]
    energy = np.sum(post_cone**2, axis=1)
    if len(energy) == 0 or np.max(energy) < 1e-20:
        return 0
    peak = np.max(energy)
    threshold = peak * threshold_frac
    return int(np.sum(energy >= threshold))


def run_collision_test(model, device, T, in_features, t1, t2, pulse_dim, amplitude, noise_std=0.001):
    """
    双脉冲碰撞测试：分别记录 R(P1), R(P2), R(P1+P2)，计算非线性残差 Δ。
    使用固定种子保证背景噪声一致。
    """
    # R(P1)
    seq_p1 = create_impulse_sequence(T, in_features, t1, pulse_dim, amplitude, noise_std, seed=123)
    # R(P2)
    seq_p2 = create_impulse_sequence(T, in_features, t2, pulse_dim, amplitude, noise_std, seed=123)
    # R(P1+P2)
    seq_both = create_impulse_sequence(T, in_features, [t1, t2], pulse_dim, amplitude, noise_std, seed=123)

    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    s_p1 = run_temporal_sequence(model, seq_p1, device)
    s_p2 = run_temporal_sequence(model, seq_p2, device)
    s_both = run_temporal_sequence(model, seq_both, device)

    SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

    # 非线性残差：Δ = R(P1+P2) - R(P1) - R(P2)
    # 但我们还需要减去纯背景噪声的贡献来得到真实的信号响应
    # R_signal(P1) = R(P1) - R(noise), 类似对 P2 和 both
    # 所以 Δ = R(both) - R(P1) - R(P2) + R(noise)
    # 由于噪声很小，直接用 delta = s_both - s_p1 - s_p2 近似
    delta = s_both - (s_p1 + s_p2)

    return s_p1, s_p2, s_both, delta


def compute_collision_metrics(delta, t1, t2):
    """从碰撞残差中提取关键指标。"""
    delta_energy = np.sum(delta**2, axis=1)
    total_nonlinearity = np.sum(delta_energy)

    collision_zone = list(range(t2, min(t2 + 20, len(delta))))
    pre_zone = list(range(0, t1))

    if len(collision_zone) > 0:
        collision_energy = np.mean(delta_energy[collision_zone])
    else:
        collision_energy = 0.0

    if len(pre_zone) > 0:
        pre_energy = np.mean(delta_energy[pre_zone])
    else:
        pre_energy = 0.0

    if pre_energy < 1e-20:
        collision_ratio = float('inf') if collision_energy > 1e-20 else 1.0
    else:
        collision_ratio = collision_energy / pre_energy

    # 碰撞后恢复度
    post_zone = list(range(min(t2 + 20, len(delta)), len(delta)))
    if len(post_zone) > 0 and collision_energy > 1e-20:
        post_energy = np.mean(delta_energy[post_zone])
        recovery = 1.0 - (post_energy / collision_energy)
    else:
        recovery = 0.0

    return {
        'total_nonlinearity': total_nonlinearity,
        'collision_ratio': collision_ratio,
        'collision_energy': collision_energy,
        'recovery': recovery,
        'delta_energy': delta_energy,
    }


# =============================================================================
# 主实验
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}", flush=True)

    T = 100
    pulse_time = 10
    pulse_dim = 0
    amplitude = 5.0
    t1, t2 = 10, 30

    conditions = {
        'Linear Integrator': {'beta': 1.0 - 1e-7, 'v_threshold': 100.0},
        'Critical Sync': {'beta': 0.90, 'v_threshold': 10.0},
    }

    results = {}

    for cond_name, params in conditions.items():
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"[Condition] {cond_name}: beta={params['beta']}, V_th={params['v_threshold']}", flush=True)

        model, in_f = create_model(device, params['beta'], params['v_threshold'])

        # ---- 步骤 A: 单脉冲传播 ----
        print(f"  [Step A] Single pulse propagation...", flush=True)
        seq = create_impulse_sequence(T, in_f, pulse_time, pulse_dim, amplitude, seed=42)

        SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
        states = run_temporal_sequence(model, seq, device)
        SpikeMode.set_global_mode(SpikeMode.BIT_EXACT)

        light_cone = compute_light_cone(states)
        pr = compute_localization(light_cone, pulse_time)
        persistence = compute_persistence(light_cone, pulse_time)

        mean_pr = np.mean(pr) if len(pr) > 0 else 0.0
        max_energy = np.max(np.sum(light_cone**2, axis=1))
        print(f"    State dim: {states.shape[1]}", flush=True)
        print(f"    Max membrane energy: {max_energy:.4f}", flush=True)
        print(f"    Mean Participation Ratio (post-pulse): {mean_pr:.4f}", flush=True)
        print(f"    Persistence (steps above 10% peak): {persistence}", flush=True)

        # ---- 步骤 B: 双脉冲碰撞 ----
        print(f"  [Step B] Two-pulse collision test...", flush=True)
        model2, _ = create_model(device, params['beta'], params['v_threshold'])
        s_p1, s_p2, s_both, delta = run_collision_test(
            model2, device, T, in_f, t1, t2, pulse_dim, amplitude)
        collision_metrics = compute_collision_metrics(delta, t1, t2)

        print(f"    Total nonlinearity (||Delta||^2): {collision_metrics['total_nonlinearity']:.4f}", flush=True)
        print(f"    Collision zone energy: {collision_metrics['collision_energy']:.4f}", flush=True)
        print(f"    Collision ratio (collision/pre): {collision_metrics['collision_ratio']:.2f}", flush=True)
        print(f"    Post-collision recovery: {collision_metrics['recovery']:.4f}", flush=True)

        elapsed = time.time() - t0
        print(f"  [Time] {elapsed:.1f}s", flush=True)

        results[cond_name] = {
            'light_cone': light_cone,
            'states': states,
            'pr': pr,
            'mean_pr': mean_pr,
            'persistence': persistence,
            's_p1': s_p1,
            's_p2': s_p2,
            's_both': s_both,
            'delta': delta,
            'collision': collision_metrics,
        }

    # =============================================================================
    # 可视化
    # =============================================================================
    print("\n[Plot] Generating 6-panel figure...", flush=True)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    cond_names = list(results.keys())

    # --- Panel 1 & 2: 时空光锥图 (Linear vs Critical) ---
    for idx, cond_name in enumerate(cond_names):
        ax = fig.add_subplot(gs[0, idx])
        lc = results[cond_name]['light_cone']
        im = ax.imshow(lc.T, aspect='auto', origin='lower', cmap='hot',
                       extent=[0, T, 0, lc.shape[1]])
        ax.axvline(x=pulse_time, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Neuron index (membrane potential)')
        ax.set_title(f'Light Cone: {cond_name}\n'
                     f'PR={results[cond_name]["mean_pr"]:.3f}, '
                     f'Persist={results[cond_name]["persistence"]}')
        plt.colorbar(im, ax=ax, label='|V_membrane|')

    # --- Panel 3: 局域化对比 (Participation Ratio vs time) ---
    ax3 = fig.add_subplot(gs[0, 2])
    for cond_name in cond_names:
        pr = results[cond_name]['pr']
        t_axis = np.arange(pulse_time + 1, pulse_time + 1 + len(pr))
        ax3.plot(t_axis, pr, label=cond_name, linewidth=2)
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Uniform (PR=1)')
    ax3.set_xlabel('Time step (post-pulse)')
    ax3.set_ylabel('Participation Ratio')
    ax3.set_title('Localization: PR < 1 = focused\nPR ~ 1 = diffused')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4 & 5: 碰撞非线性残差 (Linear vs Critical) ---
    for idx, cond_name in enumerate(cond_names):
        ax = fig.add_subplot(gs[1, idx])
        delta = results[cond_name]['delta']
        im = ax.imshow(np.abs(delta).T, aspect='auto', origin='lower', cmap='inferno',
                       extent=[0, T, 0, delta.shape[1]])
        ax.axvline(x=t1, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label=f'P1 (t={t1})')
        ax.axvline(x=t2, color='lime', linestyle='--', linewidth=1, alpha=0.7, label=f'P2 (t={t2})')
        cm = results[cond_name]['collision']
        ax.set_xlabel('Time step')
        ax.set_ylabel('Neuron index')
        ax.set_title(f'Nonlinear Residual |Delta|: {cond_name}\n'
                     f'||Delta||^2={cm["total_nonlinearity"]:.2f}, '
                     f'Ratio={cm["collision_ratio"]:.1f}')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, label='|Delta|')

    # --- Panel 6: 残差能量时间曲线 ---
    ax6 = fig.add_subplot(gs[1, 2])
    for cond_name in cond_names:
        de = results[cond_name]['collision']['delta_energy']
        ax6.plot(range(T), de, label=cond_name, linewidth=2)
    ax6.axvline(x=t1, color='cyan', linestyle='--', alpha=0.5, label=f'P1 (t={t1})')
    ax6.axvline(x=t2, color='lime', linestyle='--', alpha=0.5, label=f'P2 (t={t2})')
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Delta energy (per step)')
    ax6.set_title('Nonlinearity Energy Timeline\n'
                  'Peak at collision = structured interaction')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('symlog', linthresh=1e-4)

    fig.suptitle('Experiment 10: Spatiotemporal Impulse Response & Soliton Probing\n'
                 'NEXUS Gate Circuit Network — Green\'s Function Analysis',
                 fontsize=14, fontweight='bold', y=0.98)

    out_path = os.path.join(os.path.dirname(__file__), 'soliton_probe_results.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] Saved to {out_path}", flush=True)

    # =============================================================================
    # 摘要
    # =============================================================================
    print("\n" + "="*70, flush=True)
    print("SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"{'Condition':<25} {'Mean PR':<12} {'Persist':<12} {'||Delta||^2':<15} {'Coll.Ratio':<12} {'Recovery':<10}", flush=True)
    print("-"*86, flush=True)
    for cond_name in cond_names:
        r = results[cond_name]
        cm = r['collision']
        cr_str = f"{cm['collision_ratio']:.2f}" if cm['collision_ratio'] != float('inf') else "inf"
        print(f"{cond_name:<25} {r['mean_pr']:<12.4f} {r['persistence']:<12d} "
              f"{cm['total_nonlinearity']:<15.4f} {cr_str:<12} {cm['recovery']:<10.4f}", flush=True)

    print("\n[DONE] Experiment 10 complete.", flush=True)


if __name__ == '__main__':
    main()
