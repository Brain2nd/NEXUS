"""
实验 7: 驱动-响应一致性相变 (Drive-Response Consistency Phase Transition)
=====================================================================

物理学理论依据：
- 广义同步 (Generalized Synchronization)：两个全同混沌系统被同一外部信号驱动，
  若驱动强度足够大，会从"各自为政"突变为"步调一致"。
- 序参量：条件 Lyapunov 指数 (CLE)
  - 同步相 (CLE < 0)：外部驱动力压倒内部发散力
  - 非同步相 (CLE > 0)：蝴蝶效应压倒外部驱动

能量竞争：
1. 内部势能（折叠势）：由 V_th 控制。V_th 越低，势垒越密，非线性折叠越频繁
2. 外部动能（驱动力）：由 σ_input 提供

扫描参数：
- β (耗散参数): [0.5, 0.9, 0.95, 0.99, 1.0]
- γ = σ_input / V_th (归一化驱动力): [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
  操作：固定 σ=1，手动扫描 V_th

观测物理量：
1. 同步误差 (Sync Error): <||V_A(t) - V_B(t)||>_{t→∞}
2. 活跃度 (Activity): 发生 Soft Reset 的平均频率
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
from matplotlib.colors import ListedColormap
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
    # 创建后直接修改所有内部神经元的 v_threshold
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module.v_threshold = v_threshold
    return model, in_f


def forward_one_step(model, x_float, device):
    x_pulse = float32_to_pulse(x_float, device=device)
    _ = model(x_pulse.unsqueeze(0))


def get_state_vector(model):
    """收集所有 LIF 神经元的膜电位拼接成一个向量"""
    parts = []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            parts.append(module.v.detach().cpu().numpy().flatten())
    if len(parts) == 0:
        return None
    return np.concatenate(parts)


def set_initial_perturbation(model, epsilon):
    """对 System B 的膜电位添加随机扰动"""
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            perturbation = torch.randn_like(module.v) * epsilon
            module.v.data.add_(perturbation)


def count_activity(model):
    """统计活跃度：发生 Soft Reset 的神经元比例
    软复位后 V = V - V_th，若刚发生过复位，残余 V 为正但远小于 V_th
    近似：|V| > 0 且 V < 0.5*V_th 的神经元视为刚发生过折叠"""
    n_total = 0
    n_fired = 0
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            v_flat = module.v.detach().cpu().numpy().flatten()
            vth = module.v_threshold.flatten()[0].item() if torch.is_tensor(module.v_threshold) else float(module.v_threshold)
            n_total += len(v_flat)
            # 刚发生过软复位的特征：0 < V < 0.5 * V_th
            n_fired += np.sum((np.abs(v_flat) > 1e-10) & (np.abs(v_flat) < 0.5 * abs(vth)))
    return n_fired / max(n_total, 1)


# =============================================================================
# 核心实验：相图扫描
# =============================================================================

def run_phase_diagram(device):
    """扫描 (β, γ) 相平面，测量同步误差和活跃度"""
    print("\n" + "=" * 70, flush=True)
    print("实验 7: 驱动-响应一致性相变 (Phase Diagram)", flush=True)
    print("扫描 β × γ，测量 Sync Error 和 Activity", flush=True)
    print("=" * 70, flush=True)

    betas = [0.5, 0.9, 0.95, 0.99, 1.0]
    # γ = σ / V_th, 固定 σ=1.0
    gamma_vth_pairs = [
        (0.1,  10.0),
        (0.5,   2.0),
        (1.0,   1.0),
        (2.0,   0.5),
        (5.0,   0.2),
        (10.0,  0.1),
    ]
    gammas = [g for g, _ in gamma_vth_pairs]
    sigma_input = 1.0
    T = 100  # 时间步
    epsilon = 0.1  # 初始扰动幅度

    # 结果矩阵
    sync_error_map = np.zeros((len(betas), len(gammas)))
    activity_map = np.zeros((len(betas), len(gammas)))
    cle_map = np.zeros((len(betas), len(gammas)))

    total = len(betas) * len(gammas)
    done = 0

    for i, beta in enumerate(betas):
        for j, (gamma, vth) in enumerate(gamma_vth_pairs):
            t0 = time.time()
            torch.manual_seed(1000)
            in_f = 4
            inputs = torch.randn(T, in_f) * sigma_input

            # 双胞胎系统
            model_a, _ = create_model(device, beta, v_threshold=vth, seed=42)
            model_b, _ = create_model(device, beta, v_threshold=vth, seed=42)
            model_a.reset()
            model_b.reset()

            # 跑一步建立 V，然后给 B 加扰动
            with SpikeMode.temporal():
                forward_one_step(model_a, inputs[0], device)
                forward_one_step(model_b, inputs[0], device)
                set_initial_perturbation(model_b, epsilon)

            # 收集时间序列
            sync_errors = []
            activities = []
            lyap_sum = 0.0
            n_lyap = 0
            prev_delta_norm = None

            with SpikeMode.temporal():
                for t in range(1, T):
                    forward_one_step(model_a, inputs[t], device)
                    forward_one_step(model_b, inputs[t], device)

                    v_a = get_state_vector(model_a)
                    v_b = get_state_vector(model_b)

                    if v_a is not None and v_b is not None:
                        delta_norm = np.linalg.norm(v_a - v_b)
                        sync_errors.append(delta_norm / len(v_a))

                        # CLE 增量
                        if prev_delta_norm is not None and prev_delta_norm > 1e-30 and delta_norm > 1e-30:
                            lyap_sum += np.log(delta_norm / prev_delta_norm)
                            n_lyap += 1
                        prev_delta_norm = delta_norm

                    activities.append(count_activity(model_a))

            # 稳态同步误差：取后半段均值
            half = len(sync_errors) // 2
            sync_err = np.mean(sync_errors[half:]) if half > 0 else (np.mean(sync_errors) if sync_errors else 0.0)
            activity = np.mean(activities[half:]) if half > 0 else (np.mean(activities) if activities else 0.0)
            cle = lyap_sum / n_lyap if n_lyap > 0 else 0.0

            sync_error_map[i, j] = sync_err
            activity_map[i, j] = activity
            cle_map[i, j] = cle

            done += 1
            elapsed = time.time() - t0
            phase = classify_phase(sync_err, activity)
            print(f"  [{done:2d}/{total}] β={beta:.2f}, γ={gamma:5.1f} (V_th={vth:5.1f}): "
                  f"sync_err={sync_err:.4e}, activity={activity:.3f}, CLE={cle:+.4f} "
                  f"[{phase}] ({elapsed:.1f}s)", flush=True)

    return sync_error_map, activity_map, cle_map, betas, gammas


def classify_phase(sync_err, activity, sync_threshold=1e-4):
    """分类三个物理相"""
    if activity < 0.01:
        return "线性层流"  # Laminar: no folding
    elif sync_err > sync_threshold:
        return "非同步湍流"  # Turbulent: folding but not synchronized
    else:
        return "临界同步"  # Critical: folding AND synchronized


# =============================================================================
# 可视化
# =============================================================================

def visualize(sync_error_map, activity_map, cle_map, betas, gammas):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Experiment 7: Drive-Response Consistency Phase Transition\n"
                 "Twin System Driven by Same Input — Phase Diagram in (β, γ) Space",
                 fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Plot 1: Sync Error Phase Diagram ---
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.log10(sync_error_map + 1e-15).T, aspect='auto', cmap='RdYlGn_r',
                   origin='lower')
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b:.2f}' for b in betas], fontsize=8)
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{g}' for g in gammas])
    ax.set_xlabel('β (decay)')
    ax.set_ylabel('γ = σ/V_th (drive ratio)')
    ax.set_title('log₁₀(Sync Error)', fontsize=11, fontweight='bold')
    for i in range(len(betas)):
        for j in range(len(gammas)):
            ax.text(i, j, f'{sync_error_map[i,j]:.1e}', ha='center', va='center', fontsize=6,
                    color='white' if sync_error_map[i,j] > 1e-3 else 'black')
    plt.colorbar(im, ax=ax)

    # --- Plot 2: Activity Phase Diagram ---
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(activity_map.T, aspect='auto', cmap='hot', origin='lower',
                   vmin=0, vmax=max(activity_map.max(), 0.01))
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b:.2f}' for b in betas], fontsize=8)
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{g}' for g in gammas])
    ax.set_xlabel('β (decay)')
    ax.set_ylabel('γ = σ/V_th')
    ax.set_title('Activity (Soft Reset Rate)', fontsize=11, fontweight='bold')
    for i in range(len(betas)):
        for j in range(len(gammas)):
            ax.text(i, j, f'{activity_map[i,j]:.3f}', ha='center', va='center', fontsize=6,
                    color='white' if activity_map[i,j] > 0.15 else 'black')
    plt.colorbar(im, ax=ax)

    # --- Plot 3: CLE Phase Diagram ---
    ax = fig.add_subplot(gs[0, 2])
    vmax = max(abs(cle_map.min()), abs(cle_map.max()), 0.5)
    im = ax.imshow(cle_map.T, aspect='auto', cmap='RdBu_r', origin='lower',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b:.2f}' for b in betas], fontsize=8)
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{g}' for g in gammas])
    ax.set_xlabel('β (decay)')
    ax.set_ylabel('γ = σ/V_th')
    ax.set_title('Conditional Lyapunov Exponent', fontsize=11, fontweight='bold')
    for i in range(len(betas)):
        for j in range(len(gammas)):
            ax.text(i, j, f'{cle_map[i,j]:+.3f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(cle_map[i,j]) > 0.3 else 'black')
    plt.colorbar(im, ax=ax, label='CLE')

    # --- Plot 4: Combined Phase Classification ---
    ax = fig.add_subplot(gs[1, 0])
    phase_code_map = np.zeros_like(sync_error_map)
    for i in range(len(betas)):
        for j in range(len(gammas)):
            phase = classify_phase(sync_error_map[i, j], activity_map[i, j])
            if phase == "线性层流":
                phase_code_map[i, j] = 0
            elif phase == "临界同步":
                phase_code_map[i, j] = 1
            else:  # 非同步湍流
                phase_code_map[i, j] = 2

    cmap = ListedColormap(['#3498db', '#2ecc71', '#e74c3c'])
    im = ax.imshow(phase_code_map.T, aspect='auto', cmap=cmap, origin='lower',
                   vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b:.2f}' for b in betas], fontsize=8)
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{g}' for g in gammas])
    ax.set_xlabel('β (decay)')
    ax.set_ylabel('γ = σ/V_th')
    ax.set_title('Phase Classification', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Laminar\n(linear)', 'Critical\nSync', 'Turbulent\n(chaotic)'])

    # --- Plot 5: Sync Error vs γ for each β ---
    ax = fig.add_subplot(gs[1, 1])
    for i, beta in enumerate(betas):
        ax.semilogy(gammas, sync_error_map[i, :], 'o-', label=f'β={beta}', linewidth=2, markersize=5)
    ax.set_xlabel('γ = σ/V_th')
    ax.set_ylabel('Sync Error')
    ax.set_title('Sync Error vs Drive Ratio', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Activity vs γ for each β ---
    ax = fig.add_subplot(gs[1, 2])
    for i, beta in enumerate(betas):
        ax.plot(gammas, activity_map[i, :], 's-', label=f'β={beta}', linewidth=2, markersize=5)
    ax.set_xlabel('γ = σ/V_th')
    ax.set_ylabel('Activity (Soft Reset Rate)')
    ax.set_title('Activity vs Drive Ratio', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(os.path.dirname(__file__), 'consistency_stability_results.png')
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

    t_total = time.time()

    sync_error_map, activity_map, cle_map, betas, gammas = run_phase_diagram(device)

    visualize(sync_error_map, activity_map, cle_map, betas, gammas)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"实验 7 完成 (总耗时: {elapsed_total:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    # 相图汇总
    print("\n[相图汇总]", flush=True)
    print(f"  {'β':>5s} | {'γ':>5s} | {'V_th':>5s} | {'Sync Err':>10s} | {'Activity':>8s} | {'CLE':>8s} | {'Phase'}", flush=True)
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}", flush=True)
    gamma_vth = {0.1: 10.0, 0.5: 2.0, 1.0: 1.0, 2.0: 0.5, 5.0: 0.2, 10.0: 0.1}
    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):
            phase = classify_phase(sync_error_map[i, j], activity_map[i, j])
            print(f"  {beta:5.2f} | {gamma:5.1f} | {gamma_vth[gamma]:5.1f} | "
                  f"{sync_error_map[i,j]:10.4e} | {activity_map[i,j]:8.4f} | "
                  f"{cle_map[i,j]:+8.4f} | {phase}", flush=True)

    # 关键结论
    n_laminar = np.sum((activity_map < 0.01))
    n_critical = 0
    n_turbulent = 0
    for i in range(len(betas)):
        for j in range(len(gammas)):
            p = classify_phase(sync_error_map[i,j], activity_map[i,j])
            if p == "临界同步":
                n_critical += 1
            elif p == "非同步湍流":
                n_turbulent += 1
    print(f"\n[相统计] 线性层流: {n_laminar}, 临界同步: {n_critical}, 非同步湍流: {n_turbulent}", flush=True)
    if n_critical > 0:
        print("  *** 存在临界同步岛 (Critical Synchronized Phase) ***", flush=True)
        print("  物理工作区已找到：系统既有非线性折叠，又具备同步性", flush=True)
    else:
        print("  未发现临界同步岛", flush=True)
        print("  可能的原因：同步与非线性互斥，或需要更精细的参数扫描", flush=True)
