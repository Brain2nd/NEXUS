"""
实验 6: 频域响应与共振分析 (Spectral Analysis & Resonance)
==========================================================

物理学视角：把 NEXUS 看作一个复杂的非线性滤波器。
- β ≈ 1 → 低通滤波器（积分器）
- β ≈ 0.9 → 可能产生高频谐波（混沌折叠）

验证目标：
1. NEXUS 在 Temporal 模式下对什么频率最敏感？
2. 是否自动滤除高频噪声？
3. 是否存在 1/f 噪声（自组织临界性 SOC 的标志）？
4. 是否存在频率锁定（锁相能力）？

实验设计：
A. 正弦波频率扫描 — 输入 I(t) = A·sin(ωt)，扫描 ω 和 A，观测 V(t) 的 FFT/PSD
B. 随机输入的 PSD 分析 — 对随机驱动下的 V(t) 做功率谱，检测 1/f^α
C. 频率锁定测试 — 正弦波+噪声输入，检测输出是否锁定输入频率
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
# Model & Utils (consistent with previous experiments)
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


def collect_representative_v(model):
    """收集代表性神经元的 V 标量（取第一个非零元素）"""
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            v_flat = module.v.detach().cpu().numpy().flatten()
            nz = np.nonzero(np.abs(v_flat) > 1e-30)[0]
            if len(nz) > 0:
                return float(v_flat[nz[0]]), name
    return 0.0, "none"


def collect_multi_v(model, max_neurons=5):
    """收集多个有活动的神经元的 V"""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            v_flat = module.v.detach().cpu().numpy().flatten()
            nz = np.nonzero(np.abs(v_flat) > 1e-30)[0]
            if len(nz) > 0:
                result.append((float(v_flat[nz[0]]), name))
                if len(result) >= max_neurons:
                    break
    return result


# =============================================================================
# 实验 A: 正弦波频率扫描
# =============================================================================

def run_frequency_sweep(device):
    """
    扫描输入频率和幅度，测量系统的频率响应。
    输入: I(t) = A * sin(2π * f * t / T_total)
    对每个 (f, A, β) 组合，采集 V(t) 并做 FFT。
    """
    print("\n" + "=" * 70)
    print("实验 A: 正弦波频率扫描")
    print("=" * 70)

    T = 128  # 缩短以控制总耗时（~0.5s/step on CPU）
    betas = [0.50, 0.90, 0.99]
    # 输入频率（归一化频率 f/f_s）— 精简到关键频率
    freqs = [0.02, 0.05, 0.1, 0.2, 0.4]
    amplitudes = [0.5, 2.0]  # 中幅/大幅（大幅更容易触发软复位折叠）

    results = {}  # (beta, A) -> {freq -> {input_psd, output_psd, v_trace}}

    for beta in betas:
        for A in [0.5, 2.0]:
            key = (beta, A)
            results[key] = {}

            model, in_f = create_model(device, beta)
            model.reset()

            for f_in in freqs:
                model.reset()  # 每个频率重新开始
                t_arr = np.arange(T)
                # 构造正弦输入（4 维，相位偏移使每维不同）
                input_signals = np.zeros((T, in_f))
                for dim in range(in_f):
                    input_signals[:, dim] = A * np.sin(2 * np.pi * f_in * t_arr + dim * np.pi / in_f)

                v_trace = []
                with SpikeMode.temporal():
                    for t in range(T):
                        x_float = torch.tensor(input_signals[t], dtype=torch.float32, device=device)
                        forward_one_step(model, x_float, device)
                        v_val, _ = collect_representative_v(model)
                        v_trace.append(v_val)

                v_trace = np.array(v_trace)

                # FFT
                # 去掉前 32 步 transient
                v_steady = v_trace[32:]
                v_detrend = v_steady - np.mean(v_steady)

                if np.std(v_detrend) < 1e-20:
                    psd = np.zeros(len(v_detrend) // 2 + 1)
                    fft_freqs = np.fft.rfftfreq(len(v_detrend))
                else:
                    # Welch-like: 直接 FFT + 功率
                    fft_vals = np.fft.rfft(v_detrend)
                    psd = np.abs(fft_vals) ** 2 / len(v_detrend)
                    fft_freqs = np.fft.rfftfreq(len(v_detrend))

                results[key][f_in] = {
                    'v_trace': v_trace,
                    'psd': psd,
                    'fft_freqs': fft_freqs,
                    'v_steady': v_detrend,
                }

                print(f"  β={beta:.2f}, A={A:.1f}, f={f_in:.3f}: "
                      f"V_range={v_trace.max()-v_trace.min():.3f}, "
                      f"V_std={np.std(v_trace):.3f}", flush=True)

    # 计算增益（Gain = output power at f_in / input power at f_in）
    gain_results = {}  # (beta, A) -> list of (f_in, gain)
    for (beta, A), freq_dict in results.items():
        gains = []
        for f_in, data in freq_dict.items():
            fft_freqs = data['fft_freqs']
            psd = data['psd']
            if len(psd) > 0 and np.any(psd > 0):
                # 找到最接近 f_in 的频率 bin
                idx = np.argmin(np.abs(fft_freqs - f_in))
                # gain = output PSD at f_in / A^2 (input power)
                gain = psd[idx] / (A ** 2 + 1e-30)
                gains.append((f_in, gain, psd[idx]))
            else:
                gains.append((f_in, 0.0, 0.0))
        gain_results[(beta, A)] = gains

    print("\n  增益汇总 (A=0.5):")
    for beta in betas:
        print(f"    β={beta:.2f}:", end="")
        for f_in, gain, _ in gain_results[(beta, 0.5)]:
            print(f" f={f_in:.2f}→G={gain:.2e}", end="")
        print()

    return results, gain_results


# =============================================================================
# 实验 B: 随机驱动下的 PSD 分析（1/f 噪声检测）
# =============================================================================

def run_psd_analysis(device):
    """
    用随机输入驱动系统，对 V(t) 做功率谱密度分析。
    检测是否存在 1/f^α 噪声（α≈1 为粉红噪声/SOC 标志）。
    """
    print("\n" + "=" * 70)
    print("实验 B: 随机驱动 PSD 分析 (1/f 噪声检测)")
    print("=" * 70)

    T = 256  # 平衡频率分辨率与耗时
    betas = [0.50, 0.90, 0.95, 0.99]

    results = {}

    for beta in betas:
        t0 = time.time()
        model, in_f = create_model(device, beta)
        model.reset()

        torch.manual_seed(999)
        inputs = [torch.randn(in_f, device=device) * 0.5 for _ in range(T)]

        v_traces = {}  # neuron_name -> list of V
        with SpikeMode.temporal():
            for t in range(T):
                forward_one_step(model, inputs[t], device)
                neurons = collect_multi_v(model, max_neurons=3)
                for v_val, name in neurons:
                    if name not in v_traces:
                        v_traces[name] = []
                    v_traces[name].append(v_val)
                if (t + 1) % 200 == 0:
                    print(f"    β={beta:.2f} [{t+1}/{T}]", flush=True)

        # 对每个神经元计算 PSD
        psd_data = {}
        for name, trace in v_traces.items():
            trace = np.array(trace)
            # 去掉 transient (前 64 步)
            trace = trace[64:]
            trace_detrend = trace - np.mean(trace)

            if np.std(trace_detrend) < 1e-20:
                continue

            # Welch 方法（手动分段平均提高 PSD 估计）
            seg_len = 128
            n_segs = len(trace_detrend) // seg_len
            if n_segs < 1:
                continue

            psd_sum = None
            window = np.hanning(seg_len)
            for i in range(n_segs):
                seg = trace_detrend[i * seg_len:(i + 1) * seg_len] * window
                fft_vals = np.fft.rfft(seg)
                psd_seg = np.abs(fft_vals) ** 2 / seg_len
                if psd_sum is None:
                    psd_sum = psd_seg
                else:
                    psd_sum += psd_seg
            psd_avg = psd_sum / n_segs
            fft_freqs = np.fft.rfftfreq(seg_len)

            # 拟合 1/f^α（log-log 线性回归）
            # 排除 DC (f=0) 和极高频
            valid = (fft_freqs > 0.01) & (fft_freqs < 0.45) & (psd_avg > 1e-30)
            if np.sum(valid) > 5:
                log_f = np.log10(fft_freqs[valid])
                log_p = np.log10(psd_avg[valid])
                # 线性拟合 log(PSD) = -α * log(f) + c
                coeffs = np.polyfit(log_f, log_p, 1)
                alpha = -coeffs[0]  # PSD ∝ 1/f^α
                r_squared = 1 - np.sum((log_p - np.polyval(coeffs, log_f))**2) / np.sum((log_p - np.mean(log_p))**2)
            else:
                alpha = 0.0
                r_squared = 0.0

            psd_data[name] = {
                'psd': psd_avg,
                'freqs': fft_freqs,
                'alpha': alpha,
                'r_squared': r_squared,
                'trace': trace,
            }

        elapsed = time.time() - t0
        results[beta] = psd_data

        # 报告
        if psd_data:
            first_name = list(psd_data.keys())[0]
            d = psd_data[first_name]
            print(f"  β={beta:.2f}: α={d['alpha']:.3f} (R²={d['r_squared']:.3f}), "
                  f"neuron={first_name.split('.')[-2]}.{first_name.split('.')[-1]} "
                  f"({elapsed:.1f}s)", flush=True)
        else:
            print(f"  β={beta:.2f}: 无有效 PSD 数据 ({elapsed:.1f}s)", flush=True)

    # 汇总
    print("\n  1/f^α 噪声汇总:")
    print(f"  {'β':>6} | {'α':>8} | {'R²':>8} | 解读")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")
    for beta, psd_data in results.items():
        if psd_data:
            first_name = list(psd_data.keys())[0]
            d = psd_data[first_name]
            if d['alpha'] > 0.8 and d['alpha'] < 1.2 and d['r_squared'] > 0.7:
                interp = "★ 1/f 粉红噪声 (SOC)"
            elif d['alpha'] > 1.5 and d['r_squared'] > 0.7:
                interp = "1/f² 布朗噪声 (积分器)"
            elif d['alpha'] < 0.3 and d['r_squared'] > 0.5:
                interp = "白噪声 (无相关)"
            elif d['r_squared'] < 0.5:
                interp = "非幂律谱"
            else:
                interp = f"1/f^{d['alpha']:.1f} 噪声"
            print(f"  {beta:6.2f} | {d['alpha']:8.3f} | {d['r_squared']:8.3f} | {interp}")

    return results


# =============================================================================
# 实验 C: 频率锁定测试 (Phase Locking)
# =============================================================================

def run_phase_locking(device):
    """
    输入: 正弦波 + 高斯噪声 I(t) = A·sin(2πft) + σ·η(t)
    检测输出 V(t) 的 FFT 是否在 f 处有显著峰值。
    扫描信噪比 (SNR = A/σ)，测量输出在 f 处的频谱集中度。
    """
    print("\n" + "=" * 70)
    print("实验 C: 频率锁定测试 (Phase Locking)")
    print("=" * 70)

    T = 128
    f_signal = 0.05  # 信号频率
    A = 1.0           # 信号幅度
    noise_levels = [0.0, 0.3, 1.0, 3.0]  # σ
    betas = [0.50, 0.90, 0.99]

    results = {}

    for beta in betas:
        results[beta] = {}
        for sigma in noise_levels:
            model, in_f = create_model(device, beta)
            model.reset()

            torch.manual_seed(555)
            t_arr = np.arange(T)

            # 构造输入
            input_signals = np.zeros((T, in_f))
            for dim in range(in_f):
                signal = A * np.sin(2 * np.pi * f_signal * t_arr + dim * np.pi / in_f)
                noise = sigma * np.random.randn(T)
                input_signals[:, dim] = signal + noise

            v_trace = []
            with SpikeMode.temporal():
                for t in range(T):
                    x_float = torch.tensor(input_signals[t], dtype=torch.float32, device=device)
                    forward_one_step(model, x_float, device)
                    v_val, _ = collect_representative_v(model)
                    v_trace.append(v_val)

            v_trace = np.array(v_trace)
            v_steady = v_trace[32:]
            v_detrend = v_steady - np.mean(v_steady)

            # FFT
            if np.std(v_detrend) > 1e-20:
                fft_vals = np.fft.rfft(v_detrend)
                psd = np.abs(fft_vals) ** 2 / len(v_detrend)
                fft_freqs = np.fft.rfftfreq(len(v_detrend))

                # 频谱集中度 = f_signal 处的功率 / 总功率
                idx_signal = np.argmin(np.abs(fft_freqs - f_signal))
                # 取 signal 附近 ±2 bins 的功率
                lo = max(0, idx_signal - 2)
                hi = min(len(psd), idx_signal + 3)
                power_signal = np.sum(psd[lo:hi])
                power_total = np.sum(psd[1:])  # 排除 DC
                spectral_concentration = power_signal / (power_total + 1e-30)

                # 峰值检测：f_signal 处是否为局部最大值
                is_peak = (idx_signal > 0 and idx_signal < len(psd) - 1 and
                          psd[idx_signal] > psd[idx_signal - 1] and
                          psd[idx_signal] > psd[idx_signal + 1])
            else:
                psd = np.zeros(len(v_detrend) // 2 + 1)
                fft_freqs = np.fft.rfftfreq(len(v_detrend))
                spectral_concentration = 0.0
                is_peak = False

            snr = A / (sigma + 1e-30)
            results[beta][sigma] = {
                'v_trace': v_trace,
                'psd': psd,
                'fft_freqs': fft_freqs,
                'spectral_concentration': spectral_concentration,
                'is_peak': is_peak,
                'snr': snr,
            }

            lock_str = "LOCKED" if spectral_concentration > 0.1 else "unlocked"
            print(f"  β={beta:.2f}, σ={sigma:.1f} (SNR={snr:.1f}): "
                  f"concentration={spectral_concentration:.4f} [{lock_str}]"
                  f"{' ★PEAK' if is_peak else ''}", flush=True)

    return results


# =============================================================================
# 可视化
# =============================================================================

def visualize_spectral(freq_results, gain_results, psd_results, lock_results, save_path):
    """生成综合可视化"""
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('Spectral Analysis & Resonance of NEXUS Gate Circuit Dynamics',
                 fontsize=16, fontweight='bold', y=0.98)

    betas_sweep = [0.50, 0.70, 0.90, 0.99]
    colors = {0.50: '#e74c3c', 0.70: '#f39c12', 0.90: '#2ecc71', 0.95: '#3498db', 0.99: '#9b59b6'}

    # =========================================================================
    # Row 1: 频率响应 (Bode 图风格) — 4 个 β
    # =========================================================================
    for i, beta in enumerate(betas_sweep):
        ax = fig.add_subplot(gs[0, i])
        for A in [0.5, 2.0]:
            key = (beta, A)
            if key in gain_results:
                gains = gain_results[key]
                f_vals = [g[0] for g in gains]
                g_vals = [g[1] for g in gains]
                # 用 dB 表示增益
                g_db = [10 * np.log10(g + 1e-30) for g in g_vals]
                ax.plot(f_vals, g_db, 'o-', label=f'A={A}', markersize=4, linewidth=1.5)
        ax.set_xlabel('Input Frequency (f/f_s)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'Frequency Response β={beta}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    # =========================================================================
    # Row 2: PSD (功率谱密度) — 不同 β
    # =========================================================================
    betas_psd = [0.50, 0.70, 0.90, 0.99]
    for i, beta in enumerate(betas_psd):
        ax = fig.add_subplot(gs[1, i])
        if beta in psd_results and psd_results[beta]:
            first_name = list(psd_results[beta].keys())[0]
            d = psd_results[beta][first_name]
            freqs = d['freqs']
            psd = d['psd']
            # Ensure freqs and psd have same length
            n_min = min(len(freqs), len(psd))
            freqs_plot = freqs[:n_min]
            psd_plot = psd[:n_min]
            valid = freqs_plot > 0
            if np.any(valid) and np.any(psd_plot[valid] > 0):
                ax.loglog(freqs_plot[valid], psd_plot[valid], '-', color=colors.get(beta, 'gray'),
                         linewidth=1.2, label=f'PSD (α={d["alpha"]:.2f})')
                # 拟合线
                fit_mask = (freqs_plot > 0.01) & (freqs_plot < 0.45) & (psd_plot > 1e-30)
                if np.sum(fit_mask) > 2:
                    fit_f = freqs_plot[fit_mask]
                    fit_p = fit_f ** (-d['alpha']) * np.median(psd_plot[fit_mask] * fit_f ** d['alpha'])
                    ax.loglog(fit_f, fit_p, '--', color='black', linewidth=1, alpha=0.7,
                             label=f'$1/f^{{{d["alpha"]:.2f}}}$ fit')
        ax.set_xlabel('Frequency (f/f_s)')
        ax.set_ylabel('PSD')
        title = f'PSD β={beta}'
        if beta in psd_results and psd_results[beta]:
            fn = list(psd_results[beta].keys())[0]
            title = f'Power Spectrum β={beta}\n(α={psd_results[beta][fn]["alpha"]:.2f}, R²={psd_results[beta][fn]["r_squared"]:.3f})'
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

    # =========================================================================
    # Row 3: V(t) 时间序列在正弦波驱动下 — 3 个 β × f=0.05
    # =========================================================================
    for i, beta in enumerate([0.50, 0.90, 0.99]):
        ax = fig.add_subplot(gs[2, i])
        key = (beta, 0.5)
        if key in freq_results and 0.05 in freq_results[key]:
            v_trace = freq_results[key][0.05]['v_trace']
            t_arr = np.arange(len(v_trace))
            ax.plot(t_arr, v_trace, '-', linewidth=0.8, color=colors.get(beta, 'gray'))
            # 叠加输入正弦波（缩放到 V 的范围）
            v_range = v_trace.max() - v_trace.min()
            if v_range > 0:
                sin_scaled = v_trace.mean() + 0.3 * v_range * np.sin(2 * np.pi * 0.05 * t_arr)
                ax.plot(t_arr, sin_scaled, '--', color='black', alpha=0.4, linewidth=0.8, label='input (scaled)')
        ax.set_xlabel('Time step')
        ax.set_ylabel('V(t)')
        ax.set_title(f'V(t) under sinusoid β={beta}\n(f=0.05, A=0.5)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 1/f 指数 α vs β 汇总
    ax_alpha = fig.add_subplot(gs[2, 3])
    alpha_betas = []
    alpha_vals = []
    for beta in sorted(psd_results.keys()):
        if psd_results[beta]:
            first_name = list(psd_results[beta].keys())[0]
            d = psd_results[beta][first_name]
            alpha_betas.append(beta)
            alpha_vals.append(d['alpha'])
    if alpha_betas:
        bar_colors = ['#e74c3c' if 0.8 < a < 1.2 else '#3498db' for a in alpha_vals]
        ax_alpha.bar(range(len(alpha_betas)), alpha_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax_alpha.set_xticks(range(len(alpha_betas)))
        ax_alpha.set_xticklabels([f'{b:.2f}' for b in alpha_betas], fontsize=8)
        ax_alpha.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='α=1 (pink noise/SOC)')
        ax_alpha.axhline(y=2.0, color='blue', linestyle='--', alpha=0.5, label='α=2 (Brownian)')
        ax_alpha.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='α=0 (white)')
    ax_alpha.set_xlabel('β')
    ax_alpha.set_ylabel('Spectral exponent α')
    ax_alpha.set_title('1/f^α Exponent vs β')
    ax_alpha.legend(fontsize=6)
    ax_alpha.grid(True, alpha=0.3)

    # =========================================================================
    # Row 4: 频率锁定 — spectral concentration vs SNR
    # =========================================================================
    for i, beta in enumerate([0.50, 0.90, 0.99]):
        ax = fig.add_subplot(gs[3, i])
        if beta in lock_results:
            sigmas = sorted(lock_results[beta].keys())
            snrs = [lock_results[beta][s]['snr'] for s in sigmas]
            concs = [lock_results[beta][s]['spectral_concentration'] for s in sigmas]
            peaks = [lock_results[beta][s]['is_peak'] for s in sigmas]

            ax.plot(sigmas, concs, 'o-', color=colors.get(beta, 'gray'), linewidth=1.5, markersize=6)
            # 标记有峰值的点
            for j, (s, c, p) in enumerate(zip(sigmas, concs, peaks)):
                if p:
                    ax.plot(s, c, '*', color='red', markersize=12, zorder=5)

            ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='lock threshold')
        ax.set_xlabel('Noise σ')
        ax.set_ylabel('Spectral Concentration')
        ax.set_title(f'Phase Locking β={beta}\n(f_signal=0.05)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # PSD comparison at one noise level
    ax_cmp = fig.add_subplot(gs[3, 3])
    sigma_show = 0.3
    for beta in [0.50, 0.90, 0.99]:
        if beta in lock_results and sigma_show in lock_results[beta]:
            d = lock_results[beta][sigma_show]
            freqs = d['fft_freqs']
            psd = d['psd']
            valid = freqs > 0
            if np.any(valid) and np.any(psd[valid] > 0):
                ax_cmp.semilogy(freqs[valid], psd[valid], '-', linewidth=1.2,
                               color=colors.get(beta, 'gray'), label=f'β={beta}')
    ax_cmp.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label=f'f_signal=0.05')
    ax_cmp.set_xlabel('Frequency')
    ax_cmp.set_ylabel('PSD')
    ax_cmp.set_title(f'Output PSD with noise σ={sigma_show}')
    ax_cmp.legend(fontsize=7)
    ax_cmp.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
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

    # A: Frequency Sweep
    freq_results, gain_results = run_frequency_sweep(device) if only in (None, 'A') else ({}, {})

    # B: PSD Analysis (1/f noise)
    psd_results = run_psd_analysis(device) if only in (None, 'B') else {}

    # C: Phase Locking
    lock_results = run_phase_locking(device) if only in (None, 'C') else {}

    # Visualize
    save_path = os.path.join(os.path.dirname(__file__), 'spectral_resonance_results.png')
    if only is None:
        visualize_spectral(freq_results, gain_results, psd_results, lock_results, save_path)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"频域响应与共振分析 完成 (总耗时: {elapsed:.1f}s)")
    print(f"{'=' * 70}")

    # Summary
    if psd_results:
        print("\n[B] 1/f^α 噪声指数:")
        for beta in sorted(psd_results.keys()):
            if psd_results[beta]:
                first_name = list(psd_results[beta].keys())[0]
                d = psd_results[beta][first_name]
                print(f"  β={beta:.2f}: α = {d['alpha']:.3f} (R² = {d['r_squared']:.3f})")

    if lock_results:
        print("\n[C] 频率锁定:")
        for beta in sorted(lock_results.keys()):
            for sigma in sorted(lock_results[beta].keys()):
                d = lock_results[beta][sigma]
                lock_str = "LOCKED" if d['spectral_concentration'] > 0.1 else "unlocked"
                print(f"  β={beta:.2f}, σ={sigma:.1f}: concentration={d['spectral_concentration']:.4f} [{lock_str}]")
