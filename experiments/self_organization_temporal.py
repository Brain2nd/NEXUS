"""
实验 11: 动力学参数与拓扑结构的自组织演化 (TEMPORAL 模式 + Decoder MSE Loss)
(Self-Organization of Dynamics & Topology via Zero-Order Optimization)
================================================================================

物理动机：
- 从"糟糕"初始态（β=0.5, V_th=1.0, W~N(0,1)）出发
- 通过纯任务 Loss 驱动的 SPSA 零阶优化，联合演化全部参数（W + β + V_th）
- 观察系统是否自发演化到"临界同步区"（β≈0.9, V_th≈10, γ≈0.1）
- 如果成功，证明临界计算态是任务最优解的全局吸引子

关键架构：
1. LOSS = End-to-End Decoder MSE vs Anchors
   - ANCHOR_0 = [+1, 0, 0, 0], ANCHOR_1 = [-1, 0, 0, 0]
   - TEMPORAL mode T timesteps → decode LAST output pulse → MSE vs anchor
   - NaN handling: nan_to_num(nan=1e6, posinf=1e6, neginf=-1e6), loss clamp 1e12
2. OBSERVATION = V-based centroid separation (monitoring only, NOT training loss)
3. 全参数演化：W + β + V_th 联合优化
4. 糟糕初始化：β=0.5±0.1, V_th=1.0（湍流区）
5. 相空间轨迹记录：每轮记录宏观物理量
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
    """提取所有 LIF 的 β 和 V_th（已经是连续浮点 nn.Parameter）"""
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


def full_params_to_flat(model):
    """
    全参数展平：W1_float + W2_float + β + V_th → 一维向量
    权重经过 pulse→float 解码
    """
    w1_float, w2_float = _get_weight_floats(model)
    beta_flat, vth_flat = _get_lif_params(model)
    return torch.cat([w1_float.flatten(), w2_float.flatten(),
                      beta_flat, vth_flat])


def full_flat_to_params(model, flat, w1_shape, w2_shape, n_beta, n_vth):
    """
    全参数写回：一维向量 → 模型
    权重经过 float→pulse 编码
    """
    idx = 0
    # W1
    n_w1 = w1_shape[0] * w1_shape[1]
    w1_float = flat[idx:idx+n_w1].reshape(w1_shape)
    idx += n_w1
    # W2
    n_w2 = w2_shape[0] * w2_shape[1]
    w2_float = flat[idx:idx+n_w2].reshape(w2_shape)
    idx += n_w2
    # β
    beta_flat = flat[idx:idx+n_beta]
    idx += n_beta
    # V_th
    vth_flat = flat[idx:idx+n_vth]
    idx += n_vth

    _set_weight_floats(model, w1_float, w2_float)
    _set_lif_params(model, beta_flat, vth_flat)


def count_params(model):
    """统计全部可演化参数数量（W + β + V_th）"""
    w1_float, w2_float = _get_weight_floats(model)
    beta_flat, vth_flat = _get_lif_params(model)
    return w1_float.numel() + w2_float.numel() + beta_flat.numel() + vth_flat.numel()


# =============================================================================
# 物理量提取
# =============================================================================

def extract_physics(model):
    """提取宏观物理量：mean beta, mean v_threshold, gamma, beta/vth 分布方差"""
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.cpu().numpy().flatten())
            vths.append(module._v_threshold.data.cpu().numpy().flatten())
    if not betas:
        return 0.5, 1.0, 1.0, 0.0, 0.0
    all_beta = np.concatenate(betas)
    all_vth = np.concatenate(vths)
    avg_beta = float(np.mean(all_beta))
    avg_vth = float(np.mean(all_vth))
    beta_std = float(np.std(all_beta))
    vth_std = float(np.std(all_vth))
    gamma = 1.0 / max(avg_vth, 1e-6)
    return avg_beta, avg_vth, gamma, beta_std, vth_std


def extract_physics_full(model):
    """提取完整逐维度参数：返回每个 SimpleLIFNode 的 beta 和 v_th 完整向量

    按模块名称层级分组（linear1, linear2），保存全部参数值。
    """
    params_by_component = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            # 提取顶层组件名（如 linear1.mul.sign_xor... -> linear1）
            parts = name.split('.')
            component = parts[0] if parts else name

            beta_vals = module._beta.data.cpu().numpy().flatten().tolist()
            vth_vals = module._v_threshold.data.cpu().numpy().flatten().tolist()

            if component not in params_by_component:
                params_by_component[component] = {'beta': [], 'v_th': []}
            params_by_component[component]['beta'].extend(beta_vals)
            params_by_component[component]['v_th'].extend(vth_vals)

    # 保存完整参数
    result = {}
    for comp, data in params_by_component.items():
        result[comp] = {
            'beta': data['beta'],      # 完整 beta 向量
            'v_th': data['v_th'],      # 完整 v_th 向量
            'n_params': len(data['beta'])
        }
    return result


def clamp_physics(model):
    """物理约束：强制 beta ∈ (0.01, 0.999), V_th > 0.1"""
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module._beta.data.clamp_(0.01, 0.999)
            module._v_threshold.data.clamp_(0.1, 100.0)


def collect_lif_potentials(model):
    """收集所有 LIF 膜电位（用于观察指标，非训练Loss）"""
    potentials = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode) and module.v is not None:
            potentials[name] = module.v.detach().clone()
    return potentials


def extract_spectral_radius(model):
    """提取 linear1 权重的谱半径"""
    try:
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
# 观察指标：V-based 质心分离（OBSERVATION ONLY）
# =============================================================================

def observe_centroid_separation(model, sequences, labels, device, T_use=80, warmup=10):
    """
    在 TEMPORAL 模式下收集膜电位状态，计算两类质心的欧氏距离作为观察指标。
    返回：centroid_distance (float)
    """
    model.eval()
    all_states = []

    with torch.no_grad():
        for seq_np in sequences:
            seq = torch.from_numpy(seq_np[:T_use]).to(device)
            model.reset()

            v_snapshots = []
            with SpikeMode.temporal():
                for t in range(T_use):
                    x_t = seq[t:t+1, :]
                    x_pulse = float32_to_pulse(x_t)
                    _ = model(x_pulse)

                    # 收集膜电位
                    v_dict = collect_lif_potentials(model)
                    if v_dict and t >= warmup:
                        parts = [v.cpu().numpy().flatten() for v in v_dict.values()]
                        v_snapshots.append(np.concatenate(parts))

            # 用膜电位时间均值 + 末态拼接作为样本表示
            if v_snapshots:
                v_arr = np.array(v_snapshots)
                v_mean = np.mean(v_arr, axis=0)
                v_last = v_arr[-1]
                state = np.concatenate([v_mean, v_last])
            else:
                state = np.zeros(1)

            state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
            all_states.append(state)

    # 确保等长
    max_len = max(len(s) for s in all_states)
    padded = []
    for s in all_states:
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)))
        padded.append(s)
    states = np.array(padded)
    labels_arr = np.array(labels)

    # 计算两类质心
    mask0 = labels_arr == 0
    mask1 = labels_arr == 1
    if np.sum(mask0) == 0 or np.sum(mask1) == 0:
        return 0.0

    mu0 = states[mask0].mean(axis=0)
    mu1 = states[mask1].mean(axis=0)

    # 质心欧氏距离
    centroid_distance = float(np.linalg.norm(mu0 - mu1))
    return centroid_distance


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
# TEMPORAL 模式评估函数（End-to-End Decoder MSE Loss）
# =============================================================================

ANCHOR_0 = torch.tensor([+1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
ANCHOR_1 = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def evaluate_model_decoder(model, sequences, labels, device, T_use=80):
    """
    在 TEMPORAL 模式下评估分类（End-to-End Decoder MSE）。
    - 逐时间步前向传播，膜电位跨步累积（soft reset folding）
    - 取最后时刻输出脉冲 → Decoder → MSE vs anchor
    - NaN handling: nan_to_num(nan=1e6, posinf=1e6, neginf=-1e6), loss clamp 1e12
    - 返回：(loss, acc, firing_rate, decoder_stats)
    """
    model.eval()
    anchor_0 = ANCHOR_0.to(device)
    anchor_1 = ANCHOR_1.to(device)

    total_loss = 0.0
    correct = 0
    total_spikes = 0.0
    total_elements = 0
    nan_count = 0
    inf_count = 0

    with torch.no_grad():
        for seq_np, label in zip(sequences, labels):
            seq = torch.from_numpy(seq_np[:T_use]).to(device)
            model.reset()

            output_pulse = None
            with SpikeMode.temporal():
                for t in range(T_use):
                    x_t = seq[t:t+1, :]
                    x_pulse = float32_to_pulse(x_t)
                    output_pulse = model(x_pulse)  # [1, 4, 32]

            # 解码最后时刻的输出脉冲
            if output_pulse is None:
                output_float = torch.zeros(1, 4, device=device)
            else:
                output_float = pulse_to_float32(output_pulse.float())  # [1, 4]
                total_spikes += output_pulse.float().sum().item()
                total_elements += output_pulse.numel()

            # NaN/Inf 处理
            output_float = torch.nan_to_num(output_float, nan=1e6, posinf=1e6, neginf=-1e6)
            if torch.isnan(output_float).any():
                nan_count += 1
            if torch.isinf(output_float).any():
                inf_count += 1

            # 选择对应的 anchor
            anchor = anchor_0 if label == 0 else anchor_1

            # MSE loss
            mse = torch.mean((output_float.squeeze(0) - anchor) ** 2)
            mse = torch.clamp(mse, 0.0, 1e12)  # 防止极端值
            total_loss += mse.item()

            # 准确率：根据距离哪个 anchor 更近判断
            dist0 = torch.sum((output_float.squeeze(0) - anchor_0) ** 2)
            dist1 = torch.sum((output_float.squeeze(0) - anchor_1) ** 2)
            pred = 0 if dist0 < dist1 else 1
            if pred == label:
                correct += 1

    n_samples = len(sequences)
    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples
    firing_rate = total_spikes / max(total_elements, 1)

    decoder_stats = {
        'nan_count': nan_count,
        'inf_count': inf_count,
        'firing_rate': firing_rate,
    }

    return avg_loss, accuracy, firing_rate, decoder_stats


# =============================================================================
# SPSA 零阶优化器（TEMPORAL 模式 + Decoder Loss）
# =============================================================================

def spsa_step(model, sequences, labels, device, c_k, a_k, T_use=80,
              w1_shape=None, w2_shape=None, n_beta=0, n_vth=0):
    """
    SPSA 单步：对全部参数（W + β + V_th）做高斯扰动。
    权重经过 pulse→float→perturb→float→pulse 的正确转换。
    使用 evaluate_model_decoder（TEMPORAL + Decoder MSE）。
    返回：(loss, acc, fr, stats, diff)
    """
    flat = full_params_to_flat(model)
    n_params = flat.shape[0]

    # 高斯随机扰动（比 Bernoulli 更平滑）
    delta = torch.randn(n_params, device=device)
    delta = delta / (torch.norm(delta) + 1e-8) * np.sqrt(n_params)  # 归一化

    orig_flat = flat.clone()

    # 正向扰动评估（float空间扰动后，权重部分编码回pulse）
    full_flat_to_params(model, orig_flat + c_k * delta, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)
    loss_plus, _, fr_plus, _ = evaluate_model_decoder(model, sequences, labels, device, T_use)

    # 负向扰动评估
    full_flat_to_params(model, orig_flat - c_k * delta, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)
    loss_minus, _, fr_minus, _ = evaluate_model_decoder(model, sequences, labels, device, T_use)

    # 梯度估计
    diff = loss_plus - loss_minus
    if abs(diff) < 1e-15:
        full_flat_to_params(model, orig_flat, w1_shape, w2_shape, n_beta, n_vth)
        print(f"    [SPSA] diff≈0 (L+={loss_plus:.6f}, L-={loss_minus:.6f}), "
              f"fr+={fr_plus:.4f}, fr-={fr_minus:.4f}, skip", flush=True)
        loss, acc, fr, stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
        return loss, acc, fr, stats, diff

    g_hat = diff / (2 * c_k) * delta

    # 逐元素梯度裁剪（而非全局范数裁剪，避免高维稀释）
    max_elem = 10.0
    g_hat = g_hat.clamp(-max_elem, max_elem)
    g_norm = torch.norm(g_hat).item()

    # 计算实际更新幅度
    update = a_k * g_hat
    update_abs = update.abs()
    print(f"    [SPSA] diff={diff:.6f}, |g|={g_norm:.4f}, c={c_k:.4f}, a={a_k:.6f}", flush=True)
    print(f"    [SPSA] update: mean={update_abs.mean():.8f}, max={update_abs.max():.8f}", flush=True)

    # 更新（在浮点空间做完后，权重编码回pulse）
    new_flat = orig_flat - a_k * g_hat
    full_flat_to_params(model, new_flat, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)

    # 评估当前
    loss, acc, fr, stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
    return loss, acc, fr, stats, diff


# =============================================================================
# 主实验
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}", flush=True)
    t_global = time.time()

    # --- 数据 ---
    sequences, labels = generate_spiral_sequences(n_samples_per_class=4, T=100, seed=42)
    T_use = 15
    print(f"[data] {len(sequences)} sequences, T_use={T_use}", flush=True)

    # --- 模型：糟糕初始化 ---
    torch.manual_seed(42)
    # β=0.01 → 近100%泄漏，从"无记忆"出发，让SPSA演化出所需时间残差
    template = SimpleLIFNode(beta=0.01)
    model = SimpleSpikeMLP(4, 8, 4, neuron_template=template).to(device)

    # W ~ N(0, 1)：未经雕琢的随机连接
    w1 = torch.randn(8, 4, device=device) * 1.0
    w2 = torch.randn(4, 8, device=device) * 1.0
    model.set_weights(w1, w2)

    # V_th: DO NOT override — gate-specific thresholds (AND=1.5, OR=0.5, NOT=1.0)
    # are already correctly set by _create_neuron() during model construction.
    # Overwriting would destroy gate logic semantics.

    # 加入 β 随机扰动 ±0.1
    rng = np.random.RandomState(42)
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.uniform(-0.1, 0.1, module._beta.shape),
                                 dtype=torch.float32, device=device)
            module._beta.data.add_(noise)
            module._beta.data.clamp_(0.01, 0.999)

    n_total = count_params(model)
    # 预计算全参数形状信息（供 SPSA 使用）
    w1_shape = (8, 4)   # linear1: out_features x in_features
    w2_shape = (4, 8)   # linear2: out_features x in_features
    beta_flat, vth_flat = _get_lif_params(model)
    n_beta = beta_flat.numel()
    n_vth = vth_flat.numel()

    beta0, vth0, gamma0, beta_std0, vth_std0 = extract_physics(model)
    rho0 = extract_spectral_radius(model)
    print(f"[init] {n_total} params (W1:{w1_shape}, W2:{w2_shape}, β:{n_beta}, V_th:{n_vth})", flush=True)
    print(f"  beta={beta0:.4f}±{beta_std0:.4f}, V_th={vth0:.4f}±{vth_std0:.4f}, "
          f"gamma={gamma0:.4f}, rho={rho0:.4f}", flush=True)

    # --- Phase 1: 初始评估 ---
    print("\n[Phase 1] Initial evaluation (TEMPORAL + Decoder)...", flush=True)
    loss0, acc0, fr0, stats0 = evaluate_model_decoder(model, sequences, labels, device, T_use)
    centroid_sep0 = observe_centroid_separation(model, sequences, labels, device, T_use)
    print(f"[initial] loss={loss0:.6f}, acc={acc0:.4f}, firing_rate={fr0:.4f}, "
          f"centroid_sep={centroid_sep0:.4f}", flush=True)
    print(f"  decoder_stats: {stats0}", flush=True)

    # --- Phase 2: SPSA 全参数演化 ---
    N_EPOCHS = 15
    print(f"\n[Phase 2] SPSA optimization ({N_EPOCHS} epochs, TEMPORAL + Decoder)...", flush=True)

    # SPSA 超参数（a0 调小避免第一步跳太远）
    a0 = 0.005      # 学习率（再降10x，避免beta跳出安全区）
    c0 = 0.01       # 扰动幅度（降低，避免扰动本身引发饱和）
    alpha = 0.602
    gamma_spsa = 0.101

    # 参数插针：记录完整张量快照
    w1_f0, w2_f0 = _get_weight_floats(model)
    beta_t0, vth_t0 = _get_lif_params(model)
    param_snapshots = [{
        'epoch': 0,
        'w1_float': w1_f0.detach().cpu().clone(),
        'w2_float': w2_f0.detach().cpu().clone(),
        'beta_tensor': beta_t0.detach().cpu().clone(),
        'vth_tensor': vth_t0.detach().cpu().clone(),
    }]
    print(f"  [probe] W1 range=[{w1_f0.min():.4f}, {w1_f0.max():.4f}], "
          f"W2 range=[{w2_f0.min():.4f}, {w2_f0.max():.4f}]", flush=True)
    print(f"  [probe] β tensor={beta_t0.tolist()}", flush=True)
    print(f"  [probe] V_th tensor={vth_t0.tolist()}", flush=True)

    # 提取初始完整参数
    params_full_0 = extract_physics_full(model)

    trajectory = {
        'epoch': [0], 'beta': [beta0], 'v_th': [vth0], 'gamma': [gamma0],
        'acc': [acc0], 'loss': [loss0], 'centroid_separation': [centroid_sep0],
        'spectral_radius': [rho0], 'beta_std': [beta_std0], 'vth_std': [vth_std0],
        'diff': [0.0],
        'firing_rate': [fr0],
        'nan_count': [stats0.get('nan_count', 0)],
        'inf_count': [stats0.get('inf_count', 0)],
        'params_full': [params_full_0],  # 逐维度完整参数
    }

    best_loss = loss0
    best_flat = full_params_to_flat(model).clone()
    stagnant_count = 0

    for epoch in range(1, N_EPOCHS + 1):
        a_k = a0 / (epoch + 1) ** alpha       # 标准 SPSA 调度
        c_k = c0 / (epoch + 1) ** gamma_spsa

        loss, acc, fr, stats, diff = spsa_step(model, sequences, labels, device, c_k, a_k, T_use,
                                                  w1_shape=w1_shape, w2_shape=w2_shape,
                                                  n_beta=n_beta, n_vth=n_vth)

        # 物理量采样
        beta_now, vth_now, gamma_now, beta_std_now, vth_std_now = extract_physics(model)
        rho_now = extract_spectral_radius(model)

        # 质心分离观察（每 5 轮）
        if epoch % 5 == 0:
            centroid_sep = observe_centroid_separation(model, sequences, labels, device, T_use)
        else:
            centroid_sep = trajectory['centroid_separation'][-1]

        trajectory['epoch'].append(epoch)
        trajectory['beta'].append(beta_now)
        trajectory['v_th'].append(vth_now)
        trajectory['gamma'].append(gamma_now)
        trajectory['acc'].append(acc)
        trajectory['loss'].append(loss)
        trajectory['centroid_separation'].append(centroid_sep)
        trajectory['spectral_radius'].append(rho_now)
        trajectory['beta_std'].append(beta_std_now)
        trajectory['vth_std'].append(vth_std_now)
        trajectory['diff'].append(diff)
        trajectory['firing_rate'].append(fr)
        trajectory['nan_count'].append(stats.get('nan_count', 0))
        trajectory['inf_count'].append(stats.get('inf_count', 0))
        trajectory['params_full'].append(extract_physics_full(model))

        # 参数插针快照
        w1_now, w2_now = _get_weight_floats(model)
        beta_t_now, vth_t_now = _get_lif_params(model)
        param_snapshots.append({
            'epoch': epoch,
            'w1_float': w1_now.detach().cpu().clone(),
            'w2_float': w2_now.detach().cpu().clone(),
            'beta_tensor': beta_t_now.detach().cpu().clone(),
            'vth_tensor': vth_t_now.detach().cpu().clone(),
        })
        print(f"  [probe] W1=[{w1_now.min():.4f},{w1_now.max():.4f}] "
              f"W2=[{w2_now.min():.4f},{w2_now.max():.4f}] "
              f"β={beta_t_now.tolist()} V_th={vth_t_now.tolist()}", flush=True)

        # 记录最佳
        if loss < best_loss:
            best_loss = loss
            best_flat = full_params_to_flat(model).clone()
            stagnant_count = 0
        else:
            stagnant_count += 1

        elapsed = time.time() - t_global
        print(f"[epoch {epoch:3d}] loss={loss:.6f}, acc={acc:.4f}, fr={fr:.4f}, "
              f"nan={stats.get('nan_count',0)}, "
              f"centroid_sep={centroid_sep:.4f}, "
              f"beta={beta_now:.4f}±{beta_std_now:.4f}, V_th={vth_now:.4f}, "
              f"gamma={gamma_now:.4f}, rho={rho_now:.4f}, "
              f"time={elapsed:.0f}s", flush=True)

        # 如果停滞太久，扩大扰动
        if stagnant_count >= 10 and epoch < N_EPOCHS - 5:
            print(f"  [adaptive] Stagnant for {stagnant_count} epochs, increasing c_k", flush=True)
            c0 *= 1.5
            stagnant_count = 0

    # --- Final evaluation ---
    print("\n[Final] Restoring best parameters and evaluating...", flush=True)
    full_flat_to_params(model, best_flat, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)
    final_loss, final_acc, final_fr, final_stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
    final_centroid_sep = observe_centroid_separation(model, sequences, labels, device, T_use)
    final_beta, final_vth, final_gamma, final_beta_std, final_vth_std = extract_physics(model)
    final_rho = extract_spectral_radius(model)
    print(f"[FINAL] loss={final_loss:.6f}, acc={final_acc:.4f}, "
          f"centroid_sep={final_centroid_sep:.4f}, "
          f"beta={final_beta:.4f}±{final_beta_std:.4f}, V_th={final_vth:.4f}±{final_vth_std:.4f}, "
          f"gamma={final_gamma:.4f}, rho={final_rho:.4f}", flush=True)
    print(f"  decoder_stats: {final_stats}", flush=True)

    # =============================================================================
    # 数据持久化：存 JSON（绘图/分析脚本可独立读取）
    # =============================================================================
    import json
    data_path = os.path.join(os.path.dirname(__file__), 'self_organization_temporal_data.json')
    save_data = {
        'trajectory': trajectory,
        'config': {
            'beta_init': 0.01, 'a0': a0, 'c0': c0, 'alpha': alpha,
            'gamma_spsa': gamma_spsa, 'N_EPOCHS': N_EPOCHS, 'T_use': T_use,
        },
        'final': {
            'loss': final_loss, 'acc': final_acc, 'fr': final_fr,
            'beta': final_beta, 'v_th': final_vth, 'gamma': final_gamma,
            'rho': final_rho, 'beta_std': final_beta_std, 'vth_std': final_vth_std,
        },
    }
    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Save] Trajectory data saved to {data_path}", flush=True)

    # =============================================================================
    # 可视化 (6-panel)
    # =============================================================================
    print("\n[Plot] Generating 6-panel figure...", flush=True)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    epochs = trajectory['epoch']
    betas = trajectory['beta']
    gammas = trajectory['gamma']
    accs = trajectory['acc']
    losses = trajectory['loss']
    centroid_seps = trajectory['centroid_separation']
    rhos = trajectory['spectral_radius']
    vths = trajectory['v_th']
    diffs = trajectory['diff']

    # --- Panel 1: (beta, gamma) 相空间轨迹 ---
    ax1 = fig.add_subplot(gs[0, 0])
    colors = np.array(accs)
    sc = ax1.scatter(betas, gammas, c=colors, cmap='RdYlGn', s=40, zorder=2,
                     edgecolors='gray', linewidths=0.5, vmin=0.4, vmax=1.0)
    ax1.plot(betas[0], gammas[0], 'bs', markersize=14, label='Start (turbulent)', zorder=3)
    ax1.plot(betas[-1], gammas[-1], 'r*', markersize=18, label='End', zorder=3)
    ax1.plot(betas, gammas, 'k-', alpha=0.3, linewidth=0.8, zorder=1)
    # 目标区域
    target_rect = plt.Rectangle((0.85, 0.05), 0.12, 0.15, fill=False,
                                 edgecolor='gold', linewidth=2.5, linestyle='--',
                                 label='Target zone (β≈0.9, γ≈0.1)')
    ax1.add_patch(target_rect)
    plt.colorbar(sc, ax=ax1, label='Accuracy')
    ax1.set_xlabel('β (mean dissipation)')
    ax1.set_ylabel('γ (drive ratio = 1/V_th)')
    ax1.set_title('Phase Space Trajectory\n(β, γ) colored by accuracy')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: β / V_th 演化 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    l1, = ax2.plot(epochs, betas, 'b-o', markersize=2, label='β', linewidth=2)
    l2, = ax2_twin.plot(epochs, vths, 'r-s', markersize=2, label='V_th', linewidth=2)
    ax2.axhline(y=0.9, color='blue', linestyle=':', alpha=0.5, label='target β=0.9')
    ax2_twin.axhline(y=10.0, color='red', linestyle=':', alpha=0.5, label='target V_th=10')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('β (mean)', color='blue')
    ax2_twin.set_ylabel('V_th (mean)', color='red')
    ax2.set_title('Physics Parameters Evolution')
    ax2.legend([l1, l2], ['β', 'V_th'], fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Accuracy / Loss (Decoder MSE) ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, accs, 'g-o', markersize=2, label='Accuracy', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, losses, 'r-', alpha=0.5, label='Loss (Decoder MSE)', linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy', color='green')
    ax3_twin.set_ylabel('Loss (Decoder MSE)', color='red')
    ax3.set_title('Task Performance (TEMPORAL + Decoder)')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='lower right')

    # --- Panel 4: γ 演化 + 质心分离观察 ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, gammas, 'm-o', markersize=2, linewidth=2, label='γ (mean)')
    ax4.axhline(y=0.1, color='gold', linestyle='--', linewidth=2, label='target γ=0.1')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, centroid_seps, 'c--', markersize=1, linewidth=1.5,
                  alpha=0.7, label='Centroid Sep (obs)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('γ (drive ratio)', color='magenta')
    ax4_twin.set_ylabel('Centroid Separation (observation)', color='cyan')
    ax4.set_title('Drive Ratio + Centroid Separation (V-based)')
    ax4.legend(fontsize=9, loc='upper left')
    ax4_twin.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: |SPSA diff| signal ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, [abs(d) for d in diffs], 'orange', linewidth=1.5, label='|SPSA diff|')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('|SPSA diff|', color='orange')
    ax5.set_title('SPSA Gradient Signal')
    ax5.set_yscale('symlog', linthresh=1e-6)
    ax5.legend(fontsize=8, loc='upper left')
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: 谱半径 + 参数分化度 ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, rhos, 'orange', marker='o', markersize=2, linewidth=2, label='ρ(W1)')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(epochs, trajectory['beta_std'], 'b--', linewidth=1, alpha=0.7, label='β std')
    ax6_twin.plot(epochs, trajectory['vth_std'], 'r--', linewidth=1, alpha=0.7, label='V_th std')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Spectral Radius', color='orange')
    ax6_twin.set_ylabel('Parameter Std (differentiation)')
    ax6.set_title('Weight Structure & Parameter Differentiation')
    ax6.legend(fontsize=8, loc='upper left')
    ax6_twin.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Experiment 11: Self-Organization of Dynamics & Topology\n'
                 f'(TEMPORAL + Decoder MSE, {N_EPOCHS} epochs SPSA, start: β={beta0:.2f}, V_th={vth0:.1f}, γ={gamma0:.2f})',
                 fontsize=14, fontweight='bold')

    out_path = os.path.join(os.path.dirname(__file__), 'self_organization_temporal_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}", flush=True)

    # --- 数值摘要 ---
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 11 SUMMARY (TEMPORAL Self-Organization + Decoder MSE)", flush=True)
    print("=" * 70, flush=True)
    print(f"Architecture:", flush=True)
    print(f"  LOSS = End-to-End Decoder MSE vs Anchors (ANCHOR_0=[+1,0,0,0], ANCHOR_1=[-1,0,0,0])", flush=True)
    print(f"  OBSERVATION = V-based centroid separation (monitoring only)", flush=True)
    print(f"Start:  β={beta0:.4f}, V_th={vth0:.4f}, γ={gamma0:.4f}, "
          f"loss={loss0:.6f}, acc={acc0:.4f}, centroid_sep={centroid_sep0:.4f}", flush=True)
    print(f"End:    β={final_beta:.4f}, V_th={final_vth:.4f}, γ={final_gamma:.4f}, "
          f"loss={final_loss:.6f}, acc={final_acc:.4f}, centroid_sep={final_centroid_sep:.4f}", flush=True)
    print(f"Target: β≈0.90, V_th≈10, γ≈0.10", flush=True)
    print(f"Δβ = {final_beta - beta0:+.4f}, ΔV_th = {final_vth - vth0:+.4f}, "
          f"Δγ = {final_gamma - gamma0:+.4f}", flush=True)
    print(f"Differentiation: β_std={final_beta_std:.4f}, V_th_std={final_vth_std:.4f}", flush=True)
    print(f"Spectral Radius: {final_rho:.4f}", flush=True)
    n_nonzero_diffs = sum(1 for d in diffs if abs(d) > 1e-15)
    print(f"SPSA gradient signal: {n_nonzero_diffs}/{N_EPOCHS} epochs had nonzero diff", flush=True)
    total_time = time.time() - t_global
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
