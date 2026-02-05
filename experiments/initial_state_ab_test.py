"""
Experiment 12: Guided Self-Organization via Critical Initialization and Structured SPSA
================================================================================

A/B test:
  Group A (Turbulent): beta~N(0.5,0.01), V_th~N(1.0,0.1), W~N(0,1)
    → Strategy: full-parameter SPSA (no phase distinction)
  Group B (Critical):  beta~N(0.90,0.01), V_th~N(10.0,1.0), W=orthogonal(rho~1.5)
    → Strategy: two-phase structured SPSA

Architecture (CORRECTED):
  float input → Encoder(float32_to_pulse) → SNN_Temporal(T steps) → Decoder(pulse_to_float32) → float output
  Loss = MSE(decoded_output, anchor_target)
  Anchor targets: Class 0 → [+1, 0, 0, 0], Class 1 → [-1, 0, 0, 0]

  Encoder/Decoder are framework boundary components, NOT trained.
  Only SNN internal params (W, beta, V_th) are optimized by SPSA.

Reset semantics:
  - Reset ONLY between complete sequences (after all T timesteps processed)
  - Never reset mid-sequence — temporal dynamics accumulate within a sequence
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


# Anchor targets
ANCHOR_0 = np.array([+1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Class 0
ANCHOR_1 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Class 1


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
# Weight read/write utilities
# =============================================================================

def _get_weight_floats(model):
    w1_pulse = model.linear1.weight_pulse
    w2_pulse = model.linear2.weight_pulse
    w1_float = pulse_to_float32(w1_pulse.float())
    w2_float = pulse_to_float32(w2_pulse.float())
    return w1_float, w2_float


def _set_weight_floats(model, w1_float, w2_float):
    model.linear1.set_weight_from_float(w1_float)
    model.linear2.set_weight_from_float(w2_float)


def _get_lif_params(model):
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.flatten())
            vths.append(module._v_threshold.data.flatten())
    return torch.cat(betas), torch.cat(vths)


def _set_lif_params(model, beta_flat, vth_flat):
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
    w1_float, w2_float = _get_weight_floats(model)
    beta_flat, vth_flat = _get_lif_params(model)
    return torch.cat([w1_float.flatten(), w2_float.flatten(), beta_flat, vth_flat])


def full_flat_to_params(model, flat, w1_shape, w2_shape, n_beta, n_vth):
    idx = 0
    n_w1 = w1_shape[0] * w1_shape[1]
    w1_float = flat[idx:idx+n_w1].reshape(w1_shape)
    idx += n_w1
    n_w2 = w2_shape[0] * w2_shape[1]
    w2_float = flat[idx:idx+n_w2].reshape(w2_shape)
    idx += n_w2
    beta_flat = flat[idx:idx+n_beta]
    idx += n_beta
    vth_flat = flat[idx:idx+n_vth]
    _set_weight_floats(model, w1_float, w2_float)
    _set_lif_params(model, beta_flat, vth_flat)


# =============================================================================
# Layer-shared param utilities (Phase I)
# =============================================================================

def _get_layer_shared_params(model):
    w1_float, w2_float = _get_weight_floats(model)
    layer_betas, layer_vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            layer_betas.append(module._beta.data.mean().unsqueeze(0))
            layer_vths.append(module._v_threshold.data.mean().unsqueeze(0))
    return torch.cat([w1_float.flatten(), w2_float.flatten(),
                      torch.cat(layer_betas), torch.cat(layer_vths)])


def _set_layer_shared_params(model, flat, w1_shape, w2_shape, n_lif_modules):
    idx = 0
    n_w1 = w1_shape[0] * w1_shape[1]
    w1_float = flat[idx:idx+n_w1].reshape(w1_shape)
    idx += n_w1
    n_w2 = w2_shape[0] * w2_shape[1]
    w2_float = flat[idx:idx+n_w2].reshape(w2_shape)
    idx += n_w2
    layer_betas = flat[idx:idx+n_lif_modules]
    idx += n_lif_modules
    layer_vths = flat[idx:idx+n_lif_modules]
    _set_weight_floats(model, w1_float, w2_float)
    lif_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module._beta.data.fill_(layer_betas[lif_idx].item())
            module._v_threshold.data.fill_(layer_vths[lif_idx].item())
            lif_idx += 1


def count_lif_modules(model):
    return sum(1 for _, m in model.named_modules() if isinstance(m, SimpleLIFNode))


# =============================================================================
# Physics extraction (monitoring probes — kept for visualization)
# =============================================================================

def extract_physics(model):
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


def clamp_physics(model):
    """Minimal clamp: only prevent numerical explosion, allow free evolution."""
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module._beta.data.clamp_(0.001, 0.9999)
            module._v_threshold.data.clamp_(0.001, 1000.0)


def extract_spectral_radius(model):
    try:
        w_pulse = model.linear1.weight_pulse
        w_float = pulse_to_float32(w_pulse.float())
        svs = torch.linalg.svdvals(w_float)
        return svs[0].item()
    except:
        return 0.0


def get_all_beta_vth_arrays(model):
    betas, vths = [], []
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            betas.append(module._beta.data.cpu().numpy().flatten())
            vths.append(module._v_threshold.data.cpu().numpy().flatten())
    return np.concatenate(betas), np.concatenate(vths)


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


# =============================================================================
# Data generation: Temporal Spiral Classification
# =============================================================================

def generate_spiral_sequences(n_samples_per_class=32, T=100, noise_std=0.15, seed=42):
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
# CORRECTED evaluation: Encoder → SNN(Temporal) → Decoder → MSE vs Anchor
# =============================================================================

def evaluate_model_decoder(model, sequences, labels, device, T_use=100):
    """
    End-to-end evaluation using Decoder output.

    For each sequence:
      1. Reset model (clear context from previous sequence)
      2. In TEMPORAL mode, feed T timesteps: encode → SNN → get output pulse
      3. Decode final output pulse via pulse_to_float32
      4. Compute MSE against anchor target

    Returns:
      loss: mean MSE across all samples
      acc: classification accuracy (nearest-anchor)
      firing_rate: average output firing rate
      decoder_stats: dict with mean, var, nan_ratio of decoded outputs
    """
    model.eval()
    all_decoded = []
    total_spikes = 0
    total_outputs = 0
    nan_count = 0

    with torch.no_grad():
        for seq_np, label in zip(sequences, labels):
            seq = torch.from_numpy(seq_np[:T_use]).to(device)

            # Reset between sequences (clear temporal context)
            model.reset()

            with SpikeMode.temporal():
                for t in range(T_use):
                    x_t = seq[t:t+1, :]                  # [1, 4]
                    x_pulse = float32_to_pulse(x_t)       # Encoder: [1, 4, 32]
                    out_pulse = model(x_pulse)             # SNN temporal: [1, 4, 32]

                    # Count spikes for firing rate
                    total_spikes += out_pulse.sum().item()
                    total_outputs += out_pulse.numel()

            # Decode the LAST timestep's output pulse
            decoded = pulse_to_float32(out_pulse)          # Decoder: [1, 4]
            decoded_np = decoded.cpu().numpy().flatten()    # [4]

            if np.any(np.isnan(decoded_np)) or np.any(np.isinf(decoded_np)):
                nan_count += 1

            all_decoded.append(decoded_np)

    all_decoded = np.array(all_decoded)  # [N, 4]
    labels_arr = np.array(labels)
    N = len(labels)

    # Replace NaN/Inf for loss computation (penalize with large value)
    decoded_clean = np.nan_to_num(all_decoded, nan=1e6, posinf=1e6, neginf=-1e6)

    # MSE loss against anchors
    targets = np.array([ANCHOR_0 if l == 0 else ANCHOR_1 for l in labels])  # [N, 4]
    mse_per_sample = np.mean((decoded_clean - targets) ** 2, axis=1)  # [N]
    loss = float(np.mean(mse_per_sample))

    # Clamp loss to avoid Inf propagation in SPSA
    loss = min(loss, 1e12)

    # Nearest-anchor accuracy
    correct = 0
    for i in range(N):
        d0 = np.sum((decoded_clean[i] - ANCHOR_0) ** 2)
        d1 = np.sum((decoded_clean[i] - ANCHOR_1) ** 2)
        pred = 0 if d0 < d1 else 1
        if pred == labels_arr[i]:
            correct += 1
    acc = correct / N

    # Firing rate
    firing_rate = total_spikes / max(total_outputs, 1)

    # Decoder output stats
    finite_mask = np.isfinite(all_decoded)
    if np.any(finite_mask):
        finite_vals = all_decoded[finite_mask]
        dec_mean = float(np.mean(finite_vals))
        dec_var = float(np.var(finite_vals))
    else:
        dec_mean = float('nan')
        dec_var = float('nan')
    nan_ratio = nan_count / N

    decoder_stats = {'mean': dec_mean, 'var': dec_var, 'nan_ratio': nan_ratio}

    return float(loss), acc, firing_rate, decoder_stats


# =============================================================================
# SPSA with Bernoulli perturbation
# =============================================================================

def _bernoulli_delta(n, device):
    """Bernoulli +/-1 perturbation (standard SPSA)."""
    return torch.where(torch.rand(n, device=device) > 0.5,
                       torch.ones(n, device=device),
                       -torch.ones(n, device=device))


def spsa_step_shared(model, sequences, labels, device, c_k, a_k, T_use,
                     w1_shape, w2_shape, n_lif_modules):
    """Phase I: Layer-shared beta/V_th SPSA with Bernoulli perturbation."""
    flat = _get_layer_shared_params(model)
    n_params = flat.shape[0]
    delta = _bernoulli_delta(n_params, device)
    orig_flat = flat.clone()

    _set_layer_shared_params(model, orig_flat + c_k * delta, w1_shape, w2_shape, n_lif_modules)
    clamp_physics(model)
    loss_plus, _, _, _ = evaluate_model_decoder(model, sequences, labels, device, T_use)

    _set_layer_shared_params(model, orig_flat - c_k * delta, w1_shape, w2_shape, n_lif_modules)
    clamp_physics(model)
    loss_minus, _, _, _ = evaluate_model_decoder(model, sequences, labels, device, T_use)

    diff = loss_plus - loss_minus
    if abs(diff) < 1e-15:
        _set_layer_shared_params(model, orig_flat, w1_shape, w2_shape, n_lif_modules)
        loss, acc, fr, dec_stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
        return loss, acc, diff, fr, dec_stats

    g_hat = (diff / (2 * c_k)) * (1.0 / delta)
    g_hat = g_hat.clamp(-10.0, 10.0)
    new_flat = orig_flat - a_k * g_hat
    _set_layer_shared_params(model, new_flat, w1_shape, w2_shape, n_lif_modules)
    clamp_physics(model)

    loss, acc, fr, dec_stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
    return loss, acc, diff, fr, dec_stats


def spsa_step_full(model, sequences, labels, device, c_k, a_k, T_use,
                   w1_shape, w2_shape, n_beta, n_vth, momentum_buf=None, mu=0.9):
    """Full-parameter SPSA with Bernoulli perturbation + momentum."""
    flat = full_params_to_flat(model)
    n_params = flat.shape[0]
    delta = _bernoulli_delta(n_params, device)
    orig_flat = flat.clone()

    full_flat_to_params(model, orig_flat + c_k * delta, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)
    loss_plus, _, _, _ = evaluate_model_decoder(model, sequences, labels, device, T_use)

    full_flat_to_params(model, orig_flat - c_k * delta, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)
    loss_minus, _, _, _ = evaluate_model_decoder(model, sequences, labels, device, T_use)

    diff = loss_plus - loss_minus
    if abs(diff) < 1e-15:
        full_flat_to_params(model, orig_flat, w1_shape, w2_shape, n_beta, n_vth)
        loss, acc, fr, dec_stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
        return loss, acc, diff, fr, momentum_buf, dec_stats

    g_hat = (diff / (2 * c_k)) * (1.0 / delta)
    g_hat = g_hat.clamp(-10.0, 10.0)

    # Momentum
    if momentum_buf is None:
        momentum_buf = g_hat.clone()
    else:
        momentum_buf = mu * momentum_buf + (1 - mu) * g_hat

    new_flat = orig_flat - a_k * momentum_buf
    full_flat_to_params(model, new_flat, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)

    loss, acc, fr, dec_stats = evaluate_model_decoder(model, sequences, labels, device, T_use)
    return loss, acc, diff, fr, momentum_buf, dec_stats


# =============================================================================
# Initialization
# =============================================================================

def orthogonal_init_with_spectral_radius(shape, target_rho=1.5, device='cpu'):
    rows, cols = shape
    A = torch.randn(rows, cols, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    w = U @ Vh
    svs = torch.linalg.svdvals(w)
    current_rho = svs[0].item()
    if current_rho > 0:
        w = w * (target_rho / current_rho)
    return w


def create_model_group_a(device):
    """Group A: High-leak start + random W.

    Start from near-100% leak (beta=0.01): membrane potential decays almost
    completely each step, preventing saturation. SPSA then evolves beta upward
    to discover what temporal residuals the task requires.

    V_th retains gate-specific defaults (AND=1.5, OR=0.5, NOT=1.0).
    """
    torch.manual_seed(42)
    template = SimpleLIFNode(beta=0.01)
    model = SimpleSpikeMLP(4, 8, 4, neuron_template=template).to(device)
    w1 = torch.randn(8, 4, device=device)
    w2 = torch.randn(4, 8, device=device)
    model.set_weights(w1, w2)
    # beta is already set to 0.5 via neuron_template; add small noise
    rng = np.random.RandomState(42)
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.normal(0, 0.01, module._beta.shape),
                                 dtype=torch.float32, device=device)
            module._beta.data.add_(noise)
            module._beta.data.clamp_(0.01, 0.999)
    # V_th: DO NOT modify — gate-specific thresholds are already correct
    return model


def create_model_group_b(device):
    """Group B: High-leak start + orthogonal W.

    Same near-100% leak (beta=0.01) but with structured orthogonal weights.
    Compare with Group A to see if weight structure matters when starting
    from the high-leak regime.

    V_th retains gate-specific defaults (AND=1.5, OR=0.5, NOT=1.0).
    """
    torch.manual_seed(42)
    template = SimpleLIFNode(beta=0.01)
    model = SimpleSpikeMLP(4, 8, 4, neuron_template=template).to(device)
    w1 = orthogonal_init_with_spectral_radius((8, 4), target_rho=1.5, device=device)
    w2 = orthogonal_init_with_spectral_radius((4, 8), target_rho=1.5, device=device)
    model.set_weights(w1, w2)
    # beta is already set to 0.9 via neuron_template; add small noise
    rng = np.random.RandomState(42)
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.normal(0, 0.01, module._beta.shape),
                                 dtype=torch.float32, device=device)
            module._beta.data.add_(noise)
            module._beta.data.clamp_(0.01, 0.999)
    # V_th: DO NOT modify — gate-specific thresholds are already correct
    return model


# =============================================================================
# Training loop
# =============================================================================

def train_group(group_name, model, sequences, labels, device, T_use,
                N_EPOCHS_PHASE1=50, N_EPOCHS_PHASE2=50, strategy='two_phase'):
    w1_shape = (8, 4)
    w2_shape = (4, 8)
    beta_flat, vth_flat = _get_lif_params(model)
    n_beta = beta_flat.numel()
    n_vth = vth_flat.numel()
    n_lif = count_lif_modules(model)
    N_TOTAL = N_EPOCHS_PHASE1 + N_EPOCHS_PHASE2

    # SPSA hyperparams
    a0 = 0.005      # 学习率（再降10x，避免beta跳出安全区）
    c0 = 0.01       # 扰动幅度（降低，避免扰动本身引发饱和）
    alpha = 0.602
    gamma_spsa = 0.101

    # Initial evaluation
    beta0, vth0, gamma0, beta_std0, vth_std0 = extract_physics(model)
    rho0 = extract_spectral_radius(model)
    loss0, acc0, fr0, dec_stats0 = evaluate_model_decoder(model, sequences, labels, device, T_use)
    params_full_0 = extract_physics_full(model)

    print(f"\n[{group_name}] Init: beta={beta0:.4f}+/-{beta_std0:.4f}, V_th={vth0:.4f}+/-{vth_std0:.4f}, "
          f"gamma={gamma0:.4f}, rho={rho0:.4f}", flush=True)
    print(f"[{group_name}] Init: loss={loss0:.6f}, acc={acc0:.4f}, fr={fr0:.4f}, "
          f"dec_mean={dec_stats0['mean']:.4f}, dec_var={dec_stats0['var']:.4f}, "
          f"nan_ratio={dec_stats0['nan_ratio']:.4f}", flush=True)

    trajectory = {
        'epoch': [0], 'beta': [beta0], 'v_th': [vth0], 'gamma': [gamma0],
        'acc': [acc0], 'loss': [loss0],
        'spectral_radius': [rho0], 'beta_std': [beta_std0], 'vth_std': [vth_std0],
        'diff': [0.0], 'phase': [0], 'firing_rate': [fr0],
        'dec_mean': [dec_stats0['mean']], 'dec_var': [dec_stats0['var']],
        'nan_ratio': [dec_stats0['nan_ratio']],
        'params_full': [params_full_0],  # 完整逐维度参数
    }

    best_loss = loss0
    best_flat = full_params_to_flat(model).clone()
    momentum_buf = None
    t_start = time.time()

    for epoch in range(1, N_TOTAL + 1):
        a_k = a0 / (epoch + 1) ** alpha
        c_k = c0 / (epoch + 1) ** gamma_spsa

        if strategy == 'full_only':
            # Group A: all epochs use full-parameter SPSA
            phase = 0
            loss, acc, diff, fr, momentum_buf, dec_stats = spsa_step_full(
                model, sequences, labels, device, c_k, a_k, T_use,
                w1_shape, w2_shape, n_beta, n_vth, momentum_buf)
        else:
            # Group B: two-phase structured evolution
            phase = 1 if epoch <= N_EPOCHS_PHASE1 else 2
            if phase == 1:
                loss, acc, diff, fr, dec_stats = spsa_step_shared(
                    model, sequences, labels, device, c_k, a_k, T_use,
                    w1_shape, w2_shape, n_lif)
            else:
                loss, acc, diff, fr, momentum_buf, dec_stats = spsa_step_full(
                    model, sequences, labels, device, c_k, a_k, T_use,
                    w1_shape, w2_shape, n_beta, n_vth, momentum_buf)

        beta_now, vth_now, gamma_now, beta_std_now, vth_std_now = extract_physics(model)
        rho_now = extract_spectral_radius(model)
        params_full_now = extract_physics_full(model)

        trajectory['epoch'].append(epoch)
        trajectory['beta'].append(beta_now)
        trajectory['v_th'].append(vth_now)
        trajectory['gamma'].append(gamma_now)
        trajectory['acc'].append(acc)
        trajectory['loss'].append(loss)
        trajectory['spectral_radius'].append(rho_now)
        trajectory['beta_std'].append(beta_std_now)
        trajectory['vth_std'].append(vth_std_now)
        trajectory['diff'].append(diff)
        trajectory['phase'].append(phase)
        trajectory['firing_rate'].append(fr)
        trajectory['dec_mean'].append(dec_stats['mean'])
        trajectory['dec_var'].append(dec_stats['var'])
        trajectory['nan_ratio'].append(dec_stats['nan_ratio'])
        trajectory['params_full'].append(params_full_now)

        if loss < best_loss:
            best_loss = loss
            best_flat = full_params_to_flat(model).clone()

        elapsed = time.time() - t_start
        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS_PHASE1 or epoch == N_EPOCHS_PHASE1 + 1:
            phase_str = "full" if phase == 0 else ("I (shared)" if phase == 1 else "II (indep)")
            print(f"[{group_name}] Ep {epoch:3d} Ph{phase_str}: loss={loss:.6f}, acc={acc:.4f}, "
                  f"beta={beta_now:.4f}+/-{beta_std_now:.4f}, V_th={vth_now:.2f}, "
                  f"fr={fr:.4f}, nan={dec_stats['nan_ratio']:.2f}, "
                  f"dec_m={dec_stats['mean']:.2e}, dec_v={dec_stats['var']:.2e} [{elapsed:.0f}s]", flush=True)

    # Restore best
    full_flat_to_params(model, best_flat, w1_shape, w2_shape, n_beta, n_vth)
    clamp_physics(model)
    final_loss, final_acc, final_fr, final_dec = evaluate_model_decoder(model, sequences, labels, device, T_use)
    final_beta, final_vth, final_gamma, final_beta_std, final_vth_std = extract_physics(model)
    final_rho = extract_spectral_radius(model)

    print(f"\n[{group_name}] FINAL: loss={final_loss:.6f}, acc={final_acc:.4f}, "
          f"beta={final_beta:.4f}+/-{final_beta_std:.4f}, V_th={final_vth:.2f}+/-{final_vth_std:.4f}, "
          f"gamma={final_gamma:.4f}, rho={final_rho:.4f}, fr={final_fr:.4f}, "
          f"nan={final_dec['nan_ratio']:.2f}", flush=True)

    all_betas, all_vths = get_all_beta_vth_arrays(model)
    return trajectory, all_betas, all_vths


# =============================================================================
# Visualization (10-panel)
# =============================================================================

def plot_results(traj_a, traj_b, betas_a, vths_a, betas_b, vths_b, N_P1, N_P2):
    fig = plt.figure(figsize=(28, 21))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    ca, cb = '#e74c3c', '#2ecc71'

    # Panel 1: (beta, gamma) phase space
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj_a['beta'], traj_a['gamma'], '-', color=ca, alpha=0.5, linewidth=1)
    ax1.scatter(traj_a['beta'], traj_a['gamma'], c=range(len(traj_a['beta'])),
                cmap='Reds', s=15, zorder=2, edgecolors='none')
    ax1.plot(traj_b['beta'], traj_b['gamma'], '-', color=cb, alpha=0.5, linewidth=1)
    ax1.scatter(traj_b['beta'], traj_b['gamma'], c=range(len(traj_b['beta'])),
                cmap='Greens', s=15, zorder=2, edgecolors='none')
    ax1.plot(traj_a['beta'][0], traj_a['gamma'][0], 'rs', markersize=12, label='A start')
    ax1.plot(traj_b['beta'][0], traj_b['gamma'][0], 'g^', markersize=12, label='B start')
    ax1.plot(traj_a['beta'][-1], traj_a['gamma'][-1], 'r*', markersize=16, label='A end')
    ax1.plot(traj_b['beta'][-1], traj_b['gamma'][-1], 'g*', markersize=16, label='B end')
    rect = plt.Rectangle((0.85, 0.05), 0.12, 0.15, fill=False,
                          edgecolor='gold', linewidth=2.5, linestyle='--', label='Target zone')
    ax1.add_patch(rect)
    ax1.set_xlabel('beta (mean)')
    ax1.set_ylabel('gamma = 1/V_th')
    ax1.set_title('Phase Space Trajectory (beta, gamma)')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Loss (MSE vs Anchor)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(traj_a['epoch'], traj_a['loss'], color=ca, linewidth=2, label='Group A (turbulent)')
    ax2.plot(traj_b['epoch'], traj_b['loss'], color=cb, linewidth=2, label='Group B (critical)')
    ax2.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5, label=f'Phase II start (ep {N_P1})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss (vs Anchor)')
    ax2.set_title('Loss Trajectory (Decoder Output)')
    ax2.set_yscale('symlog', linthresh=0.01)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(traj_a['epoch'], traj_a['acc'], color=ca, linewidth=2, label='Group A')
    ax3.plot(traj_b['epoch'], traj_b['acc'], color=cb, linewidth=2, label='Group B')
    ax3.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Classification Accuracy (Nearest Anchor)')
    ax3.set_ylim(0.3, 1.05)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: beta evolution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(traj_a['epoch'], traj_a['beta'], color=ca, linewidth=2, label='A: beta')
    ax4.plot(traj_b['epoch'], traj_b['beta'], color=cb, linewidth=2, label='B: beta')
    ax4.axhline(y=0.9, color='gold', linestyle='--', linewidth=1.5, label='target beta=0.9')
    ax4.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('beta (mean)')
    ax4.set_title('beta Evolution')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: V_th evolution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(traj_a['epoch'], traj_a['v_th'], color=ca, linewidth=2, label='A: V_th')
    ax5.plot(traj_b['epoch'], traj_b['v_th'], color=cb, linewidth=2, label='B: V_th')
    ax5.axhline(y=10.0, color='gold', linestyle='--', linewidth=1.5, label='target V_th=10')
    ax5.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('V_th (mean)')
    ax5.set_title('V_th Evolution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: NaN Ratio (Decoder health)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(traj_a['epoch'], traj_a['nan_ratio'], color=ca, linewidth=2, label='A: NaN ratio')
    ax6.plot(traj_b['epoch'], traj_b['nan_ratio'], color=cb, linewidth=2, label='B: NaN ratio')
    ax6.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('NaN Ratio')
    ax6.set_title('Decoder Health (NaN Ratio)')
    ax6.set_ylim(-0.05, 1.05)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Panel 7: beta histogram
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(betas_b, bins=20, color=cb, alpha=0.7, edgecolor='black', label='B: beta')
    ax7.hist(betas_a, bins=20, color=ca, alpha=0.4, edgecolor='black', label='A: beta')
    ax7.axvline(x=0.9, color='gold', linestyle='--', linewidth=2, label='target 0.9')
    ax7.set_xlabel('beta value')
    ax7.set_ylabel('Count')
    ax7.set_title('Final beta Distribution')
    ax7.legend(fontsize=8)

    # Panel 8: V_th histogram
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(vths_b, bins=20, color=cb, alpha=0.7, edgecolor='black', label='B: V_th')
    ax8.hist(vths_a, bins=20, color=ca, alpha=0.4, edgecolor='black', label='A: V_th')
    ax8.axvline(x=10.0, color='gold', linestyle='--', linewidth=2, label='target 10.0')
    ax8.set_xlabel('V_th value')
    ax8.set_ylabel('Count')
    ax8.set_title('Final V_th Distribution')
    ax8.legend(fontsize=8)

    # Panel 9: Spectral Radius + Parameter Differentiation
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(traj_a['epoch'], traj_a['spectral_radius'], color=ca, linewidth=2, label='A: rho(W1)')
    ax9.plot(traj_b['epoch'], traj_b['spectral_radius'], color=cb, linewidth=2, label='B: rho(W1)')
    ax9_t = ax9.twinx()
    ax9_t.plot(traj_b['epoch'], traj_b['beta_std'], color=cb, linestyle='--', alpha=0.6, label='B: beta_std')
    ax9_t.plot(traj_b['epoch'], traj_b['vth_std'], color=cb, linestyle=':', alpha=0.6, label='B: V_th_std')
    ax9.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Spectral Radius')
    ax9_t.set_ylabel('Parameter Std (differentiation)')
    ax9.set_title('Structure & Differentiation')
    ax9.legend(fontsize=7, loc='upper left')
    ax9_t.legend(fontsize=7, loc='upper right')
    ax9.grid(True, alpha=0.3)

    # Panel 10: Firing Rate
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(traj_a['epoch'], traj_a['firing_rate'], color=ca, linewidth=2, label='A: Firing Rate')
    ax10.plot(traj_b['epoch'], traj_b['firing_rate'], color=cb, linewidth=2, label='B: Firing Rate')
    ax10.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
    ax10.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax10.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Firing Rate')
    ax10.set_title('Activity Monitor (0=dead, 1=epileptic)')
    ax10.set_ylim(-0.05, 1.05)
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)

    # Panel 11: Decoder output variance
    ax11 = fig.add_subplot(gs[3, 1])
    dec_var_a = [v if np.isfinite(v) else 0 for v in traj_a['dec_var']]
    dec_var_b = [v if np.isfinite(v) else 0 for v in traj_b['dec_var']]
    ax11.plot(traj_a['epoch'], dec_var_a, color=ca, linewidth=2, label='A: Dec Var')
    ax11.plot(traj_b['epoch'], dec_var_b, color=cb, linewidth=2, label='B: Dec Var')
    ax11.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Decoder Output Variance')
    ax11.set_title('Decoder Output Diversity')
    ax11.set_yscale('symlog', linthresh=1e-6)
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    # Panel 12: SPSA gradient signal (diff)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(traj_a['epoch'], [abs(d) for d in traj_a['diff']], color=ca, linewidth=1, alpha=0.7, label='A: |diff|')
    ax12.plot(traj_b['epoch'], [abs(d) for d in traj_b['diff']], color=cb, linewidth=1, alpha=0.7, label='B: |diff|')
    ax12.axvline(x=N_P1, color='gray', linestyle='--', alpha=0.5)
    ax12.set_xlabel('Epoch')
    ax12.set_ylabel('|L+ - L-|')
    ax12.set_title('SPSA Gradient Signal Strength')
    ax12.set_yscale('symlog', linthresh=1e-10)
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    fig.suptitle('Experiment 12: A/B Test - Critical Init & Structured SPSA\n'
                 'Architecture: Encoder -> SNN(Temporal) -> Decoder | Loss: MSE vs Anchor',
                 fontsize=16, fontweight='bold')

    out_path = os.path.join(os.path.dirname(__file__), 'initial_state_ab_test_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}", flush=True)
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}", flush=True)
    t_global = time.time()

    # Data: B=8 (4 per class), T=15, T_use=15  (small scale for verification)
    sequences, labels = generate_spiral_sequences(n_samples_per_class=4, T=15, seed=42)
    T_use = 15
    print(f"[data] {len(sequences)} sequences, T_use={T_use}", flush=True)

    N_P1, N_P2 = 30, 20
    a0, c0 = 0.005, 0.01  # 与 train_group 内保持一致

    # Group A: High-leak (beta=0.01) + random W
    print("\n" + "=" * 70)
    print("GROUP A: HIGH-LEAK START (beta=0.01, random W, gate V_th=default)")
    print("=" * 70)
    model_a = create_model_group_a(device)
    traj_a, betas_a, vths_a = train_group("A", model_a, sequences, labels, device, T_use, N_P1, N_P2, strategy='full_only')

    # Group B: High-leak (beta=0.01) + orthogonal W
    print("\n" + "=" * 70)
    print("GROUP B: HIGH-LEAK START (beta=0.01, orthogonal W, gate V_th=default)")
    print("=" * 70)
    model_b = create_model_group_b(device)
    traj_b, betas_b, vths_b = train_group("B", model_b, sequences, labels, device, T_use, N_P1, N_P2)

    # Comparison summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 12 A/B COMPARISON (Decoder-based)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Group A (Turbulent)':<25} {'Group B (Critical)':<25}")
    print("-" * 75)
    print(f"{'Init beta':<25} {traj_a['beta'][0]:<25.4f} {traj_b['beta'][0]:<25.4f}")
    print(f"{'Init V_th':<25} {traj_a['v_th'][0]:<25.4f} {traj_b['v_th'][0]:<25.4f}")
    print(f"{'Init gamma':<25} {traj_a['gamma'][0]:<25.4f} {traj_b['gamma'][0]:<25.4f}")
    print(f"{'Init rho(W1)':<25} {traj_a['spectral_radius'][0]:<25.4f} {traj_b['spectral_radius'][0]:<25.4f}")
    print(f"{'Init Loss (MSE)':<25} {traj_a['loss'][0]:<25.6f} {traj_b['loss'][0]:<25.6f}")
    print(f"{'Init Acc':<25} {traj_a['acc'][0]:<25.4f} {traj_b['acc'][0]:<25.4f}")
    print(f"{'Init Firing Rate':<25} {traj_a['firing_rate'][0]:<25.4f} {traj_b['firing_rate'][0]:<25.4f}")
    print(f"{'Init NaN Ratio':<25} {traj_a['nan_ratio'][0]:<25.4f} {traj_b['nan_ratio'][0]:<25.4f}")
    print("-" * 75)
    print(f"{'Final beta':<25} {traj_a['beta'][-1]:<25.4f} {traj_b['beta'][-1]:<25.4f}")
    print(f"{'Final V_th':<25} {traj_a['v_th'][-1]:<25.4f} {traj_b['v_th'][-1]:<25.4f}")
    print(f"{'Final gamma':<25} {traj_a['gamma'][-1]:<25.4f} {traj_b['gamma'][-1]:<25.4f}")
    print(f"{'Final rho(W1)':<25} {traj_a['spectral_radius'][-1]:<25.4f} {traj_b['spectral_radius'][-1]:<25.4f}")
    print(f"{'Final Loss (MSE)':<25} {traj_a['loss'][-1]:<25.6f} {traj_b['loss'][-1]:<25.6f}")
    print(f"{'Final Acc':<25} {traj_a['acc'][-1]:<25.4f} {traj_b['acc'][-1]:<25.4f}")
    print(f"{'Final Firing Rate':<25} {traj_a['firing_rate'][-1]:<25.4f} {traj_b['firing_rate'][-1]:<25.4f}")
    print(f"{'Final NaN Ratio':<25} {traj_a['nan_ratio'][-1]:<25.4f} {traj_b['nan_ratio'][-1]:<25.4f}")
    print(f"{'beta std (diff)':<25} {traj_a['beta_std'][-1]:<25.6f} {traj_b['beta_std'][-1]:<25.6f}")
    print(f"{'V_th std':<25} {traj_a['vth_std'][-1]:<25.6f} {traj_b['vth_std'][-1]:<25.6f}")
    print("-" * 75)
    loss_improve_a = (traj_a['loss'][0] - traj_a['loss'][-1]) / max(traj_a['loss'][0], 1e-8) * 100
    loss_improve_b = (traj_b['loss'][0] - traj_b['loss'][-1]) / max(traj_b['loss'][0], 1e-8) * 100
    print(f"{'Loss Improvement %':<25} {loss_improve_a:<25.1f} {loss_improve_b:<25.1f}")

    total_time = time.time() - t_global
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    # 数据持久化：存 JSON（绘图/分析脚本可独立读取）
    import json
    data_path = os.path.join(os.path.dirname(__file__), 'initial_state_ab_test_data.json')
    save_data = {
        'traj_a': traj_a, 'traj_b': traj_b,
        'config': {'a0': a0, 'c0': c0, 'N_P1': N_P1, 'N_P2': N_P2, 'T_use': T_use},
    }
    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Save] Data saved to {data_path}", flush=True)

    plot_results(traj_a, traj_b, betas_a, vths_a, betas_b, vths_b, N_P1, N_P2)

    print("\n[DONE] Experiment 12 complete.", flush=True)


if __name__ == '__main__':
    main()
