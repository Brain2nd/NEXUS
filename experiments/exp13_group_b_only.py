"""
实验 13 Group B：各向异性 SPSA (Anisotropic) - 单独运行
"""

print("[import] torch, numpy...", flush=True)
import torch
import torch.nn as nn
import numpy as np
import json
import time

print("[import] matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("[import] atomic_ops...", flush=True)
import sys
sys.path.insert(0, '/Users/tangzhengzheng/Desktop/NEXUS')

from atomic_ops import (
    SpikeMode,
    SimpleLIFNode,
    SpikeFP32Linear_MultiPrecision,
)
from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

print("[import] done", flush=True)

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] {DEVICE}", flush=True)

N_EPOCHS = 100
T_USE = 15

# Group B: 各向异性 (Anisotropic) - 量纲匹配
ANISO_c_W = 0.05
ANISO_c_beta = 0.001
ANISO_c_vth = 0.1

ANISO_a_W = 0.01
ANISO_a_beta = 0.0001
ANISO_a_vth = 0.001

alpha = 0.602
gamma_spsa = 0.101

# ============================================================
# 模型定义
# ============================================================
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


# ============================================================
# 数据生成
# ============================================================
def generate_temporal_sequences(n_samples=8, seq_len=15, n_features=4):
    torch.manual_seed(42)
    sequences = []
    labels = []
    for i in range(n_samples):
        label = i % 2
        seq = torch.randn(seq_len, n_features) * 0.5
        if label == 0:
            seq[:, 0] += 0.5
        else:
            seq[:, 0] -= 0.5
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.tensor(labels)


# ============================================================
# 参数工具函数
# ============================================================
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
    if not betas:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(betas), torch.cat(vths)


def _set_lif_params(model, all_beta, all_vth):
    idx_beta, idx_vth = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            n_beta = module._beta.numel()
            n_vth = module._v_threshold.numel()
            module._beta.data.copy_(all_beta[idx_beta:idx_beta+n_beta].view(module._beta.shape))
            module._v_threshold.data.copy_(all_vth[idx_vth:idx_vth+n_vth].view(module._v_threshold.shape))
            idx_beta += n_beta
            idx_vth += n_vth


def _bernoulli_delta(n, device):
    return torch.where(torch.rand(n, device=device) > 0.5,
                       torch.ones(n, device=device),
                       -torch.ones(n, device=device))


def clamp_physics(model):
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            module._beta.data.clamp_(0.01, 0.999)
            module._v_threshold.data.clamp_(0.1, 100.0)


# ============================================================
# 物理量提取
# ============================================================
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
    return (float(np.mean(all_beta)), float(np.mean(all_vth)),
            1.0/max(np.mean(all_vth), 1e-6),
            float(np.std(all_beta)), float(np.std(all_vth)))


def extract_physics_full(model):
    params_by_component = {}
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            parts = name.split('.')
            component = parts[0] if parts else name
            beta_vals = module._beta.data.cpu().numpy().flatten().tolist()
            vth_vals = module._v_threshold.data.cpu().numpy().flatten().tolist()
            if component not in params_by_component:
                params_by_component[component] = {'beta': [], 'v_th': []}
            params_by_component[component]['beta'].extend(beta_vals)
            params_by_component[component]['v_th'].extend(vth_vals)
    result = {}
    for comp, data in params_by_component.items():
        result[comp] = {
            'beta': data['beta'],
            'v_th': data['v_th'],
            'n_params': len(data['beta'])
        }
    return result


def extract_spectral_radius(model):
    w1_float, w2_float = _get_weight_floats(model)
    try:
        s1 = torch.linalg.svdvals(w1_float)[0].item()
        s2 = torch.linalg.svdvals(w2_float)[0].item()
        return max(s1, s2)
    except:
        return 0.0


def extract_weight_stats(model):
    w1_float, w2_float = _get_weight_floats(model)
    w_all = torch.cat([w1_float.flatten(), w2_float.flatten()])
    w_mean = float(w_all.mean().item())
    w_std = float(w_all.std().item())
    w_norm = float(torch.norm(w_all).item())
    return w_mean, w_std, w_norm


# ============================================================
# 前向评估
# ============================================================
def evaluate_model(model, sequences, labels, device, T_use):
    model.eval()
    SpikeMode.set_global_mode(SpikeMode.TEMPORAL)

    total_loss = 0.0
    correct = 0
    n_samples = len(sequences)
    stats = {'nan_count': 0, 'inf_count': 0}

    anchor_0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    anchor_1 = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device)

    for i in range(n_samples):
        model.reset()
        seq = sequences[i].to(device)
        label = labels[i].item()

        for t in range(min(T_use, seq.shape[0])):
            x_t = seq[t:t+1]
            x_pulse = float32_to_pulse(x_t, device=device)
            out_pulse = model(x_pulse)

        out_float = pulse_to_float32(out_pulse).squeeze(0)

        if torch.isnan(out_float).any():
            stats['nan_count'] += 1
            loss_i = 1e12
        elif torch.isinf(out_float).any():
            stats['inf_count'] += 1
            loss_i = 1e12
        else:
            target = anchor_0 if label == 0 else anchor_1
            loss_i = torch.mean((out_float - target) ** 2).item()

        total_loss += min(loss_i, 1e12)

        if not (torch.isnan(out_float).any() or torch.isinf(out_float).any()):
            pred = 0 if out_float[0] > 0 else 1
            if pred == label:
                correct += 1

    avg_loss = total_loss / n_samples
    acc = correct / n_samples
    fr = 0.5
    return avg_loss, acc, fr, stats


# ============================================================
# 各向异性 SPSA
# ============================================================
def spsa_step_anisotropic(model, sequences, labels, device,
                          c_W, c_beta, c_vth, a_W, a_beta, a_vth, T_use,
                          w1_shape, w2_shape, n_beta, n_vth,
                          momentum_W=None, momentum_beta=None, momentum_vth=None, mu=0.9):
    w1_float, w2_float = _get_weight_floats(model)
    all_beta, all_vth = _get_lif_params(model)

    w_flat = torch.cat([w1_float.flatten(), w2_float.flatten()])
    w_dim = w_flat.shape[0]

    delta_W = _bernoulli_delta(w_dim, device)
    delta_beta = _bernoulli_delta(n_beta, device)
    delta_vth = _bernoulli_delta(n_vth, device)

    perturbation_plus = torch.cat([c_W * delta_W, c_beta * delta_beta, c_vth * delta_vth])
    perturbation_minus = -perturbation_plus
    orig_flat = torch.cat([w_flat, all_beta, all_vth])

    # 正向扰动
    perturbed_plus = orig_flat + perturbation_plus
    w1_p = perturbed_plus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_p = perturbed_plus[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_p = perturbed_plus[w_dim:w_dim+n_beta]
    vth_p = perturbed_plus[w_dim+n_beta:]
    _set_weight_floats(model, w1_p, w2_p)
    _set_lif_params(model, beta_p, vth_p)
    clamp_physics(model)
    loss_plus, _, _, _ = evaluate_model(model, sequences, labels, device, T_use)

    # 负向扰动
    perturbed_minus = orig_flat + perturbation_minus
    w1_m = perturbed_minus[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_m = perturbed_minus[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
    beta_m = perturbed_minus[w_dim:w_dim+n_beta]
    vth_m = perturbed_minus[w_dim+n_beta:]
    _set_weight_floats(model, w1_m, w2_m)
    _set_lif_params(model, beta_m, vth_m)
    clamp_physics(model)
    loss_minus, _, _, _ = evaluate_model(model, sequences, labels, device, T_use)

    diff = loss_plus - loss_minus

    if abs(diff) < 1e-15:
        w1_orig = orig_flat[:w1_shape[0]*w1_shape[1]].view(w1_shape)
        w2_orig = orig_flat[w1_shape[0]*w1_shape[1]:w_dim].view(w2_shape)
        beta_orig = orig_flat[w_dim:w_dim+n_beta]
        vth_orig = orig_flat[w_dim+n_beta:]
        _set_weight_floats(model, w1_orig, w2_orig)
        _set_lif_params(model, beta_orig, vth_orig)
        loss, acc, fr, stats = evaluate_model(model, sequences, labels, device, T_use)
        return loss, acc, diff, fr, momentum_W, momentum_beta, momentum_vth, stats

    g_W = (diff / (2 * c_W)) * (1.0 / delta_W)
    g_beta = (diff / (2 * c_beta)) * (1.0 / delta_beta)
    g_vth = (diff / (2 * c_vth)) * (1.0 / delta_vth)

    g_W = g_W.clamp(-10.0, 10.0)
    g_beta = g_beta.clamp(-10.0, 10.0)
    g_vth = g_vth.clamp(-10.0, 10.0)

    if momentum_W is None:
        momentum_W = g_W.clone()
    else:
        momentum_W = mu * momentum_W + (1 - mu) * g_W

    if momentum_beta is None:
        momentum_beta = g_beta.clone()
    else:
        momentum_beta = mu * momentum_beta + (1 - mu) * g_beta

    if momentum_vth is None:
        momentum_vth = g_vth.clone()
    else:
        momentum_vth = mu * momentum_vth + (1 - mu) * g_vth

    new_W = w_flat - a_W * momentum_W
    new_beta = all_beta - a_beta * momentum_beta
    new_vth = all_vth - a_vth * momentum_vth

    w1_new = new_W[:w1_shape[0]*w1_shape[1]].view(w1_shape)
    w2_new = new_W[w1_shape[0]*w1_shape[1]:].view(w2_shape)
    _set_weight_floats(model, w1_new, w2_new)
    _set_lif_params(model, new_beta, new_vth)
    clamp_physics(model)

    loss, acc, fr, stats = evaluate_model(model, sequences, labels, device, T_use)
    return loss, acc, diff, fr, momentum_W, momentum_beta, momentum_vth, stats


# ============================================================
# 初始化 (与 Group A 相同的初始状态)
# ============================================================
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


def create_model(device):
    torch.manual_seed(42)
    template = SimpleLIFNode(beta=0.01)
    model = SimpleSpikeMLP(4, 8, 4, neuron_template=template).to(device)

    w1 = orthogonal_init_with_spectral_radius((8, 4), target_rho=1.5, device=device)
    w2 = orthogonal_init_with_spectral_radius((4, 8), target_rho=1.5, device=device)
    model.set_weights(w1, w2)

    rng = np.random.RandomState(42)
    for name, module in model.named_modules():
        if isinstance(module, SimpleLIFNode):
            noise = torch.tensor(rng.normal(0, 0.01, module._beta.shape),
                                 dtype=torch.float32, device=device)
            module._beta.data.add_(noise)
            module._beta.data.clamp_(0.01, 0.999)

    return model


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*70, flush=True)
    print("实验 13 Group B: 各向异性 SPSA (Anisotropic)", flush=True)
    print("="*70, flush=True)

    sequences, labels = generate_temporal_sequences(n_samples=8, seq_len=T_USE, n_features=4)
    sequences = sequences.to(DEVICE)
    labels = labels.to(DEVICE)
    print(f"[data] {len(sequences)} sequences, T_use={T_USE}", flush=True)

    w1_shape = (8, 4)
    w2_shape = (4, 8)

    model = create_model(DEVICE)
    all_beta, all_vth = _get_lif_params(model)
    n_beta = len(all_beta)
    n_vth = len(all_vth)
    print(f"[params] W: {w1_shape[0]*w1_shape[1] + w2_shape[0]*w2_shape[1]}, β: {n_beta}, V_th: {n_vth}", flush=True)

    print("\n" + "="*60, flush=True)
    print("[Group B] 各向异性 SPSA (Anisotropic) - 多尺度并发演化", flush=True)
    print(f"  c_W={ANISO_c_W}, c_β={ANISO_c_beta}, c_Vth={ANISO_c_vth}", flush=True)
    print(f"  a_W={ANISO_a_W}, a_β={ANISO_a_beta}, a_Vth={ANISO_a_vth}", flush=True)
    print("="*60, flush=True)

    # 初始评估
    beta0, vth0, gamma0, beta_std0, vth_std0 = extract_physics(model)
    rho0 = extract_spectral_radius(model)
    w_mean0, w_std0, w_norm0 = extract_weight_stats(model)
    loss0, acc0, fr0, stats0 = evaluate_model(model, sequences, labels, DEVICE, T_USE)
    params_full_0 = extract_physics_full(model)

    print(f"[B] Init: Loss={loss0:.4f}, Acc={acc0:.2f}, "
          f"W={w_mean0:.4f}±{w_std0:.4f}(‖{w_norm0:.2f}‖), "
          f"β={beta0:.4f}±{beta_std0:.4f}, V_th={vth0:.4f}±{vth_std0:.4f}, ρ={rho0:.4f}", flush=True)

    trajectory = {
        'epoch': [0], 'beta': [beta0], 'v_th': [vth0],
        'beta_std': [beta_std0], 'vth_std': [vth_std0],
        'w_mean': [w_mean0], 'w_std': [w_std0], 'w_norm': [w_norm0],
        'loss': [loss0], 'acc': [acc0], 'spectral_radius': [rho0],
        'diff': [0.0], 'firing_rate': [fr0],
        'params_full': [params_full_0],
    }

    momentum_W, momentum_beta, momentum_vth = None, None, None
    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        c_W_k = ANISO_c_W / (epoch + 1) ** gamma_spsa
        c_beta_k = ANISO_c_beta / (epoch + 1) ** gamma_spsa
        c_vth_k = ANISO_c_vth / (epoch + 1) ** gamma_spsa

        a_W_k = ANISO_a_W / (epoch + 1) ** alpha
        a_beta_k = ANISO_a_beta / (epoch + 1) ** alpha
        a_vth_k = ANISO_a_vth / (epoch + 1) ** alpha

        loss, acc, diff, fr, momentum_W, momentum_beta, momentum_vth, stats = spsa_step_anisotropic(
            model, sequences, labels, DEVICE,
            c_W_k, c_beta_k, c_vth_k, a_W_k, a_beta_k, a_vth_k, T_USE,
            w1_shape, w2_shape, n_beta, n_vth,
            momentum_W, momentum_beta, momentum_vth
        )

        beta_now, vth_now, gamma_now, beta_std_now, vth_std_now = extract_physics(model)
        rho_now = extract_spectral_radius(model)
        w_mean_now, w_std_now, w_norm_now = extract_weight_stats(model)
        params_full_now = extract_physics_full(model)

        trajectory['epoch'].append(epoch)
        trajectory['beta'].append(beta_now)
        trajectory['v_th'].append(vth_now)
        trajectory['beta_std'].append(beta_std_now)
        trajectory['vth_std'].append(vth_std_now)
        trajectory['w_mean'].append(w_mean_now)
        trajectory['w_std'].append(w_std_now)
        trajectory['w_norm'].append(w_norm_now)
        trajectory['loss'].append(loss)
        trajectory['acc'].append(acc)
        trajectory['spectral_radius'].append(rho_now)
        trajectory['diff'].append(diff)
        trajectory['firing_rate'].append(fr)
        trajectory['params_full'].append(params_full_now)

        if epoch % 10 == 0 or epoch <= 5:
            elapsed = time.time() - t_start
            print(f"[B] Ep {epoch:3d}: Loss={loss:.4f}, Acc={acc:.2f}, "
                  f"W={w_mean_now:.4f}±{w_std_now:.4f}(‖{w_norm_now:.2f}‖), "
                  f"β={beta_now:.4f}±{beta_std_now:.4f}, V_th={vth_now:.4f}±{vth_std_now:.4f}, "
                  f"ρ={rho_now:.4f}, fr={fr:.3f}, diff={diff:.2e} [{elapsed:.0f}s]", flush=True)

    # 保存结果
    results = {
        'config': {
            'N_EPOCHS': N_EPOCHS,
            'T_USE': T_USE,
            'anisotropic': {
                'c_W': ANISO_c_W, 'c_beta': ANISO_c_beta, 'c_vth': ANISO_c_vth,
                'a_W': ANISO_a_W, 'a_beta': ANISO_a_beta, 'a_vth': ANISO_a_vth,
            },
        },
        'Group_B_Anisotropic': trajectory,
    }

    json_path = '/Users/tangzhengzheng/Desktop/NEXUS/experiments/exp13_group_b_data.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] JSON saved to {json_path}", flush=True)

    # 总结
    print("\n" + "="*70, flush=True)
    print("Group B 总结", flush=True)
    print("="*70, flush=True)
    print(f"Initial β: {trajectory['beta'][0]:.6f} → Final β: {trajectory['beta'][-1]:.6f} (Δ={trajectory['beta'][-1]-trajectory['beta'][0]:.6f})")
    print(f"Initial V_th: {trajectory['v_th'][0]:.4f} → Final V_th: {trajectory['v_th'][-1]:.4f} (Δ={trajectory['v_th'][-1]-trajectory['v_th'][0]:.4f})")
    print(f"Initial Loss: {trajectory['loss'][0]:.4f} → Final Loss: {trajectory['loss'][-1]:.4f}")
    print("\n[完成]", flush=True)


if __name__ == "__main__":
    main()
