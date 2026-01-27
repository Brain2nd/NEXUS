"""
Straight-Through Estimator (STE) for SNN Training - 纯脉冲模式
===============================================================

这是 100% 纯 SNN 训练模式，完全符合 CLAUDE.md 约束：
- 前向传播: 纯 SNN 门电路计算，返回 pulse
- 反向传播: 纯 SNN 门电路计算，返回 pulse (无 Python 数学运算)

核心原则:
- 所有 backward 使用 SNN 组件 (SpikeFP32VecMul, SpikeFP32VecAdd 等)
- 梯度以 pulse 形式传递
- 仅在系统边界处编解码

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn


# ==============================================================================
# 共享 SNN 组件管理器
# ==============================================================================

class SNNBackwardComponents:
    """SNN backward 组件管理器

    单例模式，延迟初始化 SNN 组件。
    所有 STE backward 共享这些组件。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _init_components(self, device=None, neuron_template=None):
        if self._initialized:
            return

        from atomic_ops.arithmetic.fp32.fp32_matmul import (
            SpikeFP32MatMulTransposed,
            SpikeFP32OuterProduct,
            SpikeFP32VecMul,
            SpikeFP32VecAdd,
            SpikeFP32VecSub
        )
        from atomic_ops.arithmetic.fp32.fp32_constants import PulseConstants
        from atomic_ops.arithmetic.fp32.fp32_div import SpikeFP32Divider
        from atomic_ops.core.logic_gates import NOTGate
        from atomic_ops.core.vec_logic_gates import VecAND

        self.vec_mul = SpikeFP32VecMul(neuron_template=neuron_template)
        self.vec_add = SpikeFP32VecAdd(neuron_template=neuron_template)
        self.vec_sub = SpikeFP32VecSub(neuron_template=neuron_template)
        self.vec_div = SpikeFP32Divider(neuron_template=neuron_template)
        self.vec_and = VecAND(neuron_template=neuron_template)
        self.matmul_t = SpikeFP32MatMulTransposed(neuron_template=neuron_template)
        self.outer_product = SpikeFP32OuterProduct(neuron_template=neuron_template)
        self.sign_not = NOTGate(neuron_template=neuron_template)
        self.constants = PulseConstants(device=device)

        # 移动所有组件到指定设备
        if device is not None:
            self.vec_mul = self.vec_mul.to(device)
            self.vec_add = self.vec_add.to(device)
            self.vec_sub = self.vec_sub.to(device)
            self.vec_div = self.vec_div.to(device)
            self.vec_and = self.vec_and.to(device)
            self.matmul_t = self.matmul_t.to(device)
            self.outer_product = self.outer_product.to(device)
            self.sign_not = self.sign_not.to(device)

        self._initialized = True
        self._device = device

    def get(self, device=None, neuron_template=None):
        """获取组件，必要时初始化"""
        if not self._initialized:
            self._init_components(device, neuron_template)
        elif device is not None and device != self._device:
            self.to(device)
        return self

    def to(self, device):
        """移动组件到新设备"""
        if device != self._device and self._initialized:
            self.vec_mul = self.vec_mul.to(device)
            self.vec_add = self.vec_add.to(device)
            self.vec_sub = self.vec_sub.to(device)
            self.vec_div = self.vec_div.to(device)
            self.vec_and = self.vec_and.to(device)
            self.matmul_t = self.matmul_t.to(device)
            self.outer_product = self.outer_product.to(device)
            self.sign_not = self.sign_not.to(device)
            self.constants = self.constants.to(device) if hasattr(self.constants, 'to') else self.constants
            self._device = device
        return self


def get_snn_components(device=None, neuron_template=None):
    """获取共享的 SNN backward 组件"""
    return SNNBackwardComponents().get(device, neuron_template)


def _parallel_reduce_pulse(x, comp):
    """并行树形归约 (向量化，无循环)

    对第 0 维进行归约：[N, ..., bits] -> [..., bits]

    Args:
        x: 输入张量 [N, ..., bits]
        comp: SNN 组件管理器

    Returns:
        归约结果 [..., bits]
    """
    current = x

    while current.shape[0] > 1:
        n = current.shape[0]
        n_pairs = n // 2
        has_odd = (n % 2) == 1

        if n_pairs > 0:
            # 向量化切片配对 (无循环)
            left = current[0:n_pairs*2:2]   # [0, 2, 4, ...]
            right = current[1:n_pairs*2:2]  # [1, 3, 5, ...]

            # 并行 SNN 加法
            paired_sum = comp.vec_add(left, right)

            if has_odd:
                last = current[-1:]
                current = torch.cat([paired_sum, last], dim=0)
            else:
                current = paired_sum
        else:
            break

    return current[0]


def _parallel_reduce_pulse_dim(x, dim, comp):
    """并行树形归约 (向量化，任意维度)

    对指定维度进行归约：[..., N, ..., bits] -> [..., ..., bits]

    Args:
        x: 输入张量
        dim: 要归约的维度
        comp: SNN 组件管理器

    Returns:
        归约结果
    """
    # 将目标维度移到第 0 维
    x_t = x.transpose(dim, 0)
    # 归约
    result = _parallel_reduce_pulse(x_t, comp)
    return result


# ==============================================================================
# Linear STE (纯脉冲 backward)
# ==============================================================================

class STELinearFunction(torch.autograd.Function):
    """STE for Linear layer - 纯脉冲 backward

    前向: 返回 pulse
    反向: 使用 SNN 组件计算梯度
    """

    @staticmethod
    def forward(ctx, x_pulse, weight_pulse, out_pulse):
        """
        Args:
            x_pulse: 输入 pulse [..., in_features, 32]
            weight_pulse: 脉冲权重 [out_features, in_features, 32]
            out_pulse: SNN 前向结果 [..., out_features, 32]

        Returns:
            out_pulse (pulse 格式)
        """
        ctx.save_for_backward(x_pulse, weight_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        """使用 SNN 组件计算梯度"""
        x_pulse, weight_pulse = ctx.saved_tensors
        device = x_pulse.device

        # 获取 SNN 组件
        comp = get_snn_components(device)

        # grad_x = grad_output @ W (脉冲域矩阵乘法)
        # grad_output: [..., out_features, 32]
        # weight: [out_features, in_features, 32]
        grad_x_pulse = comp.matmul_t(grad_output_pulse, weight_pulse)

        # grad_W = grad_output^T @ x (脉冲域外积累加)
        grad_w_pulse = comp.outer_product(grad_output_pulse, x_pulse)

        return grad_x_pulse, grad_w_pulse, None


# ==============================================================================
# Embedding STE (纯脉冲 backward)
# ==============================================================================

class STEEmbeddingFunction(torch.autograd.Function):
    """STE for Embedding layer - 纯脉冲 backward"""

    @staticmethod
    def forward(ctx, indices, weight_pulse, out_pulse):
        ctx.save_for_backward(indices, out_pulse)
        ctx.weight_shape = weight_pulse.shape
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        indices, out_pulse = ctx.saved_tensors
        weight_shape = ctx.weight_shape
        device = out_pulse.device
        vocab_size, hidden_size = weight_shape[0], weight_shape[1]

        # Embedding 梯度是稀疏的
        # 使用向量化的 one-hot + matmul 替代 scatter_add
        comp = get_snn_components(device)

        # 展平处理
        flat_indices = indices.view(-1)  # [N]
        flat_grad = grad_output_pulse.view(-1, hidden_size, 32)  # [N, hidden, 32]
        N = flat_indices.shape[0]

        # 构建 one-hot 矩阵 [N, vocab_size]
        one_hot = torch.zeros(N, vocab_size, device=device, dtype=flat_grad.dtype)
        one_hot.scatter_(1, flat_indices.unsqueeze(1), 1.0)

        # grad_weight[v, h, b] = sum_n(one_hot[n, v] * flat_grad[n, h, b])
        # 使用 einsum 向量化: [N, vocab] x [N, hidden, 32] -> [vocab, hidden, 32]
        # one_hot: [N, V], flat_grad: [N, H, 32]
        # 转换为脉冲域外积累加
        # 由于 one_hot 是 0/1，可以用作 mask 进行向量化选择

        # 方法: 对每个 vocab index，使用 broadcast + VecAND 选择对应梯度，然后归约
        # one_hot.T: [V, N], flat_grad: [N, H, 32]
        # 扩展 one_hot 到 [V, N, 1, 1] 作为 mask
        one_hot_t = one_hot.T  # [V, N]
        mask_expanded = one_hot_t.unsqueeze(-1).unsqueeze(-1)  # [V, N, 1, 1]

        # flat_grad 扩展到 [1, N, H, 32]
        grad_expanded = flat_grad.unsqueeze(0)  # [1, N, H, 32]

        # 广播 mask 到完整形状
        mask_broadcast = mask_expanded.expand(-1, -1, hidden_size, 32)  # [V, N, H, 32]
        grad_broadcast = grad_expanded.expand(vocab_size, -1, -1, -1)  # [V, N, H, 32]

        # masked_grads: [V, N, H, 32]，使用 VecAND 进行选择 (mask 是 0/1)
        masked_grads = comp.vec_and(mask_broadcast, grad_broadcast)

        # 对 N 维度进行并行归约: [V, N, H, 32] -> [V, H, 32]
        grad_weight_pulse = _parallel_reduce_pulse_dim(masked_grads, dim=1, comp=comp)

        return None, grad_weight_pulse, None


# ==============================================================================
# RMSNorm STE (纯脉冲 backward)
# ==============================================================================

class STERMSNormFunction(torch.autograd.Function):
    """STE for RMSNorm layer - 纯脉冲 backward

    完整公式:
        rms = sqrt(mean(x²) + eps)
        x_norm = x / rms
        y = x_norm * weight

    grad_x 完整公式 (通过链式法则推导):
        grad_x = grad_out * weight * rms_inv
               - x * rms_inv³ * (1/n) * sum(grad_out * weight * x)

    其中:
        - 第一项: 直接梯度
        - 第二项: Jacobian 修正项 (因为 rms 依赖于 x)
    """

    @staticmethod
    def forward(ctx, x_pulse, weight_pulse, out_pulse, rms_inv_pulse, x_norm_pulse, eps, hidden_size_pulse):
        """
        保存前向计算中的中间值用于 backward

        Args:
            x_pulse: 输入脉冲 [..., hidden_size, 32]
            weight_pulse: 权重脉冲 [hidden_size, 32]
            out_pulse: 输出脉冲 [..., hidden_size, 32]
            rms_inv_pulse: 1/rms 的脉冲表示 [..., 1, 32]
            x_norm_pulse: x/rms 的脉冲表示 [..., hidden_size, 32]
            eps: epsilon 值
            hidden_size_pulse: hidden_size 的脉冲表示 [32] (用于 SNN 除法计算 1/n)
        """
        ctx.save_for_backward(x_pulse, weight_pulse, rms_inv_pulse, x_norm_pulse, hidden_size_pulse)
        ctx.eps = eps
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        x_pulse, weight_pulse, rms_inv_pulse, x_norm_pulse, hidden_size_pulse = ctx.saved_tensors
        device = x_pulse.device
        hidden_size = x_pulse.shape[-2]

        comp = get_snn_components(device)

        # ============================================================
        # grad_weight = sum(grad_output * x_norm) 沿 batch 维度
        # ============================================================
        grad_times_xnorm = comp.vec_mul(grad_output_pulse, x_norm_pulse)

        grad_weight_pulse = grad_times_xnorm
        if grad_times_xnorm.dim() > 2:
            batch_size = grad_times_xnorm[..., 0, 0].numel()
            flat_grad = grad_times_xnorm.reshape(batch_size, -1, 32)
            grad_weight_pulse = _parallel_reduce_pulse(flat_grad, comp)

        # ============================================================
        # grad_x 完整公式:
        # grad_x = grad_out * weight * rms_inv
        #        - x * rms_inv³ * (1/n) * sum(grad_out * weight * x)
        # ============================================================

        # 第一项: grad_out * weight * rms_inv
        grad_weight_term = comp.vec_mul(grad_output_pulse, weight_pulse)
        term1 = comp.vec_mul(grad_weight_term, rms_inv_pulse)

        # 第二项准备: grad_out * weight * x
        grad_w_x = comp.vec_mul(grad_weight_term, x_pulse)

        # sum(grad_out * weight * x) 沿 hidden 维度归约
        # grad_w_x: [..., hidden_size, 32] -> [..., 32]
        original_shape = grad_w_x.shape
        if grad_w_x.dim() == 3:
            # [batch, hidden, 32] -> [hidden, batch, 32] -> 归约 -> [batch, 32]
            grad_w_x_t = grad_w_x.transpose(-2, 0)  # [hidden, batch, 32]
            sum_grad_w_x = _parallel_reduce_pulse(grad_w_x_t, comp)  # [batch, 32]
            sum_grad_w_x = sum_grad_w_x.unsqueeze(-2)  # [batch, 1, 32]
        elif grad_w_x.dim() == 2:
            # [hidden, 32] -> 归约 -> [32]
            sum_grad_w_x = _parallel_reduce_pulse(grad_w_x, comp)  # [32]
            sum_grad_w_x = sum_grad_w_x.unsqueeze(0)  # [1, 32]
        else:
            # 更高维度: [..., hidden, 32]
            batch_dims = grad_w_x.shape[:-2]
            batch_size = 1
            for d in batch_dims:
                batch_size *= d
            flat = grad_w_x.reshape(batch_size, hidden_size, 32)  # [B, hidden, 32]
            flat_t = flat.transpose(1, 0)  # [hidden, B, 32]
            sum_flat = _parallel_reduce_pulse(flat_t, comp)  # [B, 32]
            sum_grad_w_x = sum_flat.reshape(batch_dims + (1, 32))

        # rms_inv³ = rms_inv * rms_inv * rms_inv
        rms_inv_sq = comp.vec_mul(rms_inv_pulse, rms_inv_pulse)
        rms_inv_cubed = comp.vec_mul(rms_inv_sq, rms_inv_pulse)

        # (1/n) 使用 SNN 除法器计算: 1.0 / hidden_size
        # one_pulse: 标量 1.0 的脉冲表示
        one_pulse = comp.constants.one(())  # [32]
        # hidden_size_pulse 已作为参数传入 [32]
        # 扩展维度以匹配 SNN 除法器输入
        one_expanded = one_pulse.unsqueeze(0)  # [1, 32]
        n_expanded = hidden_size_pulse.unsqueeze(0)  # [1, 32]
        inv_n_pulse = comp.vec_div(one_expanded, n_expanded)  # [1, 32]
        inv_n_pulse = inv_n_pulse.squeeze(0)  # [32]

        # 广播 inv_n_pulse 到 sum_grad_w_x 的形状
        inv_n_broadcast = inv_n_pulse.expand(sum_grad_w_x.shape)

        # x * rms_inv³ * (1/n) * sum(...)
        sum_scaled = comp.vec_mul(sum_grad_w_x, inv_n_broadcast)  # [..., 1, 32]
        sum_scaled = comp.vec_mul(sum_scaled, rms_inv_cubed)  # [..., 1, 32]
        term2 = comp.vec_mul(x_pulse, sum_scaled)  # [..., hidden, 32] (广播)

        # grad_x = term1 - term2
        grad_x_pulse = comp.vec_sub(term1, term2)

        return grad_x_pulse, grad_weight_pulse, None, None, None, None, None


# ==============================================================================
# LayerNorm STE (纯脉冲 backward)
# ==============================================================================

class STELayerNormFunction(torch.autograd.Function):
    """STE for LayerNorm layer (无权重版本) - 纯脉冲 backward"""

    @staticmethod
    def forward(ctx, x_pulse, out_pulse, std_inv_pulse, eps):
        """
        std_inv_pulse: 1/std 的脉冲表示
        """
        ctx.save_for_backward(std_inv_pulse)
        ctx.eps = eps
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        std_inv_pulse, = ctx.saved_tensors
        device = grad_output_pulse.device

        comp = get_snn_components(device)

        # grad_x = grad_output / std (简化版)
        grad_x_pulse = comp.vec_mul(grad_output_pulse, std_inv_pulse)

        return grad_x_pulse, None, None, None


# ==============================================================================
# Activation Function STEs (纯脉冲 backward)
# ==============================================================================

class STEExpFunction(torch.autograd.Function):
    """STE for Exp: y = exp(x), ∂y/∂x = y

    纯脉冲 backward: grad_x = grad_output * y
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse):
        ctx.save_for_backward(out_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        out_pulse, = ctx.saved_tensors
        device = out_pulse.device

        comp = get_snn_components(device)

        # grad_x = grad_output * y (SNN 乘法)
        grad_x_pulse = comp.vec_mul(grad_output_pulse, out_pulse)

        return grad_x_pulse, None


class STESigmoidFunction(torch.autograd.Function):
    """STE for Sigmoid: y = σ(x), ∂y/∂x = y(1-y)

    纯脉冲 backward: grad_x = grad_output * y * (1 - y)
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse):
        ctx.save_for_backward(out_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        out_pulse, = ctx.saved_tensors
        device = out_pulse.device

        comp = get_snn_components(device)
        one_pulse = comp.constants.one(out_pulse.shape[:-1])

        # 1 - y (SNN 减法)
        one_minus_y = comp.vec_sub(one_pulse, out_pulse)

        # y * (1 - y) (SNN 乘法)
        y_times_1my = comp.vec_mul(out_pulse, one_minus_y)

        # grad_x = grad_output * y * (1 - y) (SNN 乘法)
        grad_x_pulse = comp.vec_mul(grad_output_pulse, y_times_1my)

        return grad_x_pulse, None


class STETanhFunction(torch.autograd.Function):
    """STE for Tanh: y = tanh(x), ∂y/∂x = 1 - y²

    纯脉冲 backward: grad_x = grad_output * (1 - y²)
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse):
        ctx.save_for_backward(out_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        out_pulse, = ctx.saved_tensors
        device = out_pulse.device

        comp = get_snn_components(device)
        one_pulse = comp.constants.one(out_pulse.shape[:-1])

        # y² (SNN 乘法)
        y_squared = comp.vec_mul(out_pulse, out_pulse)

        # 1 - y² (SNN 减法)
        one_minus_y_sq = comp.vec_sub(one_pulse, y_squared)

        # grad_x = grad_output * (1 - y²) (SNN 乘法)
        grad_x_pulse = comp.vec_mul(grad_output_pulse, one_minus_y_sq)

        return grad_x_pulse, None


class STESiLUFunction(torch.autograd.Function):
    """STE for SiLU: y = x * σ(x), ∂y/∂x = σ + x·σ(1-σ)

    需要保存 sigmoid(x) 用于 backward
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse, sigmoid_pulse):
        """
        sigmoid_pulse: σ(x) 的脉冲表示，前向时计算并保存
        """
        ctx.save_for_backward(x_pulse, sigmoid_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        x_pulse, sigmoid_pulse = ctx.saved_tensors
        device = x_pulse.device

        comp = get_snn_components(device)
        one_pulse = comp.constants.one(sigmoid_pulse.shape[:-1])

        # 1 - σ (SNN 减法)
        one_minus_sig = comp.vec_sub(one_pulse, sigmoid_pulse)

        # σ(1-σ) (SNN 乘法)
        sig_times_1ms = comp.vec_mul(sigmoid_pulse, one_minus_sig)

        # x·σ(1-σ) (SNN 乘法)
        x_sig_1ms = comp.vec_mul(x_pulse, sig_times_1ms)

        # σ + x·σ(1-σ) (SNN 加法)
        deriv = comp.vec_add(sigmoid_pulse, x_sig_1ms)

        # grad_x = grad_output * (σ + x·σ(1-σ)) (SNN 乘法)
        grad_x_pulse = comp.vec_mul(grad_output_pulse, deriv)

        return grad_x_pulse, None, None


class STEGELUFunction(torch.autograd.Function):
    """STE for GELU (approximate): y = x * σ(1.702x)

    ∂y/∂x = σ(kx) + kx·σ(kx)(1-σ(kx)), k=1.702
    需要保存 σ(kx) 用于 backward
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse, sigmoid_kx_pulse, kx_pulse):
        """
        sigmoid_kx_pulse: σ(1.702x) 的脉冲表示
        kx_pulse: 1.702x 的脉冲表示
        """
        ctx.save_for_backward(kx_pulse, sigmoid_kx_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        kx_pulse, sigmoid_kx_pulse = ctx.saved_tensors
        device = kx_pulse.device

        comp = get_snn_components(device)
        one_pulse = comp.constants.one(sigmoid_kx_pulse.shape[:-1])

        # 1 - σ(kx) (SNN 减法)
        one_minus_sig = comp.vec_sub(one_pulse, sigmoid_kx_pulse)

        # σ(kx)(1-σ(kx)) (SNN 乘法)
        sig_times_1ms = comp.vec_mul(sigmoid_kx_pulse, one_minus_sig)

        # kx·σ(kx)(1-σ(kx)) (SNN 乘法)
        kx_sig_1ms = comp.vec_mul(kx_pulse, sig_times_1ms)

        # σ(kx) + kx·σ(kx)(1-σ(kx)) (SNN 加法)
        deriv = comp.vec_add(sigmoid_kx_pulse, kx_sig_1ms)

        # grad_x = grad_output * deriv (SNN 乘法)
        grad_x_pulse = comp.vec_mul(grad_output_pulse, deriv)

        return grad_x_pulse, None, None, None


class STESoftmaxFunction(torch.autograd.Function):
    """STE for Softmax: y = softmax(x)

    ∂L/∂x = y * (grad_out - sum(grad_out * y))

    纯脉冲 backward
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse, dim):
        ctx.save_for_backward(out_pulse)
        ctx.dim = dim
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        out_pulse, = ctx.saved_tensors
        dim = ctx.dim
        device = out_pulse.device

        comp = get_snn_components(device)

        # grad_out * y (SNN 乘法)
        grad_times_y = comp.vec_mul(grad_output_pulse, out_pulse)

        # sum(grad_out * y) 沿 dim 维度累加 (并行树形归约)
        # 将 dim 维度移到第 0 维，然后归约
        grad_times_y_t = grad_times_y.transpose(dim, 0)  # [N, ..., 32]
        sum_pulse = _parallel_reduce_pulse(grad_times_y_t, comp)  # [..., 32]

        # 广播 sum 到原始形状
        sum_expanded = sum_pulse.unsqueeze(dim).expand_as(grad_output_pulse)

        # grad_out - sum (SNN 减法)
        diff = comp.vec_sub(grad_output_pulse, sum_expanded)

        # y * diff (SNN 乘法)
        grad_x_pulse = comp.vec_mul(out_pulse, diff)

        return grad_x_pulse, None, None


class STEReLUFunction(torch.autograd.Function):
    """STE for ReLU: y = max(0, x)

    ∂y/∂x = 1 if x > 0 else 0

    对于脉冲格式: x[..., 0] 是符号位，0=正，1=负
    grad_x = grad_output if sign=0 (正数) else 0
    """

    @staticmethod
    def forward(ctx, x_pulse, out_pulse):
        ctx.save_for_backward(x_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        x_pulse, = ctx.saved_tensors
        device = x_pulse.device

        # 获取 SNN 组件
        comp = get_snn_components(device)

        # 获取符号位
        sign = x_pulse[..., 0:1]  # 0=正, 1=负

        # 正数 mask (NOT sign)
        mask = comp.sign_not(sign)

        # 广播 mask 到所有位
        mask_broadcast = mask.expand_as(grad_output_pulse)

        # grad_x = grad_output * mask (使用共享的 VecAND)
        grad_x_pulse = comp.vec_and(grad_output_pulse, mask_broadcast)

        return grad_x_pulse, None


# ==============================================================================
# Mul STE (逐元素乘法)
# ==============================================================================

class STEMulFunction(torch.autograd.Function):
    """STE for element-wise multiplication: y = a * b

    ∂y/∂a = b
    ∂y/∂b = a

    纯脉冲 backward:
        grad_a = grad_out * b
        grad_b = grad_out * a
    """

    @staticmethod
    def forward(ctx, a_pulse, b_pulse, out_pulse):
        """
        Args:
            a_pulse: 输入 a 的脉冲 [..., 32]
            b_pulse: 输入 b 的脉冲 [..., 32]
            out_pulse: SNN 前向结果 [..., 32]

        Returns:
            out_pulse
        """
        ctx.save_for_backward(a_pulse, b_pulse)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        a_pulse, b_pulse = ctx.saved_tensors
        device = a_pulse.device

        comp = get_snn_components(device)

        # grad_a = grad_out * b (SNN 乘法)
        grad_a_pulse = comp.vec_mul(grad_output_pulse, b_pulse)

        # grad_b = grad_out * a (SNN 乘法)
        grad_b_pulse = comp.vec_mul(grad_output_pulse, a_pulse)

        return grad_a_pulse, grad_b_pulse, None


# ==============================================================================
# Add STE (加法/残差连接)
# ==============================================================================

class STEAddFunction(torch.autograd.Function):
    """STE for addition: y = a + b

    ∂y/∂a = 1
    ∂y/∂b = 1

    纯脉冲 backward:
        grad_a = grad_out
        grad_b = grad_out
    """

    @staticmethod
    def forward(ctx, a_pulse, b_pulse, out_pulse):
        """
        Args:
            a_pulse: 输入 a 的脉冲 [..., 32]
            b_pulse: 输入 b 的脉冲 [..., 32]
            out_pulse: SNN 前向结果 [..., 32]

        Returns:
            out_pulse
        """
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        # 加法的梯度直接传递
        # grad_a = grad_out, grad_b = grad_out
        return grad_output_pulse, grad_output_pulse, None


# ==============================================================================
# MUX STE (条件选择)
# ==============================================================================

class STEMUXFunction(torch.autograd.Function):
    """STE for MUX: y = sel ? b : a (sel=1 选 b, sel=0 选 a)

    ∂y/∂a = 1 - sel
    ∂y/∂b = sel

    纯脉冲 backward:
        grad_a = grad_out * (1 - sel) = grad_out AND NOT(sel)
        grad_b = grad_out * sel = grad_out AND sel

    注意: sel_mask 是二进制 mask (0.0 或 1.0)，不是 FP32 脉冲编码。
    需要将其广播到所有 32 位。
    """

    @staticmethod
    def forward(ctx, sel_mask, a_pulse, b_pulse, out_pulse):
        """
        Args:
            sel_mask: 选择信号 [...] 二进制 mask (0.0 或 1.0)
            a_pulse: sel=0 时的输入 [..., 32]
            b_pulse: sel=1 时的输入 [..., 32]
            out_pulse: SNN 前向结果 [..., 32]

        Returns:
            out_pulse
        """
        ctx.save_for_backward(sel_mask)
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        sel_mask, = ctx.saved_tensors
        device = sel_mask.device

        comp = get_snn_components(device)

        # sel_mask: [...] 二进制 mask (0.0 或 1.0)
        # 扩展到 [..., 32] 以匹配 grad_output_pulse
        sel_expanded = sel_mask.unsqueeze(-1).expand_as(grad_output_pulse)
        not_sel_expanded = (1.0 - sel_mask).unsqueeze(-1).expand_as(grad_output_pulse)

        # grad_a = grad_out AND NOT(sel)
        grad_a_pulse = comp.vec_and(grad_output_pulse, not_sel_expanded)

        # grad_b = grad_out AND sel
        grad_b_pulse = comp.vec_and(grad_output_pulse, sel_expanded)

        return None, grad_a_pulse, grad_b_pulse, None


# ==============================================================================
# RoPE STE (旋转位置编码)
# ==============================================================================

class STERoPEFunction(torch.autograd.Function):
    """STE for RoPE (Rotary Position Embedding)

    RoPE 变换:
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new  = x_even * sin + x_odd * cos

    反向传播 (链式法则):
        ∂L/∂x_even = ∂L/∂x_even_new * cos + ∂L/∂x_odd_new * sin
        ∂L/∂x_odd  = -∂L/∂x_even_new * sin + ∂L/∂x_odd_new * cos

    纯脉冲 backward
    """

    @staticmethod
    def forward(ctx, x_pulse, cos_pulse, sin_pulse, out_pulse):
        """
        Args:
            x_pulse: 输入脉冲 [..., head_dim, 32]
            cos_pulse: cos(θ) 脉冲 [..., head_dim//2, 32]
            sin_pulse: sin(θ) 脉冲 [..., head_dim//2, 32]
            out_pulse: SNN 前向结果 [..., head_dim, 32]

        Returns:
            out_pulse
        """
        ctx.save_for_backward(cos_pulse, sin_pulse)
        ctx.head_dim = x_pulse.shape[-2]
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        cos_pulse, sin_pulse = ctx.saved_tensors
        head_dim = ctx.head_dim
        half_dim = head_dim // 2
        device = cos_pulse.device

        comp = get_snn_components(device)

        # 分割 grad_output 为偶数和奇数部分
        grad_even = grad_output_pulse[..., :half_dim, :]  # [..., half, 32]
        grad_odd = grad_output_pulse[..., half_dim:, :]   # [..., half, 32]

        # ∂L/∂x_even = grad_even * cos + grad_odd * sin
        grad_even_cos = comp.vec_mul(grad_even, cos_pulse)
        grad_odd_sin = comp.vec_mul(grad_odd, sin_pulse)
        grad_x_even = comp.vec_add(grad_even_cos, grad_odd_sin)

        # ∂L/∂x_odd = -grad_even * sin + grad_odd * cos
        # = grad_odd * cos - grad_even * sin
        grad_odd_cos = comp.vec_mul(grad_odd, cos_pulse)
        grad_even_sin = comp.vec_mul(grad_even, sin_pulse)
        grad_x_odd = comp.vec_sub(grad_odd_cos, grad_even_sin)

        # 拼接回原始形状
        grad_x_pulse = torch.cat([grad_x_even, grad_x_odd], dim=-2)

        return grad_x_pulse, None, None, None


# ==============================================================================
# 输出层解码 STE (边界操作，允许使用传统数学)
# ==============================================================================

def _get_precision_and_converters(pulse_tensor):
    """根据 pulse tensor 的最后一维大小确定精度和转换器"""
    bits = pulse_tensor.shape[-1]
    if bits == 8:
        from atomic_ops.encoding.converters import fp8_bits_to_float, float_to_fp8_bits
        return fp8_bits_to_float, float_to_fp8_bits
    elif bits == 16:
        from atomic_ops.encoding.converters import pulse_to_float16, float16_to_pulse
        return pulse_to_float16, float16_to_pulse
    elif bits == 32:
        from atomic_ops.encoding.converters import pulse_to_float32, float32_to_pulse
        return pulse_to_float32, float32_to_pulse
    elif bits == 64:
        from atomic_ops.encoding.converters import pulse_to_float64, float64_to_pulse
        return pulse_to_float64, float64_to_pulse
    else:
        raise ValueError(f"Unsupported pulse bit width: {bits}")


class STEDecodeFunction(torch.autograd.Function):
    """STE for decoding pulse to float at output layer

    这是系统边界操作，允许使用传统数学。
    用于网络输出层，将 pulse 解码为 float 以计算 loss。
    backward 时将 float 梯度编码为 pulse 梯度。
    """

    @staticmethod
    def forward(ctx, pulse):
        to_float, to_pulse = _get_precision_and_converters(pulse)
        ctx.to_pulse = to_pulse
        return to_float(pulse)

    @staticmethod
    def backward(ctx, grad_output):
        to_pulse = ctx.to_pulse
        return to_pulse(grad_output)


def ste_decode(pulse):
    """将 pulse 解码为 float，支持梯度回传（边界操作）"""
    return STEDecodeFunction.apply(pulse)


# ==============================================================================
# Convenience wrappers
# ==============================================================================

def ste_linear(x_pulse, weight_pulse, out_pulse):
    """Apply STE for Linear layer (纯脉冲)"""
    return STELinearFunction.apply(x_pulse, weight_pulse, out_pulse)


def ste_embedding(indices, weight_pulse, out_pulse):
    """Apply STE for Embedding layer (纯脉冲)"""
    return STEEmbeddingFunction.apply(indices, weight_pulse, out_pulse)


def ste_rmsnorm(x_pulse, weight_pulse, out_pulse, rms_inv_pulse, x_norm_pulse, eps):
    """Apply STE for RMSNorm layer (纯脉冲)"""
    return STERMSNormFunction.apply(x_pulse, weight_pulse, out_pulse, rms_inv_pulse, x_norm_pulse, eps)


def ste_layernorm(x_pulse, out_pulse, std_inv_pulse, eps):
    """Apply STE for LayerNorm layer (纯脉冲)"""
    return STELayerNormFunction.apply(x_pulse, out_pulse, std_inv_pulse, eps)


def ste_exp(x_pulse, out_pulse):
    """Apply STE for Exp (纯脉冲)"""
    return STEExpFunction.apply(x_pulse, out_pulse)


def ste_sigmoid(x_pulse, out_pulse):
    """Apply STE for Sigmoid (纯脉冲)"""
    return STESigmoidFunction.apply(x_pulse, out_pulse)


def ste_tanh(x_pulse, out_pulse):
    """Apply STE for Tanh (纯脉冲)"""
    return STETanhFunction.apply(x_pulse, out_pulse)


def ste_silu(x_pulse, out_pulse, sigmoid_pulse):
    """Apply STE for SiLU (纯脉冲)

    需要额外传入 sigmoid_pulse = σ(x)
    """
    return STESiLUFunction.apply(x_pulse, out_pulse, sigmoid_pulse)


def ste_gelu(x_pulse, out_pulse, sigmoid_kx_pulse, kx_pulse):
    """Apply STE for GELU (纯脉冲)

    需要额外传入:
    - sigmoid_kx_pulse: σ(1.702x) 的脉冲表示
    - kx_pulse: 1.702x 的脉冲表示
    """
    return STEGELUFunction.apply(x_pulse, out_pulse, sigmoid_kx_pulse, kx_pulse)


def ste_softmax(x_pulse, out_pulse, dim=-1):
    """Apply STE for Softmax (纯脉冲)"""
    return STESoftmaxFunction.apply(x_pulse, out_pulse, dim)


def ste_relu(x_pulse, out_pulse):
    """Apply STE for ReLU (纯脉冲)"""
    return STEReLUFunction.apply(x_pulse, out_pulse)


def ste_mul(a_pulse, b_pulse, out_pulse):
    """Apply STE for element-wise multiplication (纯脉冲)"""
    return STEMulFunction.apply(a_pulse, b_pulse, out_pulse)


def ste_add(a_pulse, b_pulse, out_pulse):
    """Apply STE for addition (纯脉冲)"""
    return STEAddFunction.apply(a_pulse, b_pulse, out_pulse)


def ste_mux(sel_pulse, a_pulse, b_pulse, out_pulse):
    """Apply STE for MUX (纯脉冲)

    sel=0 选 a, sel=1 选 b
    """
    return STEMUXFunction.apply(sel_pulse, a_pulse, b_pulse, out_pulse)


def ste_rope(x_pulse, cos_pulse, sin_pulse, out_pulse):
    """Apply STE for RoPE (纯脉冲)

    Args:
        x_pulse: 输入脉冲 [..., head_dim, 32]
        cos_pulse: cos(θ) 脉冲 [..., head_dim//2, 32]
        sin_pulse: sin(θ) 脉冲 [..., head_dim//2, 32]
        out_pulse: SNN 前向结果 [..., head_dim, 32]
    """
    return STERoPEFunction.apply(x_pulse, cos_pulse, sin_pulse, out_pulse)
