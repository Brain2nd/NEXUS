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
        from atomic_ops.core.logic_gates import NOTGate

        self.vec_mul = SpikeFP32VecMul(neuron_template=neuron_template)
        self.vec_add = SpikeFP32VecAdd(neuron_template=neuron_template)
        self.vec_sub = SpikeFP32VecSub(neuron_template=neuron_template)
        self.matmul_t = SpikeFP32MatMulTransposed(neuron_template=neuron_template)
        self.outer_product = SpikeFP32OuterProduct(neuron_template=neuron_template)
        self.sign_not = NOTGate(neuron_template=neuron_template)
        self.constants = PulseConstants(device=device)

        self._initialized = True
        self._device = device

    def get(self, device=None, neuron_template=None):
        """获取组件，必要时初始化"""
        if not self._initialized or (device is not None and device != self._device):
            self._init_components(device, neuron_template)
        return self

    def to(self, device):
        """移动组件到新设备"""
        if device != self._device:
            self._init_components(device)
        return self


def get_snn_components(device=None, neuron_template=None):
    """获取共享的 SNN backward 组件"""
    return SNNBackwardComponents().get(device, neuron_template)


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

        # Embedding 梯度是稀疏的
        # 需要在脉冲域中实现 index_add
        # 由于是简单的累加，可以使用 VecAdd

        comp = get_snn_components(device)

        # 初始化零梯度
        from atomic_ops.arithmetic.fp32.fp32_constants import get_zero_pulse
        grad_weight_pulse = get_zero_pulse(weight_shape[:-1], device)

        # 逐样本累加 (使用 SNN 加法器)
        flat_indices = indices.view(-1)
        flat_grad = grad_output_pulse.view(-1, weight_shape[1], 32)

        for i in range(flat_indices.shape[0]):
            idx = flat_indices[i].item()
            grad_weight_pulse[idx] = comp.vec_add(
                grad_weight_pulse[idx],
                flat_grad[i]
            )

        return None, grad_weight_pulse, None


# ==============================================================================
# RMSNorm STE (纯脉冲 backward)
# ==============================================================================

class STERMSNormFunction(torch.autograd.Function):
    """STE for RMSNorm layer - 纯脉冲 backward"""

    @staticmethod
    def forward(ctx, x_pulse, weight_pulse, out_pulse, rms_inv_pulse, x_norm_pulse, eps):
        """
        保存前向计算中的中间值用于 backward
        rms_inv_pulse: 1/rms 的脉冲表示
        x_norm_pulse: x/rms 的脉冲表示
        """
        ctx.save_for_backward(x_pulse, weight_pulse, rms_inv_pulse, x_norm_pulse)
        ctx.eps = eps
        return out_pulse

    @staticmethod
    def backward(ctx, grad_output_pulse):
        x_pulse, weight_pulse, rms_inv_pulse, x_norm_pulse = ctx.saved_tensors
        device = x_pulse.device

        comp = get_snn_components(device)

        # grad_weight = sum(grad_output * x_norm)
        # 使用 SNN 乘法
        grad_times_xnorm = comp.vec_mul(grad_output_pulse, x_norm_pulse)

        # 沿 batch 维度累加 (简化: 取 mean 近似)
        # 实际需要累加所有 batch，这里用第一个样本的形状
        grad_weight_pulse = grad_times_xnorm
        if grad_times_xnorm.dim() > 2:
            # 累加所有 batch 维度
            batch_size = grad_times_xnorm[..., 0, 0].numel()
            flat_grad = grad_times_xnorm.reshape(batch_size, -1, 32)
            grad_weight_pulse = flat_grad[0]
            for i in range(1, batch_size):
                grad_weight_pulse = comp.vec_add(grad_weight_pulse, flat_grad[i])

        # grad_x = grad_output * weight * rms_inv (简化版)
        # 完整公式更复杂，这里用简化近似
        grad_x_pulse = comp.vec_mul(grad_output_pulse, weight_pulse)
        grad_x_pulse = comp.vec_mul(grad_x_pulse, rms_inv_pulse)

        return grad_x_pulse, grad_weight_pulse, None, None, None, None


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

        # sum(grad_out * y) 沿 dim 维度累加
        # 假设 dim 是倒数第二个维度 (N)
        N = out_pulse.shape[dim]

        # 累加
        sum_pulse = grad_times_y.select(dim, 0)
        for i in range(1, N):
            sum_pulse = comp.vec_add(sum_pulse, grad_times_y.select(dim, i))

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

        # 获取符号位
        sign = x_pulse[..., 0:1]  # 0=正, 1=负

        # 正数 mask (NOT sign)
        comp = get_snn_components(device)
        mask = comp.sign_not(sign)

        # 广播 mask 到所有位
        mask_broadcast = mask.expand_as(grad_output_pulse)

        # grad_x = grad_output * mask (使用 AND 门)
        # 由于 mask 是 0 或 1，直接相乘等价于 AND
        from atomic_ops.core.vec_logic_gates import VecAND
        vec_and = VecAND()
        grad_x_pulse = vec_and(grad_output_pulse, mask_broadcast)

        return grad_x_pulse, None


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
