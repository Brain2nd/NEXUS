"""
神经元模块 (Neuron Module)
=========================

提供 IF/LIF 神经元的纯 PyTorch 实现，支持向量化参数和预分配切片机制。

核心原理
--------
- IF 神经元: V += I, 无泄漏
- LIF 神经元: V = β×V + I, 有泄漏
- 软复位: V = V - V_th (保留残差)
- 硬复位: V = 0 (清零)

预分配切片机制
--------------
- 参数在 __init__ 时预分配最大尺寸 (默认 64 位，覆盖 FP64)
- forward 时根据实际输入位宽切片使用: param[..., :input_bits]
- reset 时保留参数，只重置膜电位
- 可训练参数：trainable_threshold/trainable_beta

作者: MofNeuroSim Project
"""

import torch
import torch.nn as nn
from typing import Union, Optional

# 全局默认预分配形状 (64位，覆盖 FP8/16/32/64)
DEFAULT_MAX_PARAM_SHAPE = (64,)


class SimpleIFNode(nn.Module):
    """简化的 IF 神经元 (无泄漏) - 支持向量化阈值

    动力学方程:
    ```
    V(t+1) = V(t) + I(t)           # 膜电位积累
    S(t) = H(V(t) - V_th)          # 脉冲发放
    V(t) = V(t) - S(t) × V_th      # 软复位 (默认)
    ```

    预分配切片机制:
    - __init__ 时预分配 max_param_shape 尺寸的参数
    - forward 时根据输入位宽切片: threshold[..., :input_bits]
    - reset 时保留参数，只重置膜电位

    Args:
        v_threshold: 发放阈值 (float 或 Tensor)
        v_reset: 复位电压，None 表示软复位，数值表示硬复位到该值
        trainable_threshold: 是否允许训练阈值
        max_param_shape: 预分配参数形状，默认 DEFAULT_MAX_PARAM_SHAPE (64,)
    """
    def __init__(self,
                 v_threshold: Union[float, torch.Tensor] = 1.0,
                 v_reset: Optional[float] = None,
                 trainable_threshold: bool = False,
                 max_param_shape: Optional[tuple] = None):
        super().__init__()

        self.v_reset = v_reset
        self.trainable_threshold = trainable_threshold
        # 默认使用全局预分配形状
        self.max_param_shape = max_param_shape if max_param_shape is not None else DEFAULT_MAX_PARAM_SHAPE

        # 存储默认阈值
        if isinstance(v_threshold, torch.Tensor):
            self._v_threshold_default = float(v_threshold.mean().item())
        else:
            self._v_threshold_default = float(v_threshold)

        # 预分配参数（包括膜电位 v）
        self._preallocate_params(self.max_param_shape)

    @property
    def v_threshold(self) -> Union[float, torch.Tensor]:
        """获取阈值，支持标量和张量"""
        if self._v_threshold is not None:
            return self._v_threshold
        return self._v_threshold_default

    @v_threshold.setter
    def v_threshold(self, value):
        """设置阈值 - 支持标量覆盖（用于 _create_neuron）"""
        if isinstance(value, (int, float)):
            self._v_threshold_default = float(value)
            self._threshold_initialized = False
            if self._v_threshold is not None and not isinstance(self._v_threshold, nn.Parameter):
                self._v_threshold = None
        elif isinstance(value, torch.Tensor):
            if self.trainable_threshold:
                self._v_threshold = nn.Parameter(value.clone())
            else:
                self._v_threshold = value.clone()
            self._threshold_initialized = True

    def _preallocate_params(self, shape: tuple):
        """在 __init__ 时一次性预分配最大尺寸参数

        Args:
            shape: 要预分配的参数形状 (例如 (32,) 表示32位)
        """
        threshold_tensor = torch.full(shape, self._v_threshold_default, dtype=torch.float32)
        if self.trainable_threshold:
            self._v_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer('_v_threshold', threshold_tensor)
        self._threshold_initialized = True

        # 膜电位 v 需要匹配动态 batch 维度，无法预分配固定形状
        # 但需要追踪设备，用 _v_threshold 的设备作为参考
        self.v = None

    def forward(self, x):
        # 预分配模式：参数已在 __init__ 中创建，根据输入位宽切片
        input_bits = x.shape[-1] if x.dim() > 0 else 1
        batch_shape = x.shape[:-1]
        device = x.device  # 使用输入的设备

        # 从预分配的 buffer 切片，确保在正确设备上
        threshold = self._v_threshold[..., :input_bits].to(device)

        # 膜电位：batch 维度动态，bits 维度预分配切片
        # 预加载切片机制：batch 变化时复制扩张，不丢失状态
        max_bits = self.max_param_shape[-1]
        if self.v is None:
            # 首次创建
            self.v = torch.zeros(*batch_shape, max_bits, device=device, dtype=x.dtype)
        elif self.v.device != device:
            # 设备不匹配：移动到正确设备
            self.v = self.v.to(device)
        if self.v.shape[:-1] != batch_shape:
            # batch 维度变化：复制扩张（保留已有状态）
            new_v = torch.zeros(*batch_shape, max_bits, device=device, dtype=x.dtype)
            # 计算可复制的最小 batch 范围
            old_shape = self.v.shape[:-1]
            min_dims = min(len(old_shape), len(batch_shape))
            slices = tuple(slice(0, min(old_shape[i], batch_shape[i])) for i in range(min_dims))
            new_v[slices] = self.v[slices]
            self.v = new_v

        # 切片到当前 bits
        v = self.v[..., :input_bits]

        # 膜电位积累 (就地操作避免内存分配)
        v.add_(x)

        # 发放判断
        spike = (v >= threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            v.sub_(spike * threshold)
        else:
            # 硬复位
            reset_val = torch.full_like(v, self.v_reset)
            v.copy_(torch.where(spike > 0, reset_val, v))

        # 确保输出在输入设备上（防止设备不一致）
        if spike.device != device:
            spike = spike.to(device)

        return spike

    def reset_state(self):
        """重置膜电位"""
        self.v = None

    def reset(self):
        """重置膜电位，保留预分配参数"""
        self.v = None

    def _reset(self):
        """内部reset方法 - 由父组件调用"""
        self.v = None

    # 兼容接口
    def neuronal_charge(self, x):
        """膜电位积累 (兼容接口)"""
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + x

    def neuronal_fire(self):
        """发放判断 (兼容接口)"""
        threshold = self.v_threshold
        if isinstance(threshold, (int, float)):
            return (self.v >= threshold).float()
        return (self.v >= threshold).float()

    def neuronal_reset(self, spike):
        """复位 (兼容接口)"""
        threshold = self.v_threshold
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(threshold, device=self.v.device, dtype=self.v.dtype)

        if self.v_reset is None:
            self.v = self.v - spike * threshold
        else:
            self.v = torch.where(spike > 0,
                                 torch.full_like(self.v, self.v_reset),
                                 self.v)


class SimpleLIFNode(nn.Module):
    """简化的 LIF 神经元 (有泄漏) - 支持向量化阈值和泄漏率

    动力学方程:
    ```
    V(t+1) = β × V(t) + I(t)       # 膜电位泄漏 + 积累
    S(t) = H(V(t) - V_th)          # 脉冲发放
    V(t) = V(t) - S(t) × V_th      # 软复位 (默认)
    ```

    预分配切片机制:
    - __init__ 时预分配 max_param_shape 尺寸的参数
    - forward 时根据输入位宽切片: param[..., :input_bits]
    - reset 时保留参数，只重置膜电位

    默认值 (位精确模式):
    - beta = 1.0 - 1e-7 (极小泄漏，保持位精确)

    Args:
        beta: 膜电位泄漏因子 (0 < beta ≤ 1)
        v_threshold: 发放阈值
        v_reset: 复位电压，None 表示软复位
        trainable_beta: 是否训练泄漏率
        trainable_threshold: 是否训练阈值
        max_param_shape: 预分配参数形状，默认 DEFAULT_MAX_PARAM_SHAPE (64,)
    """
    # 默认泄漏因子：极小泄漏，保持位精确但增加信息熵
    DEFAULT_BETA = 1.0 - 1e-7

    def __init__(self,
                 beta: Union[float, torch.Tensor] = None,
                 v_threshold: Union[float, torch.Tensor] = 1.0,
                 v_reset: Optional[float] = None,
                 trainable_beta: bool = False,
                 trainable_threshold: bool = False,
                 max_param_shape: Optional[tuple] = None):
        super().__init__()

        # 处理默认beta值
        if beta is None:
            beta = self.DEFAULT_BETA

        self.v_reset = v_reset
        self.trainable_beta = trainable_beta
        self.trainable_threshold = trainable_threshold
        # 默认使用全局预分配形状
        self.max_param_shape = max_param_shape if max_param_shape is not None else DEFAULT_MAX_PARAM_SHAPE

        # 存储默认值
        self._beta_default = beta if isinstance(beta, (int, float)) else float(beta.mean().item())
        self._threshold_default = v_threshold if isinstance(v_threshold, (int, float)) else float(v_threshold.mean().item())

        # 预分配参数（包括膜电位 v）
        self._preallocate_params(self.max_param_shape)

    @property
    def beta(self) -> torch.Tensor:
        """获取泄漏因子"""
        return self._beta

    @property
    def v_threshold(self) -> torch.Tensor:
        """获取阈值"""
        return self._v_threshold

    @v_threshold.setter
    def v_threshold(self, value):
        """设置阈值 - 用于 _create_neuron 覆盖模板阈值

        预分配切片机制：更新默认值并重新填充预分配张量
        """
        if isinstance(value, (int, float)):
            self._threshold_default = float(value)
        else:
            self._threshold_default = float(value.mean().item())
        # 重新填充预分配的参数
        self._v_threshold.data.fill_(self._threshold_default)

    def _preallocate_params(self, shape: tuple):
        """在 __init__ 时一次性预分配最大尺寸参数

        Args:
            shape: 要预分配的参数形状 (例如 (32,) 表示32位)
        """
        # 检测现有参数的设备（用于 deepcopy 后重新分配时保持设备一致）
        device = None
        if hasattr(self, '_beta') and self._beta is not None:
            device = self._beta.device
        elif hasattr(self, '_v_threshold') and self._v_threshold is not None:
            device = self._v_threshold.device

        # 预分配 beta
        beta_tensor = torch.full(shape, self._beta_default, dtype=torch.float32)
        if device is not None:
            beta_tensor = beta_tensor.to(device)
        if self.trainable_beta:
            self._beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer('_beta', beta_tensor)

        # 预分配 threshold
        threshold_tensor = torch.full(shape, self._threshold_default, dtype=torch.float32)
        if device is not None:
            threshold_tensor = threshold_tensor.to(device)
        if self.trainable_threshold:
            self._v_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer('_v_threshold', threshold_tensor)

        # 膜电位 v 需要匹配动态 batch 维度，无法预分配固定形状
        # 用 _v_threshold 的设备作为参考
        self.v = None

    def forward(self, x):
        # 预分配模式：参数已在 __init__ 中创建，根据输入位宽切片
        input_bits = x.shape[-1] if x.dim() > 0 else 1
        batch_shape = x.shape[:-1]
        device = x.device  # 使用输入的设备

        # 从预分配的 buffer 切片，确保在正确设备上
        # 如果 input_bits > max_param_shape，需要广播参数
        max_bits = self.max_param_shape[-1]
        if input_bits <= max_bits:
            beta = self._beta[..., :input_bits].to(device)
            threshold = self._v_threshold[..., :input_bits].to(device)
        else:
            # input_bits 超过预分配大小，需要广播扩展
            # 使用预分配参数的值进行广播
            beta = self._beta.to(device).expand(*([1] * (x.dim() - 1)), input_bits).contiguous()
            threshold = self._v_threshold.to(device).expand(*([1] * (x.dim() - 1)), input_bits).contiguous()

        # 膜电位：batch 维度动态，bits 维度根据 input_bits 确定
        # 使用 max(input_bits, max_bits) 确保足够容量
        v_bits = max(input_bits, max_bits)
        if self.v is None:
            # 首次创建
            self.v = torch.zeros(*batch_shape, v_bits, device=device, dtype=x.dtype)
        elif self.v.device != device:
            # 设备不匹配：移动到正确设备
            self.v = self.v.to(device)

        # 检查是否需要重新分配 (batch 维度变化或位宽不足)
        need_realloc = (self.v.shape[:-1] != batch_shape) or (self.v.shape[-1] < input_bits)
        if need_realloc:
            # batch 维度或位宽变化：复制扩张（保留已有状态）
            new_v = torch.zeros(*batch_shape, v_bits, device=device, dtype=x.dtype)
            # 计算可复制的最小范围
            old_shape = self.v.shape[:-1]
            old_bits = self.v.shape[-1]
            if len(old_shape) > 0 and len(batch_shape) > 0:
                min_dims = min(len(old_shape), len(batch_shape))
                slices = tuple(slice(0, min(old_shape[i], batch_shape[i])) for i in range(min_dims))
                copy_bits = min(old_bits, v_bits)
                new_v[slices + (slice(0, copy_bits),)] = self.v[slices + (slice(0, copy_bits),)]
            self.v = new_v

        # 切片到当前 bits
        v = self.v[..., :input_bits]

        # LIF 动力学: V = beta * V + I
        v.mul_(beta).add_(x)

        # 发放判断
        spike = (v >= threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            v.sub_(spike * threshold)
        else:
            # 硬复位
            reset_val = torch.full_like(v, self.v_reset)
            v.copy_(torch.where(spike > 0, reset_val, v))

        # 确保输出在输入设备上（防止设备不一致）
        if spike.device != device:
            spike = spike.to(device)

        return spike

    def reset_state(self):
        """重置膜电位"""
        self.v = None

    def reset(self):
        """重置膜电位，保留预分配参数"""
        self.v = None

    def _reset(self):
        """内部reset方法 - 由父组件调用"""
        self.v = None


class DynamicThresholdIFNode(nn.Module):
    """动态阈值 IF 神经元 (用于浮点编码)

    实现 SAR ADC 风格的二进制扫描:
    - 阈值从 2^(N-1) 递减到 2^(-NT)
    - 使用软复位保留残差
    - 阈值预计算为 tensor，提高效率

    Args:
        N: 整数部分位宽，最高位阈值 2^(N-1)
        NT: 小数部分位宽，最低位阈值 2^(-NT)

    Attributes:
        thresholds: 预计算的阈值 tensor，形状为 [N + NT]
                    thresholds[t] = 2^(N-1-t)，t = 0, 1, ..., N+NT-1
    """
    def __init__(self, N: int, NT: int = 0):
        super().__init__()
        self.N = N
        self.NT = NT
        self.total_steps = N + NT
        self.step_counter = 0
        self.register_buffer('v', None)

        # 预计算所有时间步的阈值: [2^(N-1), 2^(N-2), ..., 2^0, 2^-1, ..., 2^-NT]
        exponents = torch.arange(N - 1, -NT - 1, -1, dtype=torch.float32)
        self.register_buffer('thresholds', (2.0 ** exponents))

    def forward(self, x: torch.Tensor):
        # 1. 积分 (仅第一步有效，后续靠残差)
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + x

        # 2. 获取当前时间步的阈值
        if self.step_counter < self.total_steps:
            v_threshold = self.thresholds[self.step_counter]
        else:
            # 超出范围时使用超大阈值（不再发放）
            v_threshold = torch.tensor(1e9, device=x.device, dtype=x.dtype)

        # 3. 发放
        spike = (self.v >= v_threshold).float()

        # 4. 软复位 (减法重置，保留残差)
        self.v = self.v - spike * v_threshold

        # 5. 更新计数
        self.step_counter += 1

        return spike

    def reset(self):
        """重置神经元状态"""
        self.v = None
        self.step_counter = 0

    def _reset(self):
        """内部reset方法 - 由父组件调用"""
        self.v = None
        self.step_counter = 0


class SignBitNode(nn.Module):
    """符号位检测神经元

    使用抑制性突触 (权重=-1) 检测负数:
    - 输入 x < 0 → 电流 = -x > 0 → 发放 (1)
    - 输入 x ≥ 0 → 电流 = -x ≤ 0 → 不发放 (0)
    """
    def __init__(self):
        super().__init__()
        self.v_threshold = 1e-6  # 极小阈值 (近似0)
        self.synaptic_weight = -1.0
        self.register_buffer('v', None)

    def forward(self, x: torch.Tensor):
        if self.v is None:
            self.v = torch.zeros_like(x)

        # 抑制性突触: 电流 = x * (-1)
        synaptic_current = x * self.synaptic_weight

        # 积分
        self.v = self.v + synaptic_current

        # 发放
        spike = (self.v >= self.v_threshold).float()

        # 软复位
        self.v = self.v - spike * self.v_threshold

        return spike

    def reset(self):
        """重置神经元状态"""
        self.v = None

    def _reset(self):
        """内部reset方法 - 由父组件调用"""
        self.v = None
