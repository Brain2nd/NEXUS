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
- 所有参数均为 nn.Parameter (可训练)

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
    - __init__ 时预分配 max_param_shape 尺寸的 nn.Parameter
    - forward 时根据输入位宽切片: threshold[..., :input_bits]
    - reset 时保留参数，只重置膜电位

    Args:
        v_threshold: 发放阈值 (float 或 Tensor)
        v_reset: 复位电压，None 表示软复位，数值表示硬复位到该值
        max_param_shape: 预分配参数形状，默认 DEFAULT_MAX_PARAM_SHAPE (64,)
    """
    def __init__(self,
                 v_threshold: Union[float, torch.Tensor] = 1.0,
                 v_reset: Optional[float] = None,
                 max_param_shape: Optional[tuple] = None,
                 # 兼容旧接口，忽略该参数
                 trainable_threshold: bool = True):
        super().__init__()

        self.v_reset = v_reset
        # 默认使用全局预分配形状
        self.max_param_shape = max_param_shape if max_param_shape is not None else DEFAULT_MAX_PARAM_SHAPE

        # 提取初始阈值标量
        if isinstance(v_threshold, torch.Tensor):
            self._init_threshold_val = float(v_threshold.mean().item())
        else:
            self._init_threshold_val = float(v_threshold)

        self.register_buffer('v', None)

        # 预分配参数 (始终为 nn.Parameter)
        self._preallocate_params(self.max_param_shape)

    @property
    def v_threshold(self) -> nn.Parameter:
        """获取阈值 (始终为 nn.Parameter)"""
        return self._v_threshold

    @v_threshold.setter
    def v_threshold(self, value):
        """设置阈值 - 就地修改预分配 Parameter（用于 _create_neuron deepcopy 后覆盖）"""
        if isinstance(value, (int, float)):
            self._init_threshold_val = float(value)
            self._v_threshold.data.fill_(float(value))
        elif isinstance(value, torch.Tensor):
            self._init_threshold_val = float(value.mean().item())
            self._v_threshold.data.fill_(float(value.mean().item()))

    def _preallocate_params(self, shape: tuple):
        """在 __init__ 时一次性预分配最大尺寸参数 (nn.Parameter)

        Args:
            shape: 要预分配的参数形状 (例如 (32,) 表示32位)
        """
        threshold_tensor = torch.full(shape, self._init_threshold_val, dtype=torch.float32)
        self._v_threshold = nn.Parameter(threshold_tensor)

    def forward(self, x):
        # 预分配模式：参数已在 __init__ 中创建，根据输入位宽切片
        input_bits = x.shape[-1] if x.dim() > 0 else 1
        threshold = self._v_threshold[..., :input_bits]

        # v 预分配切片机制:
        # - 首次 forward: 预分配 (*batch_dims, max_v_bits)
        # - max_v_bits = max(max_param_shape, input_bits) 确保覆盖实际输入位宽
        # - 参数 _v_threshold 可通过广播覆盖更大输入（如 param=(1,) 广播到 input=5）
        #   但 v 必须与输入严格匹配，不能依赖广播
        max_v_bits = max(self.max_param_shape[-1] if self.max_param_shape else 64, input_bits)
        if self.v is None:
            v_shape = list(x.shape)
            v_shape[-1] = max_v_bits
            self.v = torch.zeros(v_shape, dtype=x.dtype, device=x.device)
        elif self.v.shape[:-1] != x.shape[:-1] or self.v.shape[-1] < input_bits:
            # batch/spatial 维度变化 或 位宽不足: 重新预分配
            new_max = max(self.v.shape[-1], max_v_bits)
            v_shape = list(x.shape)
            v_shape[-1] = new_max
            self.v = torch.zeros(v_shape, dtype=x.dtype, device=x.device)

        # 切片当前位宽
        v_slice = self.v[..., :input_bits]

        # 膜电位积累 (就地操作避免内存分配)
        v_slice.add_(x)

        # 发放判断
        spike = (v_slice >= threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            v_slice.sub_(spike * threshold)
        else:
            # 硬复位
            reset_val = torch.full_like(v_slice, self.v_reset)
            v_slice.copy_(torch.where(spike > 0, reset_val, v_slice))

        return spike

    def reset_state(self):
        """重置膜电位，释放内存

        用于 SpikeMode.BIT_EXACT 模式下的调用。
        释放 self.v 以防止显存累积。

        注意：设置 self.v = None 释放内存，而不是 zero_() 保留张量。
        虽然每次 forward 需要重新分配，但这是防止显存滚雪球的必要代价。
        """
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
        threshold = self._v_threshold
        return (self.v >= threshold).float()

    def neuronal_reset(self, spike):
        """复位 (兼容接口)"""
        threshold = self._v_threshold
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
    - __init__ 时预分配 max_param_shape 尺寸的 nn.Parameter
    - forward 时根据输入位宽切片: param[..., :input_bits]
    - reset 时保留参数，只重置膜电位

    默认值 (位精确模式):
    - beta = 1.0 - 1e-7 (极小泄漏，保持位精确)

    Args:
        beta: 膜电位泄漏因子 (0 < beta ≤ 1)
        v_threshold: 发放阈值
        v_reset: 复位电压，None 表示软复位
        max_param_shape: 预分配参数形状，默认 DEFAULT_MAX_PARAM_SHAPE (64,)
    """
    # 默认泄漏因子：极小泄漏，保持位精确但增加信息熵
    DEFAULT_BETA = 1.0 - 1e-7

    def __init__(self,
                 beta: Union[float, torch.Tensor] = None,
                 v_threshold: Union[float, torch.Tensor] = 1.0,
                 v_reset: Optional[float] = None,
                 max_param_shape: Optional[tuple] = None,
                 # 兼容旧接口，忽略这些参数
                 trainable_beta: bool = True,
                 trainable_threshold: bool = True):
        super().__init__()

        # 处理默认beta值
        if beta is None:
            beta = self.DEFAULT_BETA

        self.v_reset = v_reset
        # 默认使用全局预分配形状
        self.max_param_shape = max_param_shape if max_param_shape is not None else DEFAULT_MAX_PARAM_SHAPE

        # 提取初始标量值
        self._init_beta_val = beta if isinstance(beta, (int, float)) else float(beta.mean().item())
        self._init_threshold_val = v_threshold if isinstance(v_threshold, (int, float)) else float(v_threshold.mean().item())

        self.register_buffer('v', None)

        # 预分配参数 (始终为 nn.Parameter)
        self._preallocate_params(self.max_param_shape)

    @property
    def beta(self) -> nn.Parameter:
        """获取泄漏因子 (始终为 nn.Parameter)"""
        return self._beta

    @property
    def v_threshold(self) -> nn.Parameter:
        """获取阈值 (始终为 nn.Parameter)"""
        return self._v_threshold

    @v_threshold.setter
    def v_threshold(self, value):
        """设置阈值 - 就地修改预分配 Parameter（用于 _create_neuron deepcopy 后覆盖）"""
        if isinstance(value, (int, float)):
            self._init_threshold_val = float(value)
            self._v_threshold.data.fill_(float(value))
        elif isinstance(value, torch.Tensor):
            self._init_threshold_val = float(value.mean().item())
            self._v_threshold.data.fill_(float(value.mean().item()))

    def _preallocate_params(self, shape: tuple):
        """在 __init__ 时一次性预分配最大尺寸参数 (nn.Parameter)

        Args:
            shape: 要预分配的参数形状 (例如 (32,) 表示32位)
        """
        # 预分配 beta
        beta_tensor = torch.full(shape, self._init_beta_val, dtype=torch.float32)
        self._beta = nn.Parameter(beta_tensor)

        # 预分配 threshold
        threshold_tensor = torch.full(shape, self._init_threshold_val, dtype=torch.float32)
        self._v_threshold = nn.Parameter(threshold_tensor)

    def forward(self, x):
        # 预分配模式：参数已在 __init__ 中创建，根据输入位宽切片
        input_bits = x.shape[-1] if x.dim() > 0 else 1
        beta = self._beta[..., :input_bits]
        threshold = self._v_threshold[..., :input_bits]

        # v 预分配切片机制:
        # - 首次 forward: 预分配 (*batch_dims, max_v_bits)
        # - max_v_bits = max(max_param_shape, input_bits) 确保覆盖实际输入位宽
        # - 参数 _beta/_v_threshold 可通过广播覆盖更大输入（如 param=(1,) 广播到 input=5）
        #   但 v 必须与输入严格匹配，不能依赖广播
        max_v_bits = max(self.max_param_shape[-1] if self.max_param_shape else 64, input_bits)
        if self.v is None:
            v_shape = list(x.shape)
            v_shape[-1] = max_v_bits
            self.v = torch.zeros(v_shape, dtype=x.dtype, device=x.device)
        elif self.v.shape[:-1] != x.shape[:-1] or self.v.shape[-1] < input_bits:
            # batch/spatial 维度变化 或 位宽不足: 重新预分配
            new_max = max(self.v.shape[-1], max_v_bits)
            v_shape = list(x.shape)
            v_shape[-1] = new_max
            self.v = torch.zeros(v_shape, dtype=x.dtype, device=x.device)

        # 切片当前位宽
        v_slice = self.v[..., :input_bits]

        # LIF 动力学: V = beta * V + I
        v_slice.mul_(beta).add_(x)

        # 发放判断
        spike = (v_slice >= threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            v_slice.sub_(spike * threshold)
        else:
            # 硬复位
            reset_val = torch.full_like(v_slice, self.v_reset)
            v_slice.copy_(torch.where(spike > 0, reset_val, v_slice))

        return spike

    def reset_state(self):
        """重置膜电位，保留预分配参数"""
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
