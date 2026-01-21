"""
神经元模块 (Neuron Module)
=========================

提供 IF/LIF 神经元的纯 PyTorch 实现，支持向量化参数和延迟初始化。

核心原理
--------
- IF 神经元: V += I, 无泄漏
- LIF 神经元: V = β×V + I, 有泄漏
- 软复位: V = V - V_th (保留残差)
- 硬复位: V = 0 (清零)

向量化支持
----------
- 阈值/泄漏率支持标量、向量、张量
- 动态扩展初始化：param_shape='auto' 时根据输入形状自动扩展
  - 首次调用：初始化为当前输入位宽
  - 后续调用：如果输入位宽更大，自动扩展参数
  - forward时：根据实际输入位宽动态切片参数
- 可训练参数：trainable_threshold/trainable_beta

作者: MofNeuroSim Project
"""

import torch
import torch.nn as nn
from typing import Union, Optional


class SimpleIFNode(nn.Module):
    """简化的 IF 神经元 (无泄漏) - 支持向量化阈值

    动力学方程:
    ```
    V(t+1) = V(t) + I(t)           # 膜电位积累
    S(t) = H(V(t) - V_th)          # 脉冲发放
    V(t) = V(t) - S(t) × V_th      # 软复位 (默认)
    ```

    支持三种阈值模式:
    1. 标量 (scalar): 所有神经元共享阈值 (默认，保持位精确)
    2. 向量 (vector): 每个输出通道独立阈值
    3. 张量 (tensor): 任意形状阈值 (需匹配激活形状)

    Args:
        v_threshold: 发放阈值
            - float: 标量阈值 (广播到所有维度)
            - Tensor: 向量化阈值
        v_reset: 复位电压，None 表示软复位，数值表示硬复位到该值
        trainable_threshold: 是否允许训练阈值
        threshold_shape: 延迟初始化时的目标形状
            - None: 使用标量阈值 (默认)
            - 'auto': 动态扩展模式 - 自动扩展以适应不同位宽输入
            - tuple: 指定形状
    """
    def __init__(self,
                 v_threshold: Union[float, torch.Tensor] = 1.0,
                 v_reset: Optional[float] = None,
                 trainable_threshold: bool = False,
                 threshold_shape: Optional[Union[str, tuple]] = None):
        super().__init__()

        self.v_reset = v_reset
        self.trainable_threshold = trainable_threshold
        self.threshold_shape = threshold_shape
        self._threshold_initialized = False

        # 处理不同的阈值初始化模式
        if isinstance(v_threshold, torch.Tensor):
            # 预初始化的张量
            if trainable_threshold:
                self._v_threshold = nn.Parameter(v_threshold.clone())
            else:
                self.register_buffer('_v_threshold', v_threshold.clone())
            self._threshold_initialized = True
            self._v_threshold_default = float(v_threshold.mean().item())
        else:
            # 标量 (默认，位精确模式)
            self._v_threshold_default = float(v_threshold)
            self.register_buffer('_v_threshold', None)

        self.register_buffer('v', None)

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

    def _maybe_expand_threshold(self, x: torch.Tensor):
        """动态扩展初始化：根据输入形状创建或扩展参数

        - 首次调用：初始化为当前输入位宽
        - 后续调用：如果输入位宽更大，扩展参数
        """
        if self.threshold_shape is None:
            return

        # 确定当前输入的目标形状
        if self.threshold_shape == 'auto':
            input_shape = x.shape[1:] if x.dim() > 1 else (1,)
        else:
            input_shape = self.threshold_shape

        # 首次初始化
        if not self._threshold_initialized or self._v_threshold is None:
            threshold_tensor = torch.full(
                input_shape,
                self._v_threshold_default,
                device=x.device,
                dtype=x.dtype
            )
            if self.trainable_threshold:
                self._v_threshold = nn.Parameter(threshold_tensor)
            else:
                self.register_buffer('_v_threshold', threshold_tensor)
            self._threshold_initialized = True
            return

        # 检查是否需要扩展（仅对 'auto' 模式）
        if self.threshold_shape == 'auto':
            current_shape = self._v_threshold.shape
            # 比较最后一个维度（位宽）
            if len(input_shape) > 0 and len(current_shape) > 0:
                input_bits = input_shape[-1]
                current_bits = current_shape[-1]
                if input_bits > current_bits:
                    # 需要扩展：创建新的更大的 tensor
                    new_shape = input_shape[:-1] + (input_bits,)
                    new_threshold = torch.full(
                        new_shape,
                        self._v_threshold_default,
                        device=self._v_threshold.device,
                        dtype=self._v_threshold.dtype
                    )
                    # 复制旧值到新 tensor 的前 current_bits 位
                    new_threshold[..., :current_bits] = self._v_threshold
                    if self.trainable_threshold:
                        self._v_threshold = nn.Parameter(new_threshold)
                    else:
                        self._v_threshold = new_threshold

    def forward(self, x):
        # 动态扩展初始化（如果配置了 threshold_shape）
        self._maybe_expand_threshold(x)

        # 重新初始化 v 如果形状不匹配（支持多次调用不同大小输入）
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)

        # 膜电位积累
        self.v = self.v + x

        # 获取阈值（标量或张量，支持动态切片）
        threshold = self.v_threshold
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(threshold, device=x.device, dtype=x.dtype)
        elif self.threshold_shape == 'auto' and threshold.dim() > 0:
            # 动态切片：根据输入位宽切片阈值
            input_bits = x.shape[-1] if x.dim() > 0 else 1
            if threshold.shape[-1] > input_bits:
                threshold = threshold[..., :input_bits]

        # 发放判断（支持广播）
        spike = (self.v >= threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th（支持广播）
            self.v = self.v - spike * threshold
        else:
            # 硬复位: V = v_reset (发放时) 或保持 (未发放时)
            self.v = torch.where(spike > 0,
                                 torch.full_like(self.v, self.v_reset),
                                 self.v)

        return spike

    def reset_state(self):
        """只重置膜电位，保留参数初始化状态（高效版本）

        用于 SpikeMode.BIT_EXACT 模式下的高频调用。
        与 reset() 不同，此方法不会重置参数初始化状态。
        """
        self.v = None

    def reset(self):
        """重置神经元状态"""
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

    支持向量化参数:
    - v_threshold: 标量/向量/张量
    - beta: 泄漏因子，标量/向量/张量

    默认值 (位精确模式):
    - beta = 1.0 - 1e-7 (极小泄漏，增加信息熵但保持位精确)

    Args:
        beta: 膜电位泄漏因子 (0 < beta ≤ 1)
        v_threshold: 发放阈值
        v_reset: 复位电压，None 表示软复位
        trainable_beta: 是否训练泄漏率
        trainable_threshold: 是否训练阈值
        param_shape: 延迟初始化的参数形状
            - None: 使用标量参数 (默认)
            - 'auto': 动态扩展模式 - 自动扩展以适应不同位宽输入
            - tuple: 指定形状
    """
    # 默认泄漏因子：极小泄漏，保持位精确但增加信息熵
    DEFAULT_BETA = 1.0 - 1e-7

    def __init__(self,
                 beta: Union[float, torch.Tensor] = None,
                 v_threshold: Union[float, torch.Tensor] = 1.0,
                 v_reset: Optional[float] = None,
                 trainable_beta: bool = False,
                 trainable_threshold: bool = False,
                 param_shape: Optional[Union[str, tuple]] = None):
        super().__init__()

        # 处理默认beta值
        if beta is None:
            beta = self.DEFAULT_BETA

        self.v_reset = v_reset
        self.trainable_beta = trainable_beta
        self.trainable_threshold = trainable_threshold
        self.param_shape = param_shape
        self._params_initialized = False

        # 存储默认值（用于延迟初始化）
        self._beta_default = beta if isinstance(beta, (int, float)) else float(beta.mean().item())
        self._threshold_default = v_threshold if isinstance(v_threshold, (int, float)) else float(v_threshold.mean().item())

        # 初始化 beta
        if isinstance(beta, torch.Tensor):
            if trainable_beta:
                self._beta = nn.Parameter(beta.clone())
            else:
                self.register_buffer('_beta', beta.clone())
            self._beta_initialized = True
        else:
            self.register_buffer('_beta', None)
            self._beta_initialized = False

        # 初始化 threshold
        if isinstance(v_threshold, torch.Tensor):
            if trainable_threshold:
                self._v_threshold = nn.Parameter(v_threshold.clone())
            else:
                self.register_buffer('_v_threshold', v_threshold.clone())
            self._threshold_initialized = True
        else:
            self.register_buffer('_v_threshold', None)
            self._threshold_initialized = False

        self.register_buffer('v', None)

    @property
    def beta(self) -> Union[float, torch.Tensor]:
        """获取泄漏因子"""
        if self._beta is not None:
            return self._beta
        return self._beta_default

    @beta.setter
    def beta(self, value):
        """设置泄漏因子"""
        if isinstance(value, (int, float)):
            self._beta_default = float(value)
            self._beta_initialized = False
            if self._beta is not None and not isinstance(self._beta, nn.Parameter):
                self._beta = None
        elif isinstance(value, torch.Tensor):
            if self.trainable_beta:
                self._beta = nn.Parameter(value.clone())
            else:
                self._beta = value.clone()
            self._beta_initialized = True

    @property
    def v_threshold(self) -> Union[float, torch.Tensor]:
        """获取阈值"""
        if self._v_threshold is not None:
            return self._v_threshold
        return self._threshold_default

    @v_threshold.setter
    def v_threshold(self, value):
        """设置阈值 - 支持标量覆盖"""
        if isinstance(value, (int, float)):
            self._threshold_default = float(value)
            self._threshold_initialized = False
            if self._v_threshold is not None and not isinstance(self._v_threshold, nn.Parameter):
                self._v_threshold = None
        elif isinstance(value, torch.Tensor):
            if self.trainable_threshold:
                self._v_threshold = nn.Parameter(value.clone())
            else:
                self._v_threshold = value.clone()
            self._threshold_initialized = True

    def _maybe_expand_params(self, x: torch.Tensor):
        """动态扩展初始化：根据输入形状创建或扩展参数

        - 首次调用：初始化为当前输入位宽
        - 后续调用：如果输入形状变化，重新初始化参数
        """
        if self.param_shape is None:
            return

        # 确定当前输入的目标形状
        if self.param_shape == 'auto':
            input_shape = x.shape[1:] if x.dim() > 1 else (1,)
        else:
            input_shape = self.param_shape

        # 检查形状是否变化（支持实例复用不同输入形状）
        if self._params_initialized and self._beta is not None:
            current_shape = tuple(self._beta.shape)
            if current_shape != input_shape:
                # 形状变化，需要重新初始化
                self._params_initialized = False
                self._beta_initialized = False
                self._threshold_initialized = False
                self._beta = None
                self._v_threshold = None

        # 首次初始化
        if not self._params_initialized:
            # 初始化 beta
            if not self._beta_initialized:
                beta_tensor = torch.full(input_shape, self._beta_default,
                                        device=x.device, dtype=x.dtype)
                if self.trainable_beta:
                    self._beta = nn.Parameter(beta_tensor)
                else:
                    self.register_buffer('_beta', beta_tensor)
                self._beta_initialized = True

            # 初始化 threshold
            if not self._threshold_initialized:
                threshold_tensor = torch.full(input_shape, self._threshold_default,
                                             device=x.device, dtype=x.dtype)
                if self.trainable_threshold:
                    self._v_threshold = nn.Parameter(threshold_tensor)
                else:
                    self.register_buffer('_v_threshold', threshold_tensor)
                self._threshold_initialized = True

            self._params_initialized = True
            return

        # 检查是否需要扩展（仅对 'auto' 模式）
        if self.param_shape == 'auto' and len(input_shape) > 0:
            input_bits = input_shape[-1]

            # 扩展 beta
            if self._beta is not None and self._beta.dim() > 0:
                current_bits = self._beta.shape[-1]
                if input_bits > current_bits:
                    new_shape = input_shape[:-1] + (input_bits,)
                    new_beta = torch.full(new_shape, self._beta_default,
                                         device=self._beta.device, dtype=self._beta.dtype)
                    new_beta[..., :current_bits] = self._beta
                    if self.trainable_beta:
                        self._beta = nn.Parameter(new_beta)
                    else:
                        self._beta = new_beta

            # 扩展 threshold
            if self._v_threshold is not None and self._v_threshold.dim() > 0:
                current_bits = self._v_threshold.shape[-1]
                if input_bits > current_bits:
                    new_shape = input_shape[:-1] + (input_bits,)
                    new_threshold = torch.full(new_shape, self._threshold_default,
                                              device=self._v_threshold.device,
                                              dtype=self._v_threshold.dtype)
                    new_threshold[..., :current_bits] = self._v_threshold
                    if self.trainable_threshold:
                        self._v_threshold = nn.Parameter(new_threshold)
                    else:
                        self._v_threshold = new_threshold

    def forward(self, x):
        # 动态扩展初始化（会检测形状变化并重新初始化参数）
        self._maybe_expand_params(x)

        # 重新初始化 v 如果形状不匹配（支持多次调用不同大小输入）
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)

        # 获取参数（支持广播和动态切片）
        beta = self.beta
        threshold = self.v_threshold
        input_bits = x.shape[-1] if x.dim() > 0 else 1

        if isinstance(beta, (int, float)):
            beta = torch.tensor(beta, device=x.device, dtype=x.dtype)
        elif self.param_shape == 'auto' and beta.dim() > 0:
            # 动态切片
            if beta.shape[-1] > input_bits:
                beta = beta[..., :input_bits]

        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(threshold, device=x.device, dtype=x.dtype)
        elif self.param_shape == 'auto' and threshold.dim() > 0:
            # 动态切片
            if threshold.shape[-1] > input_bits:
                threshold = threshold[..., :input_bits]

        # LIF 动力学: V = beta * V + I（支持广播）
        self.v = beta * self.v + x

        # 发放判断
        spike = (self.v >= threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            self.v = self.v - spike * threshold
        else:
            # 硬复位: V = v_reset (发放时) 或保持 (未发放时)
            self.v = torch.where(spike > 0,
                                 torch.full_like(self.v, self.v_reset),
                                 self.v)

        return spike

    def reset_state(self):
        """只重置膜电位，保留参数初始化状态（高效版本）

        用于 SpikeMode.BIT_EXACT 模式下的高频调用。
        与 reset() 不同，此方法不会重置参数初始化状态，
        避免在每次 forward 调用时重新初始化张量参数。
        """
        self.v = None

    def reset(self):
        """重置神经元状态"""
        self.v = None
        # 重置参数初始化状态，以支持不同输入形状
        self._params_initialized = False
        self._beta_initialized = False
        self._threshold_initialized = False
        # 清除已初始化的参数
        if hasattr(self, '_beta') and self._beta is not None:
            self._beta = None
        if hasattr(self, '_v_threshold') and self._v_threshold is not None:
            self._v_threshold = None

    def _reset(self):
        """内部reset方法 - 由父组件调用"""
        self.v = None
        self._params_initialized = False
        self._beta_initialized = False
        self._threshold_initialized = False
        if hasattr(self, '_beta') and self._beta is not None:
            self._beta = None
        if hasattr(self, '_v_threshold') and self._v_threshold is not None:
            self._v_threshold = None


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
