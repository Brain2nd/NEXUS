"""
神经元模块 (Neuron Module)
=========================

提供 IF/LIF 神经元的纯 PyTorch 实现。

核心原理
--------
- IF 神经元: V += I, 无泄漏
- LIF 神经元: V = β×V + I, 有泄漏
- 软复位: V = V - V_th (保留残差)
- 硬复位: V = 0 (清零)

作者: MofNeuroSim Project
"""

import torch
import torch.nn as nn


class SimpleIFNode(nn.Module):
    """简化的 IF 神经元 (无泄漏)

    动力学方程:
    ```
    V(t+1) = V(t) + I(t)           # 膜电位积累
    S(t) = H(V(t) - V_th)          # 脉冲发放
    V(t) = V(t) - S(t) × V_th      # 软复位 (默认)
    ```

    Args:
        v_threshold: 发放阈值
        v_reset: 复位电压，None 表示软复位，数值表示硬复位到该值
    """
    def __init__(self, v_threshold=1.0, v_reset=None):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset  # None = 软复位, 数值 = 硬复位
        self.register_buffer('v', None)

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)

        # 膜电位积累
        self.v = self.v + x

        # 发放判断
        spike = (self.v >= self.v_threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            self.v = self.v - spike * self.v_threshold
        else:
            # 硬复位: V = v_reset (发放时) 或保持 (未发放时)
            self.v = torch.where(spike > 0,
                                 torch.full_like(self.v, self.v_reset),
                                 self.v)

        return spike

    def reset(self):
        """重置神经元状态"""
        self.v = None

    # 兼容接口
    def neuronal_charge(self, x):
        """膜电位积累 (兼容接口)"""
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + x

    def neuronal_fire(self):
        """发放判断 (兼容接口)"""
        return (self.v >= self.v_threshold).float()

    def neuronal_reset(self, spike):
        """复位 (兼容接口)"""
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = torch.where(spike > 0,
                                 torch.full_like(self.v, self.v_reset),
                                 self.v)


class SimpleLIFNode(nn.Module):
    """简化的 LIF 神经元 (有泄漏)

    动力学方程:
    ```
    V(t+1) = β × V(t) + I(t)       # 膜电位泄漏 + 积累
    S(t) = H(V(t) - V_th)          # 脉冲发放
    V(t) = V(t) - S(t) × V_th      # 软复位 (默认)
    ```

    Args:
        beta: 膜电位泄漏因子 (0 < beta ≤ 1)
              beta = 1.0 时退化为 IF 神经元
        v_threshold: 发放阈值
        v_reset: 复位电压，None 表示软复位，数值表示硬复位到该值
    """
    def __init__(self, beta=1.0, v_threshold=1.0, v_reset=None):
        super().__init__()
        self.beta = beta
        self.v_threshold = v_threshold
        self.v_reset = v_reset  # None = 软复位, 数值 = 硬复位
        self.register_buffer('v', None)

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)

        # LIF 动力学: V = beta * V + I
        self.v = self.beta * self.v + x

        # 发放判断
        spike = (self.v >= self.v_threshold).float()

        # 复位
        if self.v_reset is None:
            # 软复位: V = V - spike × V_th
            self.v = self.v - spike * self.v_threshold
        else:
            # 硬复位: V = v_reset (发放时) 或保持 (未发放时)
            self.v = torch.where(spike > 0,
                                 torch.full_like(self.v, self.v_reset),
                                 self.v)

        return spike

    def reset(self):
        """重置神经元状态"""
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
