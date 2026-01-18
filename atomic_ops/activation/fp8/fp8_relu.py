"""
FP8 ReLU - 100% 纯 SNN 门电路实现
=================================

ReLU(x) = max(0, x)

对于 FP8 脉冲表示：
- x[..., 0] 是符号位 (0=正, 1=负)
- 正数保持不变，负数变为 +0

实现原理：
- 使用 VecNOT 对符号位取反得到 mask
- 使用 VecAND 将 mask 与每一位进行 AND 运算
- 负数 (符号位=1) → mask=0 → 所有位变为 0 → +0
- 正数 (符号位=0) → mask=1 → 所有位保持 → 原值

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.vec_logic_gates import VecNOT, VecAND


class SpikeFP8ReLU(nn.Module):
    """FP8 ReLU - 100% 纯 SNN 门电路实现

    输入: [..., 8] FP8 脉冲序列，x[..., 0] 是符号位
    输出: [..., 8] ReLU 后的 FP8 脉冲序列

    符合 SNN 基本原则:
    - 使用 VecNOT 替代 (1 - x)
    - 使用 VecAND 替代 (a * b)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式（梯度流过）
    """

    def __init__(self, neuron_template=None, trainable=False):
        super().__init__()
        self.trainable = trainable
        nt = neuron_template
        self.vec_not = VecNOT(neuron_template=nt)
        self.vec_and = VecAND(neuron_template=nt)
    
    def forward(self, x_pulse):
        """
        Args:
            x_pulse: [..., 8] FP8 脉冲序列
        Returns:
            [..., 8] ReLU(x) 脉冲序列
        """
        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # 提取符号位: shape [..., 1]
            sign = x_pulse[..., 0:1]

            # mask = NOT(sign): 正数时 mask=1, 负数时 mask=0
            # 符合 SNN 原则: 使用 VecNOT 而非 (1 - sign)
            mask = self.vec_not(sign)

            # 广播 mask 到所有 8 位: [..., 1] -> [..., 8]
            mask_broadcast = mask.expand_as(x_pulse)

            # 使用 AND 门屏蔽负数: 负数时 mask=0 → 输出全 0
            # 符合 SNN 原则: 使用 VecAND 而非 (x_pulse * mask)
            out_pulse = self.vec_and(x_pulse, mask_broadcast)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_relu
            return ste_relu(x_pulse, out_pulse)

        return out_pulse

    def reset(self):
        """重置所有神经元状态"""
        self.vec_not.reset()
        self.vec_and.reset()


class SpikeFP32ReLU(nn.Module):
    """FP32 ReLU - 100% 纯 SNN 门电路实现

    输入: [..., 32] FP32 脉冲序列，x[..., 0] 是符号位
    输出: [..., 32] ReLU 后的 FP32 脉冲序列

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式（梯度流过）
    """

    def __init__(self, neuron_template=None, trainable=False):
        super().__init__()
        self.trainable = trainable
        nt = neuron_template
        self.vec_not = VecNOT(neuron_template=nt)
        self.vec_and = VecAND(neuron_template=nt)

    def forward(self, x_pulse):
        """
        Args:
            x_pulse: [..., 32] FP32 脉冲序列
        Returns:
            [..., 32] ReLU(x) 脉冲序列
        """
        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # 提取符号位: shape [..., 1]
            sign = x_pulse[..., 0:1]

            # mask = NOT(sign)
            mask = self.vec_not(sign)

            # 广播到所有 32 位
            mask_broadcast = mask.expand_as(x_pulse)

            # AND 门屏蔽
            out_pulse = self.vec_and(x_pulse, mask_broadcast)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_relu
            return ste_relu(x_pulse, out_pulse)

        return out_pulse

    def reset(self):
        self.vec_not.reset()
        self.vec_and.reset()


class SpikeFP64ReLU(nn.Module):
    """FP64 ReLU - 100% 纯 SNN 门电路实现

    输入: [..., 64] FP64 脉冲序列，x[..., 0] 是符号位
    输出: [..., 64] ReLU 后的 FP64 脉冲序列

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式（梯度流过）
    """

    def __init__(self, neuron_template=None, trainable=False):
        super().__init__()
        self.trainable = trainable
        nt = neuron_template
        self.vec_not = VecNOT(neuron_template=nt)
        self.vec_and = VecAND(neuron_template=nt)

    def forward(self, x_pulse):
        """
        Args:
            x_pulse: [..., 64] FP64 脉冲序列
        Returns:
            [..., 64] ReLU(x) 脉冲序列
        """
        # SNN 前向 (纯门电路)
        with torch.no_grad():
            sign = x_pulse[..., 0:1]
            mask = self.vec_not(sign)
            mask_broadcast = mask.expand_as(x_pulse)
            out_pulse = self.vec_and(x_pulse, mask_broadcast)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_relu
            return ste_relu(x_pulse, out_pulse)

        return out_pulse

    def reset(self):
        self.vec_not.reset()
        self.vec_and.reset()
