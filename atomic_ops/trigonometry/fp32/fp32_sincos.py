"""
FP32 三角函数 Sin/Cos - 100%纯SNN门电路实现
============================================

内部使用 FP64 精度计算以确保精确度。

流程: FP32 输入 → FP64 转换 → FP64计算 → FP32 输出

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children

from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from atomic_ops.trigonometry.fp64.fp64_sincos import SpikeFP64Sin, SpikeFP64Cos, SpikeFP64SinCos


class SpikeFP32Sin(nn.Module):
    """FP32 正弦函数 - 内部使用 FP64 精度

    流程: FP32 → FP64 → sin → FP64 → FP32

    Args:
        neuron_template: 神经元模板，None 使用默认神经元

    输入: [..., 32] FP32脉冲
    输出: [..., 32] FP32脉冲
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_sin = SpikeFP64Sin(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)

    def forward(self, x):
        """计算 sin(x)

        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲
        """
        # FP32 → FP64
        x_fp64 = self.fp32_to_fp64(x)

        # sin in FP64
        result_fp64 = self.fp64_sin(x_fp64)

        # FP64 → FP32
        result = self.fp64_to_fp32(result_fp64)

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


class SpikeFP32Cos(nn.Module):
    """FP32 余弦函数 - 内部使用 FP64 精度

    流程: FP32 → FP64 → cos → FP64 → FP32

    Args:
        neuron_template: 神经元模板，None 使用默认神经元

    输入: [..., 32] FP32脉冲
    输出: [..., 32] FP32脉冲
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_cos = SpikeFP64Cos(neuron_template=nt)
        self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)

    def forward(self, x):
        """计算 cos(x)

        输入: [..., 32] FP32脉冲
        输出: [..., 32] FP32脉冲
        """
        # FP32 → FP64
        x_fp64 = self.fp32_to_fp64(x)

        # cos in FP64
        result_fp64 = self.fp64_cos(x_fp64)

        # FP64 → FP32
        result = self.fp64_to_fp32(result_fp64)

        return result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)


class SpikeFP32SinCos(nn.Module):
    """FP32 同时计算 Sin 和 Cos - 优化版本

    同时返回 sin(x) 和 cos(x)，共享 FP64 计算。

    Args:
        neuron_template: 神经元模板，None 使用默认神经元

    输入: [..., 32] FP32脉冲
    输出: (sin_result, cos_result) 各 [..., 32] FP32脉冲
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
        self.fp64_sincos = SpikeFP64SinCos(neuron_template=nt)
        self.fp64_to_fp32_sin = FP64ToFP32Converter(neuron_template=nt)
        self.fp64_to_fp32_cos = FP64ToFP32Converter(neuron_template=nt)

    def forward(self, x):
        """同时计算 sin(x) 和 cos(x)

        输入: [..., 32] FP32脉冲
        输出: (sin_result, cos_result)
        """
        # FP32 → FP64
        x_fp64 = self.fp32_to_fp64(x)

        # sin/cos in FP64
        sin_fp64, cos_fp64 = self.fp64_sincos(x_fp64)

        # FP64 → FP32
        sin_result = self.fp64_to_fp32_sin(sin_fp64)
        cos_result = self.fp64_to_fp32_cos(cos_fp64)

        return sin_result, cos_result

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)
