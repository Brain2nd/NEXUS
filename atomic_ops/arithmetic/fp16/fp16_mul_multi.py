"""
FP16 多精度乘法器 - 100%纯SNN门电路实现
==========================================

支持 FP16 × FP16 → FP16，中间精度可配置。

架构：
    FP16 输入 → 中间精度乘法 → FP16 输出

设计原则：
    - 输入输出精度一致（FP16）
    - 中间计算精度可配置（FP16/FP32）
    - 更高的中间精度 = 更高的计算准确度
    - 100% 复用现有 SNN 组件

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children

from .fp16_mul_to_fp32 import SpikeFP16MulToFP32
from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP16Converter


class SpikeFP16Multiplier_MultiPrecision(nn.Module):
    """FP16 × FP16 → FP16 多精度乘法器（纯SNN门电路）

    输入输出均为 FP16，中间计算精度可配置。

    Args:
        intermediate_precision: 中间计算精度，'fp16' / 'fp32'
            - 'fp32': FP32中间精度（最高准确度，推荐）
            - 'fp16': FP16中间精度（直接乘法，通过FP32实现后转回）
        neuron_template: 神经元模板，None 使用默认 SimpleLIFNode

    输入:
        A, B: [..., 16] FP16 脉冲

    输出:
        [..., 16] FP16 脉冲

    示例:
        >>> mul = SpikeFP16Multiplier_MultiPrecision(intermediate_precision='fp32')
        >>> a = fp16_encoder(torch.tensor([1.5]))  # FP16 脉冲 [1, 16]
        >>> b = fp16_encoder(torch.tensor([2.0]))  # FP16 脉冲 [1, 16]
        >>> result = mul(a, b)  # FP16 脉冲 [1, 16]
    """

    def __init__(self, intermediate_precision='fp32', neuron_template=None):
        super().__init__()
        self.intermediate_precision = intermediate_precision.lower()
        assert self.intermediate_precision in ('fp16', 'fp32'), \
            f"intermediate_precision must be 'fp16' or 'fp32', got {intermediate_precision}"

        nt = neuron_template

        # FP16 乘法始终通过 FP32 实现（无原生 FP16 乘法器）
        # FP16 → FP32 乘法 → FP16
        self.mul = SpikeFP16MulToFP32(neuron_template=nt)
        self.output_converter = FP32ToFP16Converter(neuron_template=nt)

    def forward(self, A, B):
        """
        Args:
            A, B: [..., 16] FP16 脉冲

        Returns:
            [..., 16] FP16 脉冲
        """
        # 支持广播
        A, B = torch.broadcast_tensors(A, B)

        # FP16 × FP16 → FP32 → FP16
        result_fp32 = self.mul(A, B)
        return self.output_converter(result_fp32)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)
