"""
FP8 多精度乘法器 - 100%纯SNN门电路实现
==========================================

支持 FP8 × FP8 → FP8，中间精度可配置。

架构：
    FP8 输入 → 中间精度乘法 → FP8 输出

设计原则：
    - 输入输出精度一致（FP8）
    - 中间计算精度可配置（FP8/FP16/FP32）
    - 更高的中间精度 = 更高的计算准确度
    - 100% 复用现有 SNN 组件

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children

from .fp8_mul import SpikeFP8Multiplier
from .fp8_mul_to_fp32 import SpikeFP8MulToFP32
from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP8Converter, FP32ToFP16Converter
from atomic_ops.arithmetic.fp16.fp16_components import FP16ToFP8Converter


class SpikeFP8Multiplier_MultiPrecision(nn.Module):
    """FP8 × FP8 → FP8 多精度乘法器（纯SNN门电路）

    输入输出均为 FP8，中间计算精度可配置。

    Args:
        intermediate_precision: 中间计算精度，'fp8' / 'fp16' / 'fp32'
            - 'fp32': FP32中间精度（最高准确度，推荐）
            - 'fp16': FP16中间精度（中等准确度）
            - 'fp8':  FP8中间精度（最快但准确度较低）
        neuron_template: 神经元模板，None 使用默认 SimpleLIFNode

    输入:
        A, B: [..., 8] FP8 脉冲

    输出:
        [..., 8] FP8 脉冲

    示例:
        >>> mul = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp32')
        >>> a = encoder(torch.tensor([1.5]))  # FP8 脉冲 [1, 8]
        >>> b = encoder(torch.tensor([2.0]))  # FP8 脉冲 [1, 8]
        >>> result = mul(a, b)  # FP8 脉冲 [1, 8]
    """

    def __init__(self, intermediate_precision='fp32', neuron_template=None):
        super().__init__()
        self.intermediate_precision = intermediate_precision.lower()
        assert self.intermediate_precision in ('fp8', 'fp16', 'fp32'), \
            f"intermediate_precision must be 'fp8', 'fp16', or 'fp32', got {intermediate_precision}"

        nt = neuron_template

        if self.intermediate_precision == 'fp32':
            # FP8 → FP32 乘法 → FP8
            self.mul = SpikeFP8MulToFP32(neuron_template=nt)
            self.output_converter = FP32ToFP8Converter(neuron_template=nt)
        elif self.intermediate_precision == 'fp16':
            # FP8 → FP32 乘法 → FP16 → FP8
            self.mul = SpikeFP8MulToFP32(neuron_template=nt)
            self.fp32_to_fp16 = FP32ToFP16Converter(neuron_template=nt)
            self.output_converter = FP16ToFP8Converter(neuron_template=nt)
        else:  # fp8
            # 直接 FP8 乘法
            self.mul = SpikeFP8Multiplier(neuron_template=nt)
            # 无需转换器

    def forward(self, A, B):
        """
        Args:
            A, B: [..., 8] FP8 脉冲

        Returns:
            [..., 8] FP8 脉冲
        """
        # 支持广播
        A, B = torch.broadcast_tensors(A, B)

        if self.intermediate_precision == 'fp32':
            # FP8 × FP8 → FP32 → FP8
            result_fp32 = self.mul(A, B)
            return self.output_converter(result_fp32)
        elif self.intermediate_precision == 'fp16':
            # FP8 × FP8 → FP32 → FP16 → FP8
            result_fp32 = self.mul(A, B)
            result_fp16 = self.fp32_to_fp16(result_fp32)
            return self.output_converter(result_fp16)
        else:  # fp8
            # 直接 FP8 × FP8 → FP8
            return self.mul(A, B)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)
