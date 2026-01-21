"""
FP8 脉冲域矩阵乘法 - 100%纯SNN门电路实现
=============================================

用于激活值×激活值的矩阵乘法（如 Attention 的 Q×K^T）。

支持多种累加精度以平衡精度和性能。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn

from atomic_ops.core.accumulator import SequentialAccumulator, ParallelAccumulator


class SpikeFP8MatMul(nn.Module):
    """FP8 脉冲域矩阵乘法: C = A @ B^T

    用于激活值×激活值的计算（如 Q×K^T）。

    A: [..., M, K, 8] pulse
    B: [..., N, K, 8] pulse (注意: 计算 A @ B^T)
    C: [..., M, N, bits] pulse (bits 取决于 accum_precision)

    Args:
        accum_precision: 累加精度，'fp8' / 'fp16' / 'fp32'
            - 'fp32': FP32累加（最高精度，推荐）→ 输出 FP32 脉冲 [32位]
            - 'fp16': FP16累加 → 输出 FP16 脉冲 [16位]
            - 'fp8':  FP8累加 → 输出 FP8 脉冲 [8位]
        accum_mode: 累加策略，'sequential' / 'parallel'
            - 'sequential': 顺序累加，位精确确定性（默认）
            - 'parallel': 树形归约，快速
        neuron_template: 神经元模板

    输出精度:
        - accum_precision='fp8'  → [..., M, N, 8]
        - accum_precision='fp16' → [..., M, N, 16]
        - accum_precision='fp32' → [..., M, N, 32]
    """

    def __init__(self, accum_precision='fp32', accum_mode='sequential', neuron_template=None):
        super().__init__()
        self.accum_precision = accum_precision.lower()
        self.accum_mode = accum_mode
        nt = neuron_template

        assert self.accum_precision in ('fp8', 'fp16', 'fp32'), \
            f"accum_precision must be 'fp8', 'fp16', or 'fp32', got {accum_precision}"

        # 选择 Accumulator 类
        AccumulatorClass = ParallelAccumulator if accum_mode == 'parallel' else SequentialAccumulator

        # 乘法器: FP8 × FP8 → FP32 (无舍入)
        from .fp8_mul_to_fp32 import SpikeFP8MulToFP32
        self.mul = SpikeFP8MulToFP32(neuron_template=nt)

        if self.accum_precision == 'fp32':
            # FP32 累加 → 输出 FP32 脉冲
            from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
            self.adder = SpikeFP32Adder(neuron_template=nt)
            self.accumulator = AccumulatorClass(self.adder)
            self.output_bits = 32
        elif self.accum_precision == 'fp16':
            # FP16 累加 → 输出 FP16 脉冲
            from atomic_ops.arithmetic.fp16.fp16_adder import SpikeFP16Adder
            from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP16Converter
            self.fp32_to_fp16 = FP32ToFP16Converter(neuron_template=nt)
            self.adder = SpikeFP16Adder(neuron_template=nt)
            self.accumulator = AccumulatorClass(self.adder)
            self.output_bits = 16
        else:
            # FP8 累加 → 输出 FP8 脉冲
            from .fp8_adder_spatial import SpikeFP8Adder_Spatial
            from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP8Converter
            self.fp32_to_fp8 = FP32ToFP8Converter(neuron_template=nt)
            self.adder = SpikeFP8Adder_Spatial(neuron_template=nt)
            self.accumulator = AccumulatorClass(self.adder)
            self.output_bits = 8

    def forward(self, A, B):
        """计算 C = A @ B^T

        Args:
            A: [..., M, K, 8] FP8 脉冲
            B: [..., N, K, 8] FP8 脉冲

        Returns:
            [..., M, N, output_bits] 脉冲 (output_bits=8/16/32)
        """
        # A: [..., M, K, 8] -> [..., M, 1, K, 8]
        # B: [..., N, K, 8] -> [..., 1, N, K, 8]
        A_expanded = A.unsqueeze(-3)
        B_expanded = B.unsqueeze(-4)

        # 广播乘法: [..., M, N, K, 32] (FP32 输出)
        products = self.mul(A_expanded, B_expanded)

        if self.accum_precision == 'fp32':
            # FP32 累加
            return self.accumulator.reduce(products, dim=-2)
        elif self.accum_precision == 'fp16':
            # FP16 累加
            products_fp16 = self.fp32_to_fp16(products)
            return self.accumulator.reduce(products_fp16, dim=-2)
        else:
            # FP8 累加
            products_fp8 = self.fp32_to_fp8(products)
            return self.accumulator.reduce(products_fp8, dim=-2)

    def reset(self):
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()


class SpikeFP8MatMulAB(nn.Module):
    """FP8 脉冲域矩阵乘法: C = A @ B (不转置)

    A: [..., M, K, 8] pulse
    B: [K, N, 8] pulse
    C: [..., M, N, bits] pulse

    Args:
        accum_precision: 累加精度，'fp8' / 'fp16' / 'fp32'
        accum_mode: 累加策略，'sequential' / 'parallel'
        neuron_template: 神经元模板
    """

    def __init__(self, accum_precision='fp32', accum_mode='sequential', neuron_template=None):
        super().__init__()
        self.accum_precision = accum_precision.lower()
        self.accum_mode = accum_mode
        nt = neuron_template

        AccumulatorClass = ParallelAccumulator if accum_mode == 'parallel' else SequentialAccumulator

        from .fp8_mul_to_fp32 import SpikeFP8MulToFP32
        self.mul = SpikeFP8MulToFP32(neuron_template=nt)

        if self.accum_precision == 'fp32':
            from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
            self.adder = SpikeFP32Adder(neuron_template=nt)
            self.accumulator = AccumulatorClass(self.adder)
            self.output_bits = 32
        elif self.accum_precision == 'fp16':
            from atomic_ops.arithmetic.fp16.fp16_adder import SpikeFP16Adder
            from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP16Converter
            self.fp32_to_fp16 = FP32ToFP16Converter(neuron_template=nt)
            self.adder = SpikeFP16Adder(neuron_template=nt)
            self.accumulator = AccumulatorClass(self.adder)
            self.output_bits = 16
        else:
            from .fp8_adder_spatial import SpikeFP8Adder_Spatial
            from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP8Converter
            self.fp32_to_fp8 = FP32ToFP8Converter(neuron_template=nt)
            self.adder = SpikeFP8Adder_Spatial(neuron_template=nt)
            self.accumulator = AccumulatorClass(self.adder)
            self.output_bits = 8

    def forward(self, A, B):
        """计算 C = A @ B

        Args:
            A: [..., M, K, 8] FP8 脉冲
            B: [K, N, 8] FP8 脉冲

        Returns:
            [..., M, N, output_bits] 脉冲
        """
        A_expanded = A.unsqueeze(-2)  # [..., M, K, 1, 8]
        B_expanded = B.unsqueeze(0)   # [1, K, N, 8]

        products = self.mul(A_expanded, B_expanded)  # [..., M, K, N, 32]

        if self.accum_precision == 'fp32':
            return self.accumulator.reduce(products, dim=-3)
        elif self.accum_precision == 'fp16':
            products_fp16 = self.fp32_to_fp16(products)
            return self.accumulator.reduce(products_fp16, dim=-3)
        else:
            products_fp8 = self.fp32_to_fp8(products)
            return self.accumulator.reduce(products_fp8, dim=-3)

    def reset(self):
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()
