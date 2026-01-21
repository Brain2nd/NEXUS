"""
Atomic Ops Package - 100% 纯 SNN 门电路实现的浮点运算模块

支持统一的神经元模板机制，可在 IF/LIF 之间切换用于物理仿真。
支持位精确模式 (BIT_EXACT) 和时间动力学模式 (TEMPORAL) 灵活切换。

训练模式 (TrainingMode):
- TrainingMode.NONE: 纯推理模式 (默认)
- TrainingMode.STE: 位精确 STE 训练模式
- TrainingMode.TEMPORAL: 时间动力学训练模式 (未来扩展)

使用示例:
```python
from atomic_ops import ANDGate, SimpleLIFNode, SpikeFP32Adder, SpikeMode, TrainingMode

# 默认 IF 神经元 (理想数字逻辑)
and_gate = ANDGate()

# LIF 神经元 (物理仿真)
lif_template = SimpleLIFNode(beta=0.9)
and_gate_lif = ANDGate(neuron_template=lif_template)
adder_lif = SpikeFP32Adder(neuron_template=lif_template)

# SpikeMode 模式控制 (类似 torch.no_grad())
with SpikeMode.temporal():
    # 训练模式 - 保留膜电位残差
    loss = model(input)
    loss.backward()

with SpikeMode.bit_exact():
    # 推理模式 - 每次 forward 前清除膜电位
    result = model(input)

# TrainingMode 训练模式控制
linear = SpikeFP32Linear_MultiPrecision(64, 32, training_mode=TrainingMode.STE)
```
"""
# SpikeMode 模式控制器 (位精确模式 vs 时间动力学模式)
from .core.spike_mode import SpikeMode

# TrainingMode 训练模式枚举
from .core.training_mode import TrainingMode

# Accumulator 累加器 (解耦的归约策略)
from .core.accumulator import (
    Accumulator,
    SequentialAccumulator,
    ParallelAccumulator,
    PartialProductAccumulator,
    create_accumulator,
    create_partial_product_accumulator,
)

# STE (Straight-Through Estimator) for SNN training
from .core.ste import (
    # Autograd Functions
    STELinearFunction,
    STEEmbeddingFunction,
    STERMSNormFunction,
    STELayerNormFunction,
    STEExpFunction,
    STESigmoidFunction,
    STETanhFunction,
    STESiLUFunction,
    STEGELUFunction,
    STESoftmaxFunction,
    STEReLUFunction,
    # Convenience wrapper functions
    ste_linear,
    ste_embedding,
    ste_rmsnorm,
    ste_layernorm,
    ste_exp,
    ste_sigmoid,
    ste_tanh,
    ste_silu,
    ste_gelu,
    ste_softmax,
    ste_relu,
)

# 转换工具函数
from .encoding.converters import (
    float_to_fp8_bits, fp8_bits_to_float,
    float16_to_pulse, pulse_to_float16,
    float32_to_pulse, pulse_to_float32,
    float64_to_pulse, pulse_to_float64,
    float32_to_bits, bits_to_float32,
    float64_to_bits, bits_to_float64,
    float_to_pulse, pulse_to_bits
)

# 神经元模板 (用于物理仿真)
from .core.logic_gates import SimpleLIFNode, _create_neuron

# 基础门电路
from .core.logic_gates import *
from .core.vec_logic_gates import (
    VecAND, VecOR, VecNOT, VecXOR, VecMUX,
    VecORTree, VecANDTree,
    VecHalfAdder, VecFullAdder,
    VecAdder, VecSubtractor, VecComparator
)

# 辅助功能模块
from .encoding.decimal_scanner import DecimalScanner
from .core.sign_bit import SignBitNode
from .core.dynamic_if import DynamicThresholdIFNode

# 编码器/解码器
from .encoding.floating_point import PulseFloatingPointEncoder
from .encoding.pulse_decoder import (
    PulseFloatingPointDecoder,
    PulseFP16Decoder,
    PulseFP32Decoder
)

# FP8 模块
from .arithmetic.fp8.fp8_mul import SpikeFP8Multiplier
from .arithmetic.fp8.fp8_mul_multi import SpikeFP8Multiplier_MultiPrecision
from .arithmetic.fp8.fp8_adder_spatial import SpikeFP8Adder_Spatial
from .linear.fp8.fp8_linear_multi import SpikeFP8Linear_MultiPrecision
from .arithmetic.fp8.fp8_mul_to_fp32 import SpikeFP8MulToFP32
from .activation.fp8.fp8_relu import SpikeFP8ReLU, SpikeFP32ReLU, SpikeFP64ReLU

# FP16 模块
from .arithmetic.fp16.fp16_adder import SpikeFP16Adder
from .arithmetic.fp16.fp16_components import FP8ToFP16Converter, FP16ToFP8Converter
from .arithmetic.fp16.fp16_mul_to_fp32 import FP16ToFP32Converter, SpikeFP16MulToFP32
from .arithmetic.fp16.fp16_mul_multi import SpikeFP16Multiplier_MultiPrecision
from .linear.fp16.fp16_linear import SpikeFP16Linear_MultiPrecision

# FP32 模块
from .arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from .arithmetic.fp32.fp32_div import SpikeFP32Divider
from .arithmetic.fp32.fp32_adder import SpikeFP32Adder
from .linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision

# 向后兼容别名
SpikeFP32Linear = SpikeFP32Linear_MultiPrecision
from .activation.fp32.fp32_exp import SpikeFP32Exp
from .arithmetic.fp32.fp32_sqrt import SpikeFP32Sqrt
from .arithmetic.fp32.fp32_recip import SpikeFP32Reciprocal
from .activation.fp32.fp32_sigmoid import SpikeFP32Sigmoid
from .activation.fp32.fp32_silu import SpikeFP32SiLU
from .activation.fp32.fp32_softmax import SpikeFP32Softmax
from .activation.fp32.fp32_gelu import SpikeFP32GELU, SpikeFP32GELUExact
from .activation.fp32.fp32_tanh import SpikeFP32Tanh
from .normalization.fp32.fp32_layernorm import SpikeFP32LayerNorm
from .normalization.fp32.fp32_rmsnorm import SpikeFP32RMSNormFullFP64
from .linear.fp32.fp32_embedding import SpikeFP32Embedding
from .arithmetic.fp32.fp32_components import (
    FP8ToFP32Converter,
    FP32ToFP8Converter,
    FP32ToFP16Converter
)

# FP64 模块
from .arithmetic.fp64.fp64_adder import SpikeFP64Adder
from .arithmetic.fp64.fp64_mul import SpikeFP64Multiplier
from .arithmetic.fp64.fp64_div import SpikeFP64Divider
from .arithmetic.fp64.fp64_sqrt import SpikeFP64Sqrt
from .activation.fp64.fp64_exp import (
    SpikeFP64Exp,
    SpikeFP32ExpHighPrecision,
    SpikeFP32SigmoidFullFP64,
    SpikeFP32SiLUFullFP64,
    SpikeFP32SoftmaxFullFP64
)
from .arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
from .trigonometry.fp64.fp64_sincos import SpikeFP64Sin, SpikeFP64Cos, SpikeFP64SinCos

# FP32 三角函数
from .trigonometry.fp32.fp32_sincos import SpikeFP32Sin, SpikeFP32Cos, SpikeFP32SinCos

# 旋转位置编码 (RoPE)
from .attention.rope import (
    SpikeRoPE_MultiPrecision,
    SpikeFP32RoPE,
    SpikeFP16RoPE,
    SpikeFP8RoPE
)

# 多头注意力机制 (MultiHeadAttention)
from .attention.attention import (
    SpikeMultiHeadAttention,
    SpikeFP8MultiHeadAttention,
    SpikeFP16MultiHeadAttention,
    SpikeFP32MultiHeadAttention,
)

# 纯脉冲优化器
from .optim import PulseSGD, PulseSGDWithMomentum
