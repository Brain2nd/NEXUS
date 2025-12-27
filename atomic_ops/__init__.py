"""
Atomic Ops Package - 100% 纯 SNN 门电路实现的浮点运算模块
"""
# 转换工具函数
from .converters import (
    float_to_fp8_bits, fp8_bits_to_float,
    float32_to_pulse, pulse_to_float32,
    float64_to_pulse, pulse_to_float64,
    float32_to_bits, bits_to_float32,
    float64_to_bits, bits_to_float64,
    float_to_pulse, pulse_to_bits
)

# 基础门电路
from .logic_gates import *
from .vec_logic_gates import (
    VecAND, VecOR, VecNOT, VecXOR, VecMUX,
    VecORTree, VecANDTree,
    VecHalfAdder, VecFullAdder,
    VecAdder, VecSubtractor, VecComparator
)

# 编码器/解码器
from .floating_point import PulseFloatingPointEncoder
from .pulse_decoder import (
    PulseFloatingPointDecoder, 
    PulseFP16Decoder, 
    PulseFP32Decoder
)

# FP8 模块
from .fp8_mul import SpikeFP8Multiplier
from .fp8_adder_spatial import SpikeFP8Adder_Spatial
from .fp8_linear_fast import SpikeFP8Linear_Fast
from .fp8_linear_multi import SpikeFP8Linear_MultiPrecision
from .fp8_mul_to_fp32 import SpikeFP8MulToFP32
from .fp8_relu import SpikeFP8ReLU, SpikeFP32ReLU, SpikeFP64ReLU

# 兼容性别名
SpikeFP8Linear = SpikeFP8Linear_Fast  # 默认使用 Fast 版本

# FP16 模块
from .fp16_adder import SpikeFP16Adder
from .fp16_components import FP8ToFP16Converter, FP16ToFP8Converter

# FP32 模块
from .fp32_mul import SpikeFP32Multiplier
from .fp32_div import SpikeFP32Divider
from .fp32_adder import SpikeFP32Adder
from .fp32_exp import SpikeFP32Exp
from .fp32_sqrt import SpikeFP32Sqrt
from .fp32_recip import SpikeFP32Reciprocal
from .fp32_sigmoid import SpikeFP32Sigmoid
from .fp32_silu import SpikeFP32SiLU
from .fp32_softmax import SpikeFP32Softmax
from .fp32_gelu import SpikeFP32GELU, SpikeFP32GELUExact
from .fp32_tanh import SpikeFP32Tanh
from .fp32_layernorm import SpikeFP32LayerNorm
from .fp32_rmsnorm import SpikeFP32RMSNormFullFP64
from .fp32_embedding import SpikeFP32Embedding
from .fp32_components import (
    FP8ToFP32Converter, 
    FP32ToFP8Converter, 
    FP32ToFP16Converter
)

# FP64 模块
from .fp64_adder import SpikeFP64Adder
from .fp64_mul import SpikeFP64Multiplier
from .fp64_div import SpikeFP64Divider
from .fp64_sqrt import SpikeFP64Sqrt
from .fp64_exp import (
    SpikeFP64Exp, 
    SpikeFP32ExpHighPrecision, 
    SpikeFP32SigmoidFullFP64, 
    SpikeFP32SiLUFullFP64, 
    SpikeFP32SoftmaxFullFP64
)
from .fp64_components import FP32ToFP64Converter, FP64ToFP32Converter
