# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MofNeuroSim is a **100% pure Spiking Neural Network (SNN)** implementation of IEEE floating-point arithmetic for MOF chip simulation. Unlike traditional neuromorphic frameworks, this achieves **bit-exact** results (0 ULP error) by implementing all computations entirely in the pulse domain using Integrate-and-Fire (IF) neurons.

**Key Constraint**: Traditional Python arithmetic (`+`, `-`, `*`) is forbidden in computation paths. All operations must use SNN gates (ANDGate, ORGate, NOTGate, XORGate, etc.).

---

## ⚠️ ABSOLUTE BASELINE (Highest Priority)

**Never Distort User Instructions**: Understand the user's true intent precisely. Do not arbitrarily reinterpret, simplify, or substitute the user's suggestions. This is the fundamental baseline for an AI tool.

- If uncertain about the user's meaning, **ASK FOR CLARIFICATION FIRST** instead of making assumptions
- Do not replace the user's suggestions with what you think is a "better" approach
- Do not act on your own interpretation after the user has clearly expressed their intent
- The user's original words and intent take precedence over all speculation

---

## Strict Constraints (MUST NOT VIOLATE)

1. **No Batch Verification**: Never use batch commands (grep, find, etc.) to verify code compliance. Read each file individually and provide feedback only after actually reading and understanding the code.

2. **Vectorization Required**: All components must be vectorized for performance. Loops are **forbidden** unless the next bit's computation depends on the previous bit (e.g., ripple carry). Use parallel tensor operations instead of Python loops.
   ```python
   # ❌ FORBIDDEN - Loop over bits
   for i in range(32):
       result[i] = gate(a[i], b[i])

   # ✅ REQUIRED - Vectorized
   result = gate(a, b)  # Process all bits in parallel
   ```

3. **Pure SNN is Non-Negotiable**: All implementations MUST use pulse neuron gates (ANDGate, ORGate, XORGate, etc.). This is the absolute baseline - never ask whether to follow this constraint, just follow it.

4. **Sequential Task Completion**: When executing a plan, NEVER skip ahead to later tasks. Each task must be 100% completed and verified compliant before starting the next one. No exceptions.

5. **No Unsolicited Git Push**: NEVER push to GitHub unless explicitly requested by the user. Commits can be created locally, but `git push` requires explicit user permission.

6. **Extend Existing Classes, Don't Create Demos**: When adding new features, MUST modify existing framework classes. Do NOT create simplified demo implementations for convenience - this causes fragmentation from the main framework. Always integrate new functionality into the existing architecture.
   ```python
   # ❌ FORBIDDEN - Creating standalone demo
   class SimpleFP32Adder_Demo(nn.Module):  # 脱节的简化实现
       ...

   # ✅ REQUIRED - Extend existing class
   class SpikeFP32Adder(nn.Module):  # 在现有类上添加功能
       def new_feature(self, ...):
           ...
   ```

7. **SpikeMode Mechanism - No Manual Reset in forward()**: Gate-based components must NEVER call manual `.reset()` inside `forward()` methods. The SpikeMode system automatically handles reset based on the current mode. Manual reset breaks TEMPORAL mode (训练模式).
   ```python
   # ❌ FORBIDDEN - Manual reset in gate-based forward()
   def forward(self, a, b):
       self.reset()  # 破坏 TEMPORAL 模式!
       self.gate.reset()  # 破坏 TEMPORAL 模式!
       return self.gate(a, b)

   # ✅ REQUIRED - Let SpikeMode handle reset automatically
   def forward(self, a, b):
       return self.gate(a, b)  # 内部门电路根据 SpikeMode 自动决定是否 reset
   ```

   **Why this matters**:
   - **BIT_EXACT mode** (default): Gates auto-reset before each forward() - for inference/verification
   - **TEMPORAL mode**: Gates preserve membrane potential residuals - for training/temporal dynamics
   - Manual reset forces BIT_EXACT behavior regardless of mode, breaking training capabilities

   **Exception - Temporal Boundary Components**: The `PulseFloatingPointEncoder` is a special case. It uses temporal dynamics WITHIN a single forward() call (scanning through time steps to extract bits). Its internal temporal state machine (binary_scanner, exp_generator, delay_nodes) MUST be reset at the START of each forward() for functional correctness. This is NOT about SpikeMode - it's about initializing the temporal scan:
   ```python
   # ✅ CORRECT for temporal boundary components (Encoder only)
   def forward(self, x):
       # Reset temporal state machine for new encoding
       self.binary_scanner.reset()
       self.exp_generator.reset()
       for d in self.delay_nodes: d.reset()
       # ... temporal scanning loop ...
   ```

8. **Test Coverage - Random + Boundary Values**: All test cases MUST use a combination of random values and boundary values to ensure comprehensive coverage. Testing only fixed values may miss edge cases (e.g., negative numbers, zero, denormals, infinity).
   ```python
   # ❌ FORBIDDEN - Only fixed test values
   test_values = [0.0, 1.0, 2.0, 3.14]  # 可能遗漏负数、边界情况

   # ✅ REQUIRED - Random + Boundary values
   boundary_values = [
       0.0, -0.0,                    # 零
       1.0, -1.0,                    # 单位值
       float('inf'), float('-inf'), # 无穷大
       float('nan'),                 # NaN
       1e-38, -1e-38,               # 接近下溢
       1e38, -1e38,                 # 接近上溢
   ]
   random_values = torch.randn(100).tolist()  # 随机值
   test_values = boundary_values + random_values
   ```

   **Why this matters**: The sin(-1.0) bug was only caught because the test included negative values. Boundary values catch:
   - Sign handling errors (正负数)
   - Overflow/underflow behavior (溢出/下溢)
   - Special value handling (NaN, Inf, denormals)
   - Zero and near-zero edge cases

9. **GPU Usage in Tests**: All test files MUST use GPU when available. This ensures that components are tested in the same environment they will be deployed. Use the standard device selection pattern:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Move modules to device
   encoder = PulseFloatingPointEncoder().to(device)
   adder = SpikeFP32Adder().to(device)

   # Move tensors to device
   x = torch.randn(batch_size, features).to(device)
   ```

   **Why this matters**:
   - GPU execution may have different numerical behavior than CPU
   - Memory management and tensor operations differ between devices
   - Tests must validate functionality on the target deployment hardware

## Build & Test Commands

```bash
# Install dependencies
pip install torch numpy

# Install package
pip install -e .

# Run core test suite
python tests/test_suite.py

# Run specific test category
python tests/test_suite.py --only logic_gates     # options: logic_gates, arithmetic, encoder_decoder, multiplier, linear

# Run 100% alignment verification
python tests/test_all_precision_alignment.py

# Run individual component tests
python tests/test_fp8_mul.py
python tests/test_fp32_adder.py
python tests/test_logic_gates.py

# Run physical simulation tests (LIF neurons)
python tests/test_robustness.py
```

## Architecture

### Computation Flow
```
Float Input → [Encoder] → Pulse Sequence → [SNN Gates] → Pulse Sequence → [Decoder] → Float Output
```

### Hierarchical Structure
- **Level 0**: IF/LIF Neurons (threshold + reset)
- **Level 1**: Logic Gates (AND, OR, NOT, XOR, MUX) - `atomic_ops/logic_gates.py`
- **Level 2**: Arithmetic Units (Adder, Multiplier) - `atomic_ops/vec_logic_gates.py`
- **Level 3**: Floating-Point Operators - `atomic_ops/fp{8,16,32,64}_*.py`
- **Level 4**: Neural Network Layers - `atomic_ops/fp*_linear*.py`, `fp32_layernorm.py`, etc.

### Key Components

**Boundary Components** (Float ↔ Pulse):
- `PulseFloatingPointEncoder` - Converts float to pulse using dynamic threshold IF neurons
- `PulseFloatingPointDecoder` / `PulseFP{16,32}Decoder` - Converts pulse back to float (traditional math allowed here)

**Logic Gates** (fixed thresholds for digital logic):
| Gate | Threshold | Implementation |
|------|-----------|----------------|
| AND  | 1.5       | H(A + B - 1.5) |
| OR   | 0.5       | H(A + B - 0.5) |
| NOT  | 0.5       | H(1 - A - 0.5) |
| XOR  | -         | OR(A,B) - AND(A,B) (3 neurons) |
| MUX  | -         | OR(AND(A,S), AND(B,NOT(S))) |

**Neuron Template System**: All components support `neuron_template` parameter. Default is `SimpleLIFNode` with:
- `DEFAULT_BETA = 1.0 - 1e-7` (near-zero leak, maintains bit-exact results)
- `DEFAULT_MAX_PARAM_SHAPE = (64,)` (预分配64位参数，覆盖 FP8/16/32/64)
- `trainable_threshold=True`, `trainable_beta=True` (trainable parameters enabled by default)

```python
from atomic_ops import ANDGate, SimpleLIFNode, SimpleIFNode, SpikeFP32Adder

# Default: 自动预分配 64 位参数
and_gate = ANDGate()

# Custom LIF with different beta (for physical simulation)
lif_template = SimpleLIFNode(beta=0.9)
and_gate_lif = ANDGate(neuron_template=lif_template)

# 指定更小的预分配形状（节省显存）
gate_8bit = ANDGate(max_param_shape=(8,))
```

**Preload Slice Mechanism (预分配切片机制)**:

**工作原理**：
1. `__init__()` 时预分配参数（默认 64 位）
2. `forward()` 时根据实际输入位宽**切片**使用：`param[..., :input_bits]`
3. `reset()` 时保留参数，只重置膜电位 `self.v = None`

```python
# 默认预分配 64 位，支持所有精度
gate = ANDGate()
result = gate(a_32bit, b_32bit)  # 使用 param[:32]

# 传播到子组件
class MyComponent(nn.Module):
    def __init__(self, max_param_shape=None):
        self.gate = ANDGate(max_param_shape=max_param_shape)
```

**优势**：
- 避免动态内存分配导致的显存碎片
- `reset()` 不销毁参数，避免反复创建/销毁开销
- 默认覆盖所有精度 (FP8/16/32/64)

### Reset Mechanisms
- **All components use soft reset** (V = V - V_threshold) by default
- **Encoder** (`DynamicThresholdIFNode`): Soft reset preserves residual for multi-bit extraction
- **Logic Gates**: Soft reset ensures consistent behavior across IF/LIF neuron templates

### SpikeMode - Dual Mode Control System
The `SpikeMode` system provides flexible control over neuron state management:

```python
from atomic_ops import SpikeMode, VecAND

# 1. Default: BIT_EXACT mode (推理/验证)
gate = VecAND()
result = gate(a, b)  # Gates auto-reset before forward, no residual

# 2. TEMPORAL mode (训练/时间动力学)
SpikeMode.set_global_mode(SpikeMode.TEMPORAL)
for epoch in range(100):
    output = model(input)  # Residuals accumulate, enabling temporal dynamics

# 3. Context manager (类似 torch.no_grad())
with SpikeMode.temporal():
    loss = model(input)
    loss.backward()  # Training with temporal residuals

with SpikeMode.bit_exact():
    result = model(input)  # Bit-exact inference

# 4. Instance-level override (特定组件固定行为)
gate = VecAND(mode=SpikeMode.TEMPORAL)  # This gate always preserves residuals
```

**Mode Priority**: Instance mode > Context mode > Global mode

**Key API**:
- `SpikeMode.set_global_mode(mode)` - Set global default
- `SpikeMode.get_mode()` - Get current effective mode
- `SpikeMode.temporal()` / `SpikeMode.bit_exact()` - Context managers
- `SpikeMode.should_reset(instance_mode)` - Used internally by gates

## Code Organization

- `atomic_ops/` - Core SNN components
  - `logic_gates.py` - Basic gates (AND, OR, NOT, XOR, MUX) and neuron templates
  - `vec_logic_gates.py` - Vectorized parallel gates for batch operations
  - `neurons.py` - IF/LIF neuron implementations
  - `floating_point.py` - FP8 encoder
  - `pulse_decoder.py` - Multi-precision decoders
  - `fp{8,16,32,64}_*.py` - Precision-specific arithmetic modules
  - `dual_rail/` - Dual-rail logic implementation

- `tests/` - Comprehensive test suite
  - `test_suite.py` - Main test runner
  - `test_all_precision_alignment.py` - 100% alignment verification

- `models/` - Example SNN inference models

## Supported Precisions

| Precision | Bit-exact | Key Modules |
|-----------|-----------|-------------|
| FP8 (E4M3) | Yes | `SpikeFP8Multiplier`, `SpikeFP8Adder_Spatial`, `SpikeFP8Linear_*` |
| FP16 | Yes | `SpikeFP16Adder`, `FP8ToFP16Converter` |
| FP32 | Yes | Full suite: adder, mul, div, sqrt, exp, sigmoid, tanh, GELU, softmax, LayerNorm, RMSNorm |
| FP64 | Yes | `SpikeFP64Adder`, `SpikeFP64Multiplier`, `SpikeFP64Divider`, `SpikeFP64Sqrt`, `SpikeFP64Exp` |

FP64 is also used internally for high-precision FP32 variants: `SpikeFP32ExpHighPrecision`, `SpikeFP32SigmoidFullFP64`, `SpikeFP32SiLUFullFP64`, `SpikeFP32SoftmaxFullFP64`, `SpikeFP32RMSNormFullFP64`.

## Converter Utilities

The `atomic_ops/converters.py` module provides boundary conversion functions:
```python
from atomic_ops import (
    float_to_fp8_bits, fp8_bits_to_float,     # FP8
    float16_to_pulse, pulse_to_float16,       # FP16
    float32_to_pulse, pulse_to_float32,       # FP32
    float64_to_pulse, pulse_to_float64,       # FP64
)
```

Precision converters for in-network conversion:
```python
from atomic_ops import (
    FP8ToFP16Converter, FP16ToFP8Converter,   # FP8 ↔ FP16
    FP8ToFP32Converter, FP32ToFP8Converter,   # FP8 ↔ FP32
    FP16ToFP32Converter, FP32ToFP16Converter, # FP16 ↔ FP32
    FP32ToFP64Converter, FP64ToFP32Converter, # FP32 ↔ FP64
)
```

## Device Support

All modules support both CPU and CUDA:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = PulseFloatingPointEncoder().to(device)
adder = SpikeFP32Adder().to(device)
```

## Development Guidelines

1. **Pure SNN Constraint**: Never use Python arithmetic in computation paths. Use:
   - `not_gate(x)` instead of `1 - x`
   - `and_gate(a, b)` instead of `a * b`
   - `xor_gate(a, b)` instead of `a + b - 2*a*b`

2. **Boundary-Only Encoding/Decoding**: All intermediate components MUST accept and output **pulse sequences only**. Float ↔ Pulse conversion is ONLY allowed at system boundaries:
   ```
   [User Float Input] → Encoder (边界) → [Pulse] → SNN组件A → [Pulse] → SNN组件B → [Pulse] → Decoder (边界) → [User Float Output]
                        ↑                                                                      ↑
                     仅此处允许                                                              仅此处允许
                   float→pulse                                                            pulse→float
   ```

   **Boundary components** (allowed to use Python math for conversion):
   - `PulseFloatingPointEncoder` / `float32_to_pulse()` / `float64_to_pulse()` - Float → Pulse
   - `PulseFloatingPointDecoder` / `pulse_to_float32()` / `pulse_to_float64()` - Pulse → Float
   - `FP8ToFP32Converter`, `FP16ToFP32Converter`, etc. - Precision conversion (pulse → pulse, internally uses encoding)

   **Non-boundary components** (pulse in, pulse out, NO float conversion):
   - `SpikeFP32Adder`, `SpikeFP32Multiplier`, `SpikeFP32RoPE`, etc.
   - These components receive pulse tensors and return pulse tensors
   - Internal computations use SNN gates only

   ```python
   # ❌ FORBIDDEN - Float conversion inside non-boundary component
   class SpikeFP32SomeOp(nn.Module):
       def forward(self, x_pulse):
           x_float = pulse_to_float32(x_pulse)  # 禁止！
           result_float = some_computation(x_float)
           return float32_to_pulse(result_float)  # 禁止！

   # ✅ REQUIRED - Pulse in, pulse out
   class SpikeFP32SomeOp(nn.Module):
       def forward(self, x_pulse):
           # 全部使用 SNN 门电路计算
           intermediate = self.snn_gate_a(x_pulse)
           return self.snn_gate_b(intermediate)  # 返回脉冲
   ```

3. **Pulse = Binary**: After encoding, pulses are directly treated as binary bits (0.0 or 1.0)

4. **Precision Alignment**:
   - FP32 accumulation should achieve 100% bit-exact match with PyTorch
   - FP16 accumulation should achieve ~95% alignment
   - FP8 accumulation has inherent rounding limitations

5. **SpikeMode Compliance**: NEVER add manual `.reset()` calls inside `forward()` methods. The SpikeMode system handles this automatically:
   - Base gates (VecAND, VecOR, etc.) check `SpikeMode.should_reset()` internally
   - Higher-level components just call their sub-gates normally
   - User calls `module.reset()` only when explicitly wanting to clear all state (e.g., before a new sequence)
