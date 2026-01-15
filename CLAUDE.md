# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MofNeuroSim is a **100% pure Spiking Neural Network (SNN)** implementation of IEEE floating-point arithmetic for MOF chip simulation. Unlike traditional neuromorphic frameworks, this achieves **bit-exact** results (0 ULP error) by implementing all computations entirely in the pulse domain using Integrate-and-Fire (IF) neurons.

**Key Constraint**: Traditional Python arithmetic (`+`, `-`, `*`) is forbidden in computation paths. All operations must use SNN gates (ANDGate, ORGate, NOTGate, XORGate, etc.).

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

**Neuron Template System**: All components support `neuron_template` parameter to switch between IF (ideal digital) and LIF (physical simulation):
```python
from atomic_ops import ANDGate, SimpleLIFNode, SpikeFP32Adder

# Default IF neurons
and_gate = ANDGate()

# LIF neurons for physical simulation
lif_template = SimpleLIFNode(beta=0.9)
and_gate_lif = ANDGate(neuron_template=lif_template)
adder_lif = SpikeFP32Adder(neuron_template=lif_template)
```

### Reset Mechanisms
- **All components use soft reset** (V = V - V_threshold) by default
- **Encoder** (`DynamicThresholdIFNode`): Soft reset preserves residual for multi-bit extraction
- **Logic Gates**: Soft reset ensures consistent behavior across IF/LIF neuron templates

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

2. **Pulse = Binary**: After encoding, pulses are directly treated as binary bits (0.0 or 1.0)

3. **Precision Alignment**:
   - FP32 accumulation should achieve 100% bit-exact match with PyTorch
   - FP16 accumulation should achieve ~95% alignment
   - FP8 accumulation has inherent rounding limitations

4. **State Management**: Call `.reset()` on stateful modules (accumulators, Linear layers) before processing new sequences to clear membrane potentials
