# NEXUS

> **License: GPLv3 | Academic Use Only**
>
> For commercial licensing, please contact: zztangbu@bu.edu or tangzhengzheng.ai@gmail.com

![Status](https://img.shields.io/badge/Status-Preprint-orange)
![arXiv](https://img.shields.io/badge/arXiv-2601.21279-b31b1b.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**NEXUS: Bit-Exact ANN-to-SNN Equivalence via Neuromorphic Gate Circuits with Surrogate-Free Training**

Paper: [arXiv:2601.21279](https://arxiv.org/abs/2601.21279) | Code: [https://github.com/Brain2nd/NEXUS](https://github.com/Brain2nd/NEXUS)

---

## Overview

NEXUS achieves **bit-exact ANN-to-SNN equivalence** -- not approximate, but mathematically identical outputs. All arithmetic operations (both linear and nonlinear) are constructed from pure IF neuron logic gates implementing IEEE-754 compliant floating-point arithmetic.

Key results:
- **0.00% accuracy degradation** on models up to LLaMA-2 70B
- **Mean ULP error of 6.19** (outputs differ by ~6 representable floating-point values)
- **27-168,000x energy reduction** on Loihi 2 neuromorphic hardware
- **100% accuracy at all LIF decay factors** (beta from 0.1 to 1.0) -- inherent immunity to membrane leakage
- **>98% gate-level accuracy** under synaptic noise sigma <= 0.2

### Core Idea

```
Float Input --> [Spatial Bit Encoding] --> [IF Neuron Gate Circuits] --> [Decoding] --> Float Output
                  (zero error)             (IEEE-754 compliant)           (lossless)
```

Instead of approximating continuous values with spike rates or timing, we treat each spike as a **binary digit** and construct **digital logic gates** from IF neurons. This transforms SNNs from analog computers into bit-exact digital circuits.

---

## Three Contributions

### 1. Spatial Bit Encoding (Zero Encoding Error)

Direct bit-level mapping between IEEE-754 floating-point representations and parallel spike channels. The 32-bit pattern of an FP32 value maps to 32 parallel spike channels via bit reinterpretation -- zero information loss by construction.

| Encoding Scheme | Time Steps | MSE |
|-----------------|-----------|-----|
| Rate Coding | 32 | 4.46 x 10^4 |
| TTFS | 1024 | 1.03 x 10^-9 |
| **Ours** | **16** | **1.03 x 10^-9** |
| **Ours** | **32** | **1.33 x 10^-14** |

### 2. Neuromorphic Gate Circuits (Exact Arithmetic)

All operations built from IF neurons with carefully designed thresholds:

| Gate | Formula | Threshold | IF Neurons |
|------|---------|-----------|------------|
| AND | IF_{1.5}(a + b) | 1.5 | 1 |
| OR | IF_{0.5}(a + b) | 0.5 | 1 |
| NOT | IF_{1.0}(1.5 - x) | 1.0 | 1 |
| XOR | (a AND NOT b) OR (NOT a AND b) | -- | 5 |
| MUX | (s AND a) OR (NOT s AND b) | -- | 5 |

Hierarchical construction:
```
Level 1: IF Neuron Logic Primitives (AND, OR, NOT)
Level 2: Multi-bit Arithmetic (Full Adder, Ripple-Carry Adder)
Level 3: IEEE-754 FP32 Operations (Adder, Multiplier, Divider, Sqrt)
Level 4: Nonlinear Functions (Exp, Sigmoid, GELU, SiLU, Softmax, RMSNorm)
```

### 3. Surrogate-Free STE Training (Exact Identity Mapping)

Since the forward pass is bit-exact (SNN output = ANN output), the Straight-Through Estimator is not an approximation but a mathematically exact identity mapping. No surrogate gradient functions needed.

---

## Experimental Results

### Component-Level Precision (ULP Metrics)

| Operation | Max ULP | Mean ULP | 0-ULP Rate |
|-----------|---------|----------|------------|
| Linear (Fwd+Bwd) | 4 | 1.20 | 37.5% |
| RMSNorm (Fwd+Bwd) | 1 | 0.30 | 75.0% |
| SiLU (Fwd+Bwd) | 11 | 2.10 | 46.9% |
| Softmax (Fwd+Bwd) | 6 | 0.80 | 87.5% |

### End-to-End Task Performance

| Model | MMLU | HellaSwag | ARC | TruthfulQA |
|-------|------|-----------|-----|------------|
| Qwen3-0.6B (ANN) | 52.30 | 68.20 | 48.70 | 38.90 |
| Qwen3-0.6B (SNN) | 52.30 | 68.20 | 48.70 | 38.90 |
| LLaMA-2 7B (ANN) | 60.04 | 79.13 | 56.14 | 40.95 |
| LLaMA-2 7B (SNN) | 60.04 | 79.13 | 56.14 | 40.95 |
| LLaMA-2 70B (ANN) | 65.40 | 86.90 | 67.20 | 44.90 |
| LLaMA-2 70B (SNN) | 65.40 | 86.90 | 67.20 | 44.90 |

**Identical accuracy** -- 0.00% degradation across all models and benchmarks.

### Energy Efficiency (Loihi 2)

| Component | Savings |
|-----------|---------|
| FP32 Addition | 33x |
| FP32 Multiplication | 27x |
| Exp / Sigmoid / Tanh | 153-187x |
| RMSNorm / LayerNorm | 877-890x |
| Embedding Lookup | 168,000x |
| **Full Transformer Block** | **58x** |

### Robustness Under Physical Non-Idealities

| Test | Result |
|------|--------|
| LIF Leakage (beta = 0.1 to 1.0) | **100% accuracy** -- inherent immunity |
| Synaptic Noise (sigma <= 0.2) | **>98% gate accuracy** |
| Threshold Variation (delta <= 0.10) | **>96% accuracy** |

---

## Quick Start

### Installation

```bash
git clone https://github.com/Brain2nd/NEXUS.git
cd NEXUS
pip install torch numpy
```

### Basic Usage

```python
import torch
from atomic_ops import (
    SpikeFP32Adder,
    float32_to_pulse,
    pulse_to_float32
)

# FP32 addition via pure SNN gate circuits
adder = SpikeFP32Adder()
a = float32_to_pulse(3.14159, device='cpu')
b = float32_to_pulse(2.71828, device='cpu')
result = adder(a, b)
value = pulse_to_float32(result)
print(f"pi + e = {value}")  # Bit-exact IEEE-754 result
```

### Transformer Model

```python
from models import SpikeQwen3ForCausalLM, SpikeQwen3Config
from atomic_ops import pulse_to_float32

config = SpikeQwen3Config(
    vocab_size=1000, hidden_size=64,
    intermediate_size=172, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4, head_dim=16,
)
model = SpikeQwen3ForCausalLM(config).to('cuda')
input_ids = torch.randint(0, 1000, (1, 16), device='cuda')
model.reset()
logits_pulse = model(input_ids)
logits = pulse_to_float32(logits_pulse)
```

---

## Supported Precisions

| Precision | Bit-exact | Key Modules |
|-----------|-----------|-------------|
| FP8 (E4M3) | Yes | Multiplier, Adder, Linear (multi-precision accum) |
| FP16 | Yes | Adder, Linear, Converters (FP8<->FP16) |
| FP32 | Yes | Full suite: add, mul, div, sqrt, exp, sigmoid, tanh, GELU, SiLU, softmax, LayerNorm, RMSNorm, RoPE, Attention |
| FP64 | Yes | Adder, Multiplier, Divider, Sqrt, Exp |

---

## Testing

```bash
# Core test suite
python tests/test_suite.py

# Specific categories
python tests/test_suite.py --only logic_gates
python tests/test_suite.py --only arithmetic
python tests/test_suite.py --only linear

# End-to-end model validation
python tests/test_qwen3_e2e_full.py

# Physical robustness (LIF leakage, noise, threshold variation)
python tests/test_robustness.py

# Full precision alignment verification
python tests/test_all_precision_alignment.py
```

---

## Project Structure

```
NEXUS/
├── atomic_ops/                    # Core SNN components
│   ├── core/                     # IF/LIF neurons, logic gates, vectorized gates
│   ├── encoding/                 # Spatial bit encoding (float <-> pulse)
│   ├── arithmetic/               # FP8/16/32/64 arithmetic circuits
│   ├── activation/               # Exp, sigmoid, tanh, GELU, SiLU, softmax
│   ├── normalization/            # LayerNorm, RMSNorm
│   ├── linear/                   # Linear layers (multi-precision)
│   ├── attention/                # Multi-head attention, RoPE
│   └── trigonometry/             # Sin/Cos (FP32/FP64)
│
├── models/                       # Complete SNN models (Qwen3 architecture)
├── tests/                        # Comprehensive test suite
├── CLAUDE.md                     # Development guidelines
└── README.md
```

---

## Citation

```bibtex
@article{tang2026nexus,
  title     = {NEXUS: Bit-Exact ANN-to-SNN Equivalence via Neuromorphic
               Gate Circuits with Surrogate-Free Training},
  author    = {Tang, Zhengzheng},
  journal   = {arXiv preprint arXiv:2601.21279},
  year      = {2026},
  note      = {7 pages, 6 tables, 2 figures},
}
```

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)** for **Academic Use Only**.

For commercial licensing inquiries, please contact:
- Email: zztangbu@bu.edu
- Email: tangzhengzheng.ai@gmail.com

---

<p align="center">
  <b>NEXUS</b> -- Bit-Exact Neuromorphic Computing
</p>
