# SNNTorch: Pure SNN Floating-Point Arithmetic

**100% 纯脉冲神经网络浮点运算库**

基于 Integrate-and-Fire (IF) 神经元实现的完整浮点运算系统，所有计算均在脉冲域内完成。

## ✨ 特性

- 🧠 **100% 纯 SNN**: 所有运算使用 IF 神经元门电路，无传统数值计算
- 🎯 **100% 对齐 PyTorch**: FP32 累加模式与 `nn.Linear` 完全一致
- 📐 **多精度支持**: FP8 E4M3 / FP16 / FP32 累加精度可选
- 🔧 **任意维度**: 编码器/解码器支持任意张量形状
- ⚡ **高效实现**: 基于 SpikingJelly 框架优化

## 🏗️ 架构

```
                    ┌─────────────────────────────────────────┐
                    │           纯 SNN 计算域                  │
                    │                                         │
ANN 浮点输入 ──→ [编码器] ──→ [SNN 运算] ──→ [解码器] ──→ ANN 浮点输出
                    │           ↑                             │
                    │     门电路组成:                          │
                    │     • AND/OR/XOR/NOT 门                 │
                    │     • 半加器/全加器                      │
                    │     • 行波进位加法器                     │
                    │     • 阵列乘法器                         │
                    │     • 桶形移位器                         │
                    └─────────────────────────────────────────┘
```

## 📦 组件

### 边界组件 (ANN ↔ SNN)

| 组件 | 功能 | 输入 → 输出 |
|------|------|-------------|
| `PulseFloatingPointEncoder` | 浮点→脉冲 | `[...]` → `[..., 8]` |
| `PulseFloatingPointDecoder` | 脉冲→浮点 | `[..., 8]` → `[...]` |

### 核心门电路

| 组件 | 公式 | IF 神经元数 |
|------|------|-------------|
| `ANDGate` | `H(A + B - 1.5)` | 1 |
| `ORGate` | `H(A + B - 0.5)` | 1 |
| `XORGate` | `(A + B) - 2×AND(A,B)` | 2 |
| `NOTGate` | `H(1 - A - 0.5)` | 1 |
| `FullAdder` | `S = A⊕B⊕C, Cout = ...` | 7 |

### 浮点运算

| 组件 | 功能 | 精度 |
|------|------|------|
| `SpikeFP8Multiplier` | FP8 × FP8 → FP8 | 8-bit |
| `SpikeFP8Adder_Spatial` | FP8 + FP8 → FP8 | 8-bit |
| `SpikeFP16Adder` | FP16 + FP16 → FP16 | 16-bit |
| `SpikeFP32Adder` | FP32 + FP32 → FP32 | 32-bit |
| `SpikeFP8Linear_MultiPrecision` | 全连接层 | 可选 |

## 🚀 快速开始

### 安装依赖

```bash
pip install torch spikingjelly
```

### 基本使用

```python
import torch
from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder,
    PulseFloatingPointDecoder,
    SpikeFP8Linear_MultiPrecision
)

# 1. 创建编码器/解码器
encoder = PulseFloatingPointEncoder()
decoder = PulseFloatingPointDecoder()

# 2. 编码浮点数为脉冲
x = torch.randn(32, 64)  # [batch, features]
x_pulse = encoder(x)      # [32, 64, 8]

# 3. 纯 SNN 计算
linear = SpikeFP8Linear_MultiPrecision(64, 32, accum_precision='fp32')
linear.set_weight_from_float(weight, encoder)
y_pulse = linear(x_pulse)  # [32, 32, 8]

# 4. 解码脉冲为浮点数
y = decoder(y_pulse)       # [32, 32]
```

### 多层网络

```python
# 构建 3 层 SNN 网络
layer1 = SpikeFP8Linear_MultiPrecision(128, 64, accum_precision='fp32')
layer2 = SpikeFP8Linear_MultiPrecision(64, 32, accum_precision='fp32')
layer3 = SpikeFP8Linear_MultiPrecision(32, 10, accum_precision='fp32')

# 前向传播 (全程脉冲域)
x_pulse = encoder(x)
h1 = layer1(x_pulse)
h2 = layer2(h1)
y_pulse = layer3(h2)
y = decoder(y_pulse)
```

## 📊 精度对齐

与 PyTorch `nn.Linear` 的对齐测试结果:

| 累加精度 | 对齐率 | 说明 |
|----------|--------|------|
| FP8 | ~50% | 每步舍入累积误差 |
| FP16 | ~95% | 接近 PyTorch 行为 |
| **FP32** | **100%** | **完全对齐** |

## 🔬 技术细节

### FP8 E4M3 格式

```
[S | E3 E2 E1 E0 | M2 M1 M0]
 ↑   \_________/   \_______/
符号    指数(4位)    尾数(3位)

bias = 7
Normal:    value = (-1)^S × 2^(E-7) × (1 + M/8)
Subnormal: value = (-1)^S × 2^(-6) × (M/8)
```

### 纯 SNN 原则

所有核心运算**仅使用**:
- ✅ IF 神经元 (阈值 + 复位)
- ✅ 兴奋性/抑制性突触权重
- ✅ 脉冲 (0/1) 信号

**禁止使用**:
- ❌ Python 算术运算 (`+`, `-`, `*`, `/`)
- ❌ 比较运算符 (`>`, `<`, `>=`)
- ❌ 高级张量操作 (`.sum()`, `.clamp()`)

## 📁 目录结构

```
SNNTorch/
├── atomic_ops/
│   ├── __init__.py              # 模块导出
│   ├── logic_gates.py           # 基础逻辑门 (IF 神经元)
│   ├── logic_gates_lif.py       # LIF 版本逻辑门 (物理模拟)
│   ├── floating_point.py        # FP8 编码器
│   ├── pulse_decoder.py         # FP8/16/32 解码器
│   ├── fp8_mul.py               # FP8 乘法器
│   ├── fp8_mul_to_fp32.py       # FP8→FP32 高精度乘法器
│   ├── fp8_adder_spatial.py     # FP8 加法器
│   ├── fp16_components.py       # FP8↔FP16 转换器
│   ├── fp16_adder.py            # FP16 加法器
│   ├── fp32_components.py       # FP8↔FP32 转换器
│   ├── fp32_adder.py            # FP32 加法器
│   ├── fp8_linear_multi.py      # 多精度 Linear 层
│   └── ...
├── tests/
│   ├── test_suite.py            # 核心测试套件
│   ├── test_all_precision_alignment.py  # 100% 对齐测试
│   ├── test_robustness.py       # 物理鲁棒性测试
│   └── ...
└── models/
    ├── mnist_fp8_train.py       # MNIST 训练示例
    └── mnist_snn_infer.py       # SNN 推理示例
```

## 🧪 运行测试

### 功能正确性测试

```bash
python SNNTorch/tests/test_all_precision_alignment.py
```

预期输出:
```
✓ FP8 累加:  100% 对齐
✓ FP16 累加: 100% 对齐
✓ FP32 累加: 100% 对齐
```

### 物理鲁棒性测试

模拟真实神经形态硬件的非理想特性：

```bash
python SNNTorch/tests/test_robustness.py
```

**测试内容**:

| 实验 | 参数 | 说明 |
|------|------|------|
| β 扫描 | 0.01 - 1.0 | LIF 神经元膜电位泄漏 |
| σ 扫描 | 0.0 - 1.0 | 输入高斯噪声 |
| 加法器 | 4-bit RCA | 复杂电路鲁棒性 |

**典型结果**:
```
β 扫描: 即使 β=0.01，基本门仍保持 100% 正确率
σ 扫描: σ<0.15 时保持 >99% 准确率
        σ>0.30 时准确率开始显著下降
```

## 📜 许可证

MIT License

## 🙏 致谢

- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - SNN 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**HumanBrain Project** - 探索类脑计算的边界

