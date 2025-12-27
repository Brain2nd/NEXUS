# SNNTorch 测试目录

## 📁 目录结构

```
tests/
├── README.md                          # 本文件
├── test_suite.py                      # ★ 核心测试套件 (推荐)
├── test_all_precision_alignment.py    # ★ 100% 对齐测试
│
├── 核心组件测试/
│   ├── test_logic_gates.py            # 逻辑门测试
│   ├── test_fp8_encoder.py            # FP8 编码器测试
│   ├── test_fp8_mul.py                # FP8 乘法器测试
│   ├── test_fp8_adder_spatial.py      # FP8 加法器测试
│   ├── test_fp16_adder.py             # FP16 加法器测试
│   ├── test_fp16_converter.py         # FP16 转换器测试
│   ├── test_fp32_components.py        # FP32 组件测试
│   └── test_multi_precision_linear.py # 多精度 Linear 测试
│
├── 端到端测试/
│   ├── test_e2e_linear.py             # 端到端 Linear 测试
│   ├── test_e2e_mnist.py              # MNIST 端到端测试
│   └── test_corner_cases.py           # 边界情况测试
│
└── 调试/开发用 (可删除)/
    ├── debug_*.py                     # 调试脚本
    ├── trace_*.py                     # 追踪脚本
    └── test_find_mismatch*.py         # 问题定位脚本
```

## 🚀 快速开始

### 运行核心测试套件

```bash
# 完整测试
python SNNTorch/tests/test_suite.py

# 只测试逻辑门
python SNNTorch/tests/test_suite.py --only logic_gates

# 只测试 Linear 层
python SNNTorch/tests/test_suite.py --only linear
```

### 运行 100% 对齐测试

```bash
python SNNTorch/tests/test_all_precision_alignment.py
```

预期输出:
```
FP8 累加:  ✓ 100% 对齐
FP16 累加: ✓ 100% 对齐
FP32 累加: ✓ 100% 对齐
```

## 📊 测试覆盖

### 功能正确性测试

| 组件 | 测试文件 | 覆盖内容 |
|------|----------|----------|
| 逻辑门 | `test_logic_gates.py` | AND/OR/XOR/NOT 真值表 |
| 编码器 | `test_fp8_encoder.py` | 浮点→脉冲转换 |
| 解码器 | `test_pulse_decoder.py` | 脉冲→浮点、任意维度 |
| FP8乘法 | `test_fp8_mul.py` | 穷举/随机测试 |
| FP8→FP32乘法 | `test_fp8_mul_to_fp32.py` | 高精度乘法 |
| FP8加法 | `test_fp8_adder_spatial.py` | 各种数值组合 |
| FP16/32 | `test_fp16_*.py`, `test_fp32_*.py` | 精度转换 |
| Linear | `test_all_precision_alignment.py` | 100% 对齐验证 |

### 物理硬件模拟测试

| 测试 | 文件 | 说明 |
|------|------|------|
| 膜电位泄漏 (β扫描) | `test_robustness.py` | LIF 神经元泄漏特性 |
| 输入噪声 (σ扫描) | `test_robustness.py` | 高斯噪声鲁棒性 |
| LIF 逻辑门 | `logic_gates_lif.py` | 物理模拟组件 |

## ⚠️ 注意事项

1. **GPU 测试**: 大部分测试自动检测 CUDA 可用性
2. **随机种子**: 对齐测试使用固定种子确保可复现
3. **调试文件**: `debug_*` 和 `trace_*` 文件为开发用，可在发布前删除

## 🧹 清理说明

调试/开发用文件已移至 `_debug_archive/` 目录:

```
_debug_archive/
├── debug_*.py           # 调试脚本
├── trace_*.py           # 追踪脚本
├── test_find_mismatch*.py
├── test_*_debug.py
└── ablation_*.py
```

**开源前建议**: 删除整个 `_debug_archive/` 目录

```bash
rm -rf SNNTorch/tests/_debug_archive
```

