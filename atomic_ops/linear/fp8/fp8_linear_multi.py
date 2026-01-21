"""
多精度累加 SNN Linear 层 (Multi-Precision SNN Linear Layer)
==========================================================

100% 纯 SNN 门电路实现的全连接层，支持 FP8/FP16/FP32 累加精度。

重要变更 (纯脉冲模式)
--------------------
- 权重以脉冲格式存储 (weight_pulse)
- 训练时 backward 使用纯 SNN 组件
- 完全符合 CLAUDE.md 纯 SNN 约束

架构概述
--------

```
输入脉冲 X[batch, in, 8]    权重脉冲 W[out, in, 8]
        │                           │
        └─────────┬─────────────────┘
                  │
        [FP8×FP8→FP32 乘法器] (无舍入)
                  │
                  ▼
         ┌────────────────────┐
         │   累加器选择       │
         │  ┌──────────────┐  │
         │  │ FP8 加法器   │  │  ← accum_precision='fp8' → 输出FP8脉冲[8]
         │  ├──────────────┤  │
         │  │ FP16 加法器  │  │  ← accum_precision='fp16' → 输出FP16脉冲[16]
         │  ├──────────────┤  │
         │  │ FP32 加法器  │  │  ← accum_precision='fp32' → 输出FP32脉冲[32]
         │  └──────────────┘  │
         └────────────────────┘
                  │
                  ▼
         输出脉冲 Y[batch, out, bits]
         bits=8 (FP8) / 16 (FP16) / 32 (FP32)
```

累加精度对比
-----------

| 精度 | 与 PyTorch 对齐 | 相对速度 | 内部位宽 |
|------|-----------------|----------|----------|
| fp8  | ~50%            | 最快     | 8 位     |
| fp16 | ~95%            | 中等     | 16 位    |
| fp32 | 100%            | 最慢     | 32 位    |

**推荐**: 使用 `accum_precision='fp32'` 以获得与 PyTorch `nn.Linear` 完全一致的结果。

数学公式
--------

```
Y[b, o] = Σ(X[b, i] × W[o, i]) for i in [0, in_features)

其中:
- X, W 为 FP8 E4M3 格式
- 乘法阶段: FP8×FP8→FP32 (无舍入，保持完整精度)
- 累加阶段: 根据 accum_precision 选择 FP8/FP16/FP32 累加
- 输出精度: 与 accum_precision 一致
  - accum_precision='fp8'  → 输出 FP8 脉冲 [..., out_features, 8]
  - accum_precision='fp16' → 输出 FP16 脉冲 [..., out_features, 16]
  - accum_precision='fp32' → 输出 FP32 脉冲 [..., out_features, 32]
```

使用示例
--------
```python
from atomic_ops import TrainingMode

# 推理模式 (默认)
linear = SpikeFP8Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    accum_precision='fp32'  # 100% 对齐 PyTorch
)
linear.set_weight_from_float(weight_tensor)
y_pulse = linear(x_pulse)  # 纯 SNN

# 位精确 STE 训练模式
linear = SpikeFP8Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    training_mode=TrainingMode.STE
)
linear.train()
# 使用纯脉冲优化器
from atomic_ops.optim import PulseSGD
optimizer = PulseSGD(linear.pulse_parameters(), lr=0.01)
```

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.core.accumulator import SequentialAccumulator, ParallelAccumulator

from atomic_ops.arithmetic.fp8.fp8_adder_spatial import SpikeFP8Adder_Spatial
from atomic_ops.arithmetic.fp16.fp16_adder import SpikeFP16Adder
from atomic_ops.arithmetic.fp32.fp32_components import FP8ToFP32Converter, FP32ToFP8Converter, FP32ToFP16Converter
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp8.fp8_mul_to_fp32 import SpikeFP8MulToFP32


class SpikeFP8Linear_MultiPrecision(nn.Module):
    """FP8 Linear 层 - 纯脉冲权重存储

    Y = X @ W^T，其中 X 和 W 都是 FP8 脉冲编码

    输入为 FP8，输出精度与 accum_precision 一致。

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        accum_precision: 累加精度，'fp8' / 'fp16' / 'fp32'
            - 'fp32': FP32累加 → 输出FP32脉冲[32位]（最高精度，推荐）
            - 'fp16': FP32累加 → FP16输出 → 输出FP16脉冲[16位]（中等精度）
            - 'fp8':  FP8累加 → 输出FP8脉冲[8位]（最快但精度损失大）
        accum_mode: 累加策略，'sequential' / 'parallel'
            - 'sequential': 顺序累加，O(n)，位精确确定性（默认）
            - 'parallel': 树形归约，O(log n)，快速
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式
            - None (默认): 纯推理模式，权重为 buffer
            - TrainingMode.STE: 位精确 STE 训练，权重为脉冲 Parameter
            - TrainingMode.TEMPORAL: 时间动力学训练 (未来扩展)

    架构:
        输入[FP8] → FP8×FP8→FP32乘法 → 累加[accum_precision] → 输出[accum_precision位数]
    """
    def __init__(self, in_features, out_features, accum_precision='fp8',
                 accum_mode='sequential', neuron_template=None, training_mode=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.accum_precision = accum_precision
        self.accum_mode = accum_mode
        self.training_mode = TrainingMode.validate(training_mode)
        nt = neuron_template

        # 选择 Accumulator 类
        AccumulatorClass = ParallelAccumulator if accum_mode == 'parallel' else SequentialAccumulator

        # 所有模式都使用FP32乘法器避免乘法阶段舍入
        self.mul_fp32 = SpikeFP8MulToFP32(neuron_template=nt)

        if accum_precision == 'fp32':
            # FP32 累加模式
            self.fp32_adder = SpikeFP32Adder(neuron_template=nt)
            self.fp32_accumulator = AccumulatorClass(self.fp32_adder)
        elif accum_precision == 'fp16':
            # FP16 模式：FP32 累加 + FP16 输出
            self.fp32_adder = SpikeFP32Adder(neuron_template=nt)
            self.fp32_accumulator = AccumulatorClass(self.fp32_adder)
            self.fp32_to_fp16 = FP32ToFP16Converter(neuron_template=nt)
        else:
            # FP8 累加模式
            self.fp32_to_fp8_converter = FP32ToFP8Converter(neuron_template=nt)
            self.fp8_adder = SpikeFP8Adder_Spatial(neuron_template=nt)
            self.fp8_accumulator = AccumulatorClass(self.fp8_adder)

        # 脉冲权重
        if TrainingMode.is_ste(self.training_mode):
            # STE 训练模式：权重为 Parameter (脉冲格式)
            # 初始化为零脉冲，需要通过 set_weight_from_float 或 set_weight_pulse 设置
            self.weight_pulse = nn.Parameter(
                torch.zeros(out_features, in_features, 8),
                requires_grad=True
            )
            # 脉冲梯度缓存 (用于纯脉冲优化器)
            self.register_buffer('grad_pulse', None)
        else:
            # 推理模式：权重为 buffer
            self.register_buffer('weight_pulse', None)

    def set_weight_from_float(self, weight_float, encoder=None):
        """将 float 权重转换为 FP8 脉冲

        这是边界操作：在系统初始化时将外部 float 权重编码为脉冲。

        Args:
            weight_float: [out_features, in_features] 权重张量
            encoder: PulseFloatingPointEncoder 实例 (可选，向后兼容)
        """
        from atomic_ops.encoding.converters import float_to_fp8_bits
        assert weight_float.shape == (self.out_features, self.in_features)

        # 使用 encoder 或 float_to_fp8_bits
        if encoder is not None:
            weight_pulse = encoder(weight_float)  # [out, in, 1, 8]
            weight_pulse = weight_pulse.squeeze(-2)  # [out, in, 8]
        else:
            weight_pulse = float_to_fp8_bits(weight_float, device=weight_float.device)

        if TrainingMode.is_ste(self.training_mode):
            # 训练模式：更新 Parameter
            with torch.no_grad():
                self.weight_pulse.copy_(weight_pulse)
        else:
            # 推理模式：直接设置 buffer
            self.weight_pulse = weight_pulse

    def set_weight_pulse(self, weight_pulse):
        """直接设置脉冲权重

        Args:
            weight_pulse: [out_features, in_features, 8] 脉冲权重
        """
        assert weight_pulse.shape == (self.out_features, self.in_features, 8)

        if TrainingMode.is_ste(self.training_mode):
            with torch.no_grad():
                self.weight_pulse.copy_(weight_pulse)
        else:
            self.weight_pulse = weight_pulse

    def get_weight_float(self):
        """将脉冲权重解码为 float (边界操作，用于检查/导出)"""
        from atomic_ops.encoding.converters import fp8_bits_to_float
        return fp8_bits_to_float(self.weight_pulse)

    def pulse_parameters(self):
        """返回脉冲格式的可训练参数 (用于纯脉冲优化器)"""
        if TrainingMode.is_ste(self.training_mode):
            yield self.weight_pulse
        else:
            return iter([])

    def forward(self, x):
        """
        Args:
            x: [..., in_features, 8] 输入 FP8 脉冲
        Returns:
            输出脉冲，位数取决于 accum_precision:
            - 'fp8':  [..., out_features, 8]  FP8 脉冲
            - 'fp16': [..., out_features, 16] FP16 脉冲
            - 'fp32': [..., out_features, 32] FP32 脉冲
        """
        assert self.weight_pulse is not None, "需要先调用 set_weight_from_float 或 set_weight_pulse"

        # 扩展输入以进行广播乘法
        # x: [..., in_features, 8] -> [..., 1, in_features, 8]
        # weight: [out_features, in_features, 8]
        x_expanded = x.unsqueeze(-3)

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # 所有模式都使用FP32乘法器（避免乘法阶段舍入）
            products_fp32 = self.mul_fp32(x_expanded, self.weight_pulse)

            if self.accum_precision == 'fp32':
                # FP32 模式：FP32累加 → 输出FP32脉冲[32位]
                if self.in_features == 1:
                    out_pulse = products_fp32.squeeze(-2)
                else:
                    out_pulse = self._fp32_accumulate(products_fp32)
            elif self.accum_precision == 'fp16':
                # FP16 模式：FP32累加 → FP32→FP16 → 输出FP16脉冲[16位]
                if self.in_features == 1:
                    out_pulse = self.fp32_to_fp16(products_fp32.squeeze(-2))
                else:
                    out_pulse = self._fp16_accumulate(products_fp32)
            else:
                # FP8 模式：FP32转FP8 → FP8累加 → 输出FP8脉冲[8位]
                if self.in_features == 1:
                    out_pulse = self.fp32_to_fp8_converter(products_fp32.squeeze(-2))
                else:
                    out_pulse = self._fp8_accumulate(products_fp32)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_linear
            # STE: 返回 pulse，backward 使用纯 SNN 组件计算梯度
            return ste_linear(x, self.weight_pulse, out_pulse)

        return out_pulse

    def _fp8_accumulate(self, products_fp32):
        """FP8 累加：FP32转FP8 → FP8累加 → 输出FP8脉冲"""
        # 向量化转换
        products_fp8 = self.fp32_to_fp8_converter(products_fp32)
        # 使用 Accumulator 进行归约
        return self.fp8_accumulator.reduce(products_fp8, dim=-2)

    def _fp16_accumulate(self, products_fp32):
        """FP16 模式：FP32累加 → FP16 输出脉冲[16位]"""
        # 使用 Accumulator 进行 FP32 归约
        acc = self.fp32_accumulator.reduce(products_fp32, dim=-2)
        # FP32 → FP16 输出
        return self.fp32_to_fp16(acc)

    def _fp32_accumulate(self, products_fp32):
        """FP32 累加：FP32累加 → 输出FP32脉冲[32位]"""
        # 使用 Accumulator 进行归约
        return self.fp32_accumulator.reduce(products_fp32, dim=-2)

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()

    def reset(self):
        """向后兼容"""
        self.reset_all()

    def train(self, mode=True):
        """切换训练模式"""
        super().train(mode)
        return self
