"""
FP32 Linear 层 - 100%纯SNN门电路实现
====================================

FP32 输入输出的全连接层，支持 FP32/FP64 中间累加精度。

重要变更 (纯脉冲模式)
--------------------
- 权重以脉冲格式存储 (weight_pulse)
- 训练时 backward 使用纯 SNN 组件
- 完全符合 CLAUDE.md 纯 SNN 约束

架构概述
--------

```
输入脉冲 X[batch, in, 32]    权重脉冲 W[out, in, 32]
        │                           │
        └─────────┬─────────────────┘
                  │
        [FP32×FP32→FP32 乘法器]
                  │
                  ▼
         ┌────────────────────┐
         │   累加器选择       │
         │  ┌──────────────┐  │
         │  │ FP32 加法器  │  │  ← accum_precision='fp32' → 输出FP32
         │  ├──────────────┤  │
         │  │ FP64 加法器  │  │  ← accum_precision='fp64' → 转换 → 输出FP32
         │  └──────────────┘  │
         └────────────────────┘
                  │
                  ▼
         输出脉冲 Y[batch, out, 32]
```

累加精度对比
-----------

| 精度 | 与 PyTorch 对齐 | 相对速度 | 内部位宽 |
|------|-----------------|----------|----------|
| fp32 | 100%            | 快       | 32 位    |
| fp64 | 100%            | 慢       | 64 位    |

**推荐**: 使用 `accum_precision='fp32'` 以获得与 PyTorch 完全一致的结果。

使用示例
--------
```python
from atomic_ops import TrainingMode

# 推理模式 (默认)
linear = SpikeFP32Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    accum_precision='fp32'  # 100% 对齐 PyTorch
)
linear.set_weight_from_float(weight_tensor)
y_pulse = linear(x_pulse)  # 纯 SNN

# 位精确 STE 训练模式
linear = SpikeFP32Linear_MultiPrecision(
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
from atomic_ops.core.reset_utils import reset_children
from atomic_ops.core.accumulator import SequentialAccumulator, ParallelAccumulator
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter


class SpikeFP32Linear_MultiPrecision(nn.Module):
    """FP32 Linear 层 - 纯脉冲权重存储

    Y = X @ W^T，其中 X 和 W 都是 FP32 脉冲编码

    输入输出始终为 FP32，中间累加精度可选择以平衡精度和性能。

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        accum_precision: 中间累加精度，'fp32' / 'fp64'
            - 'fp32': FP32累加 → FP32输出（与PyTorch一致，推荐）
            - 'fp64': FP64累加 → FP32输出（更高精度）
        accum_mode: 累加策略，'sequential' / 'parallel'
            - 'sequential': 顺序累加，O(n)，位精确确定性（默认）
            - 'parallel': 树形归约，O(log n)，快速
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式
            - None (默认): 纯推理模式，权重为 buffer
            - TrainingMode.STE: 位精确 STE 训练，权重为脉冲 Parameter
            - TrainingMode.TEMPORAL: 时间动力学训练 (未来扩展)

    架构:
        输入[FP32] → FP32×FP32→FP32乘法 → 累加[accum_precision] → 输出[FP32]
    """
    def __init__(self, in_features, out_features, accum_precision='fp32',
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

        # FP32 乘法器 (逐元素乘法，共享)
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

        if accum_precision == 'fp64':
            # FP64 累加模式
            # FP32 → FP64 转换器
            self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
            # FP64 加法器
            self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
            # FP64 累加器
            self.fp64_accumulator = AccumulatorClass(self.fp64_adder)
            # FP64 → FP32 输出转换器
            self.output_converter = FP64ToFP32Converter(neuron_template=nt)
        else:
            # FP32 累加模式
            self.fp32_adder = SpikeFP32Adder(neuron_template=nt)
            # FP32 累加器
            self.fp32_accumulator = AccumulatorClass(self.fp32_adder)

        # 脉冲权重
        if TrainingMode.is_ste(self.training_mode):
            # STE 训练模式：权重为 Parameter (脉冲格式，float32)
            # 初始化为零脉冲，需要通过 set_weight_from_float 或 set_weight_pulse 设置
            self._weight_pulse_float = nn.Parameter(
                torch.zeros(out_features, in_features, 32),
                requires_grad=True
            )
            # 脉冲梯度缓存 (用于纯脉冲优化器)
            self.register_buffer('grad_pulse', None)
            self.register_buffer('_weight_pulse_bool', None)  # 未使用，保持一致性
        else:
            # 推理模式：权重为 bool buffer (4x 内存节省)
            self.register_buffer('_weight_pulse_bool', None)
            self._weight_pulse_float = None  # 推理模式不需要 float Parameter

    @property
    def weight_pulse(self):
        """获取脉冲权重 (按需转换为 float)

        推理模式：从 bool buffer 转换为 float
        训练模式：直接返回 float Parameter
        """
        if TrainingMode.is_ste(self.training_mode):
            return self._weight_pulse_float
        else:
            if self._weight_pulse_bool is not None:
                return self._weight_pulse_bool.float()
            return None

    @weight_pulse.setter
    def weight_pulse(self, value):
        """设置脉冲权重"""
        if TrainingMode.is_ste(self.training_mode):
            if isinstance(value, nn.Parameter):
                self._weight_pulse_float = value
            else:
                with torch.no_grad():
                    self._weight_pulse_float.copy_(value)
        else:
            # 推理模式：存储为 bool (4x 内存节省)
            if value is not None:
                self._weight_pulse_bool = (value > 0.5).bool()
            else:
                self._weight_pulse_bool = None

    def set_weight_from_float(self, weight_float):
        """将 float 权重转换为 FP32 脉冲

        这是边界操作：在系统初始化时将外部 float 权重编码为脉冲。

        Args:
            weight_float: [out_features, in_features] 权重张量
        """
        from atomic_ops.encoding.converters import float32_to_pulse
        assert weight_float.shape == (self.out_features, self.in_features)

        weight_pulse = float32_to_pulse(weight_float, device=weight_float.device)

        if TrainingMode.is_ste(self.training_mode):
            # 训练模式：更新 Parameter (保持 float)
            with torch.no_grad():
                self._weight_pulse_float.copy_(weight_pulse)
        else:
            # 推理模式：存储为 bool (4x 内存节省)
            self._weight_pulse_bool = (weight_pulse > 0.5).bool()

    def set_weight_pulse(self, weight_pulse):
        """直接设置脉冲权重

        Args:
            weight_pulse: [out_features, in_features, 32] 脉冲权重
        """
        assert weight_pulse.shape == (self.out_features, self.in_features, 32)

        if TrainingMode.is_ste(self.training_mode):
            with torch.no_grad():
                self._weight_pulse_float.copy_(weight_pulse)
        else:
            # 推理模式：存储为 bool (4x 内存节省)
            self._weight_pulse_bool = (weight_pulse > 0.5).bool()

    def get_weight_float(self):
        """将脉冲权重解码为 float (边界操作，用于检查/导出)"""
        from atomic_ops.encoding.converters import pulse_to_float32
        return pulse_to_float32(self.weight_pulse)

    def pulse_parameters(self):
        """返回脉冲格式的可训练参数 (用于纯脉冲优化器)"""
        if TrainingMode.is_ste(self.training_mode):
            yield self.weight_pulse
        else:
            return iter([])

    def forward(self, x):
        """
        Args:
            x: [..., in_features, 32] 输入 FP32 脉冲
        Returns:
            [..., out_features, 32] 输出 FP32 脉冲（所有模式输出都是FP32）
        """
        assert self.weight_pulse is not None, "需要先调用 set_weight_from_float 或 set_weight_pulse"

        # x: [..., in_features, 32] -> [..., 1, in_features, 32]
        # weight: [out_features, in_features, 32]
        x_expanded = x.unsqueeze(-3)

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # FP32 × FP32 → FP32 逐元素乘法
            # 广播: [..., 1, in_features, 32] × [out_features, in_features, 32]
            #     → [..., out_features, in_features, 32]
            products = self.mul(x_expanded, self.weight_pulse)

            # 累加
            if self.in_features == 1:
                out_pulse = products.squeeze(-2)
            elif self.accum_precision == 'fp64':
                out_pulse = self._fp64_accumulate(products)
            else:
                out_pulse = self._fp32_accumulate(products)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_linear
            # STE: 返回 pulse，backward 使用纯 SNN 组件计算梯度
            return ste_linear(x, self.weight_pulse, out_pulse)

        return out_pulse

    def _fp32_accumulate(self, products_fp32):
        """FP32 累加：FP32累加 → 输出FP32脉冲"""
        # 使用 Accumulator 进行归约，dim=-2 是 in_features 维度
        return self.fp32_accumulator.reduce(products_fp32, dim=-2)

    def _fp64_accumulate(self, products_fp32):
        """FP64 累加：FP32→FP64 → FP64累加 → FP64→FP32 → 输出FP32脉冲"""
        # 转换所有乘积到 FP64
        products_fp64 = self.fp32_to_fp64(products_fp32)

        # 使用 Accumulator 进行归约
        acc = self.fp64_accumulator.reduce(products_fp64, dim=-2)

        # 转换回 FP32 输出
        return self.output_converter(acc)

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)

    def train(self, mode=True):
        """切换训练模式"""
        super().train(mode)
        return self


# 别名，保持向后兼容
SpikeFP32Linear = SpikeFP32Linear_MultiPrecision
