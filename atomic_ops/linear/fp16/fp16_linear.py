"""
FP16 Linear 层 - 100%纯SNN门电路实现
====================================

FP16 输入输出的全连接层，支持 FP16/FP32 中间累加精度。

架构概述
--------

```
输入脉冲 X[batch, in, 16]    权重脉冲 W[out, in, 16]
        │                           │
        └─────────┬─────────────────┘
                  │
        [FP16×FP16→FP32 乘法器] (无舍入)
                  │
                  ▼
         ┌────────────────────┐
         │   累加器选择       │
         │  ┌──────────────┐  │
         │  │ FP16 加法器  │  │  ← accum_precision='fp16' → 转换 → 输出FP16
         │  ├──────────────┤  │
         │  │ FP32 加法器  │  │  ← accum_precision='fp32' → 转换 → 输出FP16
         │  └──────────────┘  │
         └────────────────────┘
                  │
                  ▼
         输出脉冲 Y[batch, out, 16]
```

累加精度对比
-----------

| 精度 | 与 PyTorch 对齐 | 相对速度 | 内部位宽 |
|------|-----------------|----------|----------|
| fp16 | ~95%            | 快       | 16 位    |
| fp32 | 100%            | 慢       | 32 位    |

**推荐**: 使用 `accum_precision='fp32'` 以获得与 PyTorch 完全一致的结果。

使用示例
--------
```python
# 推理模式 (默认)
linear = SpikeFP16Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    accum_precision='fp32'  # 100% 对齐 PyTorch
)
linear.set_weight_from_float(weight_tensor)
y_pulse = linear(x_pulse)  # 纯 SNN

# 训练模式
linear = SpikeFP16Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    trainable=True  # 启用 STE 训练
)
linear.train()
optimizer = torch.optim.Adam([linear.weight_float], lr=1e-4)
```

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn

from atomic_ops.arithmetic.fp16.fp16_mul_to_fp32 import SpikeFP16MulToFP32
from atomic_ops.arithmetic.fp16.fp16_adder import SpikeFP16Adder
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp32.fp32_components import FP32ToFP16Converter


class SpikeFP16Linear_MultiPrecision(nn.Module):
    """FP16 Linear 层 - 支持不同中间累加精度

    Y = X @ W^T，其中 X 和 W 都是 FP16 脉冲编码

    输入输出始终为 FP16，中间累加精度可选择以平衡精度和性能。

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        accum_precision: 中间累加精度，'fp16' / 'fp32'
            - 'fp32': FP32累加 → FP16输出（最高精度，推荐）
            - 'fp16': FP16累加 → FP16输出（较快但有精度损失）
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式
            - False (默认): 纯推理模式，权重为 buffer
            - True: 训练模式，权重为 Parameter，使用 STE 反向传播

    架构:
        输入[FP16] → FP16×FP16→FP32乘法 → 累加[accum_precision] → 转换 → 输出[FP16]
    """
    def __init__(self, in_features, out_features, accum_precision='fp32',
                 neuron_template=None, trainable=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.accum_precision = accum_precision
        self.trainable = trainable
        nt = neuron_template

        # 所有模式都使用 FP32 乘法器（避免乘法阶段舍入）
        self.mul_fp32 = SpikeFP16MulToFP32(neuron_template=nt)

        if accum_precision == 'fp32':
            # FP32 累加模式 (单实例，动态扩展机制支持复用)
            self.fp32_adder = SpikeFP32Adder(neuron_template=nt)
            # 输出转换：FP32 → FP16
            self.output_converter = FP32ToFP16Converter(neuron_template=nt)
        else:
            # FP16 累加模式 (单实例，动态扩展机制支持复用)
            self.fp32_to_fp16_converter = FP32ToFP16Converter(neuron_template=nt)
            self.fp16_adder = SpikeFP16Adder(neuron_template=nt)

        # 脉冲权重缓存 (推理用)
        self.register_buffer('weight_pulse', None)

        # 浮点权重 (训练用)
        if trainable:
            self.weight_float = nn.Parameter(
                torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight_float, a=5**0.5)
            self._weight_dirty = True
        else:
            self.register_buffer('weight_float', None)
            self._weight_dirty = False

    def _sync_weight_pulse(self):
        """同步 float 权重到 pulse 缓存"""
        if self.trainable and self._weight_dirty:
            from atomic_ops.encoding.converters import float16_to_pulse
            self.weight_pulse = float16_to_pulse(
                self.weight_float.data,
                device=self.weight_float.device
            )
            self._weight_dirty = False

    def set_weight_from_float(self, weight_float):
        """将 float 权重转换为 FP16 脉冲

        Args:
            weight_float: [out_features, in_features] 权重张量
        """
        from atomic_ops.encoding.converters import float16_to_pulse
        assert weight_float.shape == (self.out_features, self.in_features)

        if self.trainable:
            # 训练模式：更新 Parameter
            with torch.no_grad():
                self.weight_float.copy_(weight_float)
            self._weight_dirty = True
        else:
            # 推理模式：直接设置 buffer
            weight_pulse = float16_to_pulse(weight_float, device=weight_float.device)
            self.weight_pulse = weight_pulse

    def forward(self, x):
        """
        Args:
            x: [..., in_features, 16] 输入 FP16 脉冲
        Returns:
            [..., out_features, 16] 输出 FP16 脉冲（所有模式输出都是FP16）
        """
        # 同步权重 (如果需要)
        self._sync_weight_pulse()
        assert self.weight_pulse is not None, "需要先调用 set_weight_from_float"

        # 扩展输入以进行广播乘法
        # x: [..., in_features, 16] -> [..., 1, in_features, 16]
        # weight: [out_features, in_features, 16]
        x_expanded = x.unsqueeze(-3)

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # FP16 × FP16 → FP32 逐元素乘法
            products_fp32 = self.mul_fp32(x_expanded, self.weight_pulse)

            if self.accum_precision == 'fp32':
                # FP32 模式：FP32累加 → FP32→FP16 → 输出FP16脉冲
                if self.in_features == 1:
                    out_pulse = self.output_converter(products_fp32.squeeze(-2))
                else:
                    out_pulse = self._fp32_accumulate(products_fp32)
            else:
                # FP16 模式：FP32转FP16 → FP16累加 → 输出FP16脉冲
                if self.in_features == 1:
                    out_pulse = self.fp32_to_fp16_converter(products_fp32.squeeze(-2))
                else:
                    out_pulse = self._fp16_accumulate(products_fp32)

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_linear
            return ste_linear(x, self.weight_float, out_pulse)

        return out_pulse

    def _fp32_accumulate(self, products_fp32):
        """FP32 累加：FP32累加 → FP32→FP16 → 输出FP16脉冲"""
        # 第一个乘积
        acc = products_fp32[..., 0, :]

        # 逐个累加（单实例复用）
        for i in range(1, self.in_features):
            acc = self.fp32_adder(acc, products_fp32[..., i, :])

        # 转换为FP16输出
        return self.output_converter(acc)

    def _fp16_accumulate(self, products_fp32):
        """FP16 累加：FP32转FP16 → FP16累加 → 输出FP16脉冲"""
        # 向量化转换
        products_fp16 = self.fp32_to_fp16_converter(products_fp32)

        # 顺序累加（单实例复用）
        acc = products_fp16[..., 0, :]
        for i in range(1, self.in_features):
            acc = self.fp16_adder(acc, products_fp16[..., i, :])
        return acc

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()

    def reset(self):
        """向后兼容"""
        self.reset_all()

    def train(self, mode=True):
        """切换训练模式时标记权重需要同步"""
        super().train(mode)
        if mode and self.trainable:
            self._weight_dirty = True
        return self
