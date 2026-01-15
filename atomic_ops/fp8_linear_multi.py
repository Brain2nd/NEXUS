"""
多精度累加 SNN Linear 层 (Multi-Precision SNN Linear Layer)
==========================================================

100% 纯 SNN 门电路实现的全连接层，支持 FP8/FP16/FP32 累加精度。

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
# 创建层
linear = SpikeFP8Linear_MultiPrecision(
    in_features=64, 
    out_features=32,
    accum_precision='fp32'  # 100% 对齐 PyTorch
)

# 设置权重
linear.set_weight_from_float(weight_tensor, encoder)

# 前向传播 (纯脉冲域)
y_pulse = linear(x_pulse)  # FP8: [batch, 32, 8]
                            # FP16: [batch, 32, 16]
                            # FP32: [batch, 32, 32]
```

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn

from .fp8_mul import SpikeFP8Multiplier
from .fp8_adder_spatial import SpikeFP8Adder_Spatial
from .fp16_components import FP8ToFP16Converter, FP16ToFP8Converter
from .fp16_adder import SpikeFP16Adder
from .fp32_components import FP8ToFP32Converter, FP32ToFP8Converter, FP32ToFP16Converter
from .fp32_adder import SpikeFP32Adder
from .fp8_mul_to_fp32 import SpikeFP8MulToFP32


class SpikeFP8Linear_MultiPrecision(nn.Module):
    """FP8 Linear 层 - 支持不同中间累加精度

    Y = X @ W^T，其中 X 和 W 都是 FP8 脉冲编码

    输入为 FP8，输出精度与 accum_precision 一致。

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        accum_precision: 累加精度，'fp8' / 'fp16' / 'fp32'
            - 'fp32': FP32累加 → 输出FP32脉冲[32位]（最高精度，推荐）
            - 'fp16': FP32累加 → FP16输出 → 输出FP16脉冲[16位]（中等精度）
            - 'fp8':  FP8累加 → 输出FP8脉冲[8位]（最快但精度损失大）
        mode: 累加模式，'sequential' 或 'tree'（仅FP8模式支持tree）
        neuron_template: 神经元模板，None 使用默认 IF 神经元

    架构:
        输入[FP8] → FP8×FP8→FP32乘法 → 累加[accum_precision] → 输出[accum_precision位数]
    """
    def __init__(self, in_features, out_features, accum_precision='fp8', mode='sequential', neuron_template=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.accum_precision = accum_precision
        self.mode = mode
        nt = neuron_template

        # FP8 乘法器（共享）
        self.mul = SpikeFP8Multiplier(neuron_template=nt)

        # 所有模式都使用FP32乘法器避免乘法阶段舍入
        self.mul_fp32 = SpikeFP8MulToFP32(neuron_template=nt)

        if accum_precision == 'fp32':
            # FP32 累加模式：每次累加使用独立实例
            self.fp32_adders = nn.ModuleList([
                SpikeFP32Adder(neuron_template=nt) for _ in range(max(1, in_features - 1))
            ])
            # FP32 模式输出 FP32 脉冲，无需转换
        elif accum_precision == 'fp16':
            # FP16 模式：每次累加使用独立实例
            self.fp32_adders = nn.ModuleList([
                SpikeFP32Adder(neuron_template=nt) for _ in range(max(1, in_features - 1))
            ])
            # 输出转换：FP32 → FP16（输出 FP16 脉冲）
            self.fp32_to_fp16 = FP32ToFP16Converter(neuron_template=nt)
        else:
            # FP8 累加模式：向量化转换（单个转换器处理所有元素）
            self.fp32_to_fp8_converter = FP32ToFP8Converter(neuron_template=nt)
            self.num_adders = in_features - 1 if in_features > 1 else 0
            if mode == 'sequential':
                self.adders = nn.ModuleList([
                    SpikeFP8Adder_Spatial(neuron_template=nt) for _ in range(self.num_adders)
                ])
            else:  # tree
                import math
                depth = math.ceil(math.log2(in_features)) if in_features > 1 else 0
                self.tree_adders = nn.ModuleList()
                for d in range(depth):
                    num_adders_at_level = (in_features + (1 << d) - 1) >> (d + 1)
                    self.tree_adders.append(nn.ModuleList([
                        SpikeFP8Adder_Spatial(neuron_template=nt) for _ in range(num_adders_at_level)
                    ]))
        
        self.register_buffer('weight_pulse', None)
        
    def set_weight_from_float(self, weight_float, encoder):
        """将 float 权重转换为 FP8 脉冲
        
        Args:
            weight_float: [out_features, in_features] 权重张量
            encoder: PulseFloatingPointEncoder 实例
        """
        assert weight_float.shape == (self.out_features, self.in_features)
        
        # 编码权重
        weight_pulse = encoder(weight_float)  # [out, in, 1, 8]
        self.weight_pulse = weight_pulse.squeeze(-2)  # [out, in, 8]
        
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
        assert self.weight_pulse is not None, "需要先调用 set_weight_from_float"
        self.reset_all()  # 高层组件统一reset

        # 扩展输入以进行广播乘法
        # x: [..., in_features, 8] -> [..., 1, in_features, 8]
        # weight: [out_features, in_features, 8]
        x_expanded = x.unsqueeze(-3)

        # 所有模式都使用FP32乘法器（避免乘法阶段舍入）
        products_fp32 = self.mul_fp32(x_expanded, self.weight_pulse)

        if self.accum_precision == 'fp32':
            # FP32 模式：FP32累加 → 输出FP32脉冲[32位]
            if self.in_features == 1:
                return products_fp32.squeeze(-2)
            return self._fp32_accumulate(products_fp32)
        elif self.accum_precision == 'fp16':
            # FP16 模式：FP32累加 → FP32→FP16 → 输出FP16脉冲[16位]
            if self.in_features == 1:
                return self.fp32_to_fp16(products_fp32.squeeze(-2))
            return self._fp16_accumulate(products_fp32)
        else:
            # FP8 模式：FP32转FP8 → FP8累加 → 输出FP8脉冲[8位]
            if self.in_features == 1:
                return self.fp32_to_fp8_converter(products_fp32.squeeze(-2))
            return self._fp8_accumulate(products_fp32)
    
    def _fp8_accumulate(self, products_fp32):
        """FP8 累加：FP32转FP8 → FP8累加 → 输出FP8脉冲"""
        # 向量化转换：一次处理所有 in_features 个乘积
        # products_fp32: [..., out_features, in_features, 32]
        # products_fp8:  [..., out_features, in_features, 8]
        products_fp8 = self.fp32_to_fp8_converter(products_fp32)

        if self.mode == 'sequential':
            # 顺序累加（使用独立实例，无需循环内reset）
            acc = products_fp8[..., 0, :]
            for i in range(1, self.in_features):
                acc = self.adders[i-1](acc, products_fp8[..., i, :])
            return acc
        else:
            # 树形累加（使用独立实例，无需循环内reset）
            # 先转为列表以便逐层处理
            current = [products_fp8[..., i, :] for i in range(self.in_features)]
            for level, level_adders in enumerate(self.tree_adders):
                next_level = []
                adder_idx = 0
                i = 0
                while i < len(current):
                    if i + 1 < len(current):
                        summed = level_adders[adder_idx](current[i], current[i+1])
                        next_level.append(summed)
                        adder_idx += 1
                        i += 2
                    else:
                        next_level.append(current[i])
                        i += 1
                current = next_level
            return current[0]

    def _fp16_accumulate(self, products_fp32):
        """FP16 模式：FP32累加 → FP16 输出脉冲[16位]

        使用FP32累加（GPU上FP16 Linear内部也用FP32累加），最后转为FP16
        """
        # FP32 累加（使用独立实例）
        acc = products_fp32[..., 0, :]

        for i in range(1, self.in_features):
            acc = self.fp32_adders[i-1](acc, products_fp32[..., i, :])

        # FP32 → FP16 输出
        return self.fp32_to_fp16(acc)

    def _fp32_accumulate(self, products_fp32):
        """FP32 累加：FP32累加 → 输出FP32脉冲[32位]

        输入是 FP32 脉冲（由 FP8MulToFP32 产生）
        """
        # 第一个乘积
        acc = products_fp32[..., 0, :]

        # 逐个累加（使用独立实例，无需循环内reset）
        for i in range(1, self.in_features):
            acc = self.fp32_adders[i-1](acc, products_fp32[..., i, :])

        # 直接返回 FP32 脉冲
        return acc

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()

    def reset(self):
        """向后兼容"""
        self.reset_all()

