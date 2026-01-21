"""
Accumulator 模块 - 可插拔的归约策略
====================================

提供不同的累加策略，用于 Linear、LayerNorm、Softmax 等组件的求和操作。

策略对比
--------
| 策略 | 串行步骤 | 精度特性 | 适用场景 |
|------|---------|---------|---------|
| Sequential | O(n) | 位精确，确定性 | 验证、调试 |
| Parallel | O(log n) | 快速，舍入不同 | 生产部署 |

使用示例
--------
```python
from atomic_ops.core.accumulator import SequentialAccumulator, ParallelAccumulator

# 默认顺序累加
acc = SequentialAccumulator(adder)
result = acc.reduce(x, dim=-2)  # [..., N, 32] -> [..., 32]

# 并行树形归约
acc = ParallelAccumulator(adder)
result = acc.reduce(x, dim=-2)
```

作者: MofNeuroSim Project
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Accumulator(ABC, nn.Module):
    """累加器基类

    定义统一的 reduce 接口，子类实现具体的归约策略。

    Args:
        adder: 加法器实例 (SpikeFP32Adder, SpikeFP64Adder 等)
    """

    def __init__(self, adder):
        super().__init__()
        self.adder = adder

    @abstractmethod
    def reduce(self, x, dim):
        """对指定维度进行归约求和

        Args:
            x: 输入张量，形状 [..., N, bits]，其中 N 是要归约的维度大小
            dim: 要归约的维度索引（通常为 -2，即倒数第二维）

        Returns:
            归约后的张量，形状 [..., bits]
        """
        raise NotImplementedError

    def reset(self):
        """重置内部状态"""
        if hasattr(self.adder, 'reset'):
            self.adder.reset()


class SequentialAccumulator(Accumulator):
    """顺序累加器 - O(n) 串行累加

    严格按顺序累加：result = (...((x[0] + x[1]) + x[2]) + ... + x[n-1])

    特点：
    - O(n) 串行步骤
    - 位精确，确定性结果
    - 适用于验证、调试

    Args:
        adder: 加法器实例
    """

    def __init__(self, adder):
        super().__init__(adder)

    def reduce(self, x, dim):
        """顺序累加

        Args:
            x: [..., N, bits]
            dim: 归约维度

        Returns:
            [..., bits]
        """
        # 将目标维度移到倒数第二位，方便索引
        if dim != -2 and dim != x.dim() - 2:
            x = x.transpose(dim, -2)

        n = x.shape[-2]

        if n == 1:
            return x.squeeze(-2)

        # 顺序累加：result = x[0], then result = result + x[i] for i in 1..n-1
        result = x[..., 0, :]
        for i in range(1, n):
            self.adder.reset()
            result = self.adder(result, x[..., i, :])

        return result


class ParallelAccumulator(Accumulator):
    """并行树形归约累加器 - 快速

    树形归约：
    Layer 0: x0  x1  x2  x3  x4  x5  x6  x7
               \\  /    \\  /    \\  /    \\  /
    Layer 1:   s01     s23     s45     s67
                  \\    /          \\    /
    Layer 2:      s0123          s4567
                       \\        /
    Layer 3:           result

    特点：
    - O(log n) 层，每层可并行
    - GPU 利用率高
    - 舍入路径与顺序累加不同

    Args:
        adder: 加法器实例
    """

    def __init__(self, adder):
        super().__init__(adder)

    def reduce(self, x, dim):
        """并行树形归约

        Args:
            x: [..., N, bits]
            dim: 归约维度

        Returns:
            [..., bits]
        """
        # 将目标维度移到倒数第二位
        if dim != -2 and dim != x.dim() - 2:
            x = x.transpose(dim, -2)

        n = x.shape[-2]

        if n == 1:
            return x.squeeze(-2)

        # 树形归约
        current = x

        while current.shape[-2] > 1:
            n_current = current.shape[-2]
            n_pairs = n_current // 2
            has_odd = (n_current % 2) == 1

            if n_pairs > 0:
                # 取偶数索引和奇数索引配对
                left = current[..., 0:n_pairs*2:2, :]   # [0, 2, 4, ...]
                right = current[..., 1:n_pairs*2:2, :]  # [1, 3, 5, ...]

                # 并行加法
                self.adder.reset()
                paired_sum = self.adder(left, right)  # [..., n_pairs, bits]

                if has_odd:
                    # 奇数个元素，最后一个单独保留
                    last = current[..., -1:, :]
                    current = torch.cat([paired_sum, last], dim=-2)
                else:
                    current = paired_sum
            else:
                # 只剩一个元素
                break

        return current.squeeze(-2)


class PartialProductAccumulator(nn.Module):
    """部分积专用累加器 (用于阵列乘法器)

    与通用 Accumulator 不同:
    - 输入是 list of tensors，不是单个张量
    - forward() 直接返回累加结果

    Args:
        adder: 加法器实例 (如 VecRippleCarryAdder108Bit)
        mode: 'sequential' 或 'parallel'
    """

    def __init__(self, adder, mode='sequential'):
        super().__init__()
        self.adder = adder
        self.mode = mode

    def forward(self, partial_products):
        """累加部分积列表

        Args:
            partial_products: List of Tensor, 每个形状 [..., bits]

        Returns:
            累加结果张量，形状 [..., bits]
        """
        if len(partial_products) == 0:
            raise ValueError("partial_products cannot be empty")

        if len(partial_products) == 1:
            return partial_products[0]

        if self.mode == 'sequential':
            return self._sequential_reduce(partial_products)
        else:
            return self._tree_reduce(partial_products)

    def _sequential_reduce(self, pps):
        """顺序累加 - O(n)，位精确"""
        result = pps[0]
        for i in range(1, len(pps)):
            self.adder.reset()
            result, _ = self.adder(result, pps[i])
        return result

    def _tree_reduce(self, pps):
        """树形归约 - O(log n)，快速，层内并行（向量化实现）"""
        # 将 list 堆叠成张量 [N, ..., bits]
        current = torch.stack(pps, dim=0)

        while current.shape[0] > 1:
            n_current = current.shape[0]
            n_pairs = n_current // 2
            has_odd = (n_current % 2) == 1

            if n_pairs > 0:
                # 向量化切片（无循环）
                left = current[0:n_pairs*2:2]   # [0, 2, 4, ...]
                right = current[1:n_pairs*2:2]  # [1, 3, 5, ...]

                self.adder.reset()
                paired_sum, _ = self.adder(left, right)  # [n_pairs, ..., bits]

                if has_odd:
                    last = current[-1:]  # [1, ..., bits]
                    current = torch.cat([paired_sum, last], dim=0)
                else:
                    current = paired_sum
            else:
                break

        return current[0]

    def reset(self):
        """重置内部状态"""
        if hasattr(self.adder, 'reset'):
            self.adder.reset()


# 工厂函数
def create_accumulator(adder, mode='sequential'):
    """创建累加器

    Args:
        adder: 加法器实例
        mode: 'sequential' 或 'parallel'

    Returns:
        Accumulator 实例
    """
    if mode == 'sequential':
        return SequentialAccumulator(adder)
    elif mode == 'parallel':
        return ParallelAccumulator(adder)
    else:
        raise ValueError(f"Unknown accumulator mode: {mode}. Use 'sequential' or 'parallel'.")


def create_partial_product_accumulator(adder, mode='sequential'):
    """创建部分积累加器

    Args:
        adder: 加法器实例 (需要返回 (sum, carry) 元组)
        mode: 'sequential' 或 'parallel'

    Returns:
        PartialProductAccumulator 实例
    """
    return PartialProductAccumulator(adder, mode=mode)
