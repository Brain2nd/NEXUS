"""
Reset 工具函数 - 高效递归重置 SNN 模块
=========================================

提供统一的 reset 机制，正确处理：
- 直接子模块
- nn.ModuleList / nn.Sequential / nn.ModuleDict 容器
- 避免 O(n²) 重复调用

使用方式
--------
```python
from atomic_ops.core.reset_utils import reset_children

class MyModule(nn.Module):
    def reset(self):
        reset_children(self)
```

作者: MofNeuroSim Project
"""

import torch.nn as nn


def reset_children(module: nn.Module) -> None:
    """递归重置模块的所有直接子模块

    正确处理：
    - 有 reset() 方法的自定义模块：调用其 reset()
    - nn.ModuleList / nn.Sequential：递归处理内部模块
    - nn.ModuleDict：递归处理内部模块
    - 普通 nn.Module：递归其 children()

    Args:
        module: 要重置子模块的父模块
    """
    for child in module.children():
        _reset_module(child)


def _reset_module(module: nn.Module) -> None:
    """重置单个模块（内部函数）

    策略：
    1. 如果模块有自定义 reset() 方法，调用它（由该模块负责其子模块）
    2. 否则如果是容器类型，递归处理内部模块
    3. 否则递归处理 children()
    """
    # 检查是否有自定义 reset 方法（排除基础 nn.Module）
    if _has_custom_reset(module):
        module.reset()
        return

    # 处理容器类型
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        for item in module:
            _reset_module(item)
    elif isinstance(module, nn.ModuleDict):
        for item in module.values():
            _reset_module(item)
    else:
        # 普通 nn.Module 没有 reset：递归其 children
        for child in module.children():
            _reset_module(child)


def _has_custom_reset(module: nn.Module) -> bool:
    """检查模块是否有自定义的 reset 方法

    排除：
    - nn.Module 基类（没有 reset）
    - nn.ModuleList / nn.Sequential / nn.ModuleDict（容器类型）
    """
    # 容器类型不算有自定义 reset
    if isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
        return False

    # 检查是否有 reset 属性且可调用
    return hasattr(module, 'reset') and callable(getattr(module, 'reset'))
