"""
Dual-Rail Base Definitions
==========================

Defines the fundamental types and protocols for True Dual-Rail SNN logic.

Protocol:
    Signal is a tuple (pos, neg) representing a logical value.
    - Logic 1: pos=1, neg=0
    - Logic 0: pos=0, neg=1
    - Null:    pos=0, neg=0 (no data)
    - Invalid: pos=1, neg=1 (error state)

Pure SNN Constraints:
    - All weights must be positive (excitatory).
    - No `1.0 - x` operations allowed in logic gates.
    - NOT operation must be a wire swap.
"""

import torch
import torch.nn as nn
from atomic_ops.core.reset_utils import reset_children
from copy import deepcopy
from atomic_ops.core.neurons import SimpleLIFNode

class DualRailBlock(nn.Module):
    """Base class for all Dual-Rail components."""
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError

    def reset(self):
        """递归reset所有子模块（处理容器类型）"""
        reset_children(self)

def create_neuron(template, threshold, v_reset=None,
                  beta=None, max_param_shape=None,
                  # 兼容旧调用，忽略这些参数
                  trainable_threshold=True, trainable_beta=True):
    """从模板创建指定阈值的神经元

    所有参数始终为 nn.Parameter (可训练)，无需 trainable 开关。

    Args:
        template: 神经元模板，None 则创建默认 SimpleLIFNode
        threshold: 目标阈值 (float 或 Tensor)
        v_reset: 复位电压 (None=软复位, 数值=硬复位)
        beta: 泄漏因子 (None=DEFAULT_BETA)
        max_param_shape: 预分配参数形状，None 使用全局默认

    Returns:
        配置好的神经元实例
    """
    if template is None:
        return SimpleLIFNode(
            beta=beta,
            v_threshold=threshold,
            v_reset=v_reset,
            max_param_shape=max_param_shape
        )
    else:
        node = deepcopy(template)
        node.v_threshold = threshold
        if hasattr(node, 'v_reset'):
            node.v_reset = v_reset
        if max_param_shape is not None and hasattr(node, 'max_param_shape'):
            node.max_param_shape = max_param_shape
            if hasattr(node, '_preallocate_params'):
                node._preallocate_params(max_param_shape)
        return node
