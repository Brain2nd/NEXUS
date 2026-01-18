"""
纯脉冲域优化器
=============

提供完全在脉冲域中进行权重更新的优化器，
使用 SNN 门电路而非 Python 数学运算。

符合 CLAUDE.md 纯 SNN 约束。

作者: MofNeuroSim Project
"""
from .pulse_sgd import PulseSGD, PulseSGDWithMomentum

__all__ = [
    'PulseSGD',
    'PulseSGDWithMomentum',
]
