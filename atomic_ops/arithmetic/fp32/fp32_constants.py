"""
FP32 脉冲常量模块
=================

提供训练中常用的脉冲格式常量 (0, 1, -1, 0.5 等)。
这些常量仅在边界处预编码一次，后续使用时直接以脉冲形式参与计算。

作者: MofNeuroSim Project
"""
import torch
from atomic_ops.encoding.converters import float32_to_pulse


# 缓存已创建的常量，避免重复编码
_PULSE_CACHE = {}


def get_pulse_constant(value: float, batch_shape: tuple = (), device=None):
    """获取脉冲格式的常量

    Args:
        value: 浮点常量值
        batch_shape: 广播形状 (可选)
        device: 目标设备

    Returns:
        [..., 32] 脉冲张量
    """
    if device is None:
        device = torch.device('cpu')

    # 创建基础脉冲常量
    cache_key = (value, device.type if hasattr(device, 'type') else str(device))
    if cache_key not in _PULSE_CACHE:
        val_tensor = torch.tensor(value, dtype=torch.float32)
        _PULSE_CACHE[cache_key] = float32_to_pulse(val_tensor, device)

    base_pulse = _PULSE_CACHE[cache_key].to(device)

    # 广播到目标形状
    if batch_shape:
        target_shape = batch_shape + (32,)
        return base_pulse.expand(target_shape).clone()

    return base_pulse.clone()


def get_zero_pulse(batch_shape: tuple = (), device=None):
    """获取 0.0 的脉冲表示

    FP32 +0.0 的位模式: 0x00000000 (全零)
    """
    return get_pulse_constant(0.0, batch_shape, device)


def get_one_pulse(batch_shape: tuple = (), device=None):
    """获取 1.0 的脉冲表示

    FP32 1.0 的位模式: 0x3F800000
    = 0 01111111 00000000000000000000000
    = sign=0, exp=127, mantissa=0
    """
    return get_pulse_constant(1.0, batch_shape, device)


def get_neg_one_pulse(batch_shape: tuple = (), device=None):
    """获取 -1.0 的脉冲表示

    FP32 -1.0 的位模式: 0xBF800000
    = 1 01111111 00000000000000000000000
    = sign=1, exp=127, mantissa=0
    """
    return get_pulse_constant(-1.0, batch_shape, device)


def get_half_pulse(batch_shape: tuple = (), device=None):
    """获取 0.5 的脉冲表示

    FP32 0.5 的位模式: 0x3F000000
    = 0 01111110 00000000000000000000000
    = sign=0, exp=126, mantissa=0
    """
    return get_pulse_constant(0.5, batch_shape, device)


def get_two_pulse(batch_shape: tuple = (), device=None):
    """获取 2.0 的脉冲表示

    FP32 2.0 的位模式: 0x40000000
    = 0 10000000 00000000000000000000000
    = sign=0, exp=128, mantissa=0
    """
    return get_pulse_constant(2.0, batch_shape, device)


class PulseConstants:
    """脉冲常量管理器

    为指定设备预生成常用常量，避免重复编码。

    使用示例:
        constants = PulseConstants(device='cuda')
        one = constants.one(batch_shape=(4, 8))
        zero = constants.zero(batch_shape=(4, 8))
    """

    def __init__(self, device=None):
        if device is None:
            device = torch.device('cpu')
        self.device = device

        # 预编码常用常量
        self._zero = get_pulse_constant(0.0, device=device)
        self._one = get_pulse_constant(1.0, device=device)
        self._neg_one = get_pulse_constant(-1.0, device=device)
        self._half = get_pulse_constant(0.5, device=device)
        self._two = get_pulse_constant(2.0, device=device)

    def zero(self, batch_shape: tuple = ()):
        """获取 0.0 的脉冲，广播到指定形状"""
        if batch_shape:
            return self._zero.expand(batch_shape + (32,)).clone()
        return self._zero.clone()

    def one(self, batch_shape: tuple = ()):
        """获取 1.0 的脉冲，广播到指定形状"""
        if batch_shape:
            return self._one.expand(batch_shape + (32,)).clone()
        return self._one.clone()

    def neg_one(self, batch_shape: tuple = ()):
        """获取 -1.0 的脉冲，广播到指定形状"""
        if batch_shape:
            return self._neg_one.expand(batch_shape + (32,)).clone()
        return self._neg_one.clone()

    def half(self, batch_shape: tuple = ()):
        """获取 0.5 的脉冲，广播到指定形状"""
        if batch_shape:
            return self._half.expand(batch_shape + (32,)).clone()
        return self._half.clone()

    def two(self, batch_shape: tuple = ()):
        """获取 2.0 的脉冲，广播到指定形状"""
        if batch_shape:
            return self._two.expand(batch_shape + (32,)).clone()
        return self._two.clone()

    def custom(self, value: float, batch_shape: tuple = ()):
        """获取自定义值的脉冲"""
        return get_pulse_constant(value, batch_shape, self.device)

    def to(self, device):
        """移动到新设备"""
        return PulseConstants(device)


def clear_cache():
    """清空脉冲常量缓存"""
    global _PULSE_CACHE
    _PULSE_CACHE = {}
