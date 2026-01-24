"""
脉冲张量存储优化工具
=====================

提供 bool 存储的脉冲张量封装，支持透明的 float 转换。
内存节省: 4x (bool 1 byte vs float32 4 bytes)

核心理念:
- Level 0 (神经元内部): 膜电位是 float，不受影响
- Level 1+ (门电路及以上): 脉冲值只有 0/1，可以用 bool 存储
- 存储时用 bool，计算时按需转换为 float

用法:
    # 创建
    pt = PulseTensor.from_float(pulse_float)  # [0.0, 1.0] -> bool

    # 存储 (bool, 4x 节省)
    buffer = pt.data  # torch.bool tensor

    # 计算 (转换为 float)
    pulse_float = pt.to_float()  # -> torch.float32

    # 或使用 property
    pulse_float = pt.float  # 自动缓存

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
from typing import Tuple, Optional


class PulseTensor:
    """内存优化的脉冲张量封装

    存储为 bool (1 byte/element)，计算时按需转换为 float (4 bytes/element)。
    内存节省: 4x
    """

    def __init__(self, bool_data: torch.Tensor):
        """从 bool 张量创建 PulseTensor

        Args:
            bool_data: dtype=torch.bool 的张量
        """
        if bool_data.dtype != torch.bool:
            raise ValueError(f"Expected bool tensor, got {bool_data.dtype}")
        self._data = bool_data
        self._float_cache: Optional[torch.Tensor] = None

    @classmethod
    def from_float(cls, float_data: torch.Tensor, threshold: float = 0.5) -> 'PulseTensor':
        """从 float 脉冲创建 PulseTensor (边界转换)

        Args:
            float_data: 包含 0.0/1.0 值的 float 张量
            threshold: 阈值，大于此值视为 1 (默认 0.5)

        Returns:
            PulseTensor 实例
        """
        bool_data = (float_data > threshold).bool()
        return cls(bool_data)

    @classmethod
    def zeros(cls, *size, device=None, dtype=None) -> 'PulseTensor':
        """创建全零 PulseTensor

        Args:
            *size: 张量形状
            device: 目标设备
            dtype: 忽略，始终使用 bool

        Returns:
            PulseTensor 实例
        """
        bool_data = torch.zeros(*size, dtype=torch.bool, device=device)
        return cls(bool_data)

    @classmethod
    def ones(cls, *size, device=None, dtype=None) -> 'PulseTensor':
        """创建全一 PulseTensor

        Args:
            *size: 张量形状
            device: 目标设备
            dtype: 忽略，始终使用 bool

        Returns:
            PulseTensor 实例
        """
        bool_data = torch.ones(*size, dtype=torch.bool, device=device)
        return cls(bool_data)

    @property
    def data(self) -> torch.Tensor:
        """获取底层 bool 张量"""
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        """设置底层数据"""
        if value.dtype != torch.bool:
            raise ValueError(f"Expected bool tensor, got {value.dtype}")
        self._data = value
        self._float_cache = None

    def to_float(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """转换为 float 张量 (用于计算，不缓存)

        Args:
            dtype: 目标 dtype (默认 float32)

        Returns:
            float 张量
        """
        return self._data.to(dtype)

    @property
    def float(self) -> torch.Tensor:
        """带缓存的 float32 转换

        Returns:
            float32 张量 (缓存结果)
        """
        if self._float_cache is None:
            self._float_cache = self._data.float()
        return self._float_cache

    def invalidate_cache(self):
        """清除缓存 (当底层数据变化时调用)"""
        self._float_cache = None

    @property
    def shape(self) -> torch.Size:
        """张量形状"""
        return self._data.shape

    @property
    def device(self) -> torch.device:
        """所在设备"""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """底层 dtype (始终为 bool)"""
        return self._data.dtype

    def to(self, device: torch.device) -> 'PulseTensor':
        """移动到指定设备

        Args:
            device: 目标设备

        Returns:
            self (原地操作)
        """
        self._data = self._data.to(device)
        self._float_cache = None
        return self

    def clone(self) -> 'PulseTensor':
        """创建副本

        Returns:
            新的 PulseTensor 实例
        """
        return PulseTensor(self._data.clone())

    def __repr__(self) -> str:
        return f"PulseTensor(shape={self.shape}, device={self.device})"

    def numel(self) -> int:
        """元素总数"""
        return self._data.numel()

    def memory_bytes(self) -> int:
        """当前内存占用 (bytes)"""
        return self._data.element_size() * self._data.numel()

    def memory_saved_bytes(self) -> int:
        """相比 float32 节省的内存 (bytes)"""
        float32_bytes = 4 * self._data.numel()
        bool_bytes = self.memory_bytes()
        return float32_bytes - bool_bytes


def pulse_to_bool(pulse: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """将 float 脉冲转换为 bool 存储

    Args:
        pulse: float 脉冲张量 (值为 0.0 或 1.0)
        threshold: 阈值 (默认 0.5)

    Returns:
        bool 张量
    """
    return (pulse > threshold).bool()


def bool_to_pulse(bool_data: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """将 bool 存储转换回 float 脉冲

    Args:
        bool_data: bool 张量
        dtype: 目标 dtype (默认 float32)

    Returns:
        float 张量 (值为 0.0 或 1.0)
    """
    return bool_data.to(dtype)


def calculate_memory_savings(shape: Tuple[int, ...]) -> dict:
    """计算给定形状的内存节省

    Args:
        shape: 张量形状

    Returns:
        包含内存统计的字典
    """
    numel = 1
    for dim in shape:
        numel *= dim

    float32_bytes = 4 * numel
    bool_bytes = 1 * numel  # PyTorch bool 是 1 byte

    return {
        'shape': shape,
        'numel': numel,
        'float32_bytes': float32_bytes,
        'float32_mb': float32_bytes / (1024 * 1024),
        'bool_bytes': bool_bytes,
        'bool_mb': bool_bytes / (1024 * 1024),
        'savings_bytes': float32_bytes - bool_bytes,
        'savings_mb': (float32_bytes - bool_bytes) / (1024 * 1024),
        'savings_ratio': float32_bytes / bool_bytes,  # 4x
    }
