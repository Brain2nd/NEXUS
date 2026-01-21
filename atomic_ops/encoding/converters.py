"""
统一的浮点数-脉冲转换工具模块
==============================

提供 FP8/FP32/FP64 与脉冲张量之间的转换函数。
所有脉冲张量使用 MSB-first 格式（索引0为符号位）。

纯 PyTorch 实现，无 numpy 依赖。

作者: MofNeuroSim Project
"""
import torch
import struct


# ==============================================================================
# FP8 (E4M3) 转换函数
# ==============================================================================

def float_to_fp8_bits(x, device=None):
    """将 float 张量转换为 FP8 E4M3 的 8 位脉冲表示

    Args:
        x: float32 张量，任意形状
        device: 目标设备（默认与输入相同）

    Returns:
        脉冲张量 [..., 8]，MSB-first
    """
    if device is None:
        device = x.device
    fp8 = x.to(torch.float8_e4m3fn)
    bits_int = fp8.view(torch.uint8)
    bits = []
    for i in range(7, -1, -1):
        bits.append(((bits_int >> i) & 1).float())
    return torch.stack(bits, dim=-1).to(device)


def fp8_bits_to_float(bits):
    """将 8 位脉冲转换回 float32

    Args:
        bits: 脉冲张量 [..., 8]，MSB-first

    Returns:
        float32 张量 [...]
    """
    bits_int = torch.zeros(bits.shape[:-1], dtype=torch.uint8, device=bits.device)
    for i in range(8):
        bits_int = bits_int + (bits[..., i].to(torch.uint8) << (7 - i))
    fp8 = bits_int.view(torch.float8_e4m3fn)
    return fp8.to(torch.float32)


# ==============================================================================
# FP32 转换函数
# ==============================================================================

def float32_to_bits(f):
    """将单个 python float 转换为 32 位整数表示"""
    return struct.unpack('>I', struct.pack('>f', f))[0]


def bits_to_float32(b):
    """将 32 位整数表示转换回 python float"""
    return struct.unpack('>f', struct.pack('>I', b))[0]


def float32_to_pulse(x, device=None):
    """将 float32 张量转换为 32 位脉冲表示

    Args:
        x: float32 张量，任意形状
        device: 目标设备，None 表示使用输入张量的设备

    Returns:
        脉冲张量 [..., 32]，MSB-first
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(torch.float32)

    # 默认使用输入张量的设备
    if device is None:
        device = x.device

    original_shape = x.shape

    # 使用 view 进行位重解释: float32 -> int32
    bits_int = x.view(torch.int32)

    # 提取每一位 (MSB-first)
    pulses = []
    for i in range(31, -1, -1):
        pulses.append(((bits_int >> i) & 1).float())

    return torch.stack(pulses, dim=-1).to(device)


def pulse_to_float32(pulse):
    """将 32 位脉冲转换回 float32 张量

    Args:
        pulse: 脉冲张量 [..., 32]，MSB-first

    Returns:
        float32 张量 [...]
    """
    device = pulse.device
    shape = pulse.shape[:-1]

    # 将脉冲转为整数位
    bits_int = torch.zeros(shape, dtype=torch.int32, device=device)
    for i in range(32):
        bits_int = bits_int + ((pulse[..., i] > 0.5).int() << (31 - i))

    # 位重解释: int32 -> float32
    return bits_int.view(torch.float32)


# ==============================================================================
# FP16 转换函数
# ==============================================================================

def float16_to_pulse(x, device='cpu'):
    """将 float16 张量转换为 16 位脉冲表示

    Args:
        x: float16 张量，任意形状（会自动转换为 float16）
        device: 目标设备

    Returns:
        脉冲张量 [..., 16]，MSB-first [S, E4..E0, M9..M0]
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float16)
    x = x.to(torch.float16)
    original_shape = x.shape

    # 使用 view 进行位重解释: float16 -> int16
    bits_int = x.view(torch.int16)

    # 提取每一位 (MSB-first)
    pulses = []
    for i in range(15, -1, -1):
        pulses.append(((bits_int >> i) & 1).float())

    return torch.stack(pulses, dim=-1).to(device)


def pulse_to_float16(pulse):
    """将 16 位脉冲转换回 float16 张量

    Args:
        pulse: 脉冲张量 [..., 16]，MSB-first

    Returns:
        float16 张量 [...]
    """
    device = pulse.device
    shape = pulse.shape[:-1]

    # 将脉冲转为整数位
    bits_int = torch.zeros(shape, dtype=torch.int16, device=device)
    for i in range(16):
        bits_int = bits_int + ((pulse[..., i] > 0.5).short() << (15 - i))

    # 位重解释: int16 -> float16
    return bits_int.view(torch.float16)


# ==============================================================================
# FP64 转换函数
# ==============================================================================

def float64_to_bits(f):
    """将 python float (double) 转换为 64 位整数表示"""
    return struct.unpack('>Q', struct.pack('>d', f))[0]


def bits_to_float64(b):
    """将 64 位整数表示转换回 python float (double)"""
    return struct.unpack('>d', struct.pack('>Q', b))[0]


def float64_to_pulse(x, device=None):
    """将 float64 张量转换为 64 位脉冲表示

    Args:
        x: float64 张量，任意形状
        device: 目标设备，None 表示使用输入张量的设备

    Returns:
        脉冲张量 [..., 64]，MSB-first
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    x = x.to(torch.float64)

    # 默认使用输入张量的设备
    if device is None:
        device = x.device

    original_shape = x.shape

    # 使用 view 进行位重解释: float64 -> int64
    bits_int = x.view(torch.int64)

    # 提取每一位 (MSB-first)
    pulses = []
    for i in range(63, -1, -1):
        pulses.append(((bits_int >> i) & 1).float())

    return torch.stack(pulses, dim=-1).to(device)


def pulse_to_float64(pulse):
    """将 64 位脉冲转换回 float64 张量

    Args:
        pulse: 脉冲张量 [..., 64]，MSB-first

    Returns:
        float64 张量 [...]
    """
    device = pulse.device
    shape = pulse.shape[:-1]

    # 将脉冲转为整数位
    bits_int = torch.zeros(shape, dtype=torch.int64, device=device)
    for i in range(64):
        bits_int = bits_int + ((pulse[..., i] > 0.5).long() << (63 - i))

    # 位重解释: int64 -> float64
    return bits_int.view(torch.float64)


# ==============================================================================
# 便捷别名
# ==============================================================================

def float_to_pulse(val, device):
    """单个 FP32 值转脉冲（兼容旧接口）"""
    return float32_to_pulse(torch.tensor([val], dtype=torch.float32), device)


def pulse_to_bits(pulse):
    """脉冲转 32 位整数（兼容旧接口）

    Returns:
        单个值时返回 Python int，多个值时返回 PyTorch tensor
    """
    shape = pulse.shape[:-1]
    bits_int = torch.zeros(shape, dtype=torch.int32, device=pulse.device)
    for i in range(32):
        bits_int = bits_int + ((pulse[..., i] > 0.5).int() << (31 - i))
    if bits_int.numel() == 1:
        return int(bits_int.item())
    return bits_int  # 返回 PyTorch tensor（纯 PyTorch，无 numpy）
