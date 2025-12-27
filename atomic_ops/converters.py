"""
统一的浮点数-脉冲转换工具模块
==============================

提供 FP8/FP32/FP64 与脉冲张量之间的转换函数。
所有脉冲张量使用 MSB-first 格式（索引0为符号位）。

作者: HumanBrain Project
"""
import torch
import numpy as np
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


def float32_to_pulse(x, device='cpu'):
    """将 float32 张量转换为 32 位脉冲表示
    
    Args:
        x: float32 张量或 numpy 数组，任意形状
        device: 目标设备
        
    Returns:
        脉冲张量 [..., 32]，MSB-first
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    x = np.asarray(x, dtype=np.float32)
    original_shape = x.shape
    bits = x.ravel().view(np.uint32)
    
    n = bits.size
    pulses = np.zeros((n, 32), dtype=np.float32)
    for i in range(32):
        shift = np.uint32(31 - i)
        pulses[:, i] = ((bits >> shift) & np.uint32(1)).astype(np.float32)
    
    return torch.from_numpy(pulses.reshape(original_shape + (32,))).to(device)


def pulse_to_float32(pulse):
    """将 32 位脉冲转换回 float32 张量
    
    Args:
        pulse: 脉冲张量 [..., 32]，MSB-first
        
    Returns:
        float32 张量 [...]
    """
    device = pulse.device
    shape = pulse.shape[:-1]
    
    flat_pulse = pulse.reshape(-1, 32).cpu().numpy() > 0.5
    n = flat_pulse.shape[0]
    
    bits = np.zeros(n, dtype=np.uint32)
    for i in range(32):
        shift = np.uint32(31 - i)
        bits |= (flat_pulse[:, i].astype(np.uint32) << shift)
    
    vals = bits.view(np.float32)
    return torch.from_numpy(vals).to(device).reshape(shape)


# ==============================================================================
# FP64 转换函数
# ==============================================================================

def float64_to_bits(f):
    """将 python float (double) 转换为 64 位整数表示"""
    return struct.unpack('>Q', struct.pack('>d', f))[0]


def bits_to_float64(b):
    """将 64 位整数表示转换回 python float (double)"""
    return struct.unpack('>d', struct.pack('>Q', b))[0]


def float64_to_pulse(x, device='cpu'):
    """将 float64 张量转换为 64 位脉冲表示
    
    Args:
        x: float64 张量或 numpy 数组，任意形状
        device: 目标设备
        
    Returns:
        脉冲张量 [..., 64]，MSB-first
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    bits = x.ravel().view(np.uint64)
    
    n = bits.size
    pulses = np.zeros((n, 64), dtype=np.float32)
    for i in range(64):
        shift = np.uint64(63 - i)
        pulses[:, i] = ((bits >> shift) & np.uint64(1)).astype(np.float32)
    
    return torch.from_numpy(pulses.reshape(original_shape + (64,))).to(device)


def pulse_to_float64(pulse):
    """将 64 位脉冲转换回 float64 张量
    
    Args:
        pulse: 脉冲张量 [..., 64]，MSB-first
        
    Returns:
        float64 张量 [...]
    """
    device = pulse.device
    shape = pulse.shape[:-1]
    
    flat_pulse = pulse.reshape(-1, 64).cpu().numpy() > 0.5
    n = flat_pulse.shape[0]
    
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(64):
        shift = np.uint64(63 - i)
        bits |= (flat_pulse[:, i].astype(np.uint64) << shift)
    
    vals = bits.view(np.float64)
    return torch.from_numpy(vals).to(device).reshape(shape)


# ==============================================================================
# 便捷别名
# ==============================================================================

# FP32 单值转换（兼容旧代码）
def float_to_pulse(val, device):
    """单个 FP32 值转脉冲（兼容旧接口）"""
    return float32_to_pulse(np.array([val], dtype=np.float32), device)


def pulse_to_bits(pulse):
    """脉冲转 32 位整数（兼容旧接口）"""
    flat_pulse = pulse.reshape(-1, 32).cpu().numpy() > 0.5
    bits = np.zeros(flat_pulse.shape[0], dtype=np.uint32)
    for i in range(32):
        shift = np.uint32(31 - i)
        bits |= (flat_pulse[:, i].astype(np.uint32) << shift)
    return int(bits[0]) if bits.size == 1 else bits
