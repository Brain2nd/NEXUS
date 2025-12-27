"""
SNN 脉冲解码器 (Pulse Decoder)
==============================

将 SNN 二进制脉冲序列转换回 ANN 浮点数。

作为纯 SNN 系统的输出边界组件，支持任意维度输入。

浮点格式说明
-----------

**FP8 E4M3 格式** (8位):
```
[S | E3 E2 E1 E0 | M2 M1 M0]
 ↑   \_________/   \_______/
符号    指数(4位)    尾数(3位)
```

**解码公式**:
- Normal (E ≠ 0):     value = (-1)^S × 2^(E-7) × (1 + M/8)
- Subnormal (E = 0):  value = (-1)^S × 2^(-6) × (M/8)
- Zero (E=0, M=0):    value = ±0

**FP16 格式** (16位): bias=15, E=5位, M=10位
**FP32 格式** (32位): bias=127, E=8位, M=23位

使用示例
-------
```python
decoder = PulseFloatingPointDecoder()
pulse = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0]])  # FP8 脉冲
value = decoder(pulse)  # 转换为浮点数
```

作者: HumanBrain Project
许可: MIT License
"""
import torch
import torch.nn as nn


class PulseFloatingPointDecoder(nn.Module):
    """FP8 脉冲解码器 - 将 SNN 脉冲转换为浮点数
    
    这是 SNN 系统的输出边界组件，将纯脉冲域的计算结果
    转换回传统浮点数表示。
    
    **数学原理**:
    
    对于 FP8 E4M3 格式：
    - 指数偏置 (bias) = 2^(E_bits-1) - 1 = 7
    - Normal:    value = (-1)^S × 2^(E-bias) × (1 + M/2^M_bits)
    - Subnormal: value = (-1)^S × 2^(1-bias) × (M/2^M_bits)
    
    **支持任意维度**:
    - 输入: [..., 8] SNN 脉冲张量
    - 输出: [...] 浮点数张量
    
    **注意**: 作为边界组件，允许使用传统计算（非 SNN 门电路）。
    
    Args:
        exponent_bits: 指数位数，默认 4 (FP8 E4M3)
        mantissa_bits: 尾数位数，默认 3 (FP8 E4M3)
    
    Example:
        >>> decoder = PulseFloatingPointDecoder()
        >>> pulse = encoder(torch.tensor([1.5]))  # [1, 8]
        >>> value = decoder(pulse)  # [1]
        >>> print(value)  # tensor([1.5000])
    """
    def __init__(self, exponent_bits: int = 4, mantissa_bits: int = 3):
        super().__init__()
        self.E_bits = exponent_bits
        self.M_bits = mantissa_bits
        self.total_bits = 1 + exponent_bits + mantissa_bits
        self.bias = (2 ** (exponent_bits - 1)) - 1  # 7 for E4M3
        
    def forward(self, pulse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pulse: [..., 8] SNN 脉冲，格式 [S, E3, E2, E1, E0, M2, M1, M0]
        Returns:
            [...] 浮点数张量
        """
        # 验证输入
        assert pulse.shape[-1] == self.total_bits, \
            f"Expected last dim to be {self.total_bits}, got {pulse.shape[-1]}"
        
        # 保存原始形状（不含最后一维）
        output_shape = pulse.shape[:-1]
        device = pulse.device
        
        # 提取各部分 (MSB first)
        s = pulse[..., 0]  # 符号位
        e_bits = pulse[..., 1:1+self.E_bits]  # 指数位 [E3, E2, E1, E0]
        m_bits = pulse[..., 1+self.E_bits:]   # 尾数位 [M2, M1, M0]
        
        # 计算指数值 (MSB first → 整数)
        exp_val = torch.zeros(output_shape, device=device, dtype=torch.float32)
        for i in range(self.E_bits):
            exp_val = exp_val + e_bits[..., i] * (2 ** (self.E_bits - 1 - i))
        
        # 计算尾数值 (MSB first → 小数)
        mant_val = torch.zeros(output_shape, device=device, dtype=torch.float32)
        for i in range(self.M_bits):
            mant_val = mant_val + m_bits[..., i] * (2 ** (-(i + 1)))
        
        # 检测特殊情况
        is_zero_exp = (exp_val == 0)
        is_zero_mant = (mant_val == 0)
        is_true_zero = is_zero_exp & is_zero_mant
        is_subnormal = is_zero_exp & (~is_zero_mant)
        
        # 计算浮点值
        # Normal: (-1)^S × 2^(E-bias) × (1 + M)
        # Subnormal: (-1)^S × 2^(1-bias) × M
        
        # Normal 情况
        normal_val = (2.0 ** (exp_val - self.bias)) * (1.0 + mant_val)
        
        # Subnormal 情况: 2^(1-bias) × M = 2^(-6) × M
        subnormal_val = (2.0 ** (1 - self.bias)) * mant_val
        
        # 选择正确的值
        abs_val = torch.where(is_subnormal, subnormal_val, normal_val)
        abs_val = torch.where(is_true_zero, torch.zeros_like(abs_val), abs_val)
        
        # 应用符号
        sign = 1.0 - 2.0 * s  # s=0 → +1, s=1 → -1
        result = sign * abs_val
        
        return result
    
    def reset(self):
        """兼容 SNN 组件接口"""
        pass


class PulseFP16Decoder(nn.Module):
    """FP16 脉冲解码器 - 将 16 位 SNN 脉冲转换为浮点数
    
    FP16 格式: [S | E4..E0 | M9..M0]
    - bias = 15
    """
    def __init__(self):
        super().__init__()
        self.E_bits = 5
        self.M_bits = 10
        self.total_bits = 16
        self.bias = 15
        
    def forward(self, pulse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pulse: [..., 16] SNN 脉冲
        Returns:
            [...] 浮点数张量
        """
        assert pulse.shape[-1] == self.total_bits
        
        output_shape = pulse.shape[:-1]
        device = pulse.device
        
        s = pulse[..., 0]
        e_bits = pulse[..., 1:6]
        m_bits = pulse[..., 6:16]
        
        # 指数
        exp_val = torch.zeros(output_shape, device=device, dtype=torch.float32)
        for i in range(self.E_bits):
            exp_val = exp_val + e_bits[..., i] * (2 ** (self.E_bits - 1 - i))
        
        # 尾数
        mant_val = torch.zeros(output_shape, device=device, dtype=torch.float32)
        for i in range(self.M_bits):
            mant_val = mant_val + m_bits[..., i] * (2 ** (-(i + 1)))
        
        is_zero_exp = (exp_val == 0)
        is_zero_mant = (mant_val == 0)
        is_true_zero = is_zero_exp & is_zero_mant
        is_subnormal = is_zero_exp & (~is_zero_mant)
        
        normal_val = (2.0 ** (exp_val - self.bias)) * (1.0 + mant_val)
        subnormal_val = (2.0 ** (1 - self.bias)) * mant_val
        
        abs_val = torch.where(is_subnormal, subnormal_val, normal_val)
        abs_val = torch.where(is_true_zero, torch.zeros_like(abs_val), abs_val)
        
        sign = 1.0 - 2.0 * s
        return sign * abs_val
    
    def reset(self):
        pass


class PulseFP32Decoder(nn.Module):
    """FP32 脉冲解码器 - 将 32 位 SNN 脉冲转换为浮点数
    
    FP32 格式: [S | E7..E0 | M22..M0]
    - bias = 127
    """
    def __init__(self):
        super().__init__()
        self.E_bits = 8
        self.M_bits = 23
        self.total_bits = 32
        self.bias = 127
        
    def forward(self, pulse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pulse: [..., 32] SNN 脉冲
        Returns:
            [...] 浮点数张量
        """
        assert pulse.shape[-1] == self.total_bits
        
        output_shape = pulse.shape[:-1]
        device = pulse.device
        
        s = pulse[..., 0]
        e_bits = pulse[..., 1:9]
        m_bits = pulse[..., 9:32]
        
        # 指数
        exp_val = torch.zeros(output_shape, device=device, dtype=torch.float32)
        for i in range(self.E_bits):
            exp_val = exp_val + e_bits[..., i] * (2 ** (self.E_bits - 1 - i))
        
        # 尾数
        mant_val = torch.zeros(output_shape, device=device, dtype=torch.float32)
        for i in range(self.M_bits):
            mant_val = mant_val + m_bits[..., i] * (2 ** (-(i + 1)))
        
        is_zero_exp = (exp_val == 0)
        is_zero_mant = (mant_val == 0)
        is_true_zero = is_zero_exp & is_zero_mant
        is_subnormal = is_zero_exp & (~is_zero_mant)
        
        normal_val = (2.0 ** (exp_val - self.bias)) * (1.0 + mant_val)
        subnormal_val = (2.0 ** (1 - self.bias)) * mant_val
        
        abs_val = torch.where(is_subnormal, subnormal_val, normal_val)
        abs_val = torch.where(is_true_zero, torch.zeros_like(abs_val), abs_val)
        
        sign = 1.0 - 2.0 * s
        return sign * abs_val
    
    def reset(self):
        pass


# ==============================================================================
# 辅助函数
# ==============================================================================

def pulse_to_fp8_bits(pulse: torch.Tensor) -> torch.Tensor:
    """将 FP8 脉冲转换为字节值 (0-255)
    
    将 8 位二进制脉冲序列 [b7, b6, ..., b0] 转换为整数:
    value = Σ(b_i × 2^(7-i)) for i in [0, 7]
    
    Args:
        pulse: [..., 8] SNN 脉冲张量，最后一维是 8 位
        
    Returns:
        [...] 整数张量，范围 0-255
        
    Example:
        >>> pulse = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]])  # 128
        >>> pulse_to_fp8_bits(pulse)
        tensor([128])
    """
    bits = pulse.int()
    byte_val = torch.zeros(pulse.shape[:-1], dtype=torch.int32, device=pulse.device)
    for i in range(8):
        byte_val = byte_val + (bits[..., i] << (7 - i))
    return byte_val


def fp8_bits_to_pulse(byte_val: torch.Tensor) -> torch.Tensor:
    """将字节值 (0-255) 转换为 FP8 脉冲
    
    将整数转换为 8 位二进制脉冲序列 [b7, b6, ..., b0]:
    b_i = (value >> (7-i)) & 1
    
    Args:
        byte_val: [...] 整数张量，范围 0-255
        
    Returns:
        [..., 8] SNN 脉冲张量
        
    Example:
        >>> byte_val = torch.tensor([128])
        >>> fp8_bits_to_pulse(byte_val)
        tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
    """
    output_shape = byte_val.shape + (8,)
    pulse = torch.zeros(output_shape, device=byte_val.device, dtype=torch.float32)
    
    for i in range(8):
        pulse[..., i] = ((byte_val >> (7 - i)) & 1).float()
    
    return pulse

