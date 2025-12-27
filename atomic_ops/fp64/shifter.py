import torch
import torch.nn as nn
from ..logic_gates import MUXGate, ORGate, ANDGate, NOTGate

class BarrelShifterRight64(nn.Module):
    """64-bit 右移位器 (用于FP64尾数对齐)
    支持移位量 0-63
    输入: data [..., 64], shift [..., 6] (二进制脉冲)
    输出: shifted [..., 64]
    """
    def __init__(self):
        super().__init__()
        # 6层 MUX，每层处理一位 shift
        # Layer 0: shift 1
        # Layer 1: shift 2
        # Layer 2: shift 4
        # Layer 3: shift 8
        # Layer 4: shift 16
        # Layer 5: shift 32
        
        self.layers = nn.ModuleList()
        shifts = [1, 2, 4, 8, 16, 32]
        
        for s in shifts:
            layer = nn.ModuleList([MUXGate() for _ in range(64)])
            self.layers.append(layer)
            
    def forward(self, data, shift):
        # data: [..., 64]
        # shift: [..., 6] (LSB first or MSB first? usually we use MSB first in pulse tensor)
        # Let's assume shift is [s5, s4, s3, s2, s1, s0] where s0 is LSB (shift 1)
        # But wait, our standard is usually MSB first.
        # shift 6 bits. Max 63. 
        # [b5, b4, b3, b2, b1, b0]
        # b5 (32), b4 (16), ..., b0 (1)
        
        current = data
        zeros = torch.zeros_like(data[..., 0:1])
        
        # Bits from LSB (b0, shift 1) to MSB (b5, shift 32)
        # shift input is [b5, b4, b3, b2, b1, b0]
        
        for i in range(6):
            s_bit = shift[..., 5-i : 6-i] # b0, b1, ... b5
            shift_amount = 1 << i
            
            # Right Shift: new[k] = s ? old[k-shift] : old[k]
            # Since index is 0..63, right shift means moving data to higher indices?
            # NO. In standard array: [MSB ... LSB].
            # Value 1000... (1) >> 1 = 0100... (0.5)
            # So data moves to HIGHER indices.
            
            next_val = []
            for k in range(64):
                # If shift, take from k - shift_amount
                # If k < shift_amount, fill with 0
                if k >= shift_amount:
                    src = current[..., k-shift_amount : k-shift_amount+1]
                else:
                    src = zeros
                
                # MUX(sel, src, default) -> if sel=1 choose src
                # Our MUXGate: (sel, a, b) -> sel ? a : b
                bit = self.layers[i][k](s_bit, src, current[..., k:k+1])
                next_val.append(bit)
            
            current = torch.cat(next_val, dim=-1)
            
        return current

class BarrelShifterLeft64(nn.Module):
    """64-bit 左移位器 (用于FP64结果归一化)
    支持移位量 0-63
    输入: data [..., 64], shift [..., 6]
    输出: shifted [..., 64]
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        shifts = [1, 2, 4, 8, 16, 32]
        
        for s in shifts:
            layer = nn.ModuleList([MUXGate() for _ in range(64)])
            self.layers.append(layer)
            
    def forward(self, data, shift):
        # shift: [b5, b4, b3, b2, b1, b0]
        
        current = data
        zeros = torch.zeros_like(data[..., 0:1])
        
        for i in range(6):
            s_bit = shift[..., 5-i : 6-i]
            shift_amount = 1 << i
            
            # Left Shift: new[k] = s ? old[k+shift] : old[k]
            # [0, 1, 2] << 1 -> [1, 2, 0]
            
            next_val = []
            for k in range(64):
                if k + shift_amount < 64:
                    src = current[..., k+shift_amount : k+shift_amount+1]
                else:
                    src = zeros
                    
                bit = self.layers[i][k](s_bit, src, current[..., k:k+1])
                next_val.append(bit)
                
            current = torch.cat(next_val, dim=-1)
            
        return current
