"""
FP8 Linear层 - 使用空间编码加法器
支持两种累加模式：sequential（与PyTorch一致）和tree（低延迟）
100%纯SNN实现
"""
import torch
import torch.nn as nn
import math
from .fp8_mul import SpikeFP8Multiplier
from .fp8_adder_spatial import SpikeFP8Adder_Spatial


class SpikeFP8Linear_Fast(nn.Module):
    """FP8 Linear层（使用空间编码加法器）
    
    支持两种累加模式：
    - 'sequential': 顺序累加 (((p0+p1)+p2)+p3)...
                   与PyTorch matmul完全bit-exact一致
                   延迟: O(N)
    - 'tree':      树形累加 ((p0+p1)+(p2+p3))...
                   低延迟但与PyTorch有微小差异（FP8非结合性）
                   延迟: O(log N)
    
    输入: x [..., in_features, 8]
    权重: w [out_features, in_features, 8]
    输出: y [..., out_features, 8]
    """
    def __init__(self, in_features, out_features, mode='sequential'):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            mode: 'sequential' (默认，与PyTorch一致) 或 'tree' (低延迟)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        
        self.mul = SpikeFP8Multiplier()
        
        if mode == 'tree':
            # 树形累加：需要 log2(N) 层加法器
            self.num_add_layers = math.ceil(math.log2(in_features)) if in_features > 1 else 0
            self.adders = nn.ModuleList()
            n = in_features
            for layer in range(self.num_add_layers):
                num_adders = (n + 1) // 2
                self.adders.append(nn.ModuleList([
                    SpikeFP8Adder_Spatial() for _ in range(num_adders)
                ]))
                n = num_adders
        else:  # sequential
            # 顺序累加：需要 N-1 个加法器
            self.num_adders = in_features - 1 if in_features > 1 else 0
            self.adders = nn.ModuleList([
                SpikeFP8Adder_Spatial() for _ in range(self.num_adders)
            ])
        
        self.register_buffer('weight_pulse', None)
        
    def set_weight_from_float(self, weight_float, encoder):
        """将float权重转换为FP8脉冲"""
        assert weight_float.shape == (self.out_features, self.in_features)
        weight_pulse = encoder(weight_float)
        self.weight_pulse = weight_pulse.squeeze(-2)
        
    def forward(self, x):
        """
        Args:
            x: [..., in_features, 8] 输入脉冲
        Returns:
            [..., out_features, 8] 输出脉冲
        """
        assert self.weight_pulse is not None, "请先调用 set_weight_from_float 设置权重"
        
        # x: [..., in_features, 8]
        # weight_pulse: [out_features, in_features, 8]
        x_expanded = x.unsqueeze(-3)  # [..., 1, in_features, 8]
        products = self.mul(x_expanded, self.weight_pulse)  # [..., out_features, in_features, 8]
        
        if self.in_features == 1:
            return products.squeeze(-2)
        
        if self.mode == 'tree':
            return self._tree_accumulate(products)
        else:
            return self._sequential_accumulate(products)
    
    def _sequential_accumulate(self, products):
        """顺序累加：与PyTorch matmul完全一致"""
        # products: [..., out_features, in_features, 8]
        acc = products[..., 0, :]  # [..., out_features, 8]
        
        for i in range(1, self.in_features):
            self.adders[i-1].reset()
            acc = self.adders[i-1](acc, products[..., i, :])
        
        return acc
    
    def _tree_accumulate(self, products):
        """树形累加：低延迟"""
        current = products
        current_size = self.in_features
        
        for layer_idx, layer_adders in enumerate(self.adders):
            next_values = []
            
            for i in range(0, current_size, 2):
                if i + 1 < current_size:
                    a = current[..., i, :]
                    b = current[..., i + 1, :]
                    layer_adders[i // 2].reset()
                    s = layer_adders[i // 2](a, b)
                    next_values.append(s)
                else:
                    next_values.append(current[..., i, :])
            
            current = torch.stack(next_values, dim=-2)
            current_size = len(next_values)
        
        return current.squeeze(-2)
    
    def reset(self):
        self.mul.reset()
        if self.mode == 'tree':
            for layer_adders in self.adders:
                for adder in layer_adders:
                    adder.reset()
        else:
            for adder in self.adders:
                adder.reset()
    
    def get_latency(self):
        """返回延迟（时间步数）"""
        if self.mode == 'tree':
            num_layers = math.ceil(math.log2(self.in_features)) if self.in_features > 1 else 0
            return 1 + num_layers
        else:
            return 1 + self.num_adders
