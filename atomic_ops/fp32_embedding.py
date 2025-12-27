"""
FP32 Embedding层 - 100%纯SNN门电路实现
========================================

整数索引 → 二进制脉冲 → MUX树选择 → embedding向量脉冲

原理：
- 整数和浮点数本质上都是二进制，都用脉冲表示
- token_id编码为log2(vocab_size)位二进制脉冲
- 用MUX树根据地址位逐层选择，O(log V)复杂度

作者: HumanBrain Project
"""
import torch
import torch.nn as nn
import struct
import math
from .logic_gates import MUXGate


class SpikeFP32Embedding(nn.Module):
    """FP32 Embedding层 - 使用MUX树选择
    
    输入: token_ids [...] 整数张量
    输出: [..., embed_dim, 32] FP32脉冲
    
    实现:
    1. token_id → 二进制脉冲 (addr_bits位)
    2. MUX树：每层用1个地址位选择，共log2(vocab_size)层
    3. 输出选中行的embedding脉冲
    
    复杂度: O(log2(vocab_size)) 层MUX
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 计算地址位数
        self.addr_bits = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1
        
        # 预编码的权重 [padded_vocab_size, embed_dim, 32]
        self.register_buffer('weight_pulse', None)
        
        # MUX树：每层每个输出位置需要一个MUX
        # 共 addr_bits 层，每层 embed_dim × 32 个MUX
        self.mux_layers = nn.ModuleList()
        for layer in range(self.addr_bits):
            self.mux_layers.append(nn.ModuleList([
                MUXGate() for _ in range(embed_dim * 32)
            ]))
        
    def set_weight_from_float(self, weight_float: torch.Tensor):
        """从浮点权重设置脉冲权重"""
        assert weight_float.shape == (self.vocab_size, self.embed_dim)
        
        # 扩展到2的幂次方便MUX树处理
        padded_size = 2 ** self.addr_bits
        if padded_size > self.vocab_size:
            padding = torch.zeros(padded_size - self.vocab_size, self.embed_dim, 
                                  device=weight_float.device)
            weight_float = torch.cat([weight_float, padding], dim=0)
        
        weight_pulse = self._float_to_fp32_pulse(weight_float)
        self.weight_pulse = weight_pulse  # [padded_size, embed_dim, 32]
        
    def from_nn_embedding(self, nn_embedding: nn.Embedding):
        """从PyTorch nn.Embedding导入权重"""
        assert nn_embedding.num_embeddings == self.vocab_size
        assert nn_embedding.embedding_dim == self.embed_dim
        self.set_weight_from_float(nn_embedding.weight.data)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [...] 整数张量
        Returns:
            [..., embed_dim, 32] FP32脉冲
        """
        assert self.weight_pulse is not None, "请先设置权重"
        
        device = token_ids.device
        batch_shape = token_ids.shape
        
        # Step 1: 整数 → 二进制脉冲 (边界操作)
        addr_pulse = self._int_to_binary_pulse(token_ids)  # [..., addr_bits]
        
        # 展平batch处理
        batch_flat = token_ids.reshape(-1)
        addr_flat = addr_pulse.reshape(-1, self.addr_bits)
        N = batch_flat.shape[0]
        
        results = []
        
        for n in range(N):
            addr = addr_flat[n]  # [addr_bits]
            current = self.weight_pulse  # [V, D, 32]
            
            for layer in range(self.addr_bits):
                sel_bit = addr[layer]
                half = current.shape[0] // 2
                if half == 0:
                    break
                    
                left = current[:half]
                right = current[half:2*half]
                
                # MUX选择
                new_rows = []
                for h in range(half):
                    row_bits = []
                    for d in range(self.embed_dim):
                        for b in range(32):
                            mux_idx = d * 32 + b
                            self.mux_layers[layer][mux_idx].reset()
                            bit = self.mux_layers[layer][mux_idx](
                                sel_bit.unsqueeze(0), 
                                right[h, d, b].unsqueeze(0),
                                left[h, d, b].unsqueeze(0)
                            )
                            row_bits.append(bit)
                    new_rows.append(torch.cat(row_bits, dim=0).view(self.embed_dim, 32))
                
                current = torch.stack(new_rows, dim=0)
            
            results.append(current.squeeze(0))
        
        result = torch.stack(results, dim=0)
        return result.view(batch_shape + (self.embed_dim, 32))
    
    def _int_to_binary_pulse(self, x: torch.Tensor) -> torch.Tensor:
        """整数 → 二进制脉冲 (MSB first)"""
        device = x.device
        result = []
        for i in range(self.addr_bits - 1, -1, -1):
            bit = ((x >> i) & 1).float()
            result.append(bit)
        return torch.stack(result, dim=-1)
    
    def _float_to_fp32_pulse(self, x: torch.Tensor) -> torch.Tensor:
        """浮点 → FP32脉冲"""
        device = x.device
        shape = x.shape
        x_flat = x.flatten().cpu().numpy()
        
        pulses = []
        for val in x_flat:
            bits = struct.unpack('>I', struct.pack('>f', val))[0]
            pulse = [float((bits >> (31 - i)) & 1) for i in range(32)]
            pulses.append(pulse)
        
        return torch.tensor(pulses, dtype=torch.float32, device=device).view(shape + (32,))
    
    def reset(self):
        for layer in self.mux_layers:
            for mux in layer:
                mux.reset()


