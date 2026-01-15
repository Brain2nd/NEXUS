"""
FP32 Embedding层 - 100%纯SNN门电路实现
========================================

整数索引 → 二进制脉冲 → MUX树选择 → embedding向量脉冲

原理：
- 整数和浮点数本质上都是二进制，都用脉冲表示
- token_id编码为log2(vocab_size)位二进制脉冲
- 用MUX树根据地址位逐层选择，O(log V)复杂度

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
import math
from .vec_logic_gates import VecMUX


class SpikeFP32Embedding(nn.Module):
    """FP32 Embedding层 - 使用MUX树选择（向量化实现）

    输入: token_ids [...] 整数张量
    输出: [..., embed_dim, 32] FP32脉冲

    实现:
    1. token_id → 二进制脉冲 (addr_bits位)
    2. MUX树：每层用1个地址位选择，共log2(vocab_size)层
    3. 输出选中行的embedding脉冲

    复杂度: O(log2(vocab_size)) 层MUX
    """
    def __init__(self, vocab_size: int, embed_dim: int, neuron_template=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        nt = neuron_template

        # 计算地址位数
        self.addr_bits = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1

        # 预编码的权重 [padded_vocab_size, embed_dim, 32]
        self.register_buffer('weight_pulse', None)

        # MUX树：每层使用一个VecMUX处理所有位（向量化）
        self.vec_mux_layers = nn.ModuleList([
            VecMUX(neuron_template=nt) for _ in range(self.addr_bits)
        ])
        
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
        self.reset()  # 重置所有MUX门

        batch_shape = token_ids.shape

        # Step 1: 整数 → 二进制脉冲 (边界操作)
        addr_pulse = self._int_to_binary_pulse(token_ids)  # [..., addr_bits]

        # 展平batch处理
        addr_flat = addr_pulse.reshape(-1, self.addr_bits)
        N = addr_flat.shape[0]

        results = []

        for n in range(N):
            addr = addr_flat[n]  # [addr_bits]
            current = self.weight_pulse  # [V, D, 32]

            for layer in range(self.addr_bits):
                sel_bit = addr[layer]  # scalar
                half = current.shape[0] // 2
                if half == 0:
                    break

                left = current[:half]        # [half, D, 32]
                right = current[half:2*half]  # [half, D, 32]

                # 向量化MUX选择：一次处理所有 half * D * 32 位
                left_flat = left.reshape(-1)   # [half * D * 32]
                right_flat = right.reshape(-1)  # [half * D * 32]
                sel_expanded = sel_bit.expand_as(left_flat)  # [half * D * 32]

                result_flat = self.vec_mux_layers[layer](sel_expanded, right_flat, left_flat)
                current = result_flat.view(half, self.embed_dim, 32)

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
        """浮点 → FP32脉冲 (纯PyTorch实现)"""
        device = x.device
        original_shape = x.shape

        # 使用 view 进行位重解释: float32 -> int32
        x_flat = x.flatten().to(torch.float32)
        bits_int = x_flat.view(torch.int32)

        # 提取每一位 (MSB-first)
        pulses = []
        for i in range(31, -1, -1):
            pulses.append(((bits_int >> i) & 1).float())

        result = torch.stack(pulses, dim=-1)  # [N, 32]
        return result.view(original_shape + (32,)).to(device)
    
    def reset(self):
        for mux in self.vec_mux_layers:
            mux.reset()


