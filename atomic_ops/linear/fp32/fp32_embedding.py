"""
FP32 Embedding层 - 100%纯SNN门电路实现
========================================

整数索引 → 二进制脉冲 → MUX树选择 → embedding向量脉冲

原理：
- 整数和浮点数本质上都是二进制，都用脉冲表示
- token_id编码为log2(vocab_size)位二进制脉冲
- 用MUX树根据地址位逐层选择，O(log V)复杂度

使用示例：
```python
# 推理模式 (默认)
embedding = SpikeFP32Embedding(vocab_size=1000, embed_dim=64)
embedding.set_weight_from_float(weight_tensor)
y_pulse = embedding(token_ids)  # 纯 SNN

# 训练模式
embedding = SpikeFP32Embedding(vocab_size=1000, embed_dim=64, trainable=True)
embedding.train()
optimizer = torch.optim.Adam([embedding.weight_float], lr=1e-4)
```

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
import math
from atomic_ops.core.vec_logic_gates import VecMUX


class SpikeFP32Embedding(nn.Module):
    """FP32 Embedding层 - 使用MUX树选择（向量化实现）

    输入: token_ids [...] 整数张量
    输出: [..., embed_dim, 32] FP32脉冲

    实现:
    1. token_id → 二进制脉冲 (addr_bits位)
    2. MUX树：每层用1个地址位选择，共log2(vocab_size)层
    3. 输出选中行的embedding脉冲

    参数:
        vocab_size: 词表大小
        embed_dim: 嵌入维度
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        trainable: 是否启用 STE 训练模式
            - False (默认): 纯推理模式，权重为 buffer
            - True: 训练模式，权重为 Parameter，使用 STE 反向传播

    复杂度: O(log2(vocab_size)) 层MUX
    """
    def __init__(self, vocab_size: int, embed_dim: int, neuron_template=None,
                 trainable=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trainable = trainable
        nt = neuron_template

        # 计算地址位数
        self.addr_bits = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1

        # 脉冲权重缓存 (推理用) [padded_vocab_size, embed_dim, 32]
        self.register_buffer('weight_pulse', None)

        # 浮点权重 (训练用)
        if trainable:
            self.weight_float = nn.Parameter(
                torch.empty(vocab_size, embed_dim))
            nn.init.normal_(self.weight_float)
            self._weight_dirty = True
        else:
            self.register_buffer('weight_float', None)
            self._weight_dirty = False

        # MUX树：每层使用一个VecMUX处理所有位（向量化）
        self.vec_mux_layers = nn.ModuleList([
            VecMUX(neuron_template=nt) for _ in range(self.addr_bits)
        ])
        
    def _sync_weight_pulse(self):
        """同步 float 权重到 pulse 缓存"""
        if self.trainable and self._weight_dirty:
            # 扩展到2的幂次方便MUX树处理
            padded_size = 2 ** self.addr_bits
            weight_data = self.weight_float.data
            if padded_size > self.vocab_size:
                padding = torch.zeros(padded_size - self.vocab_size, self.embed_dim,
                                      device=weight_data.device)
                weight_data = torch.cat([weight_data, padding], dim=0)
            self.weight_pulse = self._float_to_fp32_pulse(weight_data)
            self._weight_dirty = False

    def set_weight_from_float(self, weight_float: torch.Tensor):
        """从浮点权重设置脉冲权重"""
        assert weight_float.shape == (self.vocab_size, self.embed_dim)

        if self.trainable:
            # 训练模式：更新 Parameter
            with torch.no_grad():
                self.weight_float.copy_(weight_float)
            self._weight_dirty = True
        else:
            # 推理模式：直接设置 buffer
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
        # 同步权重 (如果需要)
        self._sync_weight_pulse()
        assert self.weight_pulse is not None, "请先设置权重"

        batch_shape = token_ids.shape

        # SNN 前向 (纯门电路)
        with torch.no_grad():
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

            out_pulse = torch.stack(results, dim=0)
            out_pulse = out_pulse.view(batch_shape + (self.embed_dim, 32))

        # 如果训练模式，用 STE 包装以支持梯度
        if self.trainable and self.training:
            from atomic_ops.core.ste import ste_embedding
            return ste_embedding(token_ids, self.weight_float, out_pulse)

        return out_pulse
    
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

    def train(self, mode=True):
        """切换训练模式时标记权重需要同步"""
        super().train(mode)
        if mode and self.trainable:
            self._weight_dirty = True
        return self


