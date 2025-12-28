"""
SpikeFP32Embedding 测试（端到端浮点验证）
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import torch.nn as nn
import struct

from SNNTorch.atomic_ops.pulse_decoder import PulseFP32Decoder


def float_to_bits(x: float) -> int:
    return struct.unpack('>I', struct.pack('>f', x))[0]


def test_embedding_basic():
    """基础测试（端到端浮点验证）"""
    print("\n" + "="*60)
    print("测试: 基础Embedding (vocab=8, dim=4)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_size = 8
    embed_dim = 4
    
    torch.manual_seed(42)
    ann_emb = nn.Embedding(vocab_size, embed_dim).to(device)
    
    from SNNTorch.atomic_ops.fp32_embedding import SpikeFP32Embedding
    snn_emb = SpikeFP32Embedding(vocab_size, embed_dim).to(device)
    snn_emb.from_nn_embedding(ann_emb)
    
    decoder = PulseFP32Decoder().to(device)
    
    print(f"词表: {vocab_size}, 地址位: {snn_emb.addr_bits}")
    
    # 穷举测试
    match = 0
    total = vocab_size * embed_dim
    
    for token_id in range(vocab_size):
        token = torch.tensor([token_id], device=device)
        
        ann_out = ann_emb(token).squeeze(0)
        snn_pulse = snn_emb(token).squeeze(0)
        
        decoder.reset()
        snn_out = decoder(snn_pulse)
        
        for d in range(embed_dim):
            if float_to_bits(ann_out[d].item()) == float_to_bits(snn_out[d].item()):
                match += 1
    
    print(f"匹配率: {match}/{total} = {100*match/total:.1f}%")
    return match == total


def test_embedding_batch():
    """批量测试（端到端浮点验证）"""
    print("\n" + "="*60)
    print("测试: 批量Embedding")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_size = 16
    embed_dim = 8
    
    torch.manual_seed(123)
    ann_emb = nn.Embedding(vocab_size, embed_dim).to(device)
    
    from SNNTorch.atomic_ops.fp32_embedding import SpikeFP32Embedding
    snn_emb = SpikeFP32Embedding(vocab_size, embed_dim).to(device)
    snn_emb.from_nn_embedding(ann_emb)
    
    decoder = PulseFP32Decoder().to(device)
    
    tokens = torch.tensor([0, 5, 10, 15], device=device)
    
    ann_out = ann_emb(tokens)
    snn_pulse = snn_emb(tokens)
    
    decoder.reset()
    snn_out = decoder(snn_pulse)
    
    match = 0
    total = tokens.shape[0] * embed_dim
    for i in range(tokens.shape[0]):
        for d in range(embed_dim):
            if float_to_bits(ann_out[i,d].item()) == float_to_bits(snn_out[i,d].item()):
                match += 1
    
    print(f"匹配率: {match}/{total} = {100*match/total:.1f}%")
    return match == total


def main():
    print("="*60)
    print("SpikeFP32Embedding 测试 (MUX树实现)")
    print("="*60)
    
    r1 = test_embedding_basic()
    r2 = test_embedding_batch()
    
    print("\n" + "="*60)
    if r1 and r2:
        print("🎉 全部通过 - 100%位精确")
    else:
        print("✗ 测试失败")
    print("="*60)


if __name__ == "__main__":
    main()

