"""
SNN FP8 MNIST 推理模型

使用纯SNN脉冲网络进行推理（仅推理，不支持训练）
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import torch.nn as nn
from SNNTorch.atomic_ops import (
    PulseFloatingPointEncoder, 
    SpikeFP8Linear_Fast,
    SpikeFP8Linear,
    SpikeFP8ReLU,
    float_to_fp8_bits,
    fp8_bits_to_float
)


class SNN_MLP(nn.Module):
    """纯SNN实现的MLP
    
    使用SpikeFP8Linear_Fast（空间编码加法器，1步延迟）
    """
    def __init__(self, in_features=784, hidden_features=128, out_features=10, use_fast=True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.use_fast = use_fast
        
        # 编码器
        self.encoder = PulseFloatingPointEncoder(
            exponent_bits=4, mantissa_bits=3,
            scan_integer_bits=8, scan_decimal_bits=16
        )
        
        # SNN Linear层
        if use_fast:
            self.fc1 = SpikeFP8Linear_Fast(in_features, hidden_features)
            self.fc2 = SpikeFP8Linear_Fast(hidden_features, out_features)
        else:
            self.fc1 = SpikeFP8Linear(in_features, hidden_features)
            self.fc2 = SpikeFP8Linear(hidden_features, out_features)
        
        # SNN ReLU激活函数 (符合SNN基本原则)
        self.relu = SpikeFP8ReLU()
    
    def load_weights(self, weights_dict):
        """加载PyTorch训练的权重"""
        # weights_dict['fc1']: [128, 784]
        # weights_dict['fc2']: [10, 128]
        self.fc1.set_weight_from_float(weights_dict['fc1'], self.encoder)
        self.fc2.set_weight_from_float(weights_dict['fc2'], self.encoder)
        print(f"权重已加载: fc1={weights_dict['fc1'].shape}, fc2={weights_dict['fc2'].shape}")
    
    def forward(self, x):
        """
        Args:
            x: [batch, 784] float输入（会先转换为FP8脉冲）
        Returns:
            [batch, 10] float输出（logits）
        """
        batch_size = x.shape[0]
        
        # Step 1: 将输入编码为FP8脉冲
        # x: [batch, 784] -> x_pulse: [batch, 784, 8]
        x_fp8 = x.to(torch.float8_e4m3fn).to(torch.float32)
        x_pulse = float_to_fp8_bits(x_fp8)
        
        # Step 2: FC1
        self.fc1.reset()
        h_pulse = self.fc1(x_pulse)  # [batch, 128, 8]
        
        # Step 3: ReLU (符号位屏蔽)
        self.relu.reset()
        h_pulse = self.relu(h_pulse)
        
        # Step 4: FC2
        self.fc2.reset()
        out_pulse = self.fc2(h_pulse)  # [batch, 10, 8]
        
        # Step 5: 解码为float
        out_float = fp8_bits_to_float(out_pulse)  # [batch, 10]
        
        return out_float
    
    def forward_verbose(self, x, print_intermediate=True):
        """带详细输出的前向传播（用于调试）"""
        batch_size = x.shape[0]
        
        # Step 1: 编码
        x_fp8 = x.to(torch.float8_e4m3fn).to(torch.float32)
        x_pulse = float_to_fp8_bits(x_fp8)
        
        if print_intermediate:
            print(f"输入: shape={x.shape}, 范围=[{x.min():.4f}, {x.max():.4f}]")
            print(f"FP8输入: 范围=[{x_fp8.min():.4f}, {x_fp8.max():.4f}]")
            print(f"脉冲编码: shape={x_pulse.shape}")
        
        # Step 2: FC1
        self.fc1.reset()
        h_pulse = self.fc1(x_pulse)
        h_float = fp8_bits_to_float(h_pulse)
        
        if print_intermediate:
            print(f"FC1输出: shape={h_pulse.shape}, float范围=[{h_float.min():.4f}, {h_float.max():.4f}]")
        
        # Step 3: ReLU
        self.relu.reset()
        h_relu_pulse = self.relu(h_pulse)
        h_relu_float = fp8_bits_to_float(h_relu_pulse)
        
        if print_intermediate:
            num_neg = (h_float < 0).sum().item()
            print(f"ReLU: {num_neg}个负值被屏蔽, 范围=[{h_relu_float.min():.4f}, {h_relu_float.max():.4f}]")
        
        # Step 4: FC2
        self.fc2.reset()
        out_pulse = self.fc2(h_relu_pulse)
        out_float = fp8_bits_to_float(out_pulse)
        
        if print_intermediate:
            print(f"FC2输出: shape={out_pulse.shape}, 范围=[{out_float.min():.4f}, {out_float.max():.4f}]")
        
        return out_float


def test_snn_inference(model, test_loader, device, num_samples=None):
    """测试SNN推理准确率
    
    Args:
        model: SNN_MLP模型
        test_loader: 测试数据加载器
        device: 设备
        num_samples: 测试样本数（None表示全部）
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)
            
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if num_samples and total >= num_samples:
                break
    
    acc = 100. * correct / total
    return acc, correct, total


def main():
    print("="*60)
    print("实验四 Step 2: SNN FP8 MNIST 推理")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载权重
    weights_path = '/home/dgxspark/Desktop/HumanBrain/SNNTorch/models/mnist_fp8_weights.pt'
    weights = torch.load(weights_path, map_location=device)
    print(f"\n加载权重: {weights_path}")
    
    # 创建SNN模型
    model = SNN_MLP(use_fast=True).to(device)
    model.load_weights(weights)
    
    # 测试单样本
    print("\n--- 单样本测试 ---")
    test_input = torch.randn(1, 784, device=device) * 0.5
    output = model.forward_verbose(test_input)
    print(f"预测类别: {output.argmax(dim=1).item()}")
    
    print("\n测试完成")


if __name__ == "__main__":
    main()

