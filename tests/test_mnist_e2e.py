"""
实验四: MNIST端到端验证 (End-to-End Test)

对比PyTorch FP8和SNN FP8的推理准确率
目标：验证SNN实现与PyTorch完全一致
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

# 使用统一的转换函数和模型
from SNNTorch.atomic_ops import float_to_fp8_bits, fp8_bits_to_float
from SNNTorch.models.mnist_snn_infer import SNN_MLP


class PyTorch_FP8_MLP(nn.Module):
    """PyTorch FP8 MLP参考实现（用于对比验证）"""
    def __init__(self, weights):
        super().__init__()
        self.w1 = weights['fc1']  # [128, 784]
        self.w2 = weights['fc2']  # [10, 128]
    
    def forward(self, x):
        """
        Args:
            x: [batch, 784] 输入
        Returns:
            [batch, 10] logits
        """
        x = x.view(-1, 784)
        
        # Layer 1: FP8 matmul + ReLU
        x_fp8 = x.to(torch.float8_e4m3fn).to(torch.float32)
        w1_fp8 = self.w1.to(torch.float8_e4m3fn).to(torch.float32)
        h = F.linear(x_fp8, w1_fp8)
        h = F.relu(h)
        
        # Layer 2: FP8 matmul
        h_fp8 = h.to(torch.float8_e4m3fn).to(torch.float32)
        w2_fp8 = self.w2.to(torch.float8_e4m3fn).to(torch.float32)
        out = F.linear(h_fp8, w2_fp8)
        
        return out


def test_pytorch_fp8(model, test_loader, device, num_samples=None):
    """测试PyTorch FP8准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if num_samples and total >= num_samples:
                break
    
    return 100. * correct / total, correct, total


def test_snn_fp8(model, test_loader, device, num_samples=None):
    """测试SNN FP8准确率"""
    model.eval()
    correct = 0
    total = 0
    nan_count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 检查NaN
            if torch.isnan(output).any():
                nan_count += torch.isnan(output).any(dim=1).sum().item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if num_samples and total >= num_samples:
                break
    
    return 100. * correct / total, correct, total, nan_count


def compare_outputs(pytorch_model, snn_model, test_loader, device, num_samples=10):
    """详细对比两个模型的输出"""
    print("\n--- 输出对比（前{}个样本）---".format(num_samples))
    
    pytorch_model.eval()
    snn_model.eval()
    
    match_count = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            for i in range(min(num_samples - total, data.size(0))):
                sample = data[i:i+1]
                label = target[i].item()
                
                # PyTorch输出
                pt_out = pytorch_model(sample)
                pt_pred = pt_out.argmax(dim=1).item()
                
                # SNN输出
                snn_out = snn_model(sample)
                snn_pred = snn_out.argmax(dim=1).item()
                
                match = (pt_pred == snn_pred)
                match_count += int(match)
                total += 1
                
                status = "✓" if match else "✗"
                nan_flag = " (NaN!)" if torch.isnan(snn_out).any() else ""
                print(f"  {status} 样本{total}: Label={label}, PyTorch={pt_pred}, SNN={snn_pred}{nan_flag}")
                
                if total >= num_samples:
                    break
            
            if total >= num_samples:
                break
    
    print(f"\n预测一致率: {match_count}/{total} ({100*match_count/total:.1f}%)")
    return match_count, total


def main():
    print("="*70)
    print("实验四: MNIST端到端验证 (End-to-End Test)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据
    print("\n加载MNIST测试集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 加载权重
    weights_path = '/home/dgxspark/Desktop/HumanBrain/SNNTorch/models/mnist_fp8_weights.pt'
    weights = torch.load(weights_path, map_location=device)
    print(f"权重已加载: {weights_path}")
    
    # 创建模型
    pytorch_model = PyTorch_FP8_MLP(weights).to(device)
    snn_model = SNN_MLP(in_features=784, hidden_features=128, out_features=10, use_fast=True).to(device)
    snn_model.load_weights(weights)
    
    # ========== 小批量对比 ==========
    print("\n" + "="*70)
    print("Phase 1: 小批量详细对比")
    print("="*70)
    compare_outputs(pytorch_model, snn_model, test_loader, device, num_samples=20)
    
    # ========== 准确率测试 ==========
    print("\n" + "="*70)
    print("Phase 2: 准确率测试")
    print("="*70)
    
    # 测试样本数
    test_sizes = [100, 500, 1000]
    
    results = []
    for n in test_sizes:
        print(f"\n--- 测试 {n} 样本 ---")
        
        # PyTorch
        t0 = time.time()
        pt_acc, pt_correct, pt_total = test_pytorch_fp8(pytorch_model, test_loader, device, n)
        pt_time = time.time() - t0
        
        # SNN
        t0 = time.time()
        snn_acc, snn_correct, snn_total, nan_count = test_snn_fp8(snn_model, test_loader, device, n)
        snn_time = time.time() - t0
        
        print(f"  PyTorch FP8: {pt_acc:.2f}% ({pt_correct}/{pt_total}), 耗时={pt_time:.2f}s")
        print(f"  SNN FP8:     {snn_acc:.2f}% ({snn_correct}/{snn_total}), 耗时={snn_time:.2f}s, NaN={nan_count}")
        print(f"  差异: {abs(pt_acc - snn_acc):.2f}%")
        
        results.append({
            'n': n,
            'pt_acc': pt_acc,
            'snn_acc': snn_acc,
            'diff': abs(pt_acc - snn_acc),
            'nan_count': nan_count
        })
    
    # ========== 总结 ==========
    print("\n" + "="*70)
    print("实验四 结果汇总")
    print("="*70)
    
    print("\n| 样本数 | PyTorch FP8 | SNN FP8 | 差异 | NaN数 |")
    print("|--------|-------------|---------|------|-------|")
    for r in results:
        print(f"| {r['n']:>6} | {r['pt_acc']:>10.2f}% | {r['snn_acc']:>6.2f}% | {r['diff']:>4.2f}% | {r['nan_count']:>5} |")
    
    # 保存结果
    torch.save(results, '/home/dgxspark/Desktop/HumanBrain/SNNTorch/results/mnist_e2e_results.pt')
    
    print("\n结论:")
    avg_diff = sum(r['diff'] for r in results) / len(results)
    total_nan = sum(r['nan_count'] for r in results)
    
    if total_nan > 0:
        print(f"  ⚠ 存在{total_nan}个NaN输出，需要检查溢出处理")
    
    if avg_diff < 1.0:
        print(f"  ✓ 平均差异{avg_diff:.2f}%，SNN实现与PyTorch基本一致")
    else:
        print(f"  ✗ 平均差异{avg_diff:.2f}%，存在显著偏差")
    
    print("="*70)


if __name__ == "__main__":
    main()

