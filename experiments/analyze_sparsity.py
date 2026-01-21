"""
稀疏性分析与脉冲发放率统计
============================

统计真实ANN模型的权重和激活值稀疏性，并使用MofNeuroSim组件计算脉冲发放率。

遵循 CLAUDE.md 规范:
- 使用 atomic_ops 组件
- GPU 优先
- Random + Boundary Values 测试

作者: MofNeuroSim Project
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    SpikeFP32Linear,
    float32_to_pulse,
    pulse_to_float32
)


def get_device():
    """获取设备（CUDA优先）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 1. 权重稀疏性统计
# ============================================================================

def weight_sparsity(model, threshold=1e-6):
    """统计权重中接近零的比例

    Args:
        model: PyTorch 模型
        threshold: 判断为零的阈值

    Returns:
        dict: 总参数量、零参数量、稀疏性比例
    """
    total_params = 0
    zero_params = 0
    layer_stats = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            numel = param.numel()
            zeros = (param.abs() < threshold).sum().item()
            total_params += numel
            zero_params += zeros
            layer_stats[name] = {
                'total': numel,
                'zeros': zeros,
                'sparsity': zeros / numel if numel > 0 else 0
            }

    return {
        'total_params': total_params,
        'zero_params': zero_params,
        'overall_sparsity': zero_params / total_params if total_params > 0 else 0,
        'layer_stats': layer_stats
    }


# ============================================================================
# 2. 激活值稀疏性统计
# ============================================================================

def activation_sparsity(model, dataloader, device, threshold=1e-6, max_batches=10):
    """Hook方式统计各层激活值稀疏性

    Args:
        model: PyTorch 模型
        dataloader: 数据加载器
        device: 计算设备
        threshold: 判断为零的阈值
        max_batches: 最大批次数

    Returns:
        dict: 各层的稀疏性统计
    """
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = {'total': 0, 'zero': 0}
            out = output.detach()
            activations[name]['total'] += out.numel()
            activations[name]['zero'] += (out.abs() < threshold).sum().item()
        return hook

    # 注册hook到所有Linear/Conv层
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 运行推理
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(device)
            model(images)

    # 移除hooks
    for h in hooks:
        h.remove()

    # 计算稀疏性
    result = {}
    for k, v in activations.items():
        result[k] = {
            'total': v['total'],
            'zeros': v['zero'],
            'sparsity': v['zero'] / v['total'] if v['total'] > 0 else 0
        }

    return result


# ============================================================================
# 3. 脉冲发放率计算（使用MofNeuroSim组件）
# ============================================================================

def pulse_firing_rate(values, device):
    """统计FP32编码后每个bit位的发放率

    Args:
        values: 浮点数张量
        device: 计算设备

    Returns:
        dict: 按位发放率统计
    """
    values = values.to(device).float()

    # 使用 float32_to_pulse 进行编码（边界转换函数）
    pulses = float32_to_pulse(values.flatten(), device=device)  # [N, 32]

    # 计算每个bit位的发放率
    firing_rates = pulses.float().mean(dim=0)  # [32]

    # 按IEEE 754 FP32结构分析 (MSB-first格式)
    # 索引0: sign, 索引1-8: exponent (8 bits), 索引9-31: mantissa (23 bits)
    sign_rate = firing_rates[0].item()
    exp_rates = firing_rates[1:9].tolist()  # 8 bits
    mantissa_rates = firing_rates[9:32].tolist()  # 23 bits

    return {
        'overall_firing_rate': pulses.float().mean().item(),
        'sign_bit_rate': sign_rate,
        'exponent_avg_rate': sum(exp_rates) / len(exp_rates),
        'exponent_rates': exp_rates,
        'mantissa_avg_rate': sum(mantissa_rates) / len(mantissa_rates),
        'mantissa_rates': mantissa_rates,
        'all_bit_rates': firing_rates.tolist()
    }


# ============================================================================
# 4. ULP误差验证（使用MofNeuroSim组件）
# ============================================================================

def verify_ulp_error(pytorch_linear, test_input, device):
    """验证转换后的位精确性 - 使用MofNeuroSim组件

    Args:
        pytorch_linear: PyTorch Linear层
        test_input: 测试输入
        device: 计算设备

    Returns:
        dict: ULP误差统计
    """
    test_input = test_input.to(device)
    pytorch_linear = pytorch_linear.to(device)

    # PyTorch 参考输出
    with torch.no_grad():
        y_ref = pytorch_linear(test_input)

    # 构建SNN Linear层（复制权重）
    snn_linear = SpikeFP32Linear(
        in_features=pytorch_linear.in_features,
        out_features=pytorch_linear.out_features,
        accum_precision='fp32'  # 100%对齐PyTorch
    ).to(device)
    snn_linear.set_weight_from_float(pytorch_linear.weight.data)
    # Note: SpikeFP32Linear does not support bias currently

    # 重置神经元状态（重要：每次推理前必须重置）
    snn_linear.reset()

    # SNN 编码 → 计算 → 解码（使用边界转换函数）
    x_pulse = float32_to_pulse(test_input, device=device)
    y_pulse = snn_linear(x_pulse)
    y_snn = pulse_to_float32(y_pulse)

    # 位级比较
    ref_bits = y_ref.view(-1).view(torch.int32)
    snn_bits = y_snn.view(-1).view(torch.int32)
    ulp_error = (ref_bits.long() - snn_bits.long()).abs()

    return {
        'max_ulp': ulp_error.max().item(),
        'mean_ulp': ulp_error.float().mean().item(),
        'zero_ulp_rate': (ulp_error == 0).float().mean().item(),
        'total_elements': ulp_error.numel()
    }


def get_boundary_test_values(device):
    """生成边界测试值（遵循CLAUDE.md: Random + Boundary Values）

    注意：排除 inf/nan 等特殊值，仅测试有效浮点数范围
    """
    boundary = torch.tensor([
        0.0, -0.0,                          # 零
        1.0, -1.0, 0.5, -0.5,               # 常用值
        1e-10, -1e-10,                       # 小值
        1e10, -1e10,                          # 大值（但在FP32范围内）
        1e-38, -1e-38,                       # 接近最小正规数
        1e38, -1e38,                          # 接近最大值
    ], device=device)

    # 添加随机值
    random_values = torch.randn(1000, device=device) * 100

    return torch.cat([boundary, random_values])


# ============================================================================
# 主函数
# ============================================================================

def main():
    device = get_device()
    print(f"Using device: {device}")

    results = {
        'device': str(device),
        'models': {}
    }

    # ========================================
    # 1. 下载和加载模型
    # ========================================
    print("\n" + "="*60)
    print("1. Loading pretrained models...")
    print("="*60)

    model_configs = {
        'ResNet-18': lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        'MobileNetV2': lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        'VGG-11': lambda: models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1),
    }

    loaded_models = {}
    for name, loader in model_configs.items():
        print(f"  Loading {name}...")
        try:
            model = loader()
            model = model.to(device)
            model.eval()
            loaded_models[name] = model
            print(f"    ✓ {name} loaded successfully")
        except Exception as e:
            print(f"    ✗ Failed to load {name}: {e}")

    # ========================================
    # 2. 下载数据集
    # ========================================
    print("\n" + "="*60)
    print("2. Loading dataset (ImageNet-style transform on CIFAR-10)...")
    print("="*60)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2
    )
    print("  ✓ CIFAR-10 test set loaded")

    # ========================================
    # 3. 权重稀疏性统计
    # ========================================
    print("\n" + "="*60)
    print("3. Analyzing weight sparsity...")
    print("="*60)

    for name, model in loaded_models.items():
        print(f"\n  [{name}]")
        ws = weight_sparsity(model)
        results['models'][name] = {'weight_sparsity': ws}
        print(f"    Total params: {ws['total_params']:,}")
        print(f"    Zero params:  {ws['zero_params']:,}")
        print(f"    Sparsity:     {ws['overall_sparsity']*100:.2f}%")

    # ========================================
    # 4. 激活值稀疏性统计
    # ========================================
    print("\n" + "="*60)
    print("4. Analyzing activation sparsity...")
    print("="*60)

    for name, model in loaded_models.items():
        print(f"\n  [{name}]")
        act_sparse = activation_sparsity(model, testloader, device, max_batches=5)
        results['models'][name]['activation_sparsity'] = act_sparse

        # 计算平均稀疏性
        sparsities = [v['sparsity'] for v in act_sparse.values()]
        if sparsities:
            avg_sparsity = sum(sparsities) / len(sparsities)
            min_sparsity = min(sparsities)
            max_sparsity = max(sparsities)
            print(f"    Layers analyzed: {len(sparsities)}")
            print(f"    Avg sparsity:    {avg_sparsity*100:.2f}%")
            print(f"    Min sparsity:    {min_sparsity*100:.2f}%")
            print(f"    Max sparsity:    {max_sparsity*100:.2f}%")
            results['models'][name]['activation_summary'] = {
                'avg': avg_sparsity,
                'min': min_sparsity,
                'max': max_sparsity
            }

    # ========================================
    # 5. 脉冲发放率分析
    # ========================================
    print("\n" + "="*60)
    print("5. Analyzing pulse firing rate (using PulseFloatingPointEncoder)...")
    print("="*60)

    # 收集所有模型的激活值样本
    all_activations = []

    for name, model in loaded_models.items():
        print(f"\n  [{name}]")

        # 用hook收集一些激活值
        sample_activations = []

        def collect_hook(module, input, output):
            sample_activations.append(output.detach().flatten()[:1000])  # 每层取1000个

        hooks = []
        for n, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                hooks.append(m.register_forward_hook(collect_hook))

        # 运行一个batch
        with torch.no_grad():
            for images, _ in testloader:
                images = images.to(device)
                model(images)
                break

        for h in hooks:
            h.remove()

        if sample_activations:
            act_tensor = torch.cat(sample_activations)
            all_activations.append(act_tensor)

            # 计算脉冲发放率
            fr = pulse_firing_rate(act_tensor, device)
            results['models'][name]['firing_rate'] = fr
            print(f"    Overall firing rate:  {fr['overall_firing_rate']*100:.2f}%")
            print(f"    Sign bit rate:        {fr['sign_bit_rate']*100:.2f}%")
            print(f"    Exponent avg rate:    {fr['exponent_avg_rate']*100:.2f}%")
            print(f"    Mantissa avg rate:    {fr['mantissa_avg_rate']*100:.2f}%")

    # 合并所有激活值计算总体发放率
    if all_activations:
        all_act = torch.cat(all_activations)
        overall_fr = pulse_firing_rate(all_act, device)
        results['overall_firing_rate'] = overall_fr
        print(f"\n  [Overall Statistics]")
        print(f"    Total samples:        {all_act.numel():,}")
        print(f"    Overall firing rate:  {overall_fr['overall_firing_rate']*100:.2f}%")

    # ========================================
    # 6. ULP误差验证
    # ========================================
    print("\n" + "="*60)
    print("6. Verifying 0 ULP error (using SpikeFP32Linear)...")
    print("="*60)

    # 测试一个简单的Linear层
    print("\n  Testing with random Linear layer...")
    test_linear = nn.Linear(64, 32, bias=False).to(device)

    # 创建测试输入：随机值 + 边界值
    boundary_values = get_boundary_test_values(device)
    # 扩展到需要的大小
    n_samples = 16
    in_features = 64
    test_input = torch.randn(n_samples, in_features, device=device)
    # 在第一个样本中混入边界值
    test_input[0, :min(len(boundary_values), in_features)] = boundary_values[:min(len(boundary_values), in_features)]

    try:
        ulp_result = verify_ulp_error(test_linear, test_input, device)
        results['ulp_verification'] = ulp_result
        print(f"    Max ULP:       {ulp_result['max_ulp']}")
        print(f"    Mean ULP:      {ulp_result['mean_ulp']:.6f}")
        print(f"    0-ULP Rate:    {ulp_result['zero_ulp_rate']*100:.2f}%")
        print(f"    Total tested:  {ulp_result['total_elements']}")

        if ulp_result['max_ulp'] == 0:
            print("    ✓ Bit-exact verification PASSED!")
        else:
            print("    ✗ ULP error detected")
    except Exception as e:
        print(f"    ✗ ULP verification failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # 7. 保存结果
    # ========================================
    print("\n" + "="*60)
    print("7. Saving results...")
    print("="*60)

    output_path = os.path.join(os.path.dirname(__file__), 'results', 'sparsity_stats.json')

    # 转换为可序列化格式
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(output_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)

    print(f"  ✓ Results saved to {output_path}")

    # ========================================
    # 8. 打印总结
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nWeight Sparsity:")
    for name in loaded_models:
        ws = results['models'][name]['weight_sparsity']
        print(f"  {name}: {ws['overall_sparsity']*100:.2f}%")

    print("\nActivation Sparsity (avg):")
    for name in loaded_models:
        if 'activation_summary' in results['models'][name]:
            avg = results['models'][name]['activation_summary']['avg']
            print(f"  {name}: {avg*100:.2f}%")

    print("\nPulse Firing Rate:")
    for name in loaded_models:
        if 'firing_rate' in results['models'][name]:
            fr = results['models'][name]['firing_rate']['overall_firing_rate']
            print(f"  {name}: {fr*100:.2f}%")

    if 'ulp_verification' in results:
        print(f"\nULP Verification: Max={results['ulp_verification']['max_ulp']}, "
              f"0-ULP Rate={results['ulp_verification']['zero_ulp_rate']*100:.2f}%")

    return results


if __name__ == '__main__':
    main()
