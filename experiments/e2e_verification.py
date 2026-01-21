"""
端到端ULP验证
=============

用真实模型的Linear层权重 + 真实激活值，验证SpikeFP32Linear的位精确性。
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import float32_to_pulse, pulse_to_float32, SpikeFP32Linear


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    return device


def extract_linear_layers(model):
    """提取模型中的所有Linear层"""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))
    return layers


def verify_single_linear(name, pt_linear, input_activations, device):
    """验证单个Linear层的端到端ULP"""
    in_f = pt_linear.in_features
    out_f = pt_linear.out_features
    has_bias = pt_linear.bias is not None

    # 准备输入 - 取合适维度的激活值
    if input_activations.numel() < in_f * 10:
        return None  # 激活值不够

    # reshape为[batch, in_features]
    n_samples = min(100, input_activations.numel() // in_f)
    x = input_activations[:n_samples * in_f].reshape(n_samples, in_f).to(device)

    # PyTorch参考输出
    pt_linear = pt_linear.to(device)
    with torch.no_grad():
        y_ref = pt_linear(x)

    # SNN Linear (目前不支持bias)
    snn_linear = SpikeFP32Linear(in_f, out_f, accum_precision='fp32').to(device)
    snn_linear.set_weight_from_float(pt_linear.weight.data)
    snn_linear.reset()

    # SNN输出
    x_pulse = float32_to_pulse(x)
    y_pulse = snn_linear(x_pulse)
    y_snn = pulse_to_float32(y_pulse)

    # 如果有bias，需要手动加上（SNN Linear不支持bias）
    if has_bias:
        y_snn = y_snn + pt_linear.bias.data

    # ULP比较
    ref_bits = y_ref.view(-1).view(torch.int32).long()
    snn_bits = y_snn.view(-1).view(torch.int32).long()
    ulp_error = (ref_bits - snn_bits).abs()

    return {
        'name': name,
        'shape': f'{in_f}x{out_f}',
        'has_bias': has_bias,
        'max_ulp': ulp_error.max().item(),
        'mean_ulp': ulp_error.float().mean().item(),
        'zero_ulp_rate': (ulp_error == 0).float().mean().item(),
        'le1_ulp_rate': (ulp_error <= 1).float().mean().item(),
        'n_samples': n_samples
    }


def main():
    device = get_device()

    # 加载数据
    print("\n[1] Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # 测试模型
    model_configs = [
        ('ResNet-18', lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)),
        ('MobileNetV2', lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)),
    ]

    all_results = {}

    for model_name, loader in model_configs:
        print(f"\n{'='*60}")
        print(f"[{model_name}] End-to-End Verification")
        print('='*60)

        model = loader().to(device).eval()

        # 收集各层的输入激活值
        layer_inputs = {}

        def make_hook(name):
            def hook(module, input, output):
                if name not in layer_inputs:
                    layer_inputs[name] = []
                # 保存输入
                inp = input[0].detach().flatten().cpu()
                layer_inputs[name].append(inp[:50000])  # 限制大小
            return hook

        # 注册hook
        hooks = []
        linear_layers = extract_linear_layers(model)
        for name, module in linear_layers:
            hooks.append(module.register_forward_hook(make_hook(name)))

        # 运行推理收集激活
        print(f"  Collecting activations from {len(linear_layers)} Linear layers...")
        with torch.no_grad():
            for i, (images, _) in enumerate(testloader):
                if i >= 5:
                    break
                images = images.to(device)
                model(images)

        for h in hooks:
            h.remove()

        # 验证每个Linear层
        print(f"\n  Verifying each Linear layer:")
        results = []

        for name, module in linear_layers:
            if name not in layer_inputs or not layer_inputs[name]:
                continue

            activations = torch.cat(layer_inputs[name])
            result = verify_single_linear(name, module, activations, device)

            if result:
                results.append(result)
                status = "✓" if result['max_ulp'] <= 1 else "○"
                print(f"    {status} {name}: {result['shape']}, Max ULP={result['max_ulp']}, 0-ULP={result['zero_ulp_rate']*100:.1f}%")

        # 汇总
        if results:
            max_ulps = [r['max_ulp'] for r in results]
            zero_rates = [r['zero_ulp_rate'] for r in results]

            print(f"\n  Summary for {model_name}:")
            print(f"    Layers tested: {len(results)}")
            print(f"    Max ULP across all layers: {max(max_ulps)}")
            print(f"    Avg 0-ULP rate: {sum(zero_rates)/len(zero_rates)*100:.1f}%")
            print(f"    Layers with Max ULP ≤ 1: {sum(1 for u in max_ulps if u <= 1)}/{len(results)}")

            all_results[model_name] = {
                'layers': results,
                'summary': {
                    'n_layers': len(results),
                    'max_ulp': max(max_ulps),
                    'avg_zero_ulp_rate': sum(zero_rates)/len(zero_rates)
                }
            }

        del model
        torch.cuda.empty_cache()

    # 总结
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - End-to-End ULP Verification")
    print('='*60)

    for model_name, data in all_results.items():
        s = data['summary']
        print(f"\n{model_name}:")
        print(f"  Linear layers tested: {s['n_layers']}")
        print(f"  Max ULP: {s['max_ulp']}")
        print(f"  Avg 0-ULP rate: {s['avg_zero_ulp_rate']*100:.1f}%")


if __name__ == '__main__':
    main()
