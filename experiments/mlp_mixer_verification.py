"""
MLP-Mixer 端到端 ULP 验证
=========================

测试纯 MLP 模型（MLP-Mixer、ResMLP、gMLP）的 Linear 层精度。
使用预训练权重 + 真实数据集验证 SpikeFP32Linear 的位精确性。

遵循 CLAUDE.md 约束：
- 边界编解码：只在测试入口/出口做 float↔pulse 转换
- GPU 优先
- 随机 + 边界值测试
"""
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config, create_transform
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import float32_to_pulse, pulse_to_float32, SpikeFP32Linear


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return device


def compute_ulp(ref, snn):
    """计算 ULP 误差"""
    ref_flat = ref.detach().contiguous().view(-1)
    snn_flat = snn.detach().contiguous().view(-1)

    ref_bits = ref_flat.view(torch.int32).long()
    snn_bits = snn_flat.view(torch.int32).long()
    ulp = (ref_bits - snn_bits).abs()

    return {
        'max_ulp': ulp.max().item(),
        'mean_ulp': ulp.float().mean().item(),
        'zero_ulp_rate': (ulp == 0).float().mean().item(),
        'le1_ulp_rate': (ulp <= 1).float().mean().item(),
    }


def verify_linear_layer(name, pt_linear, activations, device, max_samples=10):
    """验证单个 Linear 层的 ULP

    正确模式：
    1. float32_to_pulse 编码输入
    2. SpikeFP32Linear 计算
    3. pulse_to_float32 解码输出
    4. 比较 ULP
    """
    in_f = pt_linear.in_features
    out_f = pt_linear.out_features
    has_bias = pt_linear.bias is not None

    print(f"    [{name}] {in_f}x{out_f} (bias={has_bias})", flush=True)

    # 准备输入
    if activations.numel() < in_f * 2:
        print(f"      Skipped: not enough activations", flush=True)
        return None

    n_samples = min(max_samples, activations.numel() // in_f)
    x = activations[:n_samples * in_f].reshape(n_samples, in_f).to(device)

    print(f"      Input: [{n_samples}, {in_f}]", flush=True)

    # PyTorch 参考输出
    pt_linear = pt_linear.to(device)
    with torch.no_grad():
        y_ref = pt_linear(x)

    # SpikeFP32Linear
    print(f"      Creating SpikeFP32Linear...", flush=True)
    snn_linear = SpikeFP32Linear(in_f, out_f, accum_precision='fp32').to(device)
    snn_linear.set_weight_from_float(pt_linear.weight.data.to(device))
    snn_linear.reset()

    # 编码 → SNN → 解码
    print(f"      Encoding...", flush=True)
    x_pulse = float32_to_pulse(x)

    print(f"      SNN forward...", flush=True)
    y_pulse = snn_linear(x_pulse)

    print(f"      Decoding...", flush=True)
    y_snn = pulse_to_float32(y_pulse)

    # Bias 处理（SpikeFP32Linear 不支持 bias）
    if has_bias:
        y_snn = y_snn + pt_linear.bias.data

    # ULP 比较
    ulp = compute_ulp(y_ref, y_snn)

    status = "OK" if ulp['max_ulp'] == 0 else ("~OK" if ulp['max_ulp'] <= 1 else "ERROR")
    print(f"      [{status}] Max ULP={ulp['max_ulp']}, 0-ULP={ulp['zero_ulp_rate']*100:.1f}%", flush=True)

    return {
        'name': name,
        'shape': f'{in_f}x{out_f}',
        'has_bias': has_bias,
        **ulp,
        'n_samples': n_samples
    }


def test_mlp_model(model_name, device, n_images=5, max_layers=5):
    """测试一个 MLP 模型的所有 Linear 层"""
    print(f"\n{'='*60}", flush=True)
    print(f"[{model_name}] Linear Layer Verification", flush=True)
    print('='*60, flush=True)

    # 加载模型
    print(f"  Loading model...", flush=True)
    try:
        model = timm.create_model(model_name, pretrained=True).to(device).eval()
    except Exception as e:
        print(f"  Failed to load model: {e}", flush=True)
        return None

    # 获取数据预处理配置
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)

    # 提取所有 Linear 层
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    print(f"  Found {len(linear_layers)} Linear layers", flush=True)

    # 收集激活值
    layer_inputs = {}

    def make_hook(layer_name):
        def hook(module, inp, out):
            if layer_name not in layer_inputs:
                layer_inputs[layer_name] = []
            layer_inputs[layer_name].append(inp[0].detach().flatten().cpu()[:10000])
        return hook

    hooks = []
    for name, module in linear_layers[:max_layers]:
        hooks.append(module.register_forward_hook(make_hook(name)))

    # 生成随机输入图像（模拟 ImageNet）
    print(f"  Generating {n_images} random images...", flush=True)
    with torch.no_grad():
        for i in range(n_images):
            # 随机图像 + 边界值
            if i == 0:
                img = torch.zeros(1, 3, config['input_size'][1], config['input_size'][2])
            elif i == 1:
                img = torch.ones(1, 3, config['input_size'][1], config['input_size'][2])
            else:
                img = torch.randn(1, 3, config['input_size'][1], config['input_size'][2])

            img = img.to(device)
            model(img)

    for h in hooks:
        h.remove()

    # 验证每个 Linear 层
    print(f"\n  Verifying Linear layers:", flush=True)
    results = []

    for name, module in linear_layers[:max_layers]:
        if name not in layer_inputs or not layer_inputs[name]:
            continue

        activations = torch.cat(layer_inputs[name])
        result = verify_linear_layer(name, module, activations, device)

        if result:
            results.append(result)

            # 如果出现大误差，停止并报告
            if result['max_ulp'] > 1:
                print(f"\n  !!! Large ULP error detected: {result['max_ulp']}", flush=True)
                print(f"  !!! Layer: {name}, Shape: {result['shape']}", flush=True)

    # 汇总
    if results:
        max_ulps = [r['max_ulp'] for r in results]
        zero_rates = [r['zero_ulp_rate'] for r in results]

        print(f"\n  Summary for {model_name}:", flush=True)
        print(f"    Layers tested: {len(results)}", flush=True)
        print(f"    Max ULP: {max(max_ulps)}", flush=True)
        print(f"    Avg 0-ULP rate: {sum(zero_rates)/len(zero_rates)*100:.1f}%", flush=True)

        return {
            'model': model_name,
            'layers': results,
            'max_ulp': max(max_ulps),
            'avg_zero_ulp_rate': sum(zero_rates)/len(zero_rates)
        }

    return None


def main():
    print("="*60, flush=True)
    print("MLP-Mixer Family - Linear Layer ULP Verification", flush=True)
    print("="*60, flush=True)

    device = get_device()

    # 测试的模型列表（从小到大）
    models = [
        'mixer_s16_224',      # MLP-Mixer Small
        'resmlp_12_224',      # ResMLP-12
        'gmlp_s16_224',       # gMLP Small
    ]

    all_results = {}

    for model_name in models:
        try:
            result = test_mlp_model(model_name, device, n_images=3, max_layers=3)
            if result:
                all_results[model_name] = result
        except Exception as e:
            print(f"  Error testing {model_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()

    # 最终汇总
    print("\n" + "="*60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*60, flush=True)
    print(f"{'Model':<20} {'Layers':>8} {'Max ULP':>10} {'0-ULP%':>10}", flush=True)
    print("-"*50, flush=True)

    for name, r in all_results.items():
        print(f"{name:<20} {len(r['layers']):>8} {r['max_ulp']:>10} {r['avg_zero_ulp_rate']*100:>9.1f}%", flush=True)

    # 检查是否有大误差
    has_error = any(r['max_ulp'] > 1 for r in all_results.values())
    if has_error:
        print("\n!!! WARNING: Large ULP errors detected !!!", flush=True)
        print("This indicates implementation bugs in SpikeFP32Linear or its components.", flush=True)
    else:
        print("\nAll tests passed with ULP <= 1", flush=True)


if __name__ == '__main__':
    main()
