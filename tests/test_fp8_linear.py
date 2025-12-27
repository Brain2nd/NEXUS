"""
测试 SpikeFP8Linear 与 PyTorch FP8 Linear 的位级精确性
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/HumanBrain')

from SNNTorch.atomic_ops import PulseFloatingPointEncoder, SpikeFP8Linear

def get_fp8_bits(x):
    """将float转换为FP8二进制位"""
    fp8 = x.to(torch.float8_e4m3fn)
    bits = fp8.view(torch.uint8)
    binary = []
    for i in range(7, -1, -1):
        binary.append(((bits >> i) & 1).float())
    return torch.stack(binary, dim=-1)

def fp8_tree_add(values, device):
    """树形累加FP8值列表"""
    if len(values) == 0:
        return torch.tensor(0.0, dtype=torch.float8_e4m3fn, device=device)
    if len(values) == 1:
        return values[0]
    
    # 树形累加
    current = values
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                # 有一对，相加
                s = (current[i].float() + current[i+1].float()).to(torch.float8_e4m3fn)
                next_level.append(s)
            else:
                # 奇数个，最后一个直接传递
                next_level.append(current[i])
        current = next_level
    return current[0]

def fp8_linear_reference(x_fp8, w_fp8):
    """PyTorch FP8 Linear参考实现（无bias）- 使用树形累加
    
    Args:
        x_fp8: [batch, in_features] FP8可表示的float值
        w_fp8: [out_features, in_features] FP8可表示的float值
    Returns:
        [batch, out_features] FP8结果
    """
    # 转为FP8进行计算
    x_f8 = x_fp8.to(torch.float8_e4m3fn)
    w_f8 = w_fp8.to(torch.float8_e4m3fn)
    
    batch, in_f = x_fp8.shape
    out_f = w_fp8.shape[0]
    
    result = torch.zeros(batch, out_f, device=x_fp8.device)
    
    for b in range(batch):
        for j in range(out_f):
            # 先计算所有乘积
            products = []
            for k in range(in_f):
                prod = (x_f8[b, k].float() * w_f8[j, k].float()).to(torch.float8_e4m3fn)
                products.append(prod)
            # 树形累加
            result[b, j] = fp8_tree_add(products, x_fp8.device).float()
    
    return result

def test_linear(in_features, out_features, batch_size, device):
    """测试指定配置的Linear层"""
    print(f"\n{'='*60}")
    print(f"测试: in={in_features}, out={out_features}, batch={batch_size}")
    print(f"{'='*60}")
    
    # 创建编码器 (FP8 E4M3)，增加scan_decimal_bits以支持完整范围
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, mantissa_bits=3,
        scan_integer_bits=8, scan_decimal_bits=16
    ).to(device)
    
    # 创建SNN Linear
    snn_linear = SpikeFP8Linear(in_features, out_features).to(device)
    
    # 生成随机FP8权重 (正数，因为当前加法器只支持正数相加)
    w_raw = torch.rand(out_features, in_features, device=device) * 2 + 0.5
    w_fp8 = w_raw.to(torch.float8_e4m3fn).float()  # 确保是FP8可表示值
    
    # 设置权重
    snn_linear.set_weight_from_float(w_fp8, encoder)
    
    # 生成随机FP8输入 (正数，避免加法器符号问题)
    x_raw = torch.rand(batch_size, in_features, device=device) * 4 + 0.5
    x_fp8 = x_raw.to(torch.float8_e4m3fn).float()
    
    print(f"输入样本: {x_fp8[0, :min(4, in_features)].tolist()}")
    print(f"权重样本: {w_fp8[0, :min(4, in_features)].tolist()}")
    
    # 编码输入为脉冲
    x_pulse = encoder(x_fp8)  # [batch, in_features, 1, 8]
    x_pulse = x_pulse.squeeze(-2)  # [batch, in_features, 8]
    
    # SNN前向
    y_pulse = snn_linear(x_pulse)  # [batch, out_features, 8]
    
    # 参考计算
    y_ref = fp8_linear_reference(x_fp8, w_fp8)  # [batch, out_features]
    y_ref_bits = get_fp8_bits(y_ref)  # [batch, out_features, 8]
    
    # 位级比较
    match = (y_pulse == y_ref_bits).all(dim=-1)
    num_match = match.sum().item()
    total = batch_size * out_features
    
    print(f"\n位级精确匹配: {num_match}/{total}")
    
    if num_match < total:
        # 显示不匹配的样本
        mismatch_idx = torch.where(~match)
        for i in range(min(3, len(mismatch_idx[0]))):
            b, j = mismatch_idx[0][i].item(), mismatch_idx[1][i].item()
            print(f"  不匹配[{b},{j}]: SNN={y_pulse[b,j].int().tolist()} vs Ref={y_ref_bits[b,j].int().tolist()}")
            print(f"    参考值: {y_ref[b,j].item()}")
    
    passed = num_match == total
    print(f"结果: {'PASS' if passed else 'FAIL'}")
    return passed

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*60)
    print("SpikeFP8Linear 位级精确测试")
    print("="*60)
    
    results = []
    
    # 测试不同配置
    configs = [
        (2, 2, 2),    # 最小配置
        (4, 3, 2),    # 小配置
        (4, 4, 4),    # 方阵
        (8, 4, 2),    # 较大输入
    ]
    
    for in_f, out_f, batch in configs:
        passed = test_linear(in_f, out_f, batch, device)
        results.append((f"in={in_f},out={out_f},batch={batch}", passed))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print("="*60)
    if all_pass:
        print("所有测试通过！(位级精确)")
    else:
        print("存在测试失败")
    print("="*60)

if __name__ == "__main__":
    main()
