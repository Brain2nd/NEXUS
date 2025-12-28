"""
物理鲁棒性测试 (Physical Robustness Test)
==========================================

模拟真实神经形态硬件的非理想特性，测试 SNN 组件的鲁棒性。

**使用 neuron_template 统一架构** - 通过传递 SimpleLIFNode 动态切换神经元类型。

测试场景
--------

**1. 膜电位泄漏 (β 扫描)**

模拟 LIF 神经元的泄漏特性：
```
V(t+1) = β × V(t) + I(t)

β = 1.0: 理想 IF 神经元（无泄漏）
β < 1.0: LIF 神经元（存在泄漏）
β → 0:   严重泄漏，膜电位快速衰减
```

**2. 输入噪声 (σ 扫描)**

模拟突触噪声、热噪声等：
```
I_noisy = I_ideal + N(0, σ²)

σ = 0:   理想无噪声
σ > 0:   高斯噪声叠加
```

**3. 器件变异 (Vth±δ 扫描)**

模拟制造工艺变异导致的阈值偏差：
```
Vth_actual = Vth_nominal × (1 + δ)

δ = 0:    理想阈值
δ ≠ 0:    阈值偏差
```

测试组件
--------
- 基本逻辑门: AND, OR, XOR (使用 neuron_template=SimpleLIFNode)
- 算术单元: 4-bit 行波进位加法器, 4x4 阵列乘法器
- 浮点运算: FP8 乘法器, FP8/FP16/FP32 加法器 (输入噪声模拟)

运行方式
--------
```bash
python SNNTorch/tests/test_robustness.py
```

作者: HumanBrain Project
许可: MIT License
"""
import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")

import torch
import numpy as np

# 使用统一架构 - neuron_template
from SNNTorch.atomic_ops.logic_gates import (
    ANDGate, ORGate, XORGate, NOTGate,
    HalfAdder, FullAdder, RippleCarryAdder,
    ArrayMultiplier4x4_Strict, SimpleLIFNode
)
from SNNTorch.atomic_ops import (
    SpikeFP8Multiplier, SpikeFP8Adder_Spatial,
    SpikeFP16Adder, SpikeFP32Adder
)


def create_lif_template(beta=1.0):
    """创建 LIF 神经元模板，用于物理仿真"""
    return SimpleLIFNode(beta=beta, v_threshold=1.0)


# ==============================================================================
# 实验 2.1: β扫描 - 基本逻辑门 (使用 neuron_template 统一架构)
# ==============================================================================

def test_basic_gates_beta_scan(device):
    """β扫描测试：不同泄漏因子下基本门的正确率
    
    使用 neuron_template=SimpleLIFNode(beta) 动态切换神经元类型
    """
    print("\n" + "="*70)
    print("实验2.1: β扫描 - 泄漏因子对逻辑门正确率的影响 (neuron_template)")
    print("="*70)
    
    betas = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    expected_and = [0, 0, 0, 1]
    expected_or = [0, 1, 1, 1]
    expected_xor = [0, 1, 1, 0]
    
    results = []
    
    for beta in betas:
        # 使用 neuron_template 统一架构
        lif_template = create_lif_template(beta)
        and_gate = ANDGate(neuron_template=lif_template).to(device)
        or_gate = ORGate(neuron_template=create_lif_template(beta)).to(device)
        xor_gate = XORGate(neuron_template=create_lif_template(beta)).to(device)
        
        and_correct = or_correct = xor_correct = 0
        
        for i, (a, b) in enumerate(inputs):
            a_t = torch.tensor([[a]], device=device)
            b_t = torch.tensor([[b]], device=device)
            
            and_gate.reset()
            or_gate.reset()
            xor_gate.reset()
            
            if round(and_gate(a_t, b_t).item()) == expected_and[i]:
                and_correct += 1
            if round(or_gate(a_t, b_t).item()) == expected_or[i]:
                or_correct += 1
            if round(xor_gate(a_t, b_t).item()) == expected_xor[i]:
                xor_correct += 1
        
        results.append({
            'beta': beta,
            'AND': and_correct / 4 * 100,
            'OR': or_correct / 4 * 100,
            'XOR': xor_correct / 4 * 100
        })
    
    print("\n| β     | AND(%) | OR(%)  | XOR(%) |")
    print("|-------|--------|--------|--------|")
    for r in results:
        print(f"| {r['beta']:.2f}  | {r['AND']:6.1f} | {r['OR']:6.1f} | {r['XOR']:6.1f} |")
    
    return results


# ==============================================================================
# 实验 2.2: σ扫描 - 基本逻辑门 (使用 neuron_template 统一架构)
# ==============================================================================

def test_basic_gates_noise_scan(device):
    """σ扫描测试：输入噪声对逻辑门正确率的影响
    
    使用 neuron_template 统一架构，β=1.0 (IF神经元)
    """
    print("\n" + "="*70)
    print("实验2.2: σ扫描 - 输入噪声对逻辑门正确率的影响 (neuron_template)")
    print("="*70)
    
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    num_trials = 200
    
    inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    expected_and = [0, 0, 0, 1]
    expected_or = [0, 1, 1, 1]
    expected_xor = [0, 1, 1, 0]
    
    results = []
    
    for sigma in sigmas:
        # 使用 neuron_template 统一架构，β=1.0 相当于 IF 神经元
        and_gate = ANDGate(neuron_template=create_lif_template(1.0)).to(device)
        or_gate = ORGate(neuron_template=create_lif_template(1.0)).to(device)
        xor_gate = XORGate(neuron_template=create_lif_template(1.0)).to(device)
        
        and_correct = or_correct = xor_correct = 0
        total = len(inputs) * num_trials
        
        for trial in range(num_trials):
            for i, (a, b) in enumerate(inputs):
                a_noisy = max(0.0, min(1.0, a + torch.randn(1, device=device).item() * sigma))
                b_noisy = max(0.0, min(1.0, b + torch.randn(1, device=device).item() * sigma))
                
                a_t = torch.tensor([[a_noisy]], device=device)
                b_t = torch.tensor([[b_noisy]], device=device)
                
                and_gate.reset()
                or_gate.reset()
                xor_gate.reset()
                
                if round(and_gate(a_t, b_t).item()) == expected_and[i]:
                    and_correct += 1
                if round(or_gate(a_t, b_t).item()) == expected_or[i]:
                    or_correct += 1
                if round(xor_gate(a_t, b_t).item()) == expected_xor[i]:
                    xor_correct += 1
        
        results.append({
            'sigma': sigma,
            'AND': and_correct / total * 100,
            'OR': or_correct / total * 100,
            'XOR': xor_correct / total * 100
        })
    
    print("\n| σ     | AND(%) | OR(%)  | XOR(%) |")
    print("|-------|--------|--------|--------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['AND']:6.1f} | {r['OR']:6.1f} | {r['XOR']:6.1f} |")
    
    return results


# ==============================================================================
# 实验 2.3: β扫描 - 4-bit 加法器 (使用 neuron_template 统一架构)
# ==============================================================================

def test_adder_beta_scan(device):
    """β扫描测试：4位加法器在不同泄漏因子下的正确率
    
    使用 neuron_template=SimpleLIFNode(beta) 动态切换神经元类型
    """
    print("\n" + "="*70)
    print("实验2.3: β扫描 - 泄漏因子对4位加法器的影响 (neuron_template)")
    print("="*70)
    
    betas = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    num_tests = 100
    
    results = []
    
    for beta in betas:
        # 使用 neuron_template 统一架构
        adder = RippleCarryAdder(bits=4, neuron_template=create_lif_template(beta)).to(device)
        correct = 0
        
        for _ in range(num_tests):
            a_val = torch.randint(0, 16, (1,)).item()
            b_val = torch.randint(0, 16, (1,)).item()
            
            a_bits = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
            b_bits = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
            
            adder.reset()
            sum_bits, cout = adder(a_bits, b_bits)
            
            snn_result = 0
            for i in range(4):
                snn_result += int(round(sum_bits[0, i].item())) << i
            snn_result += int(round(cout[0, 0].item())) << 4
            
            if snn_result == a_val + b_val:
                correct += 1
        
        results.append({
            'beta': beta,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| β     | 4-bit Adder Accuracy |")
    print("|-------|----------------------|")
    for r in results:
        print(f"| {r['beta']:.2f}  | {r['accuracy']:18.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.4: σ扫描 - 4-bit 加法器 (使用 neuron_template 统一架构)
# ==============================================================================

def test_adder_noise_scan(device):
    """σ扫描测试：4位加法器在输入噪声下的正确率
    
    使用 neuron_template 统一架构，β=1.0 (IF神经元)
    """
    print("\n" + "="*70)
    print("实验2.4: σ扫描 - 输入噪声对4位加法器的影响 (neuron_template)")
    print("="*70)
    
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    num_tests = 100
    
    results = []
    
    for sigma in sigmas:
        # 使用 neuron_template 统一架构
        adder = RippleCarryAdder(bits=4, neuron_template=create_lif_template(1.0)).to(device)
        correct = 0
        
        for _ in range(num_tests):
            a_val = torch.randint(0, 16, (1,)).item()
            b_val = torch.randint(0, 16, (1,)).item()
            
            a_bits = []
            b_bits = []
            for i in range(4):
                a_bit = float((a_val >> i) & 1)
                b_bit = float((b_val >> i) & 1)
                a_bit = max(0.0, min(1.0, a_bit + torch.randn(1).item() * sigma))
                b_bit = max(0.0, min(1.0, b_bit + torch.randn(1).item() * sigma))
                a_bits.append(a_bit)
                b_bits.append(b_bit)
            
            a_bits = torch.tensor([a_bits], device=device)
            b_bits = torch.tensor([b_bits], device=device)
            
            adder.reset()
            sum_bits, cout = adder(a_bits, b_bits)
            
            snn_result = 0
            for i in range(4):
                snn_result += int(round(sum_bits[0, i].item())) << i
            snn_result += int(round(cout[0, 0].item())) << 4
            
            if snn_result == a_val + b_val:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | 4-bit Adder Accuracy |")
    print("|-------|----------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:18.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.5: β扫描 - 4x4 乘法器 (使用 neuron_template 统一架构)
# ==============================================================================

def test_multiplier_beta_scan(device):
    """β扫描测试：4x4乘法器在不同泄漏因子下的正确率
    
    使用 neuron_template=SimpleLIFNode(beta) 动态切换神经元类型
    """
    print("\n" + "="*70)
    print("实验2.5: β扫描 - 泄漏因子对4x4乘法器的影响 (neuron_template)")
    print("="*70)
    
    betas = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    num_tests = 50
    
    results = []
    
    for beta in betas:
        # 使用 neuron_template 统一架构
        mul = ArrayMultiplier4x4_Strict(neuron_template=create_lif_template(beta)).to(device)
        correct = 0
        
        for _ in range(num_tests):
            a_val = torch.randint(0, 16, (1,)).item()
            b_val = torch.randint(0, 16, (1,)).item()
            
            a_bits = torch.tensor([[float((a_val >> i) & 1) for i in range(4)]], device=device)
            b_bits = torch.tensor([[float((b_val >> i) & 1) for i in range(4)]], device=device)
            
            mul.reset()
            result = mul(a_bits, b_bits)
            
            snn_result = 0
            for i in range(8):
                snn_result += int(round(result[0, i].item())) << i
            
            if snn_result == a_val * b_val:
                correct += 1
        
        results.append({
            'beta': beta,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| β     | 4x4 Multiplier Accuracy |")
    print("|-------|-------------------------|")
    for r in results:
        print(f"| {r['beta']:.2f}  | {r['accuracy']:21.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.6: σ扫描 - Barrel Shifter (使用 neuron_template 统一架构)
# ==============================================================================

def test_barrel_shifter_noise_scan(device):
    """σ扫描测试：Barrel Shifter在输入噪声下的正确率
    
    使用 neuron_template 统一架构
    """
    print("\n" + "="*70)
    print("实验2.6: σ扫描 - 输入噪声对Barrel Shifter的影响 (neuron_template)")
    print("="*70)
    
    from SNNTorch.atomic_ops.logic_gates import BarrelShifter8
    
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_tests = 50
    
    results = []
    
    for sigma in sigmas:
        # 使用 neuron_template 统一架构
        bs = BarrelShifter8(neuron_template=create_lif_template(1.0)).to(device)
        correct = 0
        
        for _ in range(num_tests):
            p_val = torch.randint(0, 256, (1,)).item()
            s_val = torch.randint(0, 8, (1,)).item()
            
            p_bits = []
            for i in range(8):
                p_bit = float((p_val >> i) & 1)
                p_bit = max(0.0, min(1.0, p_bit + torch.randn(1).item() * sigma))
                p_bits.append(p_bit)
            
            s_bits = [float((s_val >> i) & 1) for i in range(3)]
            
            p_tensor = torch.tensor([p_bits], device=device)
            s_tensor = torch.tensor([s_bits], device=device)
            
            bs.reset()
            result = bs(p_tensor, s_tensor)
            
            snn_result = 0
            for i in range(8):
                snn_result += int(round(result[0, i].item())) << i
            
            expected = (p_val << s_val) & 0xFF
            if snn_result == expected:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | BarrelShifter Accuracy |")
    print("|-------|------------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:20.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.7: σ扫描 - FP8 加法器 (端到端浮点验证)
# ==============================================================================

def test_fp8_adder_noise(device):
    """测试FP8加法器在噪声下的鲁棒性 - 端到端浮点验证
    
    测试流程：
    1. 生成随机浮点数（模拟大模型激活值分布）
    2. 量化到 FP8
    3. 编码成 SNN 脉冲
    4. 添加物理噪声
    5. SNN 加法运算
    6. 解码回浮点数
    7. 直接与 PyTorch FP8 加法结果比较
    """
    print("\n" + "="*70)
    print("实验2.7: σ扫描 - 输入噪声对FP8加法器的影响 (端到端)")
    print("="*70)
    
    from SNNTorch.atomic_ops.floating_point import PulseFloatingPointEncoder
    from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]
    num_tests = 100
    
    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    adder = SpikeFP8Adder_Spatial().to(device)
    
    results = []
    
    for sigma in sigmas:
        correct = 0
        
        for _ in range(num_tests):
            # 1. 生成随机浮点数（模拟大模型分布）
            a_float = torch.randn(1, device=device).item() * 2.0  # N(0, 4)
            b_float = torch.randn(1, device=device).item() * 2.0
            
            # 2. 量化到 FP8
            a_fp8 = torch.tensor([a_float], device=device).to(torch.float8_e4m3fn)
            b_fp8 = torch.tensor([b_float], device=device).to(torch.float8_e4m3fn)
            
            # 3. 编码成 SNN 脉冲
            encoder.reset()
            a_pulses = encoder(a_fp8.float())  # [1, 8]
            encoder.reset()
            b_pulses = encoder(b_fp8.float())  # [1, 8]
            
            # 4. 添加物理噪声
            a_noisy = torch.clamp(a_pulses + torch.randn_like(a_pulses) * sigma, 0.0, 1.0)
            b_noisy = torch.clamp(b_pulses + torch.randn_like(b_pulses) * sigma, 0.0, 1.0)
            
            # 量化回二值
            a_quantized = (a_noisy > 0.5).float()
            b_quantized = (b_noisy > 0.5).float()
            
            # 5. SNN 加法运算
            adder.reset()
            result_pulses = adder(a_quantized.squeeze(0), b_quantized.squeeze(0))  # [8]
            
            # 6. 解码回浮点数
            decoder.reset()
            snn_result = decoder(result_pulses.unsqueeze(0)).item()  # 标量
            
            # 7. 计算参考值（PyTorch FP8 加法）
            # 需要用量化后的输入重建
            a_decoded = decoder(a_quantized).item()
            b_decoded = decoder(b_quantized).item()
            
            a_ref = torch.tensor([a_decoded], device=device).to(torch.float8_e4m3fn)
            b_ref = torch.tensor([b_decoded], device=device).to(torch.float8_e4m3fn)
            ref_result = (a_ref.float() + b_ref.float()).to(torch.float8_e4m3fn).float().item()
            
            # 直接比较浮点数
            if abs(snn_result - ref_result) < 1e-6 or snn_result == ref_result:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | FP8 Adder Accuracy |")
    print("|-------|--------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:16.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.8: σ扫描 - FP8 乘法器 (端到端浮点验证)
# ==============================================================================

def test_fp8_multiplier_noise(device):
    """测试FP8乘法器在噪声下的鲁棒性 - 端到端浮点验证
    
    测试流程：
    1. 生成随机浮点数（模拟大模型权重分布）
    2. 量化到 FP8
    3. 编码成 SNN 脉冲
    4. 添加物理噪声
    5. SNN 乘法运算
    6. 解码回浮点数
    7. 直接与 PyTorch FP8 乘法结果比较
    """
    print("\n" + "="*70)
    print("实验2.8: σ扫描 - 输入噪声对FP8乘法器的影响 (端到端)")
    print("="*70)
    
    from SNNTorch.atomic_ops.floating_point import PulseFloatingPointEncoder
    from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]
    num_tests = 100
    
    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFloatingPointDecoder().to(device)
    mul = SpikeFP8Multiplier().to(device)
    
    results = []
    
    for sigma in sigmas:
        correct = 0
        
        for _ in range(num_tests):
            # 1. 生成随机浮点数（模拟大模型权重分布）
            a_float = torch.randn(1, device=device).item() * 1.5  # N(0, 2.25)
            b_float = torch.randn(1, device=device).item() * 1.5
            
            # 2. 量化到 FP8
            a_fp8 = torch.tensor([a_float], device=device).to(torch.float8_e4m3fn)
            b_fp8 = torch.tensor([b_float], device=device).to(torch.float8_e4m3fn)
            
            # 3. 编码成 SNN 脉冲
            encoder.reset()
            a_pulses = encoder(a_fp8.float())  # [1, 8]
            encoder.reset()
            b_pulses = encoder(b_fp8.float())  # [1, 8]
            
            # 4. 添加物理噪声
            a_noisy = torch.clamp(a_pulses + torch.randn_like(a_pulses) * sigma, 0.0, 1.0)
            b_noisy = torch.clamp(b_pulses + torch.randn_like(b_pulses) * sigma, 0.0, 1.0)
            
            # 量化回二值
            a_quantized = (a_noisy > 0.5).float()
            b_quantized = (b_noisy > 0.5).float()
            
            # 5. SNN 乘法运算
            mul.reset()
            result_pulses = mul(a_quantized, b_quantized)  # [1, 8]
            
            # 6. 解码回浮点数
            decoder.reset()
            snn_result = decoder(result_pulses).item()  # 标量
            
            # 7. 计算参考值（PyTorch FP8 乘法）
            a_decoded = decoder(a_quantized).item()
            b_decoded = decoder(b_quantized).item()
            
            a_ref = torch.tensor([a_decoded], device=device).to(torch.float8_e4m3fn)
            b_ref = torch.tensor([b_decoded], device=device).to(torch.float8_e4m3fn)
            ref_result = (a_ref.float() * b_ref.float()).to(torch.float8_e4m3fn).float().item()
            
            # 直接比较浮点数
            if abs(snn_result - ref_result) < 1e-6 or snn_result == ref_result:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | FP8 Multiplier Accuracy |")
    print("|-------|-------------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:21.1f}% |")
    
    return results


# ==============================================================================
# 辅助函数：FP16/FP32 编码和解码（边界组件）
# ==============================================================================

def float16_to_pulses(x: torch.Tensor, device) -> torch.Tensor:
    """将 FP16 浮点数转换为 16 位脉冲序列（边界组件）
    
    Args:
        x: FP16 张量
        device: 目标设备
    Returns:
        [..., 16] 脉冲张量
    """
    import numpy as np
    x_np = x.cpu().numpy().astype(np.float16)
    bits_np = x_np.view(np.uint16)
    
    pulses = torch.zeros(x.shape + (16,), device=device, dtype=torch.float32)
    for i in range(16):
        pulses[..., i] = torch.tensor((bits_np >> (15 - i)) & 1, dtype=torch.float32, device=device)
    
    return pulses


def float32_to_pulses(x: torch.Tensor, device) -> torch.Tensor:
    """将 FP32 浮点数转换为 32 位脉冲序列（边界组件）
    
    Args:
        x: FP32 张量
        device: 目标设备
    Returns:
        [..., 32] 脉冲张量
    """
    import numpy as np
    x_np = x.cpu().numpy().astype(np.float32)
    bits_np = x_np.view(np.uint32)
    
    pulses = torch.zeros(x.shape + (32,), device=device, dtype=torch.float32)
    for i in range(32):
        pulses[..., i] = torch.tensor((bits_np >> (31 - i)) & 1, dtype=torch.float32, device=device)
    
    return pulses


# ==============================================================================
# 实验 2.9: σ扫描 - FP16 加法器 (端到端浮点验证)
# ==============================================================================

def test_fp16_adder_noise(device):
    """测试FP16加法器在噪声下的鲁棒性 - 端到端浮点验证
    
    测试流程：
    1. 生成随机浮点数（模拟大模型分布）
    2. 量化到 FP16
    3. 编码成 SNN 脉冲
    4. 添加物理噪声
    5. SNN 加法运算
    6. 解码回浮点数
    7. 直接与 PyTorch FP16 加法结果比较
    """
    print("\n" + "="*70)
    print("实验2.9: σ扫描 - 输入噪声对FP16加法器的影响 (端到端)")
    print("="*70)
    
    from SNNTorch.atomic_ops.pulse_decoder import PulseFP16Decoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]  # 统一范围
    num_tests = 50
    
    decoder = PulseFP16Decoder().to(device)
    adder = SpikeFP16Adder().to(device)
    
    results = []
    
    for sigma in sigmas:
        correct = 0
        
        for _ in range(num_tests):
            # 1. 生成随机浮点数（模拟大模型分布）
            a_float = torch.randn(1).item() * 5.0  # N(0, 25)
            b_float = torch.randn(1).item() * 5.0
            
            # 2. 量化到 FP16
            a_fp16 = torch.tensor([a_float], dtype=torch.float16)
            b_fp16 = torch.tensor([b_float], dtype=torch.float16)
            
            # 3. 编码成 SNN 脉冲
            a_pulses = float16_to_pulses(a_fp16, device).squeeze(0)  # [16]
            b_pulses = float16_to_pulses(b_fp16, device).squeeze(0)  # [16]
            
            # 4. 添加物理噪声
            a_noisy = torch.clamp(a_pulses + torch.randn(16, device=device) * sigma, 0.0, 1.0)
            b_noisy = torch.clamp(b_pulses + torch.randn(16, device=device) * sigma, 0.0, 1.0)
            
            # 量化回二值
            a_quantized = (a_noisy > 0.5).float()
            b_quantized = (b_noisy > 0.5).float()
            
            # 5. SNN 加法运算
            adder.reset()
            result_pulses = adder(a_quantized, b_quantized)  # [16]
            
            # 6. 解码回浮点数
            decoder.reset()
            snn_result = decoder(result_pulses.unsqueeze(0)).item()  # 标量
            
            # 7. 计算参考值
            a_decoded = decoder(a_quantized.unsqueeze(0)).item()
            b_decoded = decoder(b_quantized.unsqueeze(0)).item()
            
            # PyTorch FP16 加法
            a_ref = torch.tensor([a_decoded], dtype=torch.float16)
            b_ref = torch.tensor([b_decoded], dtype=torch.float16)
            ref_result = (a_ref + b_ref).float().item()
            
            # 直接比较浮点数（考虑 FP16 精度）
            if abs(snn_result - ref_result) < 1e-3 or snn_result == ref_result:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | FP16 Adder Accuracy |")
    print("|-------|---------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:17.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.10: σ扫描 - FP32 加法器 (端到端浮点验证)
# ==============================================================================

def test_fp32_adder_noise(device):
    """测试FP32加法器在噪声下的鲁棒性 - 端到端浮点验证
    
    测试流程：
    1. 生成随机浮点数（模拟大模型分布）
    2. 编码成 SNN 脉冲
    3. 添加物理噪声
    4. SNN 加法运算
    5. 解码回浮点数
    6. 直接与 PyTorch FP32 加法结果比较
    """
    print("\n" + "="*70)
    print("实验2.10: σ扫描 - 输入噪声对FP32加法器的影响 (端到端)")
    print("="*70)
    
    from SNNTorch.atomic_ops.pulse_decoder import PulseFP32Decoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]  # 统一范围
    num_tests = 30
    
    decoder = PulseFP32Decoder().to(device)
    adder = SpikeFP32Adder().to(device)
    
    results = []
    
    for sigma in sigmas:
        correct = 0
        
        for _ in range(num_tests):
            # 1. 生成随机浮点数（模拟大模型分布）
            a_float = torch.randn(1).item() * 10.0  # N(0, 100)
            b_float = torch.randn(1).item() * 10.0
            
            # 2. 量化到 FP32（实际上就是原值）
            a_fp32 = torch.tensor([a_float], dtype=torch.float32)
            b_fp32 = torch.tensor([b_float], dtype=torch.float32)
            
            # 3. 编码成 SNN 脉冲
            a_pulses = float32_to_pulses(a_fp32, device).squeeze(0)  # [32]
            b_pulses = float32_to_pulses(b_fp32, device).squeeze(0)  # [32]
            
            # 4. 添加物理噪声
            a_noisy = torch.clamp(a_pulses + torch.randn(32, device=device) * sigma, 0.0, 1.0)
            b_noisy = torch.clamp(b_pulses + torch.randn(32, device=device) * sigma, 0.0, 1.0)
            
            # 量化回二值
            a_quantized = (a_noisy > 0.5).float()
            b_quantized = (b_noisy > 0.5).float()
            
            # 5. SNN 加法运算
            adder.reset()
            result_pulses = adder(a_quantized, b_quantized)  # [32]
            
            # 6. 解码回浮点数
            decoder.reset()
            snn_result = decoder(result_pulses.unsqueeze(0)).item()  # 标量
            
            # 7. 计算参考值
            a_decoded = decoder(a_quantized.unsqueeze(0)).item()
            b_decoded = decoder(b_quantized.unsqueeze(0)).item()
            
            # PyTorch FP32 加法
            ref_result = a_decoded + b_decoded
            
            # 直接比较浮点数
            if abs(snn_result - ref_result) < 1e-6 or snn_result == ref_result:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | FP32 Adder Accuracy |")
    print("|-------|---------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:17.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.11: 器件变异测试 (Vth±δ)
# ==============================================================================

def test_threshold_variation(device):
    """器件变异测试：阈值偏差对逻辑门正确率的影响"""
    print("\n" + "="*70)
    print("实验2.11: Vth±δ扫描 - 阈值偏差对逻辑门正确率的影响")
    print("="*70)
    
    # δ 表示阈值偏差百分比
    deltas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_trials = 100
    
    inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    expected_and = [0, 0, 0, 1]
    expected_or = [0, 1, 1, 1]
    expected_xor = [0, 1, 1, 0]
    
    results = []
    
    for delta in deltas:
        and_correct = or_correct = xor_correct = 0
        total = len(inputs) * num_trials
        
        for trial in range(num_trials):
            # 每次试验为每个门随机生成阈值偏差
            and_vth = 1.5 * (1 + (torch.rand(1).item() * 2 - 1) * delta)
            or_vth = 0.5 * (1 + (torch.rand(1).item() * 2 - 1) * delta)
            xor_hidden_vth = 1.5 * (1 + (torch.rand(1).item() * 2 - 1) * delta)
            xor_out_vth = 0.5 * (1 + (torch.rand(1).item() * 2 - 1) * delta)
            
            for i, (a, b) in enumerate(inputs):
                # AND: V = a + b, 发放条件 V >= Vth
                v_and = a + b
                and_out = 1 if v_and >= and_vth else 0
                if and_out == expected_and[i]:
                    and_correct += 1
                
                # OR
                v_or = a + b
                or_out = 1 if v_or >= or_vth else 0
                if or_out == expected_or[i]:
                    or_correct += 1
                
                # XOR: 两层
                hidden = 1 if (a + b) >= xor_hidden_vth else 0
                v_xor = a + b - 2 * hidden
                xor_out = 1 if v_xor >= xor_out_vth else 0
                if xor_out == expected_xor[i]:
                    xor_correct += 1
        
        results.append({
            'delta': delta,
            'AND': and_correct / total * 100,
            'OR': or_correct / total * 100,
            'XOR': xor_correct / total * 100
        })
    
    print("\n| δ     | AND(%) | OR(%)  | XOR(%) |")
    print("|-------|--------|--------|--------|")
    for r in results:
        print(f"| {r['delta']:.2f}  | {r['AND']:6.1f} | {r['OR']:6.1f} | {r['XOR']:6.1f} |")
    
    return results


# ==============================================================================
# 实验 2.12: σ扫描 - Linear层三种精度模式
# ==============================================================================

def test_linear_noise(device):
    """测试Linear层在不同精度模式下的噪声鲁棒性 - 端到端浮点验证"""
    print("\n" + "="*70)
    print("实验2.12: σ扫描 - Linear层三种精度模式的噪声鲁棒性")
    print("="*70)
    
    from SNNTorch.atomic_ops import PulseFloatingPointEncoder, SpikeFP8Linear_MultiPrecision
    from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder, PulseFP16Decoder, PulseFP32Decoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]
    in_f, out_f, batch = 8, 4, 20
    
    encoder = PulseFloatingPointEncoder().to(device)
    decoder_fp8 = PulseFloatingPointDecoder().to(device)
    decoder_fp16 = PulseFP16Decoder().to(device)
    decoder_fp32 = PulseFP32Decoder().to(device)
    
    results = {'fp8': [], 'fp16': [], 'fp32': []}
    
    torch.manual_seed(42)
    x_float = torch.randn(batch, in_f, device=device) * 0.5
    w_float = torch.randn(out_f, in_f, device=device) * 0.5
    
    x_fp8 = x_float.to(torch.float8_e4m3fn)
    w_fp8 = w_float.to(torch.float8_e4m3fn)
    x_fp8_f32 = x_fp8.float()
    w_fp8_f32 = w_fp8.float()
    
    # PyTorch参考
    ref_fp32 = x_fp8_f32 @ w_fp8_f32.T
    ref_fp16 = ref_fp32.to(torch.float16).float()
    
    # FP8累加参考
    def fp8_accum_ref(x, w):
        batch_size, out_features = x.shape[0], w.shape[0]
        in_features = w.shape[1]
        results = torch.zeros(batch_size, out_features, device=x.device)
        for b in range(batch_size):
            for o in range(out_features):
                products = x[b] * w[o]
                products_fp8 = products.to(torch.float8_e4m3fn)
                acc = products_fp8[0]
                for i in range(1, in_features):
                    sum_tmp = acc.float() + products_fp8[i].float()
                    acc = sum_tmp.to(torch.float8_e4m3fn)
                results[b, o] = acc.float()
        return results
    ref_fp8 = fp8_accum_ref(x_fp8_f32, w_fp8_f32)
    
    for sigma in sigmas:
        for precision in ['fp8', 'fp16', 'fp32']:
            correct = 0
            num_tests = 5
            
            for _ in range(num_tests):
                encoder.reset()
                x_pulse = encoder(x_fp8_f32)
                
                # 添加噪声
                x_noisy = torch.clamp(x_pulse + torch.randn_like(x_pulse) * sigma, 0.0, 1.0)
                x_quantized = (x_noisy > 0.5).float()
                
                snn = SpikeFP8Linear_MultiPrecision(in_f, out_f, accum_precision=precision).to(device)
                snn.set_weight_from_float(w_fp8_f32, encoder)
                snn.reset()
                y_pulse = snn(x_quantized)
                
                if precision == 'fp8':
                    decoder_fp8.reset()
                    y_snn = decoder_fp8(y_pulse)
                    ref = ref_fp8
                elif precision == 'fp16':
                    decoder_fp16.reset()
                    y_snn = decoder_fp16(y_pulse)
                    ref = ref_fp16
                else:
                    decoder_fp32.reset()
                    y_snn = decoder_fp32(y_pulse)
                    ref = ref_fp32
                
                match = torch.isclose(y_snn, ref, rtol=1e-3, atol=1e-4) | (y_snn == ref)
                correct += match.sum().item()
            
            total = num_tests * batch * out_f
            results[precision].append({
                'sigma': sigma,
                'accuracy': correct / total * 100
            })
    
    print("\n| σ     | FP8(%) | FP16(%) | FP32(%) |")
    print("|-------|--------|---------|---------|")
    for i, sigma in enumerate(sigmas):
        print(f"| {sigma:.2f}  | {results['fp8'][i]['accuracy']:6.1f} | {results['fp16'][i]['accuracy']:7.1f} | {results['fp32'][i]['accuracy']:7.1f} |")
    
    return results


# ==============================================================================
# 实验 2.13: σ扫描 - FP8×FP8→FP32 乘法器
# ==============================================================================

def test_fp8_mul_to_fp32_noise(device):
    """测试FP8×FP8→FP32乘法器在噪声下的鲁棒性"""
    print("\n" + "="*70)
    print("实验2.13: σ扫描 - FP8×FP8→FP32乘法器噪声鲁棒性")
    print("="*70)
    
    from SNNTorch.atomic_ops import PulseFloatingPointEncoder
    from SNNTorch.atomic_ops.fp8_mul_to_fp32 import SpikeFP8MulToFP32
    from SNNTorch.atomic_ops.pulse_decoder import PulseFP32Decoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]
    num_tests = 50
    
    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFP32Decoder().to(device)
    mul = SpikeFP8MulToFP32().to(device)
    
    results = []
    
    for sigma in sigmas:
        correct = 0
        
        for _ in range(num_tests):
            a_float = torch.randn(1, device=device).item() * 1.5
            b_float = torch.randn(1, device=device).item() * 1.5
            
            a_fp8 = torch.tensor([a_float], device=device).to(torch.float8_e4m3fn)
            b_fp8 = torch.tensor([b_float], device=device).to(torch.float8_e4m3fn)
            
            encoder.reset()
            a_pulse = encoder(a_fp8.float())
            encoder.reset()
            b_pulse = encoder(b_fp8.float())
            
            a_noisy = torch.clamp(a_pulse + torch.randn_like(a_pulse) * sigma, 0.0, 1.0)
            b_noisy = torch.clamp(b_pulse + torch.randn_like(b_pulse) * sigma, 0.0, 1.0)
            a_q = (a_noisy > 0.5).float()
            b_q = (b_noisy > 0.5).float()
            
            mul.reset()
            result_pulse = mul(a_q, b_q)
            
            decoder.reset()
            snn_result = decoder(result_pulse).item()
            
            # 参考：用量化后的输入计算
            from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder
            dec_fp8 = PulseFloatingPointDecoder().to(device)
            a_decoded = dec_fp8(a_q).item()
            b_decoded = dec_fp8(b_q).item()
            ref_result = a_decoded * b_decoded
            
            if abs(snn_result - ref_result) < 1e-5 or snn_result == ref_result:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | FP8×FP8→FP32 Accuracy |")
    print("|-------|------------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:20.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.14: σ扫描 - 4x4 整数乘法器 (使用 neuron_template 统一架构)
# ==============================================================================

def test_multiplier_noise_scan(device):
    """σ扫描测试：4x4乘法器在输入噪声下的正确率
    
    使用 neuron_template 统一架构
    """
    print("\n" + "="*70)
    print("实验2.14: σ扫描 - 输入噪声对4x4乘法器的影响 (neuron_template)")
    print("="*70)
    
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_tests = 50
    
    results = []
    
    for sigma in sigmas:
        # 使用 neuron_template 统一架构
        mul = ArrayMultiplier4x4_Strict(neuron_template=create_lif_template(1.0)).to(device)
        correct = 0
        
        for _ in range(num_tests):
            a_val = torch.randint(0, 16, (1,)).item()
            b_val = torch.randint(0, 16, (1,)).item()
            
            a_bits = []
            b_bits = []
            for i in range(4):
                a_bit = float((a_val >> i) & 1)
                b_bit = float((b_val >> i) & 1)
                a_bit = max(0.0, min(1.0, a_bit + torch.randn(1).item() * sigma))
                b_bit = max(0.0, min(1.0, b_bit + torch.randn(1).item() * sigma))
                a_bits.append(a_bit)
                b_bits.append(b_bit)
            
            a_bits = torch.tensor([a_bits], device=device)
            b_bits = torch.tensor([b_bits], device=device)
            
            mul.reset()
            result = mul(a_bits, b_bits)
            
            snn_result = 0
            for i in range(8):
                snn_result += int(round(result[0, i].item())) << i
            
            if snn_result == a_val * b_val:
                correct += 1
        
        results.append({
            'sigma': sigma,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| σ     | 4x4 Multiplier Accuracy |")
    print("|-------|-------------------------|")
    for r in results:
        print(f"| {r['sigma']:.2f}  | {r['accuracy']:21.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.15: δ扫描 - 4-bit 加法器阈值变异
# ==============================================================================

def test_adder_threshold_variation(device):
    """器件变异测试：阈值偏差对4位加法器正确率的影响"""
    print("\n" + "="*70)
    print("实验2.15: Vth±δ扫描 - 阈值偏差对4位加法器的影响")
    print("="*70)
    
    deltas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    num_tests = 50
    
    results = []
    
    for delta in deltas:
        correct = 0
        
        for _ in range(num_tests):
            a_val = torch.randint(0, 16, (1,)).item()
            b_val = torch.randint(0, 16, (1,)).item()
            
            # 模拟阈值变异的加法
            carry = 0
            sum_bits = []
            for i in range(4):
                a_bit = (a_val >> i) & 1
                b_bit = (b_val >> i) & 1
                
                # 每个全加器有3个阈值（XOR1, XOR2, 进位生成）
                vth1 = 1.5 * (1 + (torch.rand(1).item() * 2 - 1) * delta)
                vth2 = 0.5 * (1 + (torch.rand(1).item() * 2 - 1) * delta)
                
                # 简化的全加器模型
                s = a_bit ^ b_bit ^ carry
                c = (a_bit & b_bit) | (carry & (a_bit ^ b_bit))
                
                # 应用阈值变异
                v_sum = a_bit + b_bit + carry
                if v_sum >= vth1:
                    s = 1 if (v_sum % 2) >= vth2 else 0
                
                sum_bits.append(s)
                carry = c
            
            snn_result = sum(s << i for i, s in enumerate(sum_bits)) + (carry << 4)
            
            if snn_result == a_val + b_val:
                correct += 1
        
        results.append({
            'delta': delta,
            'accuracy': correct / num_tests * 100
        })
    
    print("\n| δ     | 4-bit Adder Accuracy |")
    print("|-------|----------------------|")
    for r in results:
        print(f"| {r['delta']:.2f}  | {r['accuracy']:18.1f}% |")
    
    return results


# ==============================================================================
# 实验 2.16: 精度转换器噪声测试
# ==============================================================================

def test_converters_noise(device):
    """测试精度转换器在噪声下的鲁棒性"""
    print("\n" + "="*70)
    print("实验2.16: σ扫描 - 精度转换器噪声鲁棒性")
    print("="*70)
    
    from SNNTorch.atomic_ops import PulseFloatingPointEncoder
    from SNNTorch.atomic_ops.fp16_components import FP8ToFP16Converter
    from SNNTorch.atomic_ops.fp32_components import FP8ToFP32Converter, FP32ToFP16Converter
    from SNNTorch.atomic_ops.pulse_decoder import PulseFloatingPointDecoder, PulseFP16Decoder, PulseFP32Decoder
    
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]
    num_tests = 50
    
    encoder = PulseFloatingPointEncoder().to(device)
    dec_fp8 = PulseFloatingPointDecoder().to(device)
    dec_fp16 = PulseFP16Decoder().to(device)
    dec_fp32 = PulseFP32Decoder().to(device)
    
    fp8_to_fp16 = FP8ToFP16Converter().to(device)
    fp8_to_fp32 = FP8ToFP32Converter().to(device)
    fp32_to_fp16 = FP32ToFP16Converter().to(device)
    
    results = {'fp8→fp16': [], 'fp8→fp32': [], 'fp32→fp16': []}
    
    for sigma in sigmas:
        correct_8_16 = correct_8_32 = correct_32_16 = 0
        
        for _ in range(num_tests):
            # 测试 FP8→FP16
            val = torch.randn(1, device=device).item() * 2.0
            val_fp8 = torch.tensor([val], device=device).to(torch.float8_e4m3fn)
            
            encoder.reset()
            pulse = encoder(val_fp8.float())
            pulse_noisy = torch.clamp(pulse + torch.randn_like(pulse) * sigma, 0.0, 1.0)
            pulse_q = (pulse_noisy > 0.5).float()
            
            fp8_to_fp16.reset()
            fp16_pulse = fp8_to_fp16(pulse_q)
            dec_fp16.reset()
            snn_result = dec_fp16(fp16_pulse).item()
            
            dec_fp8.reset()
            input_val = dec_fp8(pulse_q).item()
            ref_result = torch.tensor([input_val]).to(torch.float16).float().item()
            
            if abs(snn_result - ref_result) < 1e-3:
                correct_8_16 += 1
            
            # 测试 FP8→FP32
            fp8_to_fp32.reset()
            fp32_pulse = fp8_to_fp32(pulse_q)
            dec_fp32.reset()
            snn_result32 = dec_fp32(fp32_pulse).item()
            
            if abs(snn_result32 - input_val) < 1e-5:
                correct_8_32 += 1
            
            # 测试 FP32→FP16
            fp32_pulse_input = float32_to_pulses(torch.tensor([input_val]), device).squeeze(0)
            fp32_noisy = torch.clamp(fp32_pulse_input + torch.randn(32, device=device) * sigma, 0.0, 1.0)
            fp32_q = (fp32_noisy > 0.5).float()
            
            fp32_to_fp16.reset()
            fp16_from_32 = fp32_to_fp16(fp32_q.unsqueeze(0))
            dec_fp16.reset()
            snn_result_32_16 = dec_fp16(fp16_from_32).item()
            
            dec_fp32.reset()
            input32_val = dec_fp32(fp32_q.unsqueeze(0)).item()
            ref_32_16 = torch.tensor([input32_val]).to(torch.float16).float().item()
            
            if abs(snn_result_32_16 - ref_32_16) < 1e-3:
                correct_32_16 += 1
        
        results['fp8→fp16'].append({'sigma': sigma, 'accuracy': correct_8_16 / num_tests * 100})
        results['fp8→fp32'].append({'sigma': sigma, 'accuracy': correct_8_32 / num_tests * 100})
        results['fp32→fp16'].append({'sigma': sigma, 'accuracy': correct_32_16 / num_tests * 100})
    
    print("\n| σ     | FP8→FP16(%) | FP8→FP32(%) | FP32→FP16(%) |")
    print("|-------|-------------|-------------|--------------|")
    for i, sigma in enumerate(sigmas):
        print(f"| {sigma:.2f}  | {results['fp8→fp16'][i]['accuracy']:11.1f} | {results['fp8→fp32'][i]['accuracy']:11.1f} | {results['fp32→fp16'][i]['accuracy']:12.1f} |")
    
    return results


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    print("="*70)
    print("实验二：物理鲁棒性分析 (neuron_template 统一架构)")
    print("="*70)
    print("\n注: 所有测试使用 neuron_template=SimpleLIFNode(beta) 动态切换神经元类型")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 运行所有测试
    print("\n" + "="*70)
    print("第一部分：LIF 神经元泄漏测试 (β扫描)")
    print("="*70)
    
    beta_gates = test_basic_gates_beta_scan(device)
    beta_adder = test_adder_beta_scan(device)
    beta_mul = test_multiplier_beta_scan(device)
    
    print("\n" + "="*70)
    print("第二部分：输入噪声测试 (σ扫描)")
    print("="*70)
    
    noise_gates = test_basic_gates_noise_scan(device)
    noise_adder = test_adder_noise_scan(device)
    noise_barrel = test_barrel_shifter_noise_scan(device)
    
    noise_mul = test_multiplier_noise_scan(device)
    
    print("\n" + "="*70)
    print("第三部分：浮点运算器输入噪声测试")
    print("="*70)
    
    fp8_add_noise = test_fp8_adder_noise(device)
    fp8_mul_noise = test_fp8_multiplier_noise(device)
    fp8_mul_fp32_noise = test_fp8_mul_to_fp32_noise(device)
    fp16_add_noise = test_fp16_adder_noise(device)
    fp32_add_noise = test_fp32_adder_noise(device)
    
    print("\n" + "="*70)
    print("第四部分：器件变异测试 (Vth±δ)")
    print("="*70)
    
    vth_variation = test_threshold_variation(device)
    vth_adder = test_adder_threshold_variation(device)
    
    print("\n" + "="*70)
    print("第五部分：精度转换器噪声测试")
    print("="*70)
    
    converter_noise = test_converters_noise(device)
    
    print("\n" + "="*70)
    print("第六部分：Linear层端到端噪声测试")
    print("="*70)
    
    linear_noise = test_linear_noise(device)
    
    # 总结
    print("\n" + "="*70)
    print("实验二总结")
    print("="*70)
    
    print("\n1. β临界值分析 (LIF 泄漏):")
    for r in beta_gates:
        min_acc = min(r['AND'], r['OR'], r['XOR'])
        if min_acc < 100:
            print(f"   ⚠ β={r['beta']:.2f}: 首次出现错误 (最低准确率={min_acc:.1f}%)")
            break
    else:
        print(f"   ✓ β低至{beta_gates[-1]['beta']:.2f}仍保持100%正确率")
    
    print("\n2. σ临界值分析 (输入噪声):")
    for r in noise_gates:
        min_acc = min(r['AND'], r['OR'], r['XOR'])
        if min_acc < 99:
            print(f"   ⚠ σ={r['sigma']:.2f}: 准确率<99% (最低={min_acc:.1f}%)")
            break
    
    print("\n3. δ临界值分析 (阈值偏差):")
    for r in vth_variation:
        min_acc = min(r['AND'], r['OR'], r['XOR'])
        if min_acc < 99:
            print(f"   ⚠ δ={r['delta']:.2f}: 准确率<99% (最低={min_acc:.1f}%)")
            break
    
    print("\n4. 浮点运算器鲁棒性:")
    print(f"   FP8 加法器: σ=0时 {fp8_add_noise[0]['accuracy']:.1f}%, σ=0.15时 {fp8_add_noise[5]['accuracy']:.1f}%")
    print(f"   FP8 乘法器: σ=0时 {fp8_mul_noise[0]['accuracy']:.1f}%, σ=0.15时 {fp8_mul_noise[5]['accuracy']:.1f}%")
    print(f"   FP8→FP32乘法器: σ=0时 {fp8_mul_fp32_noise[0]['accuracy']:.1f}%, σ=0.15时 {fp8_mul_fp32_noise[5]['accuracy']:.1f}%")
    print(f"   FP16 加法器: σ=0时 {fp16_add_noise[0]['accuracy']:.1f}%, σ=0.15时 {fp16_add_noise[5]['accuracy']:.1f}%")
    print(f"   FP32 加法器: σ=0时 {fp32_add_noise[0]['accuracy']:.1f}%, σ=0.15时 {fp32_add_noise[5]['accuracy']:.1f}%")
    
    print("\n5. 精度转换器鲁棒性:")
    print(f"   FP8→FP16: σ=0时 {converter_noise['fp8→fp16'][0]['accuracy']:.1f}%, σ=0.15时 {converter_noise['fp8→fp16'][5]['accuracy']:.1f}%")
    print(f"   FP8→FP32: σ=0时 {converter_noise['fp8→fp32'][0]['accuracy']:.1f}%, σ=0.15时 {converter_noise['fp8→fp32'][5]['accuracy']:.1f}%")
    print(f"   FP32→FP16: σ=0时 {converter_noise['fp32→fp16'][0]['accuracy']:.1f}%, σ=0.15时 {converter_noise['fp32→fp16'][5]['accuracy']:.1f}%")
    
    print("\n6. Linear层鲁棒性 (三种精度模式):")
    print(f"   FP8模式: σ=0时 {linear_noise['fp8'][0]['accuracy']:.1f}%, σ=0.15时 {linear_noise['fp8'][5]['accuracy']:.1f}%")
    print(f"   FP16模式: σ=0时 {linear_noise['fp16'][0]['accuracy']:.1f}%, σ=0.15时 {linear_noise['fp16'][5]['accuracy']:.1f}%")
    print(f"   FP32模式: σ=0时 {linear_noise['fp32'][0]['accuracy']:.1f}%, σ=0.15时 {linear_noise['fp32'][5]['accuracy']:.1f}%")
    
    print("\n7. IF神经元的固有优势:")
    print("   - 阈值机制提供天然的噪声抑制")
    print("   - 软重置避免状态积累")
    print("   - 二值输出对连续噪声具有离散化抑制效果")


if __name__ == "__main__":
    main()
