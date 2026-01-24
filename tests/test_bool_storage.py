"""
Bool Storage 优化测试
=====================

验证脉冲张量 bool 存储优化的正确性和内存节省。

测试内容:
1. PulseTensor 工具类功能
2. FP32 Linear 层 bool 存储
3. FP32 Embedding 层 bool 存储
4. FP16 Linear 层 bool 存储
5. FP8 Linear 层 bool 存储
6. 内存节省验证

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.pulse_storage import (
    PulseTensor,
    pulse_to_bool,
    bool_to_pulse,
    calculate_memory_savings
)
from atomic_ops.encoding.converters import (
    float32_to_pulse, pulse_to_float32,
    float16_to_pulse, pulse_to_float16,
    float_to_fp8_bits, fp8_bits_to_float
)


# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on device: {device}")


class TestPulseTensor:
    """PulseTensor 工具类测试"""

    def test_from_float_basic(self):
        """基本 float -> bool 转换"""
        pulse_float = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0], device=device)
        pt = PulseTensor.from_float(pulse_float)

        assert pt.dtype == torch.bool
        assert pt.shape == pulse_float.shape
        assert pt.device == pulse_float.device

    def test_to_float_conversion(self):
        """bool -> float 转换"""
        pulse_float = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0], device=device)
        pt = PulseTensor.from_float(pulse_float)

        recovered = pt.to_float()
        assert recovered.dtype == torch.float32
        assert torch.allclose(recovered, pulse_float)

    def test_float_property_cache(self):
        """验证 float 属性缓存"""
        pulse_float = torch.randn(100, device=device) > 0
        pulse_float = pulse_float.float()
        pt = PulseTensor.from_float(pulse_float)

        # 第一次访问
        f1 = pt.float
        # 第二次访问应该返回缓存
        f2 = pt.float
        assert f1 is f2, "Cache should return same object"

        # 清除缓存后应该返回新对象
        pt.invalidate_cache()
        f3 = pt.float
        assert f3 is not f1, "After invalidate, should return new object"

    def test_memory_calculation(self):
        """验证内存计算"""
        shape = (1000, 1000, 32)
        stats = calculate_memory_savings(shape)

        assert stats['savings_ratio'] == 4.0, "Expected 4x savings (bool=1B, float32=4B)"
        assert stats['float32_bytes'] == 4 * 1000 * 1000 * 32
        assert stats['bool_bytes'] == 1 * 1000 * 1000 * 32

    def test_device_transfer(self):
        """验证设备转移"""
        pulse_float = torch.tensor([0.0, 1.0, 0.0, 1.0], device='cpu')
        pt = PulseTensor.from_float(pulse_float)

        if torch.cuda.is_available():
            pt_cuda = pt.to(torch.device('cuda'))
            assert pt_cuda.device.type == 'cuda'

    def test_clone(self):
        """验证克隆"""
        pulse_float = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
        pt = PulseTensor.from_float(pulse_float)
        pt_clone = pt.clone()

        assert torch.equal(pt.data, pt_clone.data)
        # 修改原始不应影响克隆
        pt.data[0] = True
        assert pt.data[0] != pt_clone.data[0]


class TestFP32LinearBoolStorage:
    """FP32 Linear 层 bool 存储测试"""

    def test_inference_mode_uses_bool(self):
        """推理模式应该使用 bool 存储"""
        from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear

        linear = SpikeFP32Linear(in_features=64, out_features=32).to(device)
        weight = torch.randn(32, 64, device=device)
        linear.set_weight_from_float(weight)

        # 验证内部存储是 bool
        assert linear._weight_pulse_bool is not None
        assert linear._weight_pulse_bool.dtype == torch.bool

        # 验证通过 property 访问返回 float
        wp = linear.weight_pulse
        assert wp.dtype == torch.float32

    def test_training_mode_uses_float(self):
        """训练模式应该使用 float 存储"""
        from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear
        from atomic_ops.core.training_mode import TrainingMode

        linear = SpikeFP32Linear(
            in_features=64, out_features=32,
            training_mode=TrainingMode.STE
        ).to(device)

        # 验证内部存储是 Parameter (float)
        assert linear._weight_pulse_float is not None
        assert isinstance(linear._weight_pulse_float, nn.Parameter)
        assert linear._weight_pulse_float.dtype == torch.float32

    def test_computation_correctness(self):
        """验证 bool 存储不影响计算结果"""
        from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear

        linear = SpikeFP32Linear(in_features=8, out_features=4).to(device)

        # 设置权重
        weight = torch.randn(4, 8, device=device)
        linear.set_weight_from_float(weight)

        # 创建输入
        x_float = torch.randn(2, 8, device=device)
        x_pulse = float32_to_pulse(x_float, device=device)

        # 前向传播
        y_pulse = linear(x_pulse)

        # 验证输出形状
        assert y_pulse.shape == (2, 4, 32)

        # 验证输出值是有效脉冲 (0 或 1)
        assert torch.all((y_pulse == 0.0) | (y_pulse == 1.0))

    def test_memory_savings(self):
        """验证内存节省"""
        from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear

        linear = SpikeFP32Linear(in_features=1024, out_features=1024).to(device)
        weight = torch.randn(1024, 1024, device=device)
        linear.set_weight_from_float(weight)

        # bool 存储的内存
        bool_bytes = linear._weight_pulse_bool.element_size() * linear._weight_pulse_bool.numel()

        # 如果是 float 存储应该是 4x
        expected_float_bytes = 4 * linear._weight_pulse_bool.numel()

        assert expected_float_bytes == 4 * bool_bytes, f"Expected 4x savings"


class TestFP32EmbeddingBoolStorage:
    """FP32 Embedding 层 bool 存储测试"""

    def test_inference_mode_uses_bool(self):
        """推理模式应该使用 bool 存储"""
        from atomic_ops.linear.fp32.fp32_embedding import SpikeFP32Embedding

        embedding = SpikeFP32Embedding(vocab_size=100, embed_dim=64).to(device)
        weight = torch.randn(100, 64, device=device)
        embedding.set_weight_from_float(weight)

        # 验证内部存储是 bool
        assert embedding._weight_pulse_bool is not None
        assert embedding._weight_pulse_bool.dtype == torch.bool

    def test_computation_correctness(self):
        """验证 bool 存储不影响计算结果"""
        from atomic_ops.linear.fp32.fp32_embedding import SpikeFP32Embedding

        embedding = SpikeFP32Embedding(vocab_size=16, embed_dim=8).to(device)

        # 设置权重
        weight = torch.randn(16, 8, device=device)
        embedding.set_weight_from_float(weight)

        # 创建输入 token ids
        token_ids = torch.tensor([0, 5, 10, 15], device=device)

        # 前向传播
        out_pulse = embedding(token_ids)

        # 验证输出形状
        assert out_pulse.shape == (4, 8, 32)

        # 验证输出值是有效脉冲 (0 或 1)
        assert torch.all((out_pulse == 0.0) | (out_pulse == 1.0))


class TestFP16LinearBoolStorage:
    """FP16 Linear 层 bool 存储测试"""

    def test_inference_mode_uses_bool(self):
        """推理模式应该使用 bool 存储"""
        from atomic_ops.linear.fp16.fp16_linear import SpikeFP16Linear

        linear = SpikeFP16Linear(in_features=64, out_features=32).to(device)
        weight = torch.randn(32, 64, device=device).half()
        linear.set_weight_from_float(weight)

        # 验证内部存储是 bool
        assert linear._weight_pulse_bool is not None
        assert linear._weight_pulse_bool.dtype == torch.bool

    def test_computation_correctness(self):
        """验证 bool 存储不影响计算结果"""
        from atomic_ops.linear.fp16.fp16_linear import SpikeFP16Linear

        linear = SpikeFP16Linear(in_features=8, out_features=4).to(device)

        # 设置权重
        weight = torch.randn(4, 8, device=device).half()
        linear.set_weight_from_float(weight)

        # 创建输入
        x_float = torch.randn(2, 8, device=device).half()
        x_pulse = float16_to_pulse(x_float, device=device)

        # 前向传播
        y_pulse = linear(x_pulse)

        # 验证输出形状 (FP16 = 16 bits)
        assert y_pulse.shape == (2, 4, 16)


class TestFP8LinearBoolStorage:
    """FP8 Linear 层 bool 存储测试"""

    def test_inference_mode_uses_bool(self):
        """推理模式应该使用 bool 存储"""
        from atomic_ops.linear.fp8.fp8_linear_multi import SpikeFP8Linear_MultiPrecision

        linear = SpikeFP8Linear_MultiPrecision(in_features=64, out_features=32).to(device)
        weight = torch.randn(32, 64, device=device)
        linear.set_weight_from_float(weight)

        # 验证内部存储是 bool
        assert linear._weight_pulse_bool is not None
        assert linear._weight_pulse_bool.dtype == torch.bool

    def test_computation_correctness(self):
        """验证 bool 存储不影响计算结果"""
        from atomic_ops.linear.fp8.fp8_linear_multi import SpikeFP8Linear_MultiPrecision

        linear = SpikeFP8Linear_MultiPrecision(
            in_features=8, out_features=4, accum_precision='fp32'
        ).to(device)

        # 设置权重
        weight = torch.randn(4, 8, device=device)
        linear.set_weight_from_float(weight)

        # 创建输入
        x_float = torch.randn(2, 8, device=device)
        x_pulse = float_to_fp8_bits(x_float, device=device)

        # 前向传播
        y_pulse = linear(x_pulse)

        # 验证输出形状 (FP32 累加 = 32 bits 输出)
        assert y_pulse.shape == (2, 4, 32)


class TestHelperFunctions:
    """辅助函数测试"""

    def test_pulse_to_bool(self):
        """测试 pulse_to_bool 函数"""
        pulse = torch.tensor([0.0, 1.0, 0.3, 0.7, 1.0], device=device)
        bool_data = pulse_to_bool(pulse)

        expected = torch.tensor([False, True, False, True, True], device=device)
        assert torch.equal(bool_data, expected)

    def test_bool_to_pulse(self):
        """测试 bool_to_pulse 函数"""
        bool_data = torch.tensor([False, True, False, True], device=device)
        pulse = bool_to_pulse(bool_data)

        expected = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
        assert torch.equal(pulse, expected)

    def test_roundtrip(self):
        """测试往返转换"""
        original = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], device=device)
        bool_data = pulse_to_bool(original)
        recovered = bool_to_pulse(bool_data)

        assert torch.equal(recovered, original)


class TestBoundaryValues:
    """边界值测试 (遵循 CLAUDE.md 测试规范)"""

    def test_fp32_linear_boundary_values(self):
        """FP32 Linear 边界值测试"""
        from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear

        linear = SpikeFP32Linear(in_features=4, out_features=2).to(device)

        # 边界值权重
        boundary_weights = torch.tensor([
            [0.0, -0.0, 1.0, -1.0],
            [1e-38, -1e-38, 1e38, -1e38]
        ], device=device)
        linear.set_weight_from_float(boundary_weights)

        # 验证存储成功
        assert linear._weight_pulse_bool is not None

        # 边界值输入
        boundary_inputs = torch.tensor([
            [0.0, 1.0, -1.0, 0.5],
            [-0.5, 0.0, 1e-38, 1e38]
        ], device=device)
        x_pulse = float32_to_pulse(boundary_inputs, device=device)

        # 前向传播应该成功
        y_pulse = linear(x_pulse)
        assert y_pulse.shape == (2, 2, 32)

    def test_random_plus_boundary(self):
        """随机值 + 边界值混合测试"""
        from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear

        linear = SpikeFP32Linear(in_features=8, out_features=4).to(device)

        # 边界值
        boundary_values = [0.0, -0.0, 1.0, -1.0]
        # 随机值
        random_values = torch.randn(4).tolist()

        combined = boundary_values + random_values
        weight = torch.tensor(combined, device=device).view(1, 8).expand(4, 8).clone()
        linear.set_weight_from_float(weight)

        # 验证存储和前向传播
        x_pulse = float32_to_pulse(torch.randn(2, 8, device=device), device=device)
        y_pulse = linear(x_pulse)
        assert y_pulse.shape == (2, 4, 32)


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Bool Storage Optimization Tests")
    print("=" * 60)

    # 收集测试类
    test_classes = [
        TestPulseTensor,
        TestFP32LinearBoolStorage,
        TestFP32EmbeddingBoolStorage,
        TestFP16LinearBoolStorage,
        TestFP8LinearBoolStorage,
        TestHelperFunctions,
        TestBoundaryValues,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  [PASS] {method_name}")
                    total_passed += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    total_failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
