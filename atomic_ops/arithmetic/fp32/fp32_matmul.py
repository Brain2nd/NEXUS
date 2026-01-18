"""
FP32 脉冲域矩阵乘法 - 100%纯SNN门电路实现
=============================================

用于 backward 中的梯度计算:
- grad_x = grad_output @ W
- grad_W = grad_output^T @ x

使用 SpikeFP32Multiplier 进行乘法，SpikeFP32Adder 进行累加。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn
from .fp32_mul import SpikeFP32Multiplier
from .fp32_adder import SpikeFP32Adder


class SpikeFP32MatMul(nn.Module):
    """脉冲域矩阵乘法: C = A @ B

    A: [..., M, K, 32] pulse
    B: [K, N, 32] pulse  (注意: K 在第一维)
    C: [..., M, N, 32] pulse

    使用纯 SNN 门电路实现:
    - SpikeFP32Multiplier 进行逐元素乘法
    - SpikeFP32Adder 进行累加

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """

    def __init__(self, neuron_template=None):
        super().__init__()
        self.mul = SpikeFP32Multiplier(neuron_template=neuron_template)
        self.adder = SpikeFP32Adder(neuron_template=neuron_template)

    def forward(self, A, B):
        """
        A: [..., M, K, 32] 脉冲矩阵
        B: [K, N, 32] 脉冲矩阵
        Returns: [..., M, N, 32] 脉冲矩阵

        计算 C[..., i, j] = sum_k(A[..., i, k] * B[k, j])
        """
        device = A.device

        # 获取维度
        batch_shape = A.shape[:-3]  # [...] 部分
        M = A.shape[-3]
        K = A.shape[-2]
        assert B.shape[-3] == K, f"Dimension mismatch: A has K={K}, B has K={B.shape[-3]}"
        N = B.shape[-2]

        # 结果形状: [..., M, N, 32]
        result_shape = batch_shape + (M, N, 32)

        # 逐元素计算 (可以优化为并行)
        # 为了支持 batch，我们需要扩展 B
        # A: [..., M, K, 32]
        # B: [K, N, 32]
        # 扩展 A: [..., M, 1, K, 32]
        # 扩展 B: [1, K, N, 32] -> 广播到 [..., M, K, N, 32]

        # 但这样的广播会导致内存爆炸
        # 改用循环实现，虽然慢但内存可控

        # 初始化结果
        result = []

        for i in range(M):
            row_results = []
            for j in range(N):
                # 计算 C[..., i, j] = sum_k(A[..., i, k, :] * B[k, j, :])
                # A[..., i, :, :]: [..., K, 32]
                # B[:, j, :]: [K, 32]

                A_row = A[..., i, :, :]  # [..., K, 32]
                B_col = B[:, j, :]  # [K, 32]

                # 广播并乘法
                # A_row: [..., K, 32]
                # B_col: [K, 32] -> 广播到 [..., K, 32]
                products = self.mul(A_row, B_col)  # [..., K, 32]

                # 累加
                acc = products[..., 0, :]  # [..., 32]
                for k in range(1, K):
                    acc = self.adder(acc, products[..., k, :])

                row_results.append(acc)  # [..., 32]

            # 堆叠一行: [..., N, 32]
            row_stack = torch.stack(row_results, dim=-2)
            result.append(row_stack)

        # 堆叠所有行: [..., M, N, 32]
        result = torch.stack(result, dim=-3)

        return result

    def reset(self):
        self.mul.reset()
        self.adder.reset()


class SpikeFP32MatMulTransposed(nn.Module):
    """脉冲域矩阵乘法 (B 转置): C = A @ B^T

    用于 Linear backward 中的 grad_x = grad_output @ W 计算。

    A: [..., M, K, 32] pulse  (grad_output: [..., out_features, 32] 展开)
    B: [N, K, 32] pulse       (W: [out_features, in_features, 32])
    C: [..., M, N, 32] pulse  (grad_x: [..., in_features, 32] 展开)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """

    def __init__(self, neuron_template=None):
        super().__init__()
        self.mul = SpikeFP32Multiplier(neuron_template=neuron_template)
        self.adder = SpikeFP32Adder(neuron_template=neuron_template)

    def forward(self, A, B):
        """
        A: [..., M, K, 32] 脉冲矩阵 (实际用于 [..., out_features, 32])
        B: [N, K, 32] 脉冲矩阵 (W 权重)
        Returns: [..., M, N, 32] 脉冲矩阵

        计算 C[..., i, j] = sum_k(A[..., i, k] * B[j, k])
        即 C = A @ B^T
        """
        device = A.device

        # 对于 Linear backward: grad_x = grad_output @ W
        # grad_output: [..., out_features, 32]  -> A: [..., 1, out_features, 32]
        # W: [out_features, in_features, 32]    -> B: [out_features, in_features, 32]
        # 结果: [..., 1, in_features, 32] -> squeeze -> [..., in_features, 32]

        # 获取维度
        if A.dim() == 2:
            # 特殊情况: A 是 [K, 32]
            A = A.unsqueeze(0)  # [1, K, 32]

        batch_shape = A.shape[:-2]  # [...] 部分，可能是空的
        K = A.shape[-2]  # out_features
        N = B.shape[0]  # in_features (因为 B 是 [N, K, 32])

        assert B.shape[-2] == K, f"Dimension mismatch: A has K={K}, B has K={B.shape[-2]}"

        # 结果形状: [..., N, 32]
        result = []

        for j in range(N):
            # 计算 C[..., j] = sum_k(A[..., k] * B[j, k])
            # A: [..., K, 32]
            # B[j, :, :]: [K, 32]

            B_row = B[j, :, :]  # [K, 32]

            # 广播并乘法
            products = self.mul(A, B_row)  # [..., K, 32]

            # 累加
            acc = products[..., 0, :]  # [..., 32]
            for k in range(1, K):
                acc = self.adder(acc, products[..., k, :])

            result.append(acc)  # [..., 32]

        # 堆叠: [..., N, 32]
        result = torch.stack(result, dim=-2)

        return result

    def reset(self):
        self.mul.reset()
        self.adder.reset()


class SpikeFP32OuterProduct(nn.Module):
    """脉冲域外积累加: C = sum_batch(A^T @ B)

    用于 Linear backward 中的 grad_W = sum_batch(grad_output^T @ x) 计算。

    A: [..., K, 32] pulse  (grad_output: [..., out_features, 32])
    B: [..., N, 32] pulse  (x: [..., in_features, 32])
    C: [K, N, 32] pulse    (grad_W: [out_features, in_features, 32])

    沿 batch 维度累加外积。

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """

    def __init__(self, neuron_template=None):
        super().__init__()
        self.mul = SpikeFP32Multiplier(neuron_template=neuron_template)
        self.adder = SpikeFP32Adder(neuron_template=neuron_template)

    def forward(self, A, B):
        """
        A: [..., K, 32] 脉冲 (grad_output)
        B: [..., N, 32] 脉冲 (x)
        Returns: [K, N, 32] 脉冲 (grad_W)

        计算 C[i, j] = sum_batch(A[..., i] * B[..., j])
        """
        device = A.device

        # 获取维度
        K = A.shape[-2]  # out_features
        N = B.shape[-2]  # in_features

        # 展平 batch 维度
        batch_size = A[..., 0, 0].numel()

        # A: [..., K, 32] -> [batch_size, K, 32]
        A_flat = A.reshape(batch_size, K, 32)
        # B: [..., N, 32] -> [batch_size, N, 32]
        B_flat = B.reshape(batch_size, N, 32)

        # 初始化结果
        result = torch.zeros(K, N, 32, device=device)

        # 对每个 batch 样本计算外积并累加
        for b in range(batch_size):
            for i in range(K):
                for j in range(N):
                    # 计算 A_flat[b, i, :] * B_flat[b, j, :]
                    prod = self.mul(
                        A_flat[b, i, :].unsqueeze(0),
                        B_flat[b, j, :].unsqueeze(0)
                    ).squeeze(0)  # [32]

                    # 累加到结果
                    result[i, j, :] = self.adder(
                        result[i, j, :].unsqueeze(0),
                        prod.unsqueeze(0)
                    ).squeeze(0)

        return result

    def reset(self):
        self.mul.reset()
        self.adder.reset()


class SpikeFP32VecMul(nn.Module):
    """脉冲域逐元素乘法

    对两个相同形状的脉冲张量进行逐元素乘法。
    用于激活函数的 backward 梯度计算。

    A: [..., 32] pulse
    B: [..., 32] pulse
    C: [..., 32] pulse = A * B (逐元素)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """

    def __init__(self, neuron_template=None):
        super().__init__()
        self.mul = SpikeFP32Multiplier(neuron_template=neuron_template)

    def forward(self, A, B):
        """
        A, B: [..., 32] 脉冲张量
        Returns: [..., 32] 脉冲张量 = A * B
        """
        return self.mul(A, B)

    def reset(self):
        self.mul.reset()


class SpikeFP32VecAdd(nn.Module):
    """脉冲域逐元素加法

    对两个相同形状的脉冲张量进行逐元素加法。
    用于激活函数的 backward 梯度计算。

    A: [..., 32] pulse
    B: [..., 32] pulse
    C: [..., 32] pulse = A + B (逐元素)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """

    def __init__(self, neuron_template=None):
        super().__init__()
        self.adder = SpikeFP32Adder(neuron_template=neuron_template)

    def forward(self, A, B):
        """
        A, B: [..., 32] 脉冲张量
        Returns: [..., 32] 脉冲张量 = A + B
        """
        return self.adder(A, B)

    def reset(self):
        self.adder.reset()


class SpikeFP32VecSub(nn.Module):
    """脉冲域逐元素减法

    对两个相同形状的脉冲张量进行逐元素减法。
    实现方式: A - B = A + (-B)，通过翻转 B 的符号位实现取反。

    A: [..., 32] pulse
    B: [..., 32] pulse
    C: [..., 32] pulse = A - B (逐元素)

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """

    def __init__(self, neuron_template=None):
        super().__init__()
        self.adder = SpikeFP32Adder(neuron_template=neuron_template)
        from atomic_ops.core.logic_gates import NOTGate
        self.sign_not = NOTGate(neuron_template=neuron_template)

    def forward(self, A, B):
        """
        A, B: [..., 32] 脉冲张量
        Returns: [..., 32] 脉冲张量 = A - B
        """
        # 取反 B 的符号位
        neg_B_sign = self.sign_not(B[..., 0:1])
        neg_B = torch.cat([neg_B_sign, B[..., 1:]], dim=-1)

        # A + (-B)
        return self.adder(A, neg_B)

    def reset(self):
        self.adder.reset()
        self.sign_not.reset()
