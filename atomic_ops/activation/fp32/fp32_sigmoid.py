"""
FP32 Sigmoid函数 - 100%纯SNN门电路实现
=========================================

sigmoid(x) = 1 / (1 + exp(-x))

使用已有的Exp、Adder、Divider组合实现。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.core.logic_gates import NOTGate, MUXGate, ANDGate, ORGate
from .fp32_exp import SpikeFP32Exp
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp32.fp32_div import SpikeFP32Divider


class SpikeFP32Sigmoid(nn.Module):
    """FP32 Sigmoid函数 - 100%纯SNN门电路实现

    sigmoid(x) = 1 / (1 + exp(-x))

    输入: x [..., 32] FP32脉冲
    输出: sigmoid(x) [..., 32] FP32脉冲

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式 (None/TrainingMode.STE/TrainingMode.TEMPORAL)（梯度流过）
    """
    def __init__(self, neuron_template=None, training_mode=None):
        super().__init__()
        self.training_mode = TrainingMode.validate(training_mode)
        nt = neuron_template

        # 组件
        self.exp = SpikeFP32Exp(neuron_template=nt)
        self.adder = SpikeFP32Adder(neuron_template=nt)
        self.divider = SpikeFP32Divider(neuron_template=nt)

        # 符号翻转 (用于计算-x)
        self.sign_not = NOTGate(neuron_template=nt, max_param_shape=(1,))

    def forward(self, x):
        """
        x: [..., 32] FP32脉冲
        Returns: sigmoid(x) [..., 32] FP32脉冲
        """
        device = x.device
        batch_shape = x.shape[:-1]

        # SNN 前向 (纯门电路)
        with torch.no_grad():
            # Step 1: 计算 -x (翻转符号位)
            neg_x_sign = self.sign_not(x[..., 0:1])
            neg_x = torch.cat([neg_x_sign, x[..., 1:]], dim=-1)

            # Step 2: exp(-x)
            exp_neg_x = self.exp(neg_x)

            # Step 3: 1 + exp(-x)
            c1 = self._make_constant(0x3F800000, batch_shape, device)
            one_plus_exp = self.adder(c1, exp_neg_x)

            # Step 4: 1 / (1 + exp(-x))
            out_pulse = self.divider(c1, one_plus_exp)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_sigmoid
            return ste_sigmoid(x, out_pulse)

        return out_pulse
    
    def _make_constant(self, bits, batch_shape, device):
        """从32位整数构造FP32脉冲常量"""
        pulse = torch.zeros(batch_shape + (32,), device=device)
        for i in range(32):
            pulse[..., i] = float((bits >> (31 - i)) & 1)
        return pulse
    
    def reset(self):
        self.exp.reset()
        self.adder.reset()
        self.divider.reset()
        self.sign_not.reset()

