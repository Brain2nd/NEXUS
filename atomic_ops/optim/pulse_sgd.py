"""
纯脉冲域 SGD 优化器
===================

完全在脉冲域中进行权重更新，使用 SNN 门电路:
- w_new = w - lr * grad
- 全部使用 SpikeFP32Multiplier 和 SpikeFP32Adder

符合 CLAUDE.md 纯 SNN 约束：无 Python 数学运算。

作者: MofNeuroSim Project
"""
import torch
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.core.logic_gates import NOTGate
from atomic_ops.encoding.converters import float32_to_pulse


class PulseSGD:
    """纯脉冲域 SGD 优化器

    w_new = w - lr * grad

    所有计算使用 SNN 门电路:
    - SpikeFP32Multiplier 计算 lr * grad
    - NOTGate 翻转符号位 (-lr * grad)
    - SpikeFP32Adder 计算 w + (-lr * grad)

    Args:
        pulse_params: 可迭代的脉冲参数 (generator 或 list)
        lr: 学习率 (float)
        neuron_template: 神经元模板，None 使用默认 IF 神经元

    使用示例:
        linear = SpikeFP32Linear_MultiPrecision(..., trainable=True)
        optimizer = PulseSGD(linear.pulse_parameters(), lr=0.01)

        for x, y in dataloader:
            output = model(x)
            loss = compute_loss(output, y)
            loss.backward()

            # 梯度在 backward 中已经计算为脉冲格式
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(self, pulse_params, lr=0.01, neuron_template=None):
        self.params = list(pulse_params)
        self.lr = lr

        # SNN 组件
        self.mul = SpikeFP32Multiplier(neuron_template=neuron_template)
        self.adder = SpikeFP32Adder(neuron_template=neuron_template)
        self.sign_not = NOTGate(neuron_template=neuron_template)

        # 预编码学习率为脉冲常量 (边界操作)
        self._lr_pulse = float32_to_pulse(torch.tensor(lr))
        self._device = None

    def _get_lr_pulse(self, device):
        """获取学习率脉冲 (处理设备迁移)"""
        if self._device != device:
            self._lr_pulse = self._lr_pulse.to(device)
            self._device = device
        return self._lr_pulse

    def zero_grad(self):
        """清除所有参数的脉冲梯度"""
        for p in self.params:
            if hasattr(p, 'grad_pulse'):
                p.grad_pulse = None
            # 同时清除 PyTorch 原生梯度 (如果有)
            if p.grad is not None:
                p.grad = None

    def step(self):
        """执行一步优化

        对于每个参数:
        1. 获取脉冲梯度
        2. update = lr * grad (SNN 乘法)
        3. neg_update = -update (符号位翻转)
        4. w_new = w + neg_update (SNN 加法)
        """
        for p in self.params:
            # 获取脉冲梯度
            grad_pulse = getattr(p, 'grad_pulse', None)
            if grad_pulse is None:
                continue

            device = p.device

            # lr_pulse: 广播到 grad 形状
            lr_pulse = self._get_lr_pulse(device)
            lr_expanded = lr_pulse.expand_as(grad_pulse)

            # update = lr * grad (SNN 乘法)
            update = self.mul(lr_expanded, grad_pulse)

            # neg_update = -update (符号位翻转)
            # FP32: 符号位是 bit 0 (MSB first)
            neg_update_sign = self.sign_not(update[..., 0:1])
            neg_update = torch.cat([neg_update_sign, update[..., 1:]], dim=-1)

            # w_new = w + neg_update (SNN 加法)
            with torch.no_grad():
                p.data = self.adder(p.data, neg_update)

    def reset(self):
        """重置 SNN 组件状态"""
        self.mul.reset()
        self.adder.reset()
        self.sign_not.reset()

    def state_dict(self):
        """返回优化器状态"""
        return {
            'lr': self.lr,
            'lr_pulse': self._lr_pulse.clone()
        }

    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.lr = state_dict['lr']
        self._lr_pulse = state_dict['lr_pulse'].clone()
        self._device = None


class PulseSGDWithMomentum(PulseSGD):
    """带动量的纯脉冲域 SGD 优化器

    v = momentum * v + grad
    w = w - lr * v

    Args:
        pulse_params: 可迭代的脉冲参数
        lr: 学习率 (float)
        momentum: 动量系数 (float)
        neuron_template: 神经元模板
    """

    def __init__(self, pulse_params, lr=0.01, momentum=0.9, neuron_template=None):
        super().__init__(pulse_params, lr, neuron_template)
        self.momentum = momentum

        # 动量脉冲常量 (边界操作)
        self._momentum_pulse = float32_to_pulse(torch.tensor(momentum))

        # 速度缓存 (脉冲格式)
        self.velocities = {}

    def _get_momentum_pulse(self, device):
        """获取动量脉冲"""
        if not hasattr(self, '_momentum_device') or self._momentum_device != device:
            self._momentum_pulse = self._momentum_pulse.to(device)
            self._momentum_device = device
        return self._momentum_pulse

    def step(self):
        """执行一步优化

        v = momentum * v + grad
        w = w - lr * v
        """
        for i, p in enumerate(self.params):
            grad_pulse = getattr(p, 'grad_pulse', None)
            if grad_pulse is None:
                continue

            device = p.device

            # 初始化速度
            if i not in self.velocities:
                self.velocities[i] = torch.zeros_like(p.data)

            v = self.velocities[i]

            # momentum * v (SNN 乘法)
            momentum_pulse = self._get_momentum_pulse(device)
            momentum_expanded = momentum_pulse.expand_as(v)
            mv = self.mul(momentum_expanded, v)

            # v = momentum * v + grad (SNN 加法)
            v_new = self.adder(mv, grad_pulse)
            self.velocities[i] = v_new

            # lr * v (SNN 乘法)
            lr_pulse = self._get_lr_pulse(device)
            lr_expanded = lr_pulse.expand_as(v_new)
            update = self.mul(lr_expanded, v_new)

            # neg_update = -update (符号位翻转)
            neg_update_sign = self.sign_not(update[..., 0:1])
            neg_update = torch.cat([neg_update_sign, update[..., 1:]], dim=-1)

            # w_new = w + neg_update (SNN 加法)
            with torch.no_grad():
                p.data = self.adder(p.data, neg_update)

    def zero_grad(self):
        """清除梯度 (保留速度)"""
        super().zero_grad()

    def reset(self):
        """重置 SNN 组件和速度缓存"""
        super().reset()
        self.velocities = {}

    def state_dict(self):
        """返回优化器状态"""
        state = super().state_dict()
        state['momentum'] = self.momentum
        state['velocities'] = {k: v.clone() for k, v in self.velocities.items()}
        return state

    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        super().load_state_dict(state_dict)
        self.momentum = state_dict['momentum']
        self.velocities = {k: v.clone() for k, v in state_dict['velocities'].items()}
