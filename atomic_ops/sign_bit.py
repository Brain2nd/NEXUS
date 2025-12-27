import torch
from spikingjelly.activation_based import neuron, surrogate

class SignBitNode(neuron.IFNode):
    def __init__(self, surrogate_function=surrogate.ATan(), **kwargs):
        """
        符号位判别组件。
        物理结构：前突触权重为 -1 的 0 阈值 IF 神经元。
        逻辑：
          - 输入负数 (<0) -> 突触后电流为正 -> 发放 (1)
          - 输入正数 (>=0) -> 突触后电流为负 -> 不发放 (0)
        """
        # 阈值设为极小正数 (近似为0)，防止浮点误差导致的误触
        super().__init__(v_threshold=1e-6, v_reset=None, surrogate_function=surrogate_function, **kwargs)
        
        # 定义固定的突触权重 W = -1
        self.synaptic_weight = -1.0

    def forward(self, x: torch.Tensor):
        # 1. 突触传递 (Synaptic Transmission)
        # 电流 = 输入 * 权重
        synaptic_current = x * self.synaptic_weight
        
        # 2. 胞体积分与发放 (Soma Integration & Firing)
        self.neuronal_charge(synaptic_current)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        
        return spike

    def reset(self):
        super().reset()
