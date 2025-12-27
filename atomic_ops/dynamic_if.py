import torch
from spikingjelly.activation_based import neuron, surrogate

class DynamicThresholdIFNode(neuron.IFNode):
    def __init__(self, N: int, NT: int = 0, surrogate_function=surrogate.ATan(), **kwargs):
        """
        参数:
            N (int): 整数部分位宽. 最高位阈值 2^(N-1).
            NT (int): 小数部分位宽. 最低位阈值 2^(-NT).
            surrogate_function: 替代梯度函数.
        """
        super().__init__(v_threshold=1.0, v_reset=None, surrogate_function=surrogate_function, **kwargs)
        self.N = N
        self.NT = NT
        self.step_counter = 0

    def forward(self, x: torch.Tensor):
        # 1. 积分
        self.neuronal_charge(x)
        
        # 2. 计算阈值: 2^(N-1), ..., 2^0, 2^-1, ... 2^-NT
        exponent = (self.N - 1) - self.step_counter
        
        # 停止条件: 当 exponent 小于 -NT 时停止
        if exponent < -self.NT:
             self.v_threshold = torch.tensor(1e9, device=x.device, dtype=x.dtype)
        else:
             self.v_threshold = torch.tensor(2.0 ** exponent, device=x.device, dtype=x.dtype)
            
        # 3. 发放
        spike = self.neuronal_fire()
        
        # 4. 减法重置
        self.neuronal_reset(spike)
        
        # 5. 更新计数
        self.step_counter += 1
        
        return spike

    def reset(self):
        super().reset()
        self.step_counter = 0
