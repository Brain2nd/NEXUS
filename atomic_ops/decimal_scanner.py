"""
十进制扫描器 - 实验性组件（未导出到核心系统）

注意: 此组件用于实验目的，不是核心 SNN 系统的一部分。
核心 SNN 系统使用二进制脉冲编码，不使用十进制阈值比较。
"""
import torch
from spikingjelly.activation_based import neuron, surrogate
from .vec_logic_gates import VecAND, VecNOT


class DecimalScanner(neuron.BaseNode):
    """十进制扫描器 - 实验性组件
    
    警告: 此组件使用阈值比较，不完全符合纯 SNN 原则。
    仅供实验和研究目的使用。
    """
    def __init__(self, T: int, surrogate_function=surrogate.ATan()):
        """
        纯粹的十进制扫描器。
        参数:
            T (int): 扫描总时间窗长度，也是起始最高幂次。
                     时间 t=0 对应阈值 10^T
                     时间 t=T-1 对应阈值 10^1
        """
        super().__init__(surrogate_function=surrogate_function)
        self.T = T
        
        # 状态: 记录是否已经发放过脉冲 (自我闭锁用)
        self.register_memory('fired_flag', 0)
        self.register_memory('step_counter', 0)
        
        # 纯 SNN 门电路
        self.vec_not = VecNOT()  # 用于 NOT(fired_flag)
        self.vec_and = VecAND()  # 用于 should_fire AND NOT(fired_flag)

    def forward(self, x: torch.Tensor):
        # 1. 如果已经发过脉冲，直接输出 0 (进入无限不应期)
        if isinstance(self.fired_flag, torch.Tensor) and self.fired_flag.any():
             pass
        
        # 2. 计算当前动态阈值
        current_exp = self.T - self.step_counter
        threshold = 10.0 ** current_exp
        
        # 转为 tensor
        th_tensor = torch.tensor(threshold, device=x.device, dtype=x.dtype)
        
        # 3. 比较发放
        # 注意: (x >= th_tensor) 是阈值比较，这在真正的 SNN 硬件中
        # 应该通过多位比较器实现，这里为了实验简化
        should_fire = (x >= th_tensor).float()
        
        # 4. 使用纯 SNN 门电路屏蔽已发放的样本
        # real_fire = should_fire AND NOT(fired_flag)
        if isinstance(self.fired_flag, (int, float)):
            fired_tensor = torch.full_like(should_fire, self.fired_flag)
        else:
            fired_tensor = self.fired_flag
        
        not_fired = self.vec_not(fired_tensor)
        real_fire = self.vec_and(should_fire, not_fired)
        
        # 5. 更新状态 - 使用 OR 逻辑 (fired_flag OR real_fire)
        # 但由于 fired_flag 只能从 0->1，这等价于加法
        self.fired_flag = fired_tensor + real_fire
        self.fired_flag = torch.clamp(self.fired_flag, 0.0, 1.0)
        self.step_counter += 1
        
        return real_fire

    def reset(self):
        super().reset()
        self.fired_flag = 0.
        self.step_counter = 0
        self.vec_not.reset()
        self.vec_and.reset()
