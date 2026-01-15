"""
SNN 浮点编码器 (Pulse Floating-Point Encoder)
=============================================

将 ANN 浮点数转换为 SNN 二进制脉冲序列。

这是纯 SNN 系统的输入边界组件，使用时序动态阈值扫描实现。

编码原理
--------

**二进制扫描机制**:

使用动态阈值 IF 神经元，阈值按 2^k 指数递减扫描：

```
时间步 t:  0    1    2    ...   9   10   11   ...   19
阈值:    2^9  2^8  2^7  ...  2^0  2^-1 2^-2 ...  2^-10
```

当输入值 x 首次超过阈值时，神经元发放脉冲，该位置确定指数。

**FP8 E4M3 编码**:

```
输出: [S | E3 E2 E1 E0 | M2 M1 M0]

S (符号位):
  - x >= 0: S = 0
  - x < 0:  S = 1

E (指数):
  - 首脉冲位置 t → E = (N-1) - t + bias
  - bias = 7 (FP8 E4M3)

M (尾数):
  - Normal:    M = [spike(t+1), spike(t+2), spike(t+3)]
  - Subnormal: M = [spike(16), spike(17), spike(18)]
```

**支持的数值类型**:

| 类型       | 指数 E | 尾数 M | 值                        |
|------------|--------|--------|---------------------------|
| Zero       | 0      | 0      | ±0                        |
| Subnormal  | 0      | ≠0     | ±(M/8) × 2^(-6)           |
| Normal     | 1-14   | any    | ±(1 + M/8) × 2^(E-7)      |
| Max Normal | 14     | 7      | ±448                      |

使用示例
--------
```python
encoder = PulseFloatingPointEncoder()
x = torch.tensor([1.5, -0.25, 0.0])  # 浮点输入
pulse = encoder(x)  # [..., 8] 脉冲输出
```

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn
from .sign_bit import SignBitNode
from .dynamic_if import DynamicThresholdIFNode
from .logic_gates import (
    FirstSpikeDetector, TemporalExponentGenerator, DelayNode,
    ORTree, MUXGate, ANDGate, ORGate, NOTGate
)
from .vec_logic_gates import VecAND, VecOR


class PulseFloatingPointEncoder(nn.Module):
    """纯 SNN 浮点编码器 - 将浮点数转换为脉冲序列
    
    **100% 纯 SNN 实现**，基于时序动力学：
    
    1. **符号检测**: SignBitNode (抑制性突触)
    2. **二进制扫描**: DynamicThresholdIFNode (动态阈值)
    3. **首脉冲检测**: FirstSpikeDetector (时序状态机)
    4. **指数生成**: TemporalExponentGenerator (减法计数器)
    5. **尾数提取**: DelayLine (时序相关性)
    
    **支持任意维度输入**:
    - 输入: [...] 浮点数张量
    - 输出: [..., 8] 脉冲张量 (FP8 E4M3)
    
    Args:
        exponent_bits: 指数位数，默认 4
        mantissa_bits: 尾数位数，默认 3
        scan_integer_bits: 整数部分扫描位数，默认 10
        scan_decimal_bits: 小数部分扫描位数，默认 10
    
    Example:
        >>> encoder = PulseFloatingPointEncoder()
        >>> x = torch.tensor([1.5, -2.0])
        >>> pulse = encoder(x)  # [2, 8]
    """
    def __init__(self, exponent_bits: int = 4, mantissa_bits: int = 3,
                 scan_integer_bits: int = 10, scan_decimal_bits: int = 10,
                 neuron_template=None):
        super().__init__()
        self.E_bits = exponent_bits
        self.M_bits = mantissa_bits
        self.total_bits = 1 + exponent_bits + mantissa_bits
        self.bias = (2 ** (exponent_bits - 1)) - 1  # 7
        nt = neuron_template

        self.scanner_N = scan_integer_bits
        self.scanner_NT = scan_decimal_bits
        self.total_steps = scan_integer_bits + scan_decimal_bits

        # 1. 符号
        self.sign_node = SignBitNode()

        # 2. 扫描
        self.binary_scanner = DynamicThresholdIFNode(N=scan_integer_bits, NT=scan_decimal_bits)

        # 3. 首脉冲
        self.first_spike_detector = FirstSpikeDetector(self.total_steps)

        # 4. 指数生成 (时序)
        # 初始值：t=0 时对应指数。
        # 阈值 2^(N-1) -> E = (N-1) + bias
        # 因为 exp_gen 先减后输出，所以初始值要 +1
        start_exp = scan_integer_bits + self.bias
        self.exp_generator = TemporalExponentGenerator(start_value=start_exp, bits=exponent_bits)

        # 5. 尾数提取 (延迟线)
        # 需要 M_bits 个延迟节点链
        self.delay_nodes = nn.ModuleList()
        for _ in range(mantissa_bits):
            self.delay_nodes.append(DelayNode())

        # 逻辑门 - 纯 SNN 向量化实现
        self.vec_and_latch_e = VecAND(neuron_template=nt)
        self.vec_or_accum_e = VecOR(neuron_template=nt)

        self.vec_and_sample_m = VecAND(neuron_template=nt)
        self.vec_or_accum_m = VecOR(neuron_template=nt)

        # 首脉冲检测门电路
        self.not_fired = NOTGate(neuron_template=nt)
        self.and_first_spike = ANDGate(neuron_template=nt)
        self.or_has_fired = ORGate(neuron_template=nt)

        # Subnormal 采样门电路
        self.not_for_sub = NOTGate(neuron_template=nt)
        self.and_sub_sample = ANDGate(neuron_template=nt)
        self.or_sub_m = nn.ModuleList([ORGate(neuron_template=nt) for _ in range(mantissa_bits)])
        
        # 寄存器 (累积结果)
        self.register_buffer('e_reg', None)
        self.register_buffer('m_reg', None)
        
        # Subnormal 窗口检测
        # 当 E 减到 0 时进入 subnormal 区域
        # 对应的时间步是 start_exp + 1 (因为从 start 减到 0 需要 start 步，第 start+1 步是 0)
        # 但我们用固定位置简化：位置 16, 17, 18
        self.subnormal_start_step = scan_integer_bits + 6 # N+6 (2^-7)
        
    def forward(self, x: torch.Tensor):
        input_shape = x.shape
        device = x.device
        
        flat_x = x.flatten().unsqueeze(1)
        n_samples = flat_x.shape[0]
        
        # 1. 符号
        self.sign_node.reset()
        s_out = self.sign_node(flat_x)
        
        # 2. 扫描 & 时序处理
        abs_x = torch.abs(flat_x)
        self.binary_scanner.reset()
        self.exp_generator.reset()
        self.first_spike_detector.reset()
        for d in self.delay_nodes: d.reset()
        
        # 初始化寄存器
        e_reg = torch.zeros(n_samples, self.E_bits, device=device)
        m_reg = torch.zeros(n_samples, self.M_bits, device=device)
        
        # 之前是否有首脉冲 (用于驱动指数减法)
        has_fired_accum = torch.zeros(n_samples, 1, device=device)
        
        # 延迟的首脉冲信号 (用于Normal尾数采样)
        # delayed_first_spike[0] -> 采样 M0
        # delayed_first_spike[1] -> 采样 M1 ...
        delayed_fs = [torch.zeros(n_samples, 1, device=device) for _ in range(self.M_bits)]
        
        for t in range(self.total_steps):
            # --- A. 产生脉冲 ---
            curr_in = abs_x if t == 0 else torch.zeros_like(abs_x)
            s = self.binary_scanner(curr_in) # [N, 1]
            
            # --- B. 检测首脉冲 ---
            # 使用纯 SNN 门电路: fs = s AND NOT(has_fired_accum)
            self.not_fired.reset()
            self.and_first_spike.reset()
            not_fired = self.not_fired(has_fired_accum)
            first_spike = self.and_first_spike(s, not_fired) 
            
            # --- C. 指数生成 ---
            # 时钟脉冲总是1
            clock = torch.ones_like(s)
            # 如果还没有fire，指数减1
            # exp_gen 内部会处理：update if NOT(has_fired_accum)
            current_exp_bits = self.exp_generator(has_fired_accum, clock)
            
            # --- D. 锁存指数 ---
            # 如果当前是 first_spike，锁存 current_exp
            # 只有在 Normal 区域才锁存指数 (Subnormal E=0)
            # 使用纯 SNN 门电路
            if t < self.subnormal_start_step:
                # 向量化：所有指数位同时处理
                self.vec_and_latch_e.reset()
                self.vec_or_accum_e.reset()
                # first_spike: [n_samples, 1] -> 广播到 [n_samples, E_bits]
                first_spike_expanded = first_spike.expand_as(current_exp_bits)
                bit_val = self.vec_and_latch_e(current_exp_bits, first_spike_expanded)
                e_reg = self.vec_or_accum_e(e_reg, bit_val)
            
            # --- E. 尾数采样 (Normal) ---
            # 移位延迟线：fs -> d0 -> d1 -> d2
            # 只有在 Normal 区域产生的首脉冲才进入延迟线
            # 这样避免 Subnormal 的脉冲被误当首脉冲从而错位采样
            valid_fs_for_delay = first_spike if t < self.subnormal_start_step else torch.zeros_like(first_spike)
            
            # 更新延迟线
            prev_signal = valid_fs_for_delay
            current_delayed_fs = []
            for i in range(self.M_bits):
                out = self.delay_nodes[i](prev_signal)
                current_delayed_fs.append(out)
                prev_signal = out
            
            # 采样：M[i] = s AND delayed_fs[i]
            # 向量化：所有尾数位同时处理
            self.vec_and_sample_m.reset()
            self.vec_or_accum_m.reset()
            # 堆叠延迟信号 [n_samples, M_bits]
            delayed_fs_stacked = torch.cat(current_delayed_fs, dim=-1)
            # s: [n_samples, 1] -> 广播到 [n_samples, M_bits]
            s_expanded = s.expand_as(delayed_fs_stacked)
            sampled = self.vec_and_sample_m(s_expanded, delayed_fs_stacked)
            m_reg = self.vec_or_accum_m(m_reg, sampled)
            
            # --- F. 尾数采样 (Subnormal) ---
            # 如果是 subnormal 区域 (t >= 16)，直接采集脉冲
            # subnormal_start_step = 16
            # t=16 -> M0 (2^-7), t=17 -> M1, t=18 -> M2
            if t >= self.subnormal_start_step and t < self.subnormal_start_step + self.M_bits:
                m_idx = t - self.subnormal_start_step
                # 条件：必须是 subnormal (即还没有 fire 过 normal 首脉冲)
                # has_fired_accum 在 normal 区域结束后应该为 0
                # 但这里 has_fired_accum 会在 subnormal 脉冲出现时变为 1 吗？
                # 是的，下面的更新会置1。
                # 所以我们需要判断：在 Normal 区域结束时 (t=16之前) 是否 fired。
                
                # 简化：直接 OR 进去。因为如果 Normal 已经发生，subnormal 位置不会有脉冲（已被提取），
                # 或者即使有（作为尾数），也会被正确处理？
                # 不，Normal 的尾数逻辑已经处理了所有后续脉冲。
                # 只有当 Normal 没发生时，我们才关心这些位置作为 Subnormal 尾数。
                
                # 关键：OneHotToExponent 保证了 Normal 时 E>=1。
                # 如果 E=0 (Subnormal)，则 e_reg 为 0。
                # 此时 m_reg 也是 0 (因为 first_spike 没发生，delayed_fs 全 0)。
                # 所以我们只需要把 subnormal 脉冲加到 m_reg 即可。
                
                # (废弃代码已删除 - 实际 Subnormal 采样在后面统一使用 SNN 门电路处理)
                # 注意：Subnormal 的第一个脉冲会触发 has_fired 更新，所以后续脉冲可能被屏蔽
                # 这不对。Subnormal 是一串脉冲，不是首脉冲逻辑。
                
                # 修正：Subnormal 仅仅是复制 t=16,17,18 的脉冲到 M0,M1,M2
                # 前提是：这不是 Normal 数的一部分。
                # 如果是 Normal 数，t=16 可能是 M2 (如果首脉冲在 14)。
                # 所以必须互斥。
                
                # 互斥条件：Normal 模式下，first_spike 在 t < 15 时发生。
                # 所以只要判断 has_fired_accum (在进入 subnormal 区域前)
                # 但 has_fired_accum 是动态更新的。
                
                # 让我们利用 E_reg。如果 E_reg > 0，说明是 Normal。
                # 但 E_reg 是累积的。
                
                # 简单方案：
                # Normal 采样逻辑处理了所有 Start < 16 的情况。
                # Subnormal 逻辑处理 Start >= 16 的情况。
                # 如果 Start >= 16，则 first_spike 在 t >= 16 发生。
                # 此时 delayed_fs 会在 t >= 17 激活。
                # 这会错位！Subnormal 的 M0 就是 t=16 本身。
                
                # 所以：如果 t=16 且 NOT(has_fired)，则 M0 = s
                # 如果 t=17 且 NOT(has_fired)，则 M1 = s
                # ...
                # 这里的 has_fired 指的是 "Normal 区域的首脉冲"。
                # 我们可以冻结一个 "normal_fired" 状态。
                
                # 或者：直接把 t=16,17,18 的脉冲强制作为 M0,M1,M2 的**备选**。
                # M_final = M_normal OR (M_subnormal AND NOT Is_Normal)
                # Is_Normal = OR(first_spike[0...15])
                
                # 在纯时序中，这有点难。
                # 让我们用简单的：
                # 如果 t=16，且 has_fired_accum=0 -> 这是 Subnormal M0
                # 且更新 has_fired_accum? 不，Subnormal 不应触发 FirstSpike 逻辑 (它没有隐含1)。
                
                pass # 逻辑在下面统一处理
                
            # 更新全局 has_fired
            # 注意：对于 Subnormal，我们**不**希望它触发 Normal 的采样逻辑
            # 所以 first_spike 应该只在 Normal 区域有效？
            # 或者 TemporalExponentGenerator 会自动处理？
            # ExpGen 在减到 0 后保持 0。
            # 如果 first_spike 在 t=16 发生，ExpGen=0。E_reg=0。正确。
            # 此时 delayed_fs 启动，采样 t=17,18,19 作为 M0,M1,M2。
            # 但 Subnormal 的定义是：t=16 是 M0 (2^-7)。
            # 如果按 Normal 逻辑，t=16 是首脉冲 (隐含1, 2^-7)，则 M0 是 t=17 (2^-8)。
            # 这样值就变成了 1.M * 2^-7 = (1 + M/8)*2^-7 = 2^-7 + M*2^-10。
            # 而 Subnormal 实际是 0.M * 2^-6 = M/8 * 2^-6 = M * 2^-9。
            # 2^-7 对应 M=4 (100)。
            # 如果 t=16 有脉冲：
            #   Normal解读: 1.0 * 2^-7 = 2^-7. (M=0)
            #   Subnormal解读: M=4 (100). 4 * 2^-9 = 2^-7.
            # **结果是一样的！**
            
            # **惊人的发现**：
            # 对于 FP8 E4M3，Subnormal 的表示 (0.M * 2^-6) 和 Normal 的延展 (1.M * 2^E) 在边界处是**平滑**的！
            # E=1 (min normal) -> 1.0 * 2^-6 = 2^-6. (t=15)
            # E=0 (subnormal) -> 最大是 0.111 * 2^-6 = 7/8 * 2^-6.
            # 下一个通过点是 1.000 * 2^-7 ? (t=16) -> 2^-7.
            # Subnormal M=4 (100) -> 0.100 * 2^-6 = 0.5 * 2^-6 = 2^-7.
            
            # 所以，如果我们将 t=16 视为 E=-1 (Normal)，则 val = 1.0 * 2^-8 ?
            # Bias=7. t=16 -> E = 16-16 = 0? No.
            # t=15 -> E=1.
            # t=16 -> E=0.
            # 如果 t=16 是首脉冲：
            #   ExpGen 输出 0。
            #   E_reg = 0.
            #   M 采样 t=17, 18, 19。
            #   结果：E=0, M = [s[17], s[18], s[19]].
            #   这是：1.M * 2^(0-7) = 1.M * 2^-7 ? NO!
            #   当 E=0 时，硬件解释为 0.M * 2^-6.
            #   所以如果 E=0, M=xxx，值为 0.xxx * 2^-6.
            
            # 现在的逻辑：
            # t=16 触发 -> E=0. M 采样 t=17(M0), t=18(M1)...
            # 假设只有 t=16 脉冲 (2^-7)。
            # E=0. M=0. -> 0.0 * 2^-6 = 0.
            # **错误！** 应该是 2^-7.
            
            # 结论：**Subnormal 需要特殊的采样逻辑**。
            # 在 E=0 时，首脉冲位置本身就是 M 的一部分，而不是隐含的 1。
            # t=16 -> M0. t=17 -> M1. t=18 -> M2.
            
            # 修正逻辑：
            # 1. 正常的 FirstSpike 逻辑只在 t <= 15 有效 (Normal区域)。
            # 2. 在 t >= 16 (Subnormal区域)，直接将 s 映射到 M_reg。
            #    t=16 -> M0
            #    t=17 -> M1
            #    t=18 -> M2
            # 3. 并且，如果在 t <= 15 已经发生过首脉冲，则屏蔽 Subnormal 的直接映射 (避免重复采样)。
            
            if t < self.subnormal_start_step:
                # 使用纯 SNN OR 门更新 has_fired_accum
                self.or_has_fired.reset()
                has_fired_accum = self.or_has_fired(has_fired_accum, first_spike)
            
            # Subnormal 直接采样
            if t >= self.subnormal_start_step and t < self.subnormal_start_step + self.M_bits:
                sub_idx = t - self.subnormal_start_step
                # 只有当 Normal 没发生时，使用纯 SNN 门电路
                self.not_for_sub.reset()
                self.and_sub_sample.reset()
                self.or_sub_m[sub_idx].reset()
                not_has_fired = self.not_for_sub(has_fired_accum)
                do_sub_sample = self.and_sub_sample(s, not_has_fired)
                m_reg[..., sub_idx:sub_idx+1] = self.or_sub_m[sub_idx](m_reg[..., sub_idx:sub_idx+1], do_sub_sample)
        
        # e_reg 是 Little Endian [E0, E1, E2, E3]，需要转为 Big Endian [E3, E2, E1, E0]
        e_out = e_reg.flip(-1)
        
        return torch.cat([s_out, e_out, m_reg], dim=-1).view(input_shape + (8,))

    def reset(self):
        self.sign_node.reset()
        self.binary_scanner.reset()
        self.first_spike_detector.reset()
        self.exp_generator.reset()
        for d in self.delay_nodes: d.reset()

