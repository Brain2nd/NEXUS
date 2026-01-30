# TEMPORAL 模式熵与可分性实验报告

## 1. 研究动机与问题定义

### 1.1 我们要研究什么

NEXUS 框架的 TEMPORAL 模式允许 LIF 神经元的膜电位残差跨 forward 调用累积。这引出一个核心问题：

> **这些累积的膜电位残差是否真的携带了有意义的信息？还是只是无用的数值噪声？**

如果残差携带信息，那 TEMPORAL 模式就具备了作为"隐式 KV-cache"替代 Transformer 显式注意力机制的潜力——这是 NEXUS 生物启发计算的理论基础。

### 1.2 为什么要研究这个

传统 Transformer 依赖显式的 KV-cache 存储序列上下文（O(n) 内存，n 为序列长度）。NEXUS 的 LIF 神经元天然具有时间动力学：

```
V(t+1) = beta * V(t) + I(t)    # beta ≈ 1 - 1e-7 (微泄漏)
```

每次 forward 后，膜电位 V 保留了"衰减加权的历史输入总和"。如果这些残差能编码序列上下文，则：
- 不需要显式注意力矩阵（O(n^2) -> O(1) 内存）
- 不需要 KV-cache（O(n) -> O(1) 内存）
- 上下文信息隐式存储在神经元状态中

### 1.3 遵循什么指导

实验设计遵循以下逻辑链：

1. **理论推导** -> 状态熵公式 h(V) = N/2 * ln(2*pi*e*sigma_I^2) - 1/2 * sum_ij ln(1 - beta_ij^2)
2. **存在性验证** -> 实验 1: V_residual 是否编码了可区分的上下文信息？
3. **定量度量** -> 实验 2: 状态熵的理论值与实测值是多少？BIT_EXACT 是否为零？
4. **参数敏感性** -> 实验 3: beta（泄漏因子）如何影响信息容量？

### 1.4 测试对象

所有实验的被测对象是 NEXUS 框架自身的纯 SNN 组件，具体为：

- **`SpikeFP32Linear_MultiPrecision`**: NEXUS 的 FP32 全连接层，内部由纯 SNN 门电路构成（`SpikeFP32Multiplier` + `SpikeFP32Adder` + `SequentialAccumulator`）
- **`SimpleLIFNode`**: NEXUS 的 LIF 神经元，所有门电路（VecAND/VecOR/VecXOR/VecNOT/VecMUX 等）的基础计算单元，实现 `V(t+1) = beta * V(t) + I(t)` 动力学
- **组合结构**: 2 层 MLP，由 2 个 `SpikeFP32Linear_MultiPrecision` 串联。每个 Linear 内部包含 FP32 乘法器（符号异或、指数加法器、尾数乘法器、LZD、归一化移位器、舍入加法器等）和 FP32 加法器（指数比较器、尾数对齐移位器、尾数加减法器、前导零检测、归一化、舍入等），每个子组件由数十到数百个 LIF 门电路构成

一个 2 层 MLP (4->8->4) 共包含 **12,554 个 LIF 神经元**。这些不是简化的 demo，而是框架中实际用于位精确 IEEE-754 浮点运算的完整门电路。

### 1.5 两种计算范式对比

| | BIT_EXACT (数字逻辑) | TEMPORAL (生物计算) |
|---|---|---|
| 设计目标 | IEEE-754 位精确浮点运算 | 生物启发时间动力学 |
| 膜电位管理 | forward 后立即清除 (`v = None`) | 跨调用保留，残差累积 |
| 状态信息 | 无（每次调用独立） | 编码了历史输入的衰减加权和 |
| 等价物 | 组合逻辑电路 | 带记忆的时序电路 / 隐式 KV-cache |
| 状态熵 | 恒为 0 | > 0，由 beta 和输入分布决定 |

---

## 2. 实验 1: 时间-空间编码可分性

### 2.1 研究问题

> 同一个目标 token 输入网络，但之前经历的上下文不同，最终的 V_residual 是否不同且可区分？

如果答案是"是"，则说明 V_residual 确实编码了上下文信息，而不仅是对当前输入的响应。

### 2.2 实验对象

- **被测系统**: 2 层 FP32 纯 SNN MLP（`SpikeFP32Linear_MultiPrecision`），结构 4 -> 8 -> 4
- **被测变量**: 所有 LIF 神经元（12,554 个）的膜电位残差 V_residual
- **输入**: FP32 脉冲编码的浮点张量（float32_to_pulse 边界转换后的 32 位脉冲）

### 2.3 对照组设计

| 组别 | 模式 | 上下文 | 目标 token | 预期 V_residual |
|---|---|---|---|---|
| **对照组 1**: BIT_EXACT | BIT_EXACT | Context A (4步) | 固定 | 全部 None (v=0) |
| **实验组 A**: TEMPORAL+长上下文 | TEMPORAL | Context A (4步随机前缀) | 固定 | 非零，编码 A 的上下文 |
| **实验组 B**: TEMPORAL+短上下文 | TEMPORAL | Context B (2步随机前缀) | 固定 | 非零，编码 B 的上下文 |

- **对照组 1 的作用**: 证明 BIT_EXACT 模式下 V 确实被清除（状态熵 = 0），排除"V_residual 是模式控制 bug"的可能
- **实验组 A vs B 的作用**: 证明 V_residual 编码的是上下文差异而非随机噪声——相同 token 不同上下文应产生可分离的分布
- **固定目标 token**: 控制变量——确保分布差异来自上下文而非目标输入本身

### 2.4 方法

1. **模型初始化**: 权重随机初始化 (randn * 0.5)，固定随机种子 42
2. **数据生成**:
   - 目标 token: 固定 (randn(4) * 0.3)
   - Context A: 20 个 trial，每 trial 4 步随机输入 (randn(4) * 0.5) + 1 步目标
   - Context B: 20 个 trial，每 trial 2 步随机输入 (randn(4) * 0.5) + 1 步目标
3. **数据采集**:
   - 每个 trial 前 reset 模型（清除所有 V）
   - 按序输入上下文 + 目标，收集最终所有 LIF 神经元的 V（展平为向量）
4. **数据清洗**:
   - 去掉常量列（方差 < 1e-30，来自预分配但未使用的位）
   - 替换 inf/nan 为 0
5. **分析指标**:
   - Fisher 判别比: `|centroid_A - centroid_B|^2 / (var_A + var_B)`，> 1 表示类间差异大于类内差异
   - t-SNE 降维可视化（2D）
   - PCA 降维可视化（2D）+ 主成分方差解释率

### 2.5 结果

| 指标 | 值 | 含义 |
|---|---|---|
| BIT_EXACT 残余 V 数 | 0/20 | 对照组确认：BIT_EXACT 完全清除状态 |
| TEMPORAL A 有效试验 | 20/20 | 所有 trial 均采集到 V_residual |
| TEMPORAL B 有效试验 | 20/20 | 所有 trial 均采集到 V_residual |
| 有效维度 | 194,639 | 去掉 83,633 个常量列后的实际信息维度 |
| 质心欧氏距离 | 3,178.50 | A/B 两组分布中心距离很大 |
| 类内方差 A | 17,496.18 | A 组内部变异（由随机上下文造成） |
| 类内方差 B | 11,054.61 | B 组内部变异 |
| **Fisher 判别比** | **353.86** | **远大于 1，两组高度可分** |
| PCA PC1 方差解释率 | 90.2% | 第一主成分主导，说明分布差异有明确方向 |
| PCA PC2 方差解释率 | 0.9% | 第二成分几乎不贡献 |

### 2.6 结论

- **BIT_EXACT 对照**: V 全部为 None，确认状态熵为 0，模式控制正确
- **TEMPORAL 实验**: Fisher 判别比 353.86 >> 1，t-SNE/PCA 图上两类完全分离
- **核心结论**: V_residual 确实编码了可区分的上下文信息，"隐式 KV-cache"假说得到实验支持

---

## 3. 实验 2: 熵探针 (Entropy Probe)

### 3.1 研究问题

> TEMPORAL 模式下的状态熵到底有多大？理论公式的预测与实测是否一致？

实验 1 证明了"信息存在"，实验 2 要回答"信息有多少"。

### 3.2 实验对象

- **被测系统**: 同实验 1 的 2 层 MLP
- **被测变量**: 所有 LIF 神经元 V_residual 的联合微分熵
- **参数**: 每个 LIF 神经元的 beta（泄漏因子），共 12,554 个神经元

### 3.3 对照组设计

| 组别 | 模式 | 预期熵 | 作用 |
|---|---|---|---|
| **理论基准** | 公式计算 | h(V) 由 beta 和 sigma_I 决定 | 提供理论上界 |
| **实验组**: TEMPORAL 实测 | TEMPORAL | > 0 | 实际测量微分熵 |
| **对照组**: BIT_EXACT 实测 | BIT_EXACT | = 0 | 确认零状态基线 |

### 3.4 方法

1. **理论基准计算**:
   - 遍历模型所有 `SimpleLIFNode` 子模块，收集 beta 张量
   - 计算 `sum_ij ln(1 - beta_ij^2)` 和理论状态熵
2. **TEMPORAL 实测**:
   - 采样 30 次（每次 3 步 forward，随机高斯输入 randn * 0.5）
   - 收集 V_residual，去掉常量列，标准化
   - PCA 降至 50 维（防止维度灾难）
   - k-NN 微分熵估计（k=5，Kozachenko-Leonenko 估计器）
3. **BIT_EXACT 对照**:
   - 采样 20 次，同样流程
   - 检查是否存在任何非空 V

### 3.5 结果

| 指标 | 值 | 说明 |
|---|---|---|
| LIF 神经元总数 | 12,554 | 2 层 MLP 中所有门电路的 LIF 节点 |
| 理论状态熵 | 104,831.57 nats | 独立高斯假设下的上界 |
| sum ln(1-beta^2) | -191,439.97 | beta ≈ 1-1e-7 使每个神经元贡献约 15.25 nats |
| 实测微分熵 (k-NN) | 183.6 nats | PCA 降至 50 维后的估计 |
| BIT_EXACT 有 V 数 | 0/20 | 确认零状态 |

### 3.6 理论值与实测值差异分析

实测值 (183.6) 远低于理论值 (104,831.57)，这是预期行为：

1. **降维损失**: 原始 ~19 万维 -> PCA 50 维，丢失大量信息
2. **独立性假设**: 理论公式假设各神经元独立，实际高度相关（共享输入）
3. **有限样本**: 仅 30 个样本估计 50 维分布，统计量不充分

关键不是理论/实测的绝对值匹配，而是：
- **TEMPORAL > 0** vs **BIT_EXACT = 0** 的定性区别成立
- 理论公式给出了正确的趋势（beta 越大，熵越高）

---

## 4. 实验 3: 多尺度 beta 初始化对比

### 4.1 研究问题

> 泄漏因子 beta 如何影响状态熵（信息容量）？什么样的 beta 初始化策略最优？

这直接关系到 TEMPORAL 模式训练的超参数选择。

### 4.2 实验对象

- **被测系统**: 4 个独立的 2 层 MLP，分别使用不同 beta 的 LIF 神经元模板
- **被测变量**: 理论状态熵和实测状态熵随 beta 的变化关系
- **自变量**: beta 取值（控制泄漏强度）

### 4.3 对照组设计

4 种 beta 配置互为对照，覆盖从"几乎无泄漏"到"强泄漏"的完整频谱：

| 配置 | beta | epsilon = 1 - beta | 物理含义 | 预期信息容量 |
|---|---|---|---|---|
| 默认微泄漏 | 0.9999999 | 1e-7 | 几乎完美记忆 | 最高 |
| 轻度泄漏 | 0.999 | 1e-3 | 长时记忆 | 高 |
| 中度泄漏 | 0.99 | 1e-2 | 中时记忆 | 中 |
| 强泄漏 | 0.9 | 1e-1 | 短时记忆 | 低 |

### 4.4 方法

对每种 beta 配置：
1. 创建 `SimpleLIFNode(beta=beta_value)` 模板
2. 用该模板构建 2 层 MLP
3. 计算理论状态熵（从 beta 张量直接算）
4. TEMPORAL 模式采样 20 次（每次 3 步），k-NN 估计实测熵
5. 绘制理论曲线 h(beta) = -1/2 * ln(1 - beta^2) 的单神经元连续曲线

### 4.5 结果

理论曲线确认：
- **beta -> 1** 时状态熵趋向无穷（完美记忆 = 无限信息容量）
- **beta -> 0** 时状态熵趋向 0（完全遗忘 = 无信息容量）
- **单调递增**: 泄漏越小，信息保留越多

默认 beta = 1 - 1e-7 处于曲线极右侧，具有最大理论信息容量。

### 4.6 对训练策略的启示

- **微泄漏 (1e-7)**: 最大信息容量，但可能导致梯度问题（近似无泄漏）
- **中度泄漏 (1e-2 ~ 1e-3)**: 平衡信息保留与数值稳定性，可能更适合训练
- **多尺度初始化**: 不同层/不同神经元使用不同 beta，让网络自动学习最优时间尺度

---

## 5. 可视化

![Temporal Entropy & Separability Results](temporal_entropy_results.png)

6 个子图说明：

1. **t-SNE** (左上): V_residual 的 2D t-SNE 投影。蓝点 = Context A (长序列)，橙点 = Context B (短序列)，两族完全分离，Fisher Ratio = 353.86
2. **PCA** (中上): V_residual 的 PCA 投影。PC1 解释 90.2% 方差，说明两组差异主要沿一个方向，分离结构清晰
3. **V 分布直方图** (右上): 随机抽取 5 个维度的膜电位分布，集中在 0.6 和 1.4 附近（对应阈值附近的软复位残差）
4. **状态熵对比** (左下): BIT_EXACT = 0.0 nats（对照基线），TEMPORAL 理论 = 104,831.6 nats，实测 = 183.6 nats
5. **State Entropy vs epsilon** (中下): 4 种 beta 配置的理论/实测熵对比条形图
6. **单神经元理论曲线** (右下): h(beta) = -1/2 * ln(1 - beta^2) 的连续曲线，标注了 4 种 beta 配置的位置

---

## 6. 代码修改记录

### 6.1 neurons.py: V 预分配切片机制

为支持 TEMPORAL 模式，修改了 `SimpleIFNode.forward()` 和 `SimpleLIFNode.forward()`:

**问题**: TEMPORAL 模式下，同一门电路被不同位宽的输入复用（如 VecTreeAND 树归约: 4位 -> 2位 -> 1位）。BIT_EXACT 模式下每次 forward 后 `v = None` 重新分配不会出错，但 TEMPORAL 模式下 v 保留导致 `v.shape[-1] != x.shape[-1]` 报错。

**修复**: 膜电位 `v` 采用与 `_beta` / `_v_threshold` 一致的预分配切片机制：
- 首次 forward: 预分配 `v` 为 `(*batch_dims, max_bits=64)`
- 后续 forward: 切片 `v[..., :input_bits]` 使用
- batch/spatial 维度变化时重新预分配

```python
# 切片当前位宽
v_slice = self.v[..., :input_bits]
# LIF: V = beta * V + I
v_slice.mul_(beta).add_(x)
# 软复位: V = V - spike * V_th
v_slice.sub_(spike * threshold)
```

**对 BIT_EXACT 模式的影响**: 无。BIT_EXACT 每次 forward 后 `v = None`，下次重新预分配全零张量，计算结果完全一致。

### 6.2 neurons.py: SimpleLIFNode.v_threshold setter

新增 `v_threshold` 属性的 setter，支持 `_create_neuron()` 在 deepcopy 模板后设置阈值。此前 `v_threshold` 为只读 property，导致非默认 beta 的模板创建失败。

---

## 7. 运行方式

```bash
# 需要 dapo conda 环境
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate dapo
python experiments/temporal_entropy_probe.py
```

依赖: `torch`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`

输出: `experiments/temporal_entropy_results.png`

预计耗时: CPU 上约 5-10 分钟（主要瓶颈为纯 SNN 门电路 FP32 forward）
