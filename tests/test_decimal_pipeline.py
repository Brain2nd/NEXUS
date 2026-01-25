import sys
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from atomic_ops import DecimalScanner, DynamicThresholdIFNode

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_pipeline_T14():
    print(f"Device: {device}")
    T = 14
    # 1. 初始化原子操作
    scanner = DecimalScanner(T=T).to(device)
    # 输出二进制位数：14 需要 4 位 (需要覆盖 2^3=8, 2^2=4, 2^1=2, 2^0=1)
    # 设置 N=4 (代表4位)
    encoder = DynamicThresholdIFNode(N=4).to(device)

    # 2. 准备测试数据
    # Case 1: 10^14 (t=0 应该发) -> 指数 14 -> 二进制 1110
    # Case 2: 10^13 (t=1 应该发) -> 指数 13 -> 二进制 1101
    # Case 3: 100   (t=12 应该发)-> 指数 2  -> 二进制 0010
    # Case 4: 1.5   (t=13 应该发)-> 指数 0? 不，最小阈值是 10^1=10。
    #         我们的 scanner 阈值: t=0->10^14 ... t=13->10^1. 
    #         如果输入 < 10，全为0。
    
    input_vals = torch.tensor([
        [1e14 + 1],      # 14
        [1e13 + 1],      # 13
        [1e2 + 1],       # 2
        [1e4 + 1]        # 4 -> 二进制 0100
    ], device=device)
    
    print(f"Testing Pipeline with T={T}")
    print(f"Input Values: \n{input_vals}")
    
    scanner.reset()
    encoder.reset()
    
    # 模拟时间步
    # 阶段 1: Scanner 扫描 (持续 T 个时间步)
    # 我们需要记录 scanner 的输出脉冲序列
    scanner_spikes = []
    
    print("-" * 30)
    print("Phase 1: Scanning...")
    for t in range(T):
        # 输入恒定
        s = scanner(input_vals)
        scanner_spikes.append(s)
        
        # 打印中间状态
        # 只有当有脉冲时才打印
        if s.any():
            th = 10**(T-t)
            print(f"  Step {t} (Th=10^{T-t}): Spikes at indices {torch.where(s.squeeze()==1)[0].tolist()}")

    # 堆叠 Scanner 输出: [T, Batch, 1] -> [Batch, T]
    scanner_out_tensor = torch.stack(scanner_spikes, dim=1).squeeze(-1)
    
    # -------------------------------------------------------
    # 阶段 2: 矩阵计算 (Synaptic Weights) - 纯脉冲域计算
    # -------------------------------------------------------
    # 每一个时间步 t 对应一个权重 W_t
    # t=0 (10^14) -> 权重 14
    # t=1 (10^13) -> 权重 13
    # ...
    # t=k -> 权重 T - k
    
    # 构建权重向量 [T]
    weights = torch.tensor([T - t for t in range(T)], dtype=torch.float32)
    print("-" * 30)
    print(f"Synaptic Weights (Time-to-Value): {weights.tolist()}")
    
    # 矩阵乘法: Batch x T  @  T -> Batch
    # 这相当于积分电荷
    integrated_current = torch.matmul(scanner_out_tensor, weights).unsqueeze(1)
    
    print("-" * 30)
    print(f"Integrated Intensity (Values passed to Encoder):\n{integrated_current}")
    
    # -------------------------------------------------------
    # 阶段 3: 二进制编码 (Encoder)
    # -------------------------------------------------------
    print("-" * 30)
    print("Phase 3: Binary Encoding...")
    
    encoder_spikes = []
    # Encoder 需要运行 N=4 个时间步来输出 4 位二进制
    encoder_steps = 4 
    
    for t in range(encoder_steps):
        # 仅在 t=0 时注入电荷，或者利用 IFNode 的积分特性
        # 这里我们模拟 t=0 注入脉冲转换来的瞬时大电流
        curr = integrated_current if t == 0 else torch.zeros_like(integrated_current)
        
        s = encoder(curr)
        encoder_spikes.append(s)

    # [Time, Batch, 1] -> [Batch, Time]
    final_output = torch.stack(encoder_spikes, dim=1).squeeze(-1)
    
    print(f"Final Binary Output (MSB First):\n{final_output.int()}")
    
    # 验证
    # 14 -> 1110
    # 13 -> 1101
    # 2  -> 0010
    # 4  -> 0100
    expected = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=torch.int)
    
    if torch.equal(final_output.int(), expected.int()):
        print("\nTest PASSED: Complete pipeline works correctly.")
    else:
        print("\nTest FAILED.")

if __name__ == "__main__":
    test_pipeline_T14()
