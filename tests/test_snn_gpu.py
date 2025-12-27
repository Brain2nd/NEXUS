import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
import time
from spikingjelly.activation_based import neuron, functional, layer

def test_snn(device_name):
    try:
        device = torch.device(device_name)
        print(f"\nTesting on {device_name}...")
        
        # 定义一个简单的网络：Flatten -> Linear -> LIF -> Linear -> LIF
        # 模拟处理 MNIST 大小的输入
        net = torch.nn.Sequential(
            layer.Flatten(),
            torch.nn.Linear(28 * 28, 2048),
            neuron.LIFNode(tau=2.0, detach_reset=True),
            torch.nn.Linear(2048, 10),
            neuron.LIFNode(tau=2.0, detach_reset=True)
        ).to(device)

        # 随机输入数据: [T, N, C, H, W]
        # T=时间步数, N=Batch size
        T = 16
        N = 128 
        x = torch.rand(T, N, 1, 28, 28).to(device)

        # 预热 (Warmup)
        functional.reset_net(net)
        with torch.no_grad():
            y = functional.multi_step_forward(x, net)
        if device_name.startswith('cuda'):
            torch.cuda.synchronize()
        
        # 开始计时
        start_time = time.time()
        iterations = 50
        with torch.no_grad():
            for _ in range(iterations):
                functional.reset_net(net)
                y = functional.multi_step_forward(x, net)
        
        if device_name.startswith('cuda'):
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        print(f"Avg Time per forward pass on {device_name}: {avg_time:.4f}s")
        return avg_time

    except Exception as e:
        print(f"Error on {device_name}: {e}")
        return float('inf')

if __name__ == '__main__':
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    cpu_time = test_snn('cpu')
    
    if torch.cuda.is_available():
        gpu_time = test_snn('cuda:0')
        
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"\nSUCCESS: GPU acceleration is working! Speedup: {speedup:.2f}x")
        else:
            print(f"\nWARNING: GPU was slower than CPU. This might be due to small model size or overhead.")
    else:
        print("\nCUDA not available, skipping GPU test.")

