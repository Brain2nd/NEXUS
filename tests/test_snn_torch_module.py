import sys
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from atomic_ops import DynamicThresholdIFNode

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_neuron():
    print(f"Device: {device}")
    # Case 1: 纯整数测试 (N=4, NT=0) -> 13 (1101)
    N1, NT1 = 4, 0
    print(f"Test 1: Integer Only (Config: N={N1}, NT={NT1}) -> Input: 13")

    layer1 = DynamicThresholdIFNode(N=N1, NT=NT1).to(device)
    input1 = torch.tensor([[13.0]], device=device)
    
    layer1.reset()
    out1 = []
    for t in range(N1 + NT1):
        x = input1 if t == 0 else torch.zeros_like(input1)
        out1.append(layer1(x))
    print(f"Output: {torch.stack(out1).squeeze().int().tolist()}")
    print(f"Expect: [1, 1, 0, 1]")
    print("-" * 20)

    # Case 2: 定点小数测试 (N=3, NT=2) -> 5.75 (101.11)
    N2, NT2 = 3, 2
    print(f"Test 2: Fixed Point (Config: N={N2}, NT={NT2}) -> Input: 5.75")

    layer2 = DynamicThresholdIFNode(N=N2, NT=NT2).to(device)
    input2 = torch.tensor([[5.75]], device=device)
    
    layer2.reset()
    out2 = []
    for t in range(N2 + NT2):
        x = input2 if t == 0 else torch.zeros_like(input2)
        out2.append(layer2(x))
    
    res2 = torch.stack(out2).squeeze().int().tolist()
    print(f"Output: {res2}")
    print(f"Expect: [1, 0, 1, 1, 1]")
    
    if res2 == [1, 0, 1, 1, 1]:
        print("\nALL PASSED")
    else:
        print("\nFAILED")

if __name__ == "__main__":
    test_neuron()
