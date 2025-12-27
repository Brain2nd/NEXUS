import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import PulseFloatingPointEncoder

def test_fp_encoder():
    print("Testing PulseFloatingPointEncoder (FP8: E3M4) with Tensor Shape")
    encoder = PulseFloatingPointEncoder(exponent_bits=3, mantissa_bits=4)
    
    # Input Shape: [2, 3] (Batch=2, Channel=3)
    inputs = torch.tensor([
        [1.5, -0.25, 0.0],
        [6.0, -1.5, 0.0625]
    ])
    print(f"Input Shape: {inputs.shape}")
    print(f"Input Values:\n{inputs}")
    
    output = encoder(inputs)
    
    print("-" * 30)
    print(f"Output Shape: {output.shape}")
    print(f"Output Values (Last dim is [S, E, M]):\n{output.int()}")
    
    # 验证 1.5 -> 0 011 1000
    case1 = output[0, 0].int().tolist()
    if case1 == [0, 0, 1, 1, 1, 0, 0, 0]:
        print("Case 1.5 Passed")
    else:
        print(f"Case 1.5 Failed: {case1}")

    # 验证 -0.25 -> 1 001 0000
    case2 = output[0, 1].int().tolist()
    if case2 == [1, 0, 0, 1, 0, 0, 0, 0]:
        print("Case -0.25 Passed")
    else:
        print(f"Case -0.25 Failed: {case2}")

if __name__ == "__main__":
    test_fp_encoder()
