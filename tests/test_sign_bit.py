import sys
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from atomic_ops import SignBitNode

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_sign_bit():
    print("Testing SignBitNode (W=-1, Th=0)")
    print(f"Device: {device}")
    node = SignBitNode().to(device)

    # Test cases:
    # -5.0 -> Expect 1 (Negative)
    # +5.0 -> Expect 0 (Positive)
    # 0.0  -> Expect 0 (Zero treated as Positive)
    inputs = torch.tensor([[-5.0], [5.0], [0.0]], device=device)

    print(f"Input:\n{inputs}")

    node.reset()
    # 只运行一步，因为符号判别是瞬时的
    output = node(inputs)

    print(f"Output:\n{output.int()}")

    expected = torch.tensor([[1], [0], [0]], dtype=torch.int, device=device)
    
    if torch.equal(output.int(), expected.int()):
        print("PASSED: Correctly identifies sign bit.")
    else:
        print("FAILED.")

if __name__ == "__main__":
    test_sign_bit()
