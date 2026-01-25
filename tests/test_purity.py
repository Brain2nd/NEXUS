import torch
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.logic_gates import NOTGate, XORGate
from atomic_ops.core.vec_logic_gates import VecNOT, VecXOR

# GPU 设备选择 (CLAUDE.md #9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_scalar_gates():
    print("\nTesting Scalar Gates (logic_gates.py)...")
    print(f"Device: {device}")
    not_gate = NOTGate().to(device)
    xor_gate = XORGate().to(device)

    # NOT - 使用 1D tensor 以兼容门电路
    assert not_gate(torch.tensor([0.0], device=device)).item() == 1.0, "NOT(0) failed"
    assert not_gate(torch.tensor([1.0], device=device)).item() == 0.0, "NOT(1) failed"
    print("NOTGate: PASS")

    # XOR
    assert xor_gate(torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)).item() == 0.0, "XOR(0,0) failed"
    assert xor_gate(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)).item() == 1.0, "XOR(0,1) failed"
    assert xor_gate(torch.tensor([1.0], device=device), torch.tensor([0.0], device=device)).item() == 1.0, "XOR(1,0) failed"
    assert xor_gate(torch.tensor([1.0], device=device), torch.tensor([1.0], device=device)).item() == 0.0, "XOR(1,1) failed"
    print("XORGate: PASS")

def test_vector_gates():
    print("\nTesting Vector Gates (vec_logic_gates.py)...")
    vec_not = VecNOT().to(device)
    vec_xor = VecXOR().to(device)

    x = torch.tensor([0.0, 1.0], device=device)
    y_not = vec_not(x)
    assert torch.allclose(y_not, torch.tensor([1.0, 0.0], device=device)), f"VecNOT failed: {y_not}"
    print("VecNOT: PASS")

    a = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    b = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    y_xor = vec_xor(a, b)
    assert torch.allclose(y_xor, torch.tensor([0.0, 1.0, 1.0, 0.0], device=device)), f"VecXOR failed: {y_xor}"
    print("VecXOR: PASS")

if __name__ == "__main__":
    try:
        test_scalar_gates()
        test_vector_gates()
        print("\nALL LOGIC TESTS PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
