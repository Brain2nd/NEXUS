import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import ANDGate, ORGate, XORGate, FullAdder

def test_logic_gates():
    print("=== Testing Spike Logic Gates ===")
    
    # Data: 4 cases [0,0], [0,1], [1,0], [1,1]
    a = torch.tensor([[0.], [0.], [1.], [1.]])
    b = torch.tensor([[0.], [1.], [0.], [1.]])
    
    # 1. AND
    print("\n1. AND Gate")
    gate_and = ANDGate()
    gate_and.reset()
    out_and = gate_and(a, b)
    print(f"Inputs A,B -> Out:\n{torch.cat([a, b, out_and], dim=1).int()}")
    expected_and = torch.tensor([[0], [0], [0], [1]])
    if torch.equal(out_and.int(), expected_and.int()): print("PASS")
    else: print("FAIL")

    # 2. OR
    print("\n2. OR Gate")
    gate_or = ORGate()
    gate_or.reset()
    out_or = gate_or(a, b)
    print(f"Inputs A,B -> Out:\n{torch.cat([a, b, out_or], dim=1).int()}")
    expected_or = torch.tensor([[0], [1], [1], [1]])
    if torch.equal(out_or.int(), expected_or.int()): print("PASS")
    else: print("FAIL")

    # 3. XOR
    print("\n3. XOR Gate")
    gate_xor = XORGate()
    gate_xor.reset()
    out_xor = gate_xor(a, b)
    print(f"Inputs A,B -> Out:\n{torch.cat([a, b, out_xor], dim=1).int()}")
    expected_xor = torch.tensor([[0], [1], [1], [0]])
    if torch.equal(out_xor.int(), expected_xor.int()): print("PASS")
    else: print("FAIL")

def test_full_adder():
    print("\n=== Testing Full Adder ===")
    adder = FullAdder()
    
    # Truth table for Full Adder (8 cases)
    # A B Cin -> Sum Cout
    # 0 0 0 -> 0 0
    # 0 0 1 -> 1 0
    # 0 1 0 -> 1 0
    # 0 1 1 -> 0 1
    # 1 0 0 -> 1 0
    # 1 0 1 -> 0 1
    # 1 1 0 -> 0 1
    # 1 1 1 -> 1 1
    
    cases = [
        [0,0,0], [0,0,1], [0,1,0], [0,1,1],
        [1,0,0], [1,0,1], [1,1,0], [1,1,1]
    ]
    
    a = torch.tensor([[c[0]] for c in cases])
    b = torch.tensor([[c[1]] for c in cases])
    cin = torch.tensor([[c[2]] for c in cases])
    
    adder.reset()
    # 注意：Full Adder 内部由多个 Gate 组成，信号传递是瞬时的(在前向传播中完成)
    # 但如果有时间步依赖(LIF的泄露)，需要小心。
    # 这里我们用的是无泄露IF且一步到位，所以应该能一次 forward 搞定。
    
    s_out, c_out = adder(a, b, cin)
    
    print("A B Cin | Sum Cout")
    print("-" * 20)
    for i in range(8):
        print(f"{int(a[i])} {int(b[i])} {int(cin[i])}   | {int(s_out[i])}   {int(c_out[i])}")
        
    # Verification
    expected_s = torch.tensor([[0], [1], [1], [0], [1], [0], [0], [1]])
    expected_c = torch.tensor([[0], [0], [0], [1], [0], [1], [1], [1]])
    
    if torch.equal(s_out.int(), expected_s.int()) and torch.equal(c_out.int(), expected_c.int()):
        print("\nFULL ADDER PASSED")
    else:
        print("\nFULL ADDER FAILED")

if __name__ == "__main__":
    test_logic_gates()
    test_full_adder()

