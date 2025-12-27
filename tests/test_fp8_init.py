import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch

print("=== FP8 初始化方法测试 ===\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}\n")

# 方法 1: empty + normal_
print("[方法 1] torch.empty().normal_()")
try:
    x = torch.empty((2, 2), dtype=torch.float8_e4m3fn, device=device).normal_()
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 2: empty + uniform_
print("[方法 2] torch.empty().uniform_()")
try:
    x = torch.empty((2, 2), dtype=torch.float8_e4m3fn, device=device).uniform_(-1, 1)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 3: randn 直接指定 dtype
print("[方法 3] torch.randn(dtype=float8)")
try:
    x = torch.randn((2, 2), dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 4: rand 直接指定 dtype
print("[方法 4] torch.rand(dtype=float8)")
try:
    x = torch.rand((2, 2), dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 5: zeros
print("[方法 5] torch.zeros(dtype=float8)")
try:
    x = torch.zeros((2, 2), dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 6: ones
print("[方法 6] torch.ones(dtype=float8)")
try:
    x = torch.ones((2, 2), dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 7: full
print("[方法 7] torch.full(dtype=float8)")
try:
    x = torch.full((2, 2), 1.5, dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 8: tensor 从 list
print("[方法 8] torch.tensor([...], dtype=float8)")
try:
    x = torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 9: arange
print("[方法 9] torch.arange(dtype=float8)")
try:
    x = torch.arange(0, 4, dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 10: linspace
print("[方法 10] torch.linspace(dtype=float8)")
try:
    x = torch.linspace(0, 1, steps=4, dtype=torch.float8_e4m3fn, device=device)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 11: empty 然后 fill_
print("[方法 11] torch.empty().fill_()")
try:
    x = torch.empty((2, 2), dtype=torch.float8_e4m3fn, device=device).fill_(1.5)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

# 方法 12: 从 float32 转换
print("[方法 12] torch.randn(float32).to(float8)")
try:
    x = torch.randn((2, 2), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    print(f"成功: {x}")
except Exception as e:
    print(f"失败: {e}\n")

print("\n=== 测试结束 ===")
