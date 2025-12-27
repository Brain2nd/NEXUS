import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import PulseFloatingPointEncoder

def test_fp8_real_data():
    print("=== Testing PulseFloatingPointEncoder with Native FP8 Data (E4M3) ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # E4M3 Format:
    # S: 1 bit
    # E: 4 bits (Bias = 2^(4-1)-1 = 7 ??? No, E4M3 bias is usually 7 or 8 depending on standard)
    #    Standard E4M3FN (PyTorch) Bias = 7.
    #    Range: E_stored 0~15.
    # M: 3 bits
    # Total: 8 bits
    
    # 配置编码器为 E4M3
    # 注意：Scanner bits 需要足够大以覆盖 FP8 的动态范围
    # FP8 E4M3 max ~ 448, min ~ 2^-6 ~ 0.015
    encoder = PulseFloatingPointEncoder(
        exponent_bits=4, 
        mantissa_bits=3,
        scan_integer_bits=10, # 覆盖到 2^9=512 > 448
        scan_decimal_bits=10  # 覆盖到 2^-10 足够精确
    ).to(device)
    
    # 生成 FP8 数据
    # 使用 torch.float8_e4m3fn 类型
    try:
        # Case 1: Specific Values
        val_list = [1.5, -0.25, 6.0, 448.0, 0.015625]
        fp8_tensor = torch.tensor(val_list, device=device).to(torch.float8_e4m3fn)
        
        print(f"\nInput Tensor (FP8 E4M3FN):\n{fp8_tensor}")
        print(f"Original Float Values: {fp8_tensor.float()}")
        
        # 编码器只接受 float32/float16 输入来进行 SNN 模拟
        # 所以我们需要把 FP8 转回 float32 喂给网络，验证网络的输出是否还原了 FP8 的结构
        input_for_snn = fp8_tensor.float().unsqueeze(1) # [Batch, 1]
        
        output = encoder(input_for_snn)
        
        print("-" * 30)
        print(f"SNN Encoded Output Shape: {output.shape}")
        
        # 打印结果
        # FP8 E4M3: [S(1) | E(4) | M(3)]
        print("\nEncoding Results (S | E | M):")
        for i, val in enumerate(val_list):
            # output[i] shape is [1, 8], squeeze to [8]
            bits = output[i].squeeze().int().tolist()
            s = bits[0]
            e = bits[1:5]
            m = bits[5:]
            
            print(f"Value: {val:10.6f} -> S:{s} E:{e} M:{m} -> Raw: {bits}")
            
    except AttributeError as e:
        print(f"\nSkipping FP8 test: Your PyTorch version might not support float8_e4m3fn yet.\nError: {e}")
    except Exception as e:
        print(f"\nTest Failed: {e}")

if __name__ == "__main__":
    test_fp8_real_data()
