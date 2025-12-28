#!/usr/bin/env python
"""
SNNTorch 集成测试入口
====================
运行所有现有测试文件

使用方法:
    python SNNTorch/tests/run_all_tests.py           # 运行所有测试
    python SNNTorch/tests/run_all_tests.py -v        # 详细输出
    python SNNTorch/tests/run_all_tests.py -k fp32   # 只运行包含 fp32 的测试
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建 pytest 命令
    cmd = [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short"] + sys.argv[1:]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("=" * 70)
    
    sys.exit(subprocess.call(cmd))
