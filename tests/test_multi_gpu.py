#!/usr/bin/env python3
"""
测试多GPU训练功能
"""

import torch
import argparse

def test_gpu_setup():
    """测试GPU设置"""
    print("🔍 GPU信息检测:")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / 1024**3
            print(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
        
        return gpu_count
    else:
        print("❌ 未检测到CUDA支持")
        return 0

def test_config_options():
    """测试不同配置选项"""
    print("\n🧪 测试配置选项:")
    
    configs = [
        ("单GPU Mamba", "config.yaml"),
        ("4GPU Transformer", "config_transformer_4gpu.yaml")
    ]
    
    for name, config_file in configs:
        print(f"\n--- {name} ---")
        import subprocess
        try:
            result = subprocess.run([
                "python", "train.py", 
                "--config", config_file, 
                "--dry_run"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 提取关键信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if '模型类型:' in line or '参数量:' in line or '批大小:' in line or '估算显存:' in line:
                        print(f"  {line.strip()}")
                print("  ✅ 配置验证成功")
            else:
                print("  ❌ 配置验证失败")
                print(f"  错误: {result.stderr}")
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="测试多GPU训练设置")
    parser.add_argument("--skip_config", action="store_true", help="跳过配置测试")
    args = parser.parse_args()
    
    print("🚀 RAG Transformer 多GPU训练测试")
    print("=" * 50)
    
    # 测试GPU设置
    gpu_count = test_gpu_setup()
    
    # 测试配置
    if not args.skip_config:
        test_config_options()
    
    # 总结建议
    print("\n💡 使用建议:")
    if gpu_count == 0:
        print("- 当前环境不支持CUDA，只能使用CPU训练")
        print("- 建议使用Google Colab或云端GPU进行训练")
    elif gpu_count == 1:
        print("- 推荐使用单GPU配置 (config.yaml)")
        print("- Mamba模型显存效率更高")
    elif gpu_count >= 2:
        print(f"- 可以使用{gpu_count}GPU并行训练")
        print("- 修改配置文件中的 num_gpus 参数")
        print("- 批大小会自动调整以适应多GPU")
    
    print(f"\n🎯 开始训练命令:")
    if gpu_count >= 4:
        print(f"python train.py --config config_transformer_4gpu.yaml --num_gpus {min(gpu_count, 4)}")
    else:
        print("python train.py")

if __name__ == "__main__":
    main() 