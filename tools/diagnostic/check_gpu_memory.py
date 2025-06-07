#!/usr/bin/env python3
"""
GPU显存检查和管理工具
"""

import torch
import subprocess
import sys
import os

def check_gpu_memory():
    """检查GPU显存使用情况"""
    print("🔍 检查GPU显存使用情况...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"📊 检测到 {num_gpus} 个GPU:")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1e9
        
        # 获取当前显存使用情况
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free = total_memory - reserved
        
        print(f"\n  GPU {i}: {props.name}")
        print(f"    总显存: {total_memory:.2f}GB")
        print(f"    已分配: {allocated:.2f}GB")
        print(f"    已保留: {reserved:.2f}GB")
        print(f"    可用: {free:.2f}GB")
        
        if free < 5.0:  # 小于5GB可用显存时警告
            print(f"    ⚠️  显存不足！")

def clear_gpu_cache():
    """清理GPU缓存"""
    print("\n🧹 清理GPU缓存...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ PyTorch GPU缓存已清理")
        
        # 再次检查显存
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free = (torch.cuda.get_device_properties(i).total_memory - 
                   torch.cuda.memory_reserved(i)) / 1e9
            print(f"  GPU {i} 可用显存: {free:.2f}GB")
    else:
        print("❌ CUDA不可用，无法清理")

def estimate_model_memory(model_size='7B'):
    """估算模型显存需求"""
    print(f"\n💾 估算{model_size}模型显存需求:")
    
    if model_size == '7B':
        params = 7e9
    elif model_size == '1B':
        params = 1e9
    else:
        params = float(model_size.replace('B', 'e9').replace('M', 'e6'))
    
    # FP16参数存储
    param_memory = params * 2 / 1e9
    
    # 优化器状态（Adam需要2倍参数）
    optimizer_memory = param_memory * 2
    
    # 梯度
    gradient_memory = param_memory
    
    # 激活值（估算）
    activation_memory = param_memory * 0.5
    
    total_per_gpu = (param_memory + optimizer_memory + gradient_memory + activation_memory) / 4  # 4GPU分布
    
    print(f"  参数存储: {param_memory:.2f}GB")
    print(f"  优化器状态: {optimizer_memory:.2f}GB") 
    print(f"  梯度: {gradient_memory:.2f}GB")
    print(f"  激活值: {activation_memory:.2f}GB")
    print(f"  总需求: {param_memory + optimizer_memory + gradient_memory + activation_memory:.2f}GB")
    print(f"  每GPU需求(4卡): {total_per_gpu:.2f}GB")

def kill_gpu_processes():
    """终止占用GPU的进程"""
    print("\n🔪 检查GPU进程...")
    
    try:
        # 运行nvidia-smi获取进程信息
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            print("当前GPU进程:")
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid, name, memory = parts[0], parts[1], parts[2]
                    print(f"  PID {pid}: {name} (使用 {memory}MB)")
            
            # 询问是否终止进程
            print("\n⚠️  发现GPU进程，是否终止？(y/N)")
            # 在脚本中不等待用户输入，只显示信息
            print("💡 手动终止命令: sudo kill -9 <PID>")
        else:
            print("✅ 未检测到GPU进程")
            
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到")

def main():
    parser = argparse.ArgumentParser(description="GPU显存检查和管理")
    parser.add_argument("--clear", action="store_true", help="清理GPU缓存")
    parser.add_argument("--estimate", type=str, default="7B", help="估算模型显存需求")
    parser.add_argument("--kill", action="store_true", help="检查GPU进程")
    
    args = parser.parse_args()
    
    print("🚀 GPU显存管理工具")
    print("=" * 50)
    
    check_gpu_memory()
    
    if args.clear:
        clear_gpu_cache()
    
    if args.estimate:
        estimate_model_memory(args.estimate)
    
    if args.kill:
        kill_gpu_processes()
    
    print("\n💡 解决显存不足的建议:")
    print("1. 清理GPU缓存: python check_gpu_memory.py --clear")
    print("2. 终止其他GPU进程: sudo kill -9 <PID>")
    print("3. 减小批大小: 修改config中的batch_size")
    print("4. 启用梯度检查点: 减少激活值显存占用")
    print("5. 使用DeepSpeed ZeRO: 分布式优化器状态")

if __name__ == "__main__":
    import argparse
    main() 