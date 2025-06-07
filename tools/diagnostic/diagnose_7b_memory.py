#!/usr/bin/env python3
"""
7B Mamba模型内存诊断工具
专门分析6×32GB GPU环境下的内存使用
"""

import torch
import psutil
import subprocess
import json
from configs.config_presets import CONFIG_PRESETS

def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        gpus = []
        for line in lines:
            if line.strip():
                index, name, used, total = line.split(', ')
                gpus.append({
                    'index': int(index),
                    'name': name,
                    'used_mb': int(used),
                    'total_mb': int(total),
                    'free_mb': int(total) - int(used),
                    'usage_pct': (int(used) / int(total)) * 100
                })
        return gpus
    except Exception as e:
        print(f"❌ 无法获取GPU信息: {e}")
        return []

def estimate_model_memory(config):
    """估算模型内存需求"""
    # 7B模型参数量
    param_count = 7e9
    
    # FP16每个参数2字节
    model_memory_mb = (param_count * 2) / (1024**2)
    
    # 梯度内存（FP16）
    gradient_memory_mb = model_memory_mb
    
    # 优化器状态（AdamW需要2倍参数）
    optimizer_memory_mb = model_memory_mb * 2
    
    # 激活值内存（估算）
    batch_size = config.get('train_micro_batch_size_per_gpu', 1)
    seq_length = config.get('max_seq_length', 1024)
    d_model = 4096
    n_layers = 32
    
    # 估算激活值内存
    activation_memory_mb = (batch_size * seq_length * d_model * n_layers * 2) / (1024**2)
    
    return {
        'model_mb': model_memory_mb,
        'gradient_mb': gradient_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'activation_mb': activation_memory_mb,
        'total_mb': model_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb
    }

def analyze_deepspeed_config(config_file):
    """分析DeepSpeed配置的内存优化程度"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        score = 0
        suggestions = []
        
        # 检查ZeRO stage
        zero_stage = config.get('zero_optimization', {}).get('stage', 0)
        if zero_stage == 3:
            score += 30
        elif zero_stage == 2:
            score += 20
            suggestions.append("建议使用ZeRO-3以获得更好的内存节省")
        else:
            suggestions.append("强烈建议启用ZeRO-3")
        
        # 检查CPU卸载
        if config.get('zero_optimization', {}).get('offload_optimizer'):
            score += 25
        else:
            suggestions.append("建议启用优化器CPU卸载")
            
        if config.get('zero_optimization', {}).get('offload_param'):
            score += 25
        else:
            suggestions.append("建议启用参数CPU卸载")
        
        # 检查激活检查点
        if config.get('activation_checkpointing', {}).get('partition_activations'):
            score += 15
        else:
            suggestions.append("建议启用激活检查点")
        
        # 检查batch size
        micro_batch = config.get('train_micro_batch_size_per_gpu', 4)
        if micro_batch <= 1:
            score += 5
        else:
            suggestions.append(f"micro_batch_size过大({micro_batch})，建议设为1")
        
        return score, suggestions
        
    except Exception as e:
        return 0, [f"无法读取配置文件: {e}"]

def get_optimization_recommendations():
    """获取优化建议"""
    return [
        "🔧 立即可行的优化:",
        "  1. 使用极端配置: deepspeed_6gpu_extreme.json",
        "  2. 减少序列长度: --max_seq_length 512",
        "  3. 设置内存环境变量: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
        "🔬 模型级优化:",
        "  1. 已实现：分块式selective_scan计算",
        "  2. 减少d_state参数（从16到8）",
        "  3. 减少expand参数（从2到1.5）",
        "",
        "⚡ 硬件级优化:",
        "  1. 使用更小的模型（3B或1B）",
        "  2. 考虑使用8bit量化",
        "  3. 增加系统内存用于CPU卸载",
        "",
        "🎯 终极方案:",
        "  如果仍然OOM，建议：",
        "  - 使用3B参数模型",
        "  - 或者减少到4张GPU训练",
        "  - 或者使用gradient checkpointing + CPU卸载"
    ]

def main():
    print("🔍 7B Mamba模型内存诊断工具")
    print("=" * 60)
    
    # 1. 系统信息
    print("\n📱 系统信息:")
    print(f"  CPU内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # 2. GPU信息
    print("\n🎮 GPU信息:")
    gpus = get_gpu_memory_info()
    total_gpu_memory = 0
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    总内存: {gpu['total_mb']/1024:.1f} GB")
        print(f"    已使用: {gpu['used_mb']/1024:.1f} GB ({gpu['usage_pct']:.1f}%)")
        print(f"    可用: {gpu['free_mb']/1024:.1f} GB")
        total_gpu_memory += gpu['total_mb']
    
    print(f"\n  总GPU内存: {total_gpu_memory/1024:.1f} GB")
    
    # 3. 模型内存估算
    print("\n🧠 7B Mamba模型内存估算:")
    config = CONFIG_PRESETS['7b_mamba']
    memory_est = estimate_model_memory(config)
    
    print(f"  模型参数: {memory_est['model_mb']/1024:.1f} GB")
    print(f"  梯度: {memory_est['gradient_mb']/1024:.1f} GB")
    print(f"  优化器状态: {memory_est['optimizer_mb']/1024:.1f} GB")
    print(f"  激活值: {memory_est['activation_mb']/1024:.1f} GB")
    print(f"  总计: {memory_est['total_mb']/1024:.1f} GB")
    
    # 4. 配置分析
    config_files = [
        'deepspeed_6gpu.json',
        'deepspeed_6gpu_extreme.json',
        'deepspeed_6gpu_fp16_safe.json'
    ]
    
    print("\n⚙️ 配置文件分析:")
    for config_file in config_files:
        try:
            score, suggestions = analyze_deepspeed_config(config_file)
            print(f"\n  {config_file}:")
            print(f"    内存优化评分: {score}/100")
            if suggestions:
                for suggestion in suggestions[:3]:  # 只显示前3个建议
                    print(f"    - {suggestion}")
        except:
            print(f"  {config_file}: 文件不存在")
    
    # 5. 问题诊断
    print("\n🔍 问题诊断:")
    single_gpu_memory = total_gpu_memory / len(gpus) / 1024  # GB
    required_memory = memory_est['total_mb'] / 1024  # GB
    
    if required_memory > single_gpu_memory * 6:  # 即使分布式也不够
        print("  ❌ 严重问题：即使使用6GPU + ZeRO-3也可能内存不足")
        print(f"     需要: {required_memory:.1f} GB, 可用: {single_gpu_memory*6:.1f} GB")
    elif required_memory > single_gpu_memory:
        print("  ⚠️  需要分布式训练和内存优化")
        print(f"     单GPU不足，需要ZeRO-3 + CPU卸载")
    else:
        print("  ✅ 单GPU理论上足够，问题可能在实现细节")
    
    # 6. 优化建议
    print("\n💡 优化建议:")
    for recommendation in get_optimization_recommendations():
        print(recommendation)
    
    # 7. 快速启动命令
    print("\n🚀 推荐启动命令:")
    print("./launch_6gpu_extreme_safe.sh")
    print("")
    print("或手动:")
    print("export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,roundup_power2_divisions:16'")
    print("deepspeed --num_gpus=6 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_6gpu_extreme.json --max_seq_length 512")

if __name__ == "__main__":
    main() 