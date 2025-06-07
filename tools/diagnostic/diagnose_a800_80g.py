#!/usr/bin/env python3
"""
单卡A800-80GB内存诊断工具
分析80GB大显存环境下的最优配置
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

def estimate_model_memory_single_gpu(config, batch_size=4):
    """估算单GPU模型内存需求"""
    # 7B模型参数量
    param_count = 7e9
    
    # FP16每个参数2字节
    model_memory_mb = (param_count * 2) / (1024**2)
    
    # 梯度内存（FP16）
    gradient_memory_mb = model_memory_mb
    
    # 优化器状态（可以卸载到CPU）
    optimizer_memory_mb = 0  # CPU卸载
    
    # 激活值内存（单GPU，可以用更大的batch size）
    seq_length = config.get('max_seq_length', 1024)
    d_model = 4096
    n_layers = 32
    
    # 估算激活值内存 - A800可以承受更大的批次
    activation_memory_mb = (batch_size * seq_length * d_model * n_layers * 2) / (1024**2)
    
    return {
        'model_mb': model_memory_mb,
        'gradient_mb': gradient_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'activation_mb': activation_memory_mb,
        'total_mb': model_memory_mb + gradient_memory_mb + activation_memory_mb
    }

def get_a800_recommendations():
    """获取A800优化建议"""
    return [
        "🎯 A800-80GB专用优化策略:",
        "  1. 充分利用80GB大显存优势",
        "  2. 使用ZeRO-2而非ZeRO-3（减少通信开销）",
        "  3. 仅优化器CPU卸载，参数保留在GPU",
        "  4. 可以使用较大的batch size (4-8)",
        "  5. 启用完整的1024序列长度",
        "",
        "🔧 推荐配置:",
        "  - micro_batch_per_gpu: 4-8",
        "  - gradient_accumulation_steps: 2-4",
        "  - ZeRO stage: 2",
        "  - 激活检查点: 轻度使用",
        "  - 序列长度: 1024",
        "",
        "⚡ 性能优化:",
        "  1. 关闭不必要的激活检查点",
        "  2. 使用标准FP16设置",
        "  3. 增加CPU线程数利用多核",
        "  4. 可选择更激进的学习率",
        "",
        "🚀 如果还有余量:",
        "  - 尝试micro_batch=8或更大",
        "  - 增加模型复杂度（d_state, expand）",
        "  - 使用更长的序列长度（2048）"
    ]

def suggest_batch_size(gpu_memory_gb):
    """根据显存大小建议batch size"""
    if gpu_memory_gb >= 70:  # A800/A100-80GB
        return {
            'conservative': 4,
            'recommended': 6,
            'aggressive': 8,
            'max_seq_length': 1024
        }
    elif gpu_memory_gb >= 40:  # A100-40GB
        return {
            'conservative': 2,
            'recommended': 4,
            'aggressive': 6,
            'max_seq_length': 1024
        }
    elif gpu_memory_gb >= 24:  # RTX 4090/3090
        return {
            'conservative': 1,
            'recommended': 2,
            'aggressive': 3,
            'max_seq_length': 512
        }
    else:
        return {
            'conservative': 1,
            'recommended': 1,
            'aggressive': 2,
            'max_seq_length': 256
        }

def main():
    print("🎮 A800-80GB内存诊断工具")
    print("=" * 60)
    
    # 1. 系统信息
    print("\n📱 系统信息:")
    print(f"  CPU内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # 2. GPU信息
    print("\n🎮 GPU信息:")
    gpus = get_gpu_memory_info()
    
    if not gpus:
        print("  ❌ 无法检测到GPU")
        return
    
    gpu = gpus[0]  # 单GPU
    gpu_memory_gb = gpu['total_mb'] / 1024
    
    print(f"  GPU: {gpu['name']}")
    print(f"  总内存: {gpu_memory_gb:.1f} GB")
    print(f"  已使用: {gpu['used_mb']/1024:.1f} GB ({gpu['usage_pct']:.1f}%)")
    print(f"  可用: {gpu['free_mb']/1024:.1f} GB")
    
    # 3. A800特殊检查
    is_a800 = "A800" in gpu['name'] or "A100" in gpu['name']
    if is_a800:
        print(f"  ✅ 检测到高端GPU，适合训练大模型")
    else:
        print(f"  ⚠️  当前GPU不是A800/A100，建议调整配置")
    
    # 4. Batch size建议
    print(f"\n📊 针对{gpu_memory_gb:.0f}GB显存的Batch Size建议:")
    batch_suggestions = suggest_batch_size(gpu_memory_gb)
    
    print(f"  保守设置: micro_batch={batch_suggestions['conservative']}, seq_len={batch_suggestions['max_seq_length']}")
    print(f"  推荐设置: micro_batch={batch_suggestions['recommended']}, seq_len={batch_suggestions['max_seq_length']}")
    print(f"  激进设置: micro_batch={batch_suggestions['aggressive']}, seq_len={batch_suggestions['max_seq_length']}")
    
    # 5. 内存估算
    print(f"\n🧠 7B Mamba模型内存估算 (micro_batch={batch_suggestions['recommended']}):")
    config = CONFIG_PRESETS['7b_mamba']
    memory_est = estimate_model_memory_single_gpu(config, batch_suggestions['recommended'])
    
    print(f"  模型参数: {memory_est['model_mb']/1024:.1f} GB")
    print(f"  梯度: {memory_est['gradient_mb']/1024:.1f} GB")
    print(f"  优化器状态: {memory_est['optimizer_mb']/1024:.1f} GB (CPU卸载)")
    print(f"  激活值: {memory_est['activation_mb']/1024:.1f} GB")
    print(f"  总GPU需求: {memory_est['total_mb']/1024:.1f} GB")
    
    # 6. 可行性分析
    print(f"\n🔍 可行性分析:")
    required_memory = memory_est['total_mb'] / 1024
    available_memory = gpu['free_mb'] / 1024
    
    if required_memory < available_memory * 0.8:  # 留20%余量
        print(f"  ✅ 内存充足! 需要{required_memory:.1f}GB, 可用{available_memory:.1f}GB")
        print(f"  💡 可以尝试更大的batch size或序列长度")
    elif required_memory < available_memory:
        print(f"  ⚠️  内存紧张但可行。需要{required_memory:.1f}GB, 可用{available_memory:.1f}GB")
        print(f"  💡 建议使用保守配置")
    else:
        print(f"  ❌ 内存不足! 需要{required_memory:.1f}GB, 但只有{available_memory:.1f}GB")
        print(f"  💡 需要降低batch size或使用ZeRO-3")
    
    # 7. 优化建议
    print(f"\n💡 A800-80GB优化建议:")
    for recommendation in get_a800_recommendations():
        print(recommendation)
    
    # 8. 启动命令
    print(f"\n🚀 推荐启动命令:")
    print("./launch_single_a800_80g.sh")
    print("")
    print("或手动:")
    print("python train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_single_a800_80g.json --max_seq_length 1024")

if __name__ == "__main__":
    main() 