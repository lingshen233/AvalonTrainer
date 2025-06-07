#!/usr/bin/env python3
"""
GPU显存问题诊断脚本
分析显存使用情况并提供优化建议
"""

import torch
import json
import os
import argparse

def check_gpu_memory():
    """检查GPU显存状态"""
    print("🔍 GPU显存检查")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"🎯 检测到 {gpu_count} 张GPU")
    
    total_memory = 0
    available_memory = 0
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / (1024**3)  # GB
        
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = total - allocated
        
        total_memory += total
        available_memory += free
        
        print(f"GPU {i}: {props.name}")
        print(f"  总显存: {total:.1f}GB")
        print(f"  已分配: {allocated:.1f}GB")
        print(f"  已保留: {reserved:.1f}GB") 
        print(f"  可用: {free:.1f}GB")
        print()
    
    print(f"📊 总计:")
    print(f"  总显存: {total_memory:.1f}GB")
    print(f"  可用显存: {available_memory:.1f}GB")
    
    return gpu_count, total_memory, available_memory

def estimate_model_memory(model_size_b=7):
    """估算模型显存需求"""
    print(f"\n🧮 估算 {model_size_b}B 参数模型显存需求")
    print("=" * 50)
    
    # 基本计算（FP16）
    model_params = model_size_b * 1e9
    model_memory_fp16 = model_params * 2 / (1024**3)  # 2 bytes per param
    
    # 优化器状态（AdamW: momentum + variance）
    optimizer_memory = model_params * 2 * 4 / (1024**3)  # 8 bytes per param (FP32)
    
    # 梯度
    gradient_memory = model_params * 2 / (1024**3)  # FP16 gradients
    
    # 激活值（估算）
    activation_memory_per_batch = 2.0  # GB per batch
    
    print(f"模型参数 (FP16): {model_memory_fp16:.1f}GB")
    print(f"优化器状态 (FP32): {optimizer_memory:.1f}GB") 
    print(f"梯度 (FP16): {gradient_memory:.1f}GB")
    print(f"激活值 (每batch): {activation_memory_per_batch:.1f}GB")
    
    total_per_gpu_no_zero = model_memory_fp16 + optimizer_memory + gradient_memory
    print(f"\n🚫 无ZeRO优化 - 每GPU需求: {total_per_gpu_no_zero:.1f}GB")
    
    # ZeRO-2优化
    zero2_optimizer = optimizer_memory  # 优化器状态分片
    zero2_per_gpu = model_memory_fp16 + gradient_memory + zero2_optimizer / 2
    print(f"🔄 ZeRO-2优化 - 每GPU需求: {zero2_per_gpu:.1f}GB")
    
    # ZeRO-3优化
    zero3_per_gpu = model_memory_fp16 / 4 + gradient_memory / 4  # 参数和梯度分片
    print(f"⚡ ZeRO-3优化 - 每GPU需求: {zero3_per_gpu:.1f}GB")
    
    return {
        'model_fp16': model_memory_fp16,
        'optimizer': optimizer_memory, 
        'gradient': gradient_memory,
        'activation_per_batch': activation_memory_per_batch,
        'no_zero': total_per_gpu_no_zero,
        'zero2': zero2_per_gpu,
        'zero3': zero3_per_gpu
    }

def analyze_config(config_file):
    """分析DeepSpeed配置"""
    print(f"\n📋 分析配置文件: {config_file}")
    print("=" * 50)
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 基本配置
    train_batch = config.get('train_batch_size', 'NOT SET')
    micro_batch = config.get('train_micro_batch_size_per_gpu', 'NOT SET')
    grad_acc = config.get('gradient_accumulation_steps', 'NOT SET')
    
    print(f"批次配置:")
    print(f"  train_batch_size: {train_batch}")
    print(f"  micro_batch_per_gpu: {micro_batch}")
    print(f"  gradient_accumulation_steps: {grad_acc}")
    
    # ZeRO配置
    zero_config = config.get('zero_optimization', {})
    zero_stage = zero_config.get('stage', 'NOT SET')
    offload_optimizer = zero_config.get('offload_optimizer', {}).get('device', 'none')
    offload_param = zero_config.get('offload_param', {}).get('device', 'none')
    
    print(f"\nZeRO优化:")
    print(f"  阶段: {zero_stage}")
    print(f"  优化器卸载: {offload_optimizer}")
    print(f"  参数卸载: {offload_param}")
    
    # 激活检查点
    activation_checkpointing = config.get('activation_checkpointing', {})
    cpu_checkpointing = activation_checkpointing.get('cpu_checkpointing', False)
    num_checkpoints = activation_checkpointing.get('number_checkpoints', 'NOT SET')
    
    print(f"\n激活检查点:")
    print(f"  CPU检查点: {cpu_checkpointing}")
    print(f"  检查点数量: {num_checkpoints}")
    
    return {
        'train_batch_size': train_batch,
        'micro_batch_per_gpu': micro_batch,
        'gradient_accumulation_steps': grad_acc,
        'zero_stage': zero_stage,
        'offload_optimizer': offload_optimizer,
        'offload_param': offload_param,
        'cpu_checkpointing': cpu_checkpointing
    }

def provide_recommendations(gpu_count, available_memory, memory_estimates, config_analysis):
    """提供优化建议"""
    print(f"\n💡 优化建议")
    print("=" * 50)
    
    if config_analysis is None:
        print("❌ 无法分析配置，跳过建议")
        return
    
    # 检查micro_batch_size
    micro_batch = config_analysis.get('micro_batch_per_gpu', 0)
    if isinstance(micro_batch, int) and micro_batch > 1:
        print(f"⚠️ micro_batch_per_gpu = {micro_batch} 太大")
        print(f"   建议: 设置为 1")
    
    # 检查ZeRO阶段
    zero_stage = config_analysis.get('zero_stage')
    if zero_stage != 3 and gpu_count >= 4:
        print(f"⚠️ 当前ZeRO阶段: {zero_stage}")
        print(f"   建议: 对于 {gpu_count} 张GPU，使用ZeRO-3")
    
    # 检查CPU卸载
    offload_optimizer = config_analysis.get('offload_optimizer', 'none')
    offload_param = config_analysis.get('offload_param', 'none')
    
    if offload_optimizer == 'none' or offload_param == 'none':
        print(f"⚠️ CPU卸载未启用")
        print(f"   优化器卸载: {offload_optimizer}")
        print(f"   参数卸载: {offload_param}")
        print(f"   建议: 启用CPU卸载以节省GPU显存")
    
    # 检查激活检查点
    cpu_checkpointing = config_analysis.get('cpu_checkpointing', False)
    if not cpu_checkpointing:
        print(f"⚠️ CPU激活检查点未启用")
        print(f"   建议: 启用CPU激活检查点")
    
    # 计算推荐配置
    available_per_gpu = available_memory / gpu_count
    recommended_zero_stage = 3 if available_per_gpu < 10 else 2
    
    print(f"\n📋 推荐配置:")
    print(f"  ZeRO阶段: {recommended_zero_stage}")
    print(f"  micro_batch_per_gpu: 1")
    print(f"  启用CPU卸载: 是")
    print(f"  启用CPU激活检查点: 是")
    
    # 生成具体的配置文件建议
    if available_per_gpu < 8:
        print(f"\n🚨 显存严重不足 (每GPU可用: {available_per_gpu:.1f}GB)")
        print(f"   建议使用: deepspeed_extreme_memory_safe.json")
    elif available_per_gpu < 12:
        print(f"\n⚠️ 显存紧张 (每GPU可用: {available_per_gpu:.1f}GB)")
        print(f"   建议使用: deepspeed_{gpu_count}gpu.json (如果存在)")

def main():
    parser = argparse.ArgumentParser(description="GPU显存问题诊断")
    parser.add_argument("--config", type=str, help="要分析的配置文件")
    parser.add_argument("--model_size", type=int, default=7, help="模型大小(B)")
    parser.add_argument("--fix", action="store_true", help="自动生成修复配置")
    
    args = parser.parse_args()
    
    print("🔍 GPU显存问题诊断工具")
    print("=" * 60)
    
    # 1. 检查GPU显存
    gpu_count, total_memory, available_memory = check_gpu_memory()
    
    # 2. 估算模型显存需求
    memory_estimates = estimate_model_memory(args.model_size)
    
    # 3. 分析配置文件
    config_analysis = None
    if args.config:
        config_analysis = analyze_config(args.config)
    
    # 4. 提供建议
    provide_recommendations(gpu_count, available_memory, memory_estimates, config_analysis)
    
    # 5. 自动修复
    if args.fix:
        print(f"\n🔧 自动生成优化配置...")
        os.system(f"python fix_deepspeed_batch_size.py --num_gpus {gpu_count} --generate")

if __name__ == "__main__":
    main() 