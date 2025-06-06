#!/usr/bin/env python3
"""
显存优化版多GPU训练脚本
专门针对大模型和有限显存的情况
"""

import os
import sys
import argparse
import yaml
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import subprocess
import platform
import gc

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def setup_ddp(rank: int, world_size: int):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 使用不同端口避免冲突
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式训练"""
    destroy_process_group()

def check_available_memory(device):
    """检查可用显存"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        free = total - reserved
        return free, total
    return 0, 0

def train_worker_optimized(rank: int, world_size: int, config_path: str):
    """显存优化的训练工作进程"""
    
    # 设置分布式
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # 在每个进程开始前清理显存
    clear_gpu_memory()
    
    # 检查可用显存
    free_mem, total_mem = check_available_memory(rank)
    if rank == 0:
        print(f"GPU {rank}: 可用显存 {free_mem:.2f}GB / 总显存 {total_mem:.2f}GB")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 导入必要的模块
    from configs.base import ModelConfig
    from models import create_model
    
    # 创建模型配置
    model_config = ModelConfig(
        model_type=config['model_type'],
        vocab_size=config['model']['vocab_size'],
        max_seq_length=config['model']['max_seq_length'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    )
    
    # 创建模型
    print(f"GPU {rank}: 创建模型...")
    model = create_model(model_config.model_type, model_config)
    
    # 启用梯度检查点（如果配置启用）
    if config.get('optimization', {}).get('gradient_checkpointing', False):
        print(f"GPU {rank}: 启用梯度检查点")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    # 检查模型创建后的显存
    model_mem_before = torch.cuda.memory_allocated(rank) / 1e9 if torch.cuda.is_available() else 0
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"模型创建后显存使用: {model_mem_before:.2f}GB")
    
    # 移动模型到GPU（分批进行以减少显存峰值）
    print(f"GPU {rank}: 移动模型到设备...")
    try:
        # 逐层移动模型以减少显存峰值
        if hasattr(model, 'transformer'):
            # Transformer模型
            model.transformer.wte = model.transformer.wte.to(device)
            model.transformer.wpe = model.transformer.wpe.to(device)
            
            for i, layer in enumerate(model.transformer.h):
                layer = layer.to(device)
                if i % 4 == 0:  # 每4层清理一次缓存
                    clear_gpu_memory()
            
            model.transformer.ln_f = model.transformer.ln_f.to(device)
            model.lm_head = model.lm_head.to(device)
        else:
            # 其他模型类型直接移动
            model = model.to(device)
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU {rank}: 显存不足 - {e}")
        # 尝试清理并重试
        clear_gpu_memory()
        print(f"GPU {rank}: 清理缓存后重试...")
        try:
            model = model.to(device)
        except torch.cuda.OutOfMemoryError:
            print(f"GPU {rank}: 显存仍然不足，请减小模型规模或批大小")
            return
    
    # 检查模型移动后的显存
    model_mem_after = torch.cuda.memory_allocated(rank) / 1e9 if torch.cuda.is_available() else 0
    if rank == 0:
        print(f"模型移动后显存使用: {model_mem_after:.2f}GB")
    
    # 包装为DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 简化的训练循环（演示）
    print(f"GPU {rank}: 开始训练...")
    
    # 创建虚拟批次数据进行测试
    batch_size = config['training']['batch_size']
    seq_length = config['training']['max_length']
    
    try:
        # 测试前向传播
        dummy_input = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)
        
        with torch.cuda.amp.autocast(enabled=config['training']['fp16']):
            output = model(dummy_input)
        
        if rank == 0:
            forward_mem = torch.cuda.memory_allocated(rank) / 1e9
            print(f"前向传播后显存: {forward_mem:.2f}GB")
            print(f"✅ 前向传播成功，输出形状: {output.logits.shape if hasattr(output, 'logits') else output.shape}")
            
            # 保存简单的模型检查点
            os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
            checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "test_model.pt")
            torch.save({
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'config': model_config.__dict__,
                'rank': rank
            }, checkpoint_path)
            print(f"✅ 测试模型已保存至: {checkpoint_path}")
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU {rank}: 前向传播显存不足 - {e}")
        print(f"建议: 减小batch_size或max_length")
    
    # 清理
    if world_size > 1:
        cleanup_ddp()
    clear_gpu_memory()

def main():
    parser = argparse.ArgumentParser(description="显存优化的多GPU训练")
    parser.add_argument("--config", type=str, default="config_7b_transformer_fixed.yaml", help="配置文件")
    parser.add_argument("--check_memory", action="store_true", help="只检查显存不训练")
    
    args = parser.parse_args()
    
    # 检查显存模式
    if args.check_memory:
        if torch.cuda.is_available():
            print("🔍 GPU显存检查:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                free = total - reserved
                
                print(f"GPU {i}: {props.name}")
                print(f"  总显存: {total:.2f}GB")
                print(f"  已分配: {allocated:.2f}GB") 
                print(f"  已保留: {reserved:.2f}GB")
                print(f"  可用: {free:.2f}GB")
                
                if free < 15:  # 7B模型至少需要15GB可用显存
                    print(f"  ⚠️  显存可能不足（需要约15-20GB）")
                else:
                    print(f"  ✅ 显存充足")
        return
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    world_size = config['num_gpus']
    
    print(f"🚀 启动显存优化训练...")
    print(f"配置文件: {args.config}")
    print(f"GPU数量: {world_size}")
    print(f"批大小: {config['training']['batch_size']}")
    print(f"梯度累积: {config['training']['gradient_accumulation_steps']}")
    print(f"混合精度: {config['training']['fp16']}")
    
    # 清理所有GPU的缓存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
    
    try:
        if world_size <= 1:
            train_worker_optimized(0, 1, args.config)
        else:
            mp.spawn(
                train_worker_optimized,
                args=(world_size, args.config),
                nprocs=world_size,
                join=True
            )
        print("✅ 训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("💡 建议:")
        print("1. 检查GPU显存: python train_memory_optimized.py --check_memory")
        print("2. 减小批大小或序列长度")
        print("3. 清理其他GPU进程")

if __name__ == "__main__":
    main() 