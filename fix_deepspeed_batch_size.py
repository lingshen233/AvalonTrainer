#!/usr/bin/env python3
"""
修复DeepSpeed批次大小配置问题
"""

import json
import os
import sys
import argparse

def fix_deepspeed_config(config_file="deepspeed_config.json", num_gpus=4, micro_batch=4, grad_acc=8):
    """修复现有的DeepSpeed配置文件"""
    
    print(f"🔧 修复DeepSpeed配置: {config_file}")
    print(f"   GPU数量: {num_gpus}")
    print(f"   micro_batch_per_gpu: {micro_batch}")
    print(f"   gradient_accumulation_steps: {grad_acc}")
    
    # 计算正确的train_batch_size
    correct_train_batch_size = micro_batch * grad_acc * num_gpus
    
    print(f"   计算train_batch_size: {micro_batch} × {grad_acc} × {num_gpus} = {correct_train_batch_size}")
    
    # 检查是否存在现有配置
    if os.path.exists(config_file):
        print(f"📂 发现现有配置文件: {config_file}")
        
        # 读取现有配置
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"   当前train_batch_size: {config.get('train_batch_size', 'NOT SET')}")
        print(f"   当前micro_batch_per_gpu: {config.get('train_micro_batch_size_per_gpu', 'NOT SET')}")
        print(f"   当前gradient_accumulation_steps: {config.get('gradient_accumulation_steps', 'NOT SET')}")
        
        # 更新配置
        config['train_batch_size'] = correct_train_batch_size
        config['train_micro_batch_size_per_gpu'] = micro_batch
        config['gradient_accumulation_steps'] = grad_acc
        
        # 备份原文件
        backup_file = f"{config_file}.backup"
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(config_file, backup_file)
            print(f"💾 备份原文件到: {backup_file}")
    
    else:
        print(f"📝 创建新的配置文件: {config_file}")
        
        # 创建新配置
        config = {
            "train_batch_size": correct_train_batch_size,
            "train_micro_batch_size_per_gpu": micro_batch,
            "gradient_accumulation_steps": grad_acc,
            
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
                "offload_optimizer": {
                    "device": "none"
                }
            },
            
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 10000
                }
            },
            
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": True
            }
        }
    
    # 保存修复后的配置
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 配置文件已修复: {config_file}")
    
    # 验证配置
    verify_config(config_file, num_gpus)
    
    return config

def verify_config(config_file, num_gpus):
    """验证配置文件的正确性"""
    
    print(f"\n🔍 验证配置: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    train_batch = config.get('train_batch_size', 0)
    micro_batch = config.get('train_micro_batch_size_per_gpu', 0)
    grad_acc = config.get('gradient_accumulation_steps', 0)
    
    expected = micro_batch * grad_acc * num_gpus
    
    print(f"   train_batch_size: {train_batch}")
    print(f"   micro_batch_per_gpu: {micro_batch}")
    print(f"   gradient_accumulation_steps: {grad_acc}")
    print(f"   world_size: {num_gpus}")
    print(f"   公式验证: {micro_batch} × {grad_acc} × {num_gpus} = {expected}")
    
    if train_batch == expected:
        print("✅ 配置验证通过")
        return True
    else:
        print(f"❌ 配置验证失败: {train_batch} != {expected}")
        return False

def clean_old_configs():
    """清理可能的旧配置文件"""
    print("\n🧹 清理旧配置文件...")
    
    # 可能的配置文件名
    config_files = [
        "deepspeed_config.json",
        "ds_config.json", 
        "deepspeed.json",
        ".deepspeed_config",
        "deepspeed_config.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   发现: {config_file}")
            
            if config_file.endswith('.json'):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    train_batch = config.get('train_batch_size', 'NOT SET')
                    micro_batch = config.get('train_micro_batch_size_per_gpu', 'NOT SET')
                    grad_acc = config.get('gradient_accumulation_steps', 'NOT SET')
                    
                    print(f"      train_batch_size: {train_batch}")
                    print(f"      micro_batch_per_gpu: {micro_batch}")
                    print(f"      gradient_accumulation_steps: {grad_acc}")
                    
                    # 检查是否是问题配置
                    if train_batch == 16:
                        print(f"      ⚠️ 发现问题配置 (train_batch_size=16)")
                        
                        # 重命名为备份
                        backup_name = f"{config_file}.problem.backup"
                        os.rename(config_file, backup_name)
                        print(f"      🔄 重命名为: {backup_name}")
                
                except Exception as e:
                    print(f"      ❌ 无法读取: {e}")
            else:
                print(f"      📄 非JSON文件，跳过")

def generate_correct_configs():
    """生成不同GPU数量的正确配置"""
    print("\n📝 生成标准配置文件...")
    
    configs = [
        {"gpus": 1, "micro": 2, "grad_acc": 16, "name": "deepspeed_1gpu.json"},
        {"gpus": 2, "micro": 2, "grad_acc": 16, "name": "deepspeed_2gpu.json"},
        {"gpus": 4, "micro": 4, "grad_acc": 8, "name": "deepspeed_4gpu.json"},
        {"gpus": 8, "micro": 4, "grad_acc": 4, "name": "deepspeed_8gpu.json"},
    ]
    
    for cfg in configs:
        print(f"\n   生成 {cfg['gpus']}GPU 配置: {cfg['name']}")
        fix_deepspeed_config(
            config_file=cfg['name'],
            num_gpus=cfg['gpus'],
            micro_batch=cfg['micro'],
            grad_acc=cfg['grad_acc']
        )

def main():
    parser = argparse.ArgumentParser(description="修复DeepSpeed批次大小配置")
    parser.add_argument("--config", type=str, default="deepspeed_config.json", help="配置文件路径")
    parser.add_argument("--num_gpus", type=int, default=4, help="GPU数量")
    parser.add_argument("--micro_batch", type=int, default=4, help="每GPU的micro batch size")
    parser.add_argument("--grad_acc", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--clean", action="store_true", help="清理旧配置文件")
    parser.add_argument("--generate", action="store_true", help="生成标准配置文件")
    
    args = parser.parse_args()
    
    print("🔧 DeepSpeed批次大小修复工具")
    print("=" * 50)
    
    if args.clean:
        clean_old_configs()
    
    if args.generate:
        generate_correct_configs()
    
    # 修复指定配置
    if not args.generate:
        fix_deepspeed_config(
            config_file=args.config,
            num_gpus=args.num_gpus,
            micro_batch=args.micro_batch,
            grad_acc=args.grad_acc
        )
    
    print("\n🎯 修复完成！")
    print("\n📖 使用建议:")
    print("   4GPU训练: deepspeed --num_gpus=4 train_deepspeed.py --preset 7b_mamba")
    print("   验证配置: python train_deepspeed.py --preset 7b_mamba --num_gpus 4 --dry_run")

if __name__ == "__main__":
    main() 