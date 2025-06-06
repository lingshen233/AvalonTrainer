#!/usr/bin/env python3
"""
简化的多GPU训练脚本
支持YAML配置文件和命令行参数
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

from configs.presets import get_config, calculate_model_size, estimate_memory_usage, list_available_configs
from configs.base import ModelConfig, TrainingConfig
from models import create_model
from trainers.base import BaseTrainer
from data.processor import DataProcessor

def load_config(config_path: str):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def is_docker_container():
    """检测是否在Docker容器中运行"""
    try:
        # 检查是否存在Docker特征文件
        return (os.path.exists('/.dockerenv') or 
                os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read())
    except:
        return False

def is_root_user():
    """检测是否为root用户"""
    return os.geteuid() == 0

def auto_shutdown(delay_seconds: int = 60):
    """自动关机功能"""
    print(f"\n🔄 训练完成！将在 {delay_seconds} 秒后自动关机...")
    print("按 Ctrl+C 取消自动关机")
    
    try:
        for i in range(delay_seconds, 0, -1):
            print(f"\r⏰ 倒计时: {i} 秒", end="", flush=True)
            time.sleep(1)
        
        print(f"\n💤 正在关机...")
        
        # 检测环境并选择合适的关机命令
        system = platform.system().lower()
        in_docker = is_docker_container()
        is_root = is_root_user()
        
        if in_docker:
            print("🐳 检测到Docker容器环境")
            # 在Docker容器中，通常只能停止容器，不能关机
            print("💡 容器环境无法直接关机，建议手动停止容器")
            print("   可以使用: docker stop <container_id>")
            return
        
        if system == "windows":
            subprocess.run(["shutdown", "/s", "/t", "0"])
        elif system in ["linux", "darwin"]:  # Linux或macOS
            if is_root:
                # root用户直接使用shutdown
                subprocess.run(["shutdown", "-h", "now"])
            else:
                # 非root用户使用sudo
                subprocess.run(["sudo", "shutdown", "-h", "now"])
        else:
            print("❌ 不支持的操作系统，无法自动关机")
            
    except KeyboardInterrupt:
        print(f"\n❌ 自动关机已取消")
    except FileNotFoundError as e:
        print(f"\n❌ 关机命令未找到: {e}")
        print("💡 可能的解决方案:")
        if is_docker_container():
            print("   - Docker容器环境请手动停止容器")
        else:
            print("   - 确保系统支持shutdown命令")
            print("   - 检查用户权限设置")
    except Exception as e:
        print(f"\n❌ 自动关机失败: {e}")

def create_configs_from_yaml(yaml_config):
    """从YAML配置创建模型和训练配置"""
    
    # 处理自动批大小
    batch_size = yaml_config['training']['batch_size']
    if batch_size is None:
        # 根据GPU数量和模型类型自动设置
        if yaml_config['model_type'] == 'mamba':
            batch_size = 6 if yaml_config['num_gpus'] == 1 else 8
        else:
            batch_size = 4 if yaml_config['num_gpus'] == 1 else 6
    
    # 模型配置
    model_config = ModelConfig(
        model_type=yaml_config['model_type'],
        vocab_size=yaml_config['model']['vocab_size'],
        max_seq_length=yaml_config['model']['max_seq_length'],
        d_model=yaml_config['model']['d_model'],
        n_layers=yaml_config['model']['n_layers'],
        dropout=yaml_config['model']['dropout'],
        n_heads=yaml_config['model']['n_heads'],
        d_ff=yaml_config['model']['d_ff'],
        d_state=yaml_config['model']['d_state'],
        d_conv=yaml_config['model']['d_conv'],
        expand=yaml_config['model']['expand']
    )
    
    # 训练配置
    training_config = TrainingConfig(
        dataset_name=yaml_config['training']['dataset'],
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        gradient_accumulation_steps=yaml_config['training']['gradient_accumulation_steps'],
        max_length=yaml_config['training']['max_length'],
        learning_rate=yaml_config['training']['learning_rate'],
        weight_decay=yaml_config['training']['weight_decay'],
        max_grad_norm=yaml_config['training']['max_grad_norm'],
        max_steps=yaml_config['training']['max_steps'],
        warmup_steps=yaml_config['training']['warmup_steps'],
        eval_steps=yaml_config['training']['eval_steps'],
        save_steps=yaml_config['training']['save_steps'],
        logging_steps=yaml_config['training']['logging_steps'],
        fp16=yaml_config['training']['fp16'],
        output_dir=yaml_config['training']['output_dir'],
        checkpoint_dir=yaml_config['training']['checkpoint_dir'],
        use_wandb=yaml_config['training']['use_wandb'],
        wandb_project=yaml_config['training']['wandb_project'],
        wandb_run_name=yaml_config['training']['run_name'],
        distributed=(yaml_config['num_gpus'] > 1),
        world_size=yaml_config['num_gpus']
    )
    
    return model_config, training_config

def setup_ddp(rank: int, world_size: int):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式训练"""
    destroy_process_group()

def train_worker(rank: int, world_size: int, model_config: ModelConfig, training_config: TrainingConfig):
    """多GPU训练的工作进程"""
    
    # 设置分布式
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_model(model_config.model_type, model_config)
    model = model.to(device)
    
    # 包装为DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 创建简化的训练器
    class SimpleTrainer:
        def __init__(self, model, config, device, rank):
            self.model = model
            self.config = config
            self.device = device
            self.rank = rank
            self.data_processor = DataProcessor(config)
        
        def train(self):
            print(f"GPU {self.rank}: 开始训练...")
            # 这里可以添加实际的训练循环
            # 目前只是演示框架
            
            if self.rank == 0:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
                
                # 创建输出目录
                os.makedirs(self.config.output_dir, exist_ok=True)
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                
                # 保存最终模型（示例）
                final_model_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': model_config.__dict__,
                    'total_params': total_params
                }, final_model_path)
                
                print(f"✅ 模型已保存至: {os.path.abspath(final_model_path)}")
    
    trainer = SimpleTrainer(model, training_config, device, rank)
    trainer.train()
    
    if world_size > 1:
        cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description="简化的多GPU训练脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model_type", type=str, choices=["transformer", "mamba"], help="模型类型")
    parser.add_argument("--num_gpus", type=int, help="GPU数量")
    parser.add_argument("--list_models", action="store_true", help="列出可用模型")
    parser.add_argument("--dry_run", action="store_true", help="只验证配置")
    parser.add_argument("--no_shutdown", action="store_true", help="禁用自动关机")
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print("\n🤖 可用模型:")
        for model_type, desc in list_available_configs().items():
            print(f"  {model_type}: {desc}")
        return
    
    # 加载配置
    if os.path.exists(args.config):
        yaml_config = load_config(args.config)
        print(f"✅ 加载配置文件: {args.config}")
    else:
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    # 命令行参数覆盖
    if args.model_type:
        yaml_config['model_type'] = args.model_type
    if args.num_gpus:
        yaml_config['num_gpus'] = args.num_gpus
    if args.no_shutdown:
        yaml_config['system']['auto_shutdown'] = False
    
    # 创建配置
    model_config, training_config = create_configs_from_yaml(yaml_config)
    
    # 计算资源需求
    total_params = calculate_model_size(model_config)
    memory_info = estimate_memory_usage(model_config, training_config)
    
    # 打印环境信息
    print(f"\n🌍 运行环境:")
    print(f"操作系统: {platform.system()}")
    print(f"Docker容器: {'是' if is_docker_container() else '否'}")
    print(f"Root用户: {'是' if is_root_user() else '否'}")
    
    # 打印配置信息
    print(f"\n📊 训练配置:")
    print(f"模型类型: {model_config.model_type}")
    print(f"参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"GPU数量: {yaml_config['num_gpus']}")
    print(f"批大小: {training_config.train_batch_size}")
    print(f"估算显存: {memory_info['total_memory_gb']:.1f}GB/GPU")
    print(f"输出目录: {os.path.abspath(training_config.output_dir)}")
    print(f"模型保存: {os.path.abspath(training_config.checkpoint_dir)}")
    
    # 显示自动关机状态
    auto_shutdown_enabled = yaml_config.get('system', {}).get('auto_shutdown', False)
    if auto_shutdown_enabled:
        shutdown_delay = yaml_config.get('system', {}).get('shutdown_delay', 60)
        print(f"🔄 自动关机: 启用 ({shutdown_delay}秒延迟)")
        if is_docker_container():
            print(f"⚠️  Docker环境警告: 将显示关机提示但不会实际关机")
    else:
        print(f"🔄 自动关机: 禁用")
    
    if args.dry_run:
        print("\n✅ 配置验证完成（dry_run模式）")
        return
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA，将使用CPU训练")
        yaml_config['num_gpus'] = 0
    elif yaml_config['num_gpus'] > torch.cuda.device_count():
        print(f"⚠️ 请求{yaml_config['num_gpus']}个GPU，但只有{torch.cuda.device_count()}个可用")
        yaml_config['num_gpus'] = torch.cuda.device_count()
    
    # 开始训练
    world_size = yaml_config['num_gpus']
    
    try:
        if world_size <= 1:
            # 单GPU训练
            print("🚀 启动单GPU训练...")
            train_worker(0, 1, model_config, training_config)
        else:
            # 多GPU训练
            print(f"🚀 启动{world_size}GPU训练...")
            mp.spawn(
                train_worker,
                args=(world_size, model_config, training_config),
                nprocs=world_size,
                join=True
            )
        
        print("✅ 训练完成！")
        
        # 自动关机功能
        if auto_shutdown_enabled and not args.no_shutdown:
            shutdown_delay = yaml_config.get('system', {}).get('shutdown_delay', 60)
            auto_shutdown(shutdown_delay)
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")

if __name__ == "__main__":
    main() 