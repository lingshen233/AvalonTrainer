#!/usr/bin/env python3
"""
DeepSpeed ZeRO优化训练脚本 - 修复版
解决批次大小配置问题
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import argparse
import shutil

# DeepSpeed导入
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("❌ DeepSpeed未安装，将使用标准训练")

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

def load_config(config_path: str):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_deepspeed_config(model_config, training_config, num_gpus):
    """创建DeepSpeed配置"""
    # 计算正确的批次大小关系：
    # train_batch_size = micro_batch_per_gpu * gradient_accumulation_steps * world_size
    micro_batch_per_gpu = training_config.train_batch_size
    gradient_accumulation_steps = training_config.gradient_accumulation_steps
    train_batch_size = micro_batch_per_gpu * gradient_accumulation_steps * num_gpus
    
    ds_config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        # 启用ZeRO-2优化
        "zero_optimization": {
            "stage": 2,  # ZeRO-2: 分片优化器状态和梯度
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "offload_optimizer": {
                "device": "none"  # 不使用CPU卸载，Mamba模型不适合
            }
        },
        
        # 混合精度
        "fp16": {
            "enabled": training_config.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        # 优化器配置
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": training_config.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": training_config.weight_decay
            }
        },
        
        # 学习率调度器
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config.learning_rate,
                "warmup_num_steps": training_config.warmup_steps
            }
        },
        
        # 梯度裁剪
        "gradient_clipping": training_config.max_grad_norm,
        
        # 检查点配置
        "steps_per_print": training_config.logging_steps,
        "wall_clock_breakdown": False,
        
        # 内存优化
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": True
        }
    }
    
    return ds_config

def use_prebuilt_config(num_gpus):
    """使用预先构建的正确配置"""
    config_file = f"deepspeed_{num_gpus}gpu.json"
    
    if os.path.exists(config_file):
        print(f"✅ 使用预构建配置: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 验证配置正确性
        train_batch = config.get('train_batch_size', 0)
        micro_batch = config.get('train_micro_batch_size_per_gpu', 0)
        grad_acc = config.get('gradient_accumulation_steps', 0)
        expected = micro_batch * grad_acc * num_gpus
        
        if train_batch == expected:
            print(f"✅ 配置验证通过: {train_batch} = {micro_batch} × {grad_acc} × {num_gpus}")
            return config
        else:
            print(f"❌ 配置验证失败: {train_batch} != {expected}")
            return None
    else:
        print(f"⚠️ 未找到预构建配置: {config_file}")
        return None

def create_configs_from_yaml(yaml_config):
    """从YAML配置创建模型和训练配置"""
    from configs.base import ModelConfig, TrainingConfig
    
    # 创建模型配置
    model_params = yaml_config.get('model', {})
    model_config = ModelConfig(
        model_type=yaml_config.get('model_type', 'mamba'),
        vocab_size=model_params.get('vocab_size', 50257),
        max_seq_length=model_params.get('max_seq_length', 4096),
        d_model=model_params.get('d_model', 4864),
        n_layers=model_params.get('n_layers', 45),
        n_heads=model_params.get('n_heads', 32),
        d_ff=model_params.get('d_ff', 16384),
        dropout=model_params.get('dropout', 0.1),
        # Mamba特有参数
        d_state=model_params.get('d_state', 16),
        d_conv=model_params.get('d_conv', 4),
        expand=model_params.get('expand', 2)
    )
    
    # 创建训练配置
    training_params = yaml_config.get('training', {})
    training_config = TrainingConfig(
        dataset_name=training_params.get('dataset', 'auto'),
        train_batch_size=training_params.get('batch_size', 1),
        eval_batch_size=training_params.get('eval_batch_size', 1),
        gradient_accumulation_steps=training_params.get('gradient_accumulation_steps', 8),
        max_length=training_params.get('max_length', 4096),
        learning_rate=training_params.get('learning_rate', 1e-4),
        weight_decay=training_params.get('weight_decay', 0.01),
        max_grad_norm=training_params.get('max_grad_norm', 1.0),
        max_steps=training_params.get('max_steps', 200000),
        warmup_steps=training_params.get('warmup_steps', 10000),
        eval_steps=training_params.get('eval_steps', 5000),
        save_steps=training_params.get('save_steps', 10000),
        logging_steps=training_params.get('logging_steps', 100),
        fp16=training_params.get('fp16', True),
        output_dir=training_params.get('output_dir', './outputs'),
        checkpoint_dir=training_params.get('checkpoint_dir', './checkpoints'),
        use_wandb=training_params.get('use_wandb', False),
        wandb_project=training_params.get('wandb_project', 'rag-transformer'),
        wandb_run_name=training_params.get('wandb_run_name', 'deepspeed_7b'),
        distributed=True,  # DeepSpeed需要分布式
        world_size=yaml_config.get('num_gpus', 4)
    )
    
    return model_config, training_config

class DeepSpeedTrainer:
    """DeepSpeed优化训练器"""
    
    def __init__(self, model_config, training_config, ds_config):
        self.model_config = model_config
        self.training_config = training_config
        self.ds_config = ds_config
        
        # 创建输出目录
        os.makedirs(training_config.output_dir, exist_ok=True)
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
        
        # 初始化分布式环境
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
        
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"💾 DeepSpeed训练器初始化")
        print(f"   Local Rank: {self.local_rank}")
        print(f"   World Size: {self.world_size}")
        print(f"   输出目录: {os.path.abspath(training_config.output_dir)}")
    
    def create_model(self):
        """创建模型"""
        from models import create_model
        
        print("📦 创建模型...")
        model = create_model(self.model_config.model_type, self.model_config)
        
        # 使用正确的参数量计算函数
        from configs.model_presets import calculate_model_parameters
        estimated_params = calculate_model_parameters(self.model_config)
        
        # 实际参数量（用于验证）
        actual_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.local_rank == 0:
            print(f"模型参数: {actual_params:,} ({actual_params/1e9:.2f}B)")
            print(f"可训练参数: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
        
        return model
    
    def create_dataloader(self):
        """创建简化的数据加载器"""
        # 这里使用虚拟数据进行演示
        class DummyDataset:
            def __init__(self, vocab_size, max_length, num_samples=10000):
                self.vocab_size = vocab_size
                self.max_length = max_length
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # 生成随机序列
                input_ids = torch.randint(0, self.vocab_size, (self.max_length,))
                return {'input_ids': input_ids, 'labels': input_ids.clone()}
        
        dataset = DummyDataset(
            self.model_config.vocab_size,
            self.model_config.max_seq_length
        )
        
        # 使用DeepSpeed的数据采样器
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.training_config.train_batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
        return dataloader
    
    def train(self):
        """主训练循环"""
        # 创建模型
        model = self.create_model()
        
        # 创建数据加载器
        dataloader = self.create_dataloader()
        
        # 初始化DeepSpeed引擎
        if self.local_rank == 0:
            print("🚀 初始化DeepSpeed引擎...")
        
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            config=self.ds_config
        )
        
        # 开始训练
        if self.local_rank == 0:
            print("🚀 开始DeepSpeed训练...")
        
        model_engine.train()
        global_step = 0
        
        try:
            for epoch in range(1, 6):  # 限制epoch数用于演示
                dataloader.sampler.set_epoch(epoch)
                
                for step, batch in enumerate(dataloader):
                    if global_step >= self.training_config.max_steps:
                        break
                    
                    # 将数据移动到GPU
                    input_ids = batch['input_ids'].to(model_engine.device)
                    labels = batch['labels'].to(model_engine.device)
                    
                    # 前向传播
                    outputs = model_engine(input_ids)
                    
                    # 计算损失
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    # 语言建模损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # 反向传播
                    model_engine.backward(loss)
                    model_engine.step()
                    
                    global_step += 1
                    
                    # 记录日志
                    if global_step % self.training_config.logging_steps == 0 and self.local_rank == 0:
                        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else self.training_config.learning_rate
                        
                        # GPU显存使用
                        allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
                        reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
                        
                        print(f"步骤 {global_step}: 损失 {loss.item():.4f}, "
                              f"学习率 {current_lr:.2e}, "
                              f"显存 {allocated:.1f}/{reserved:.1f}GB")
                    
                    # 保存检查点
                    if global_step % self.training_config.save_steps == 0:
                        if self.local_rank == 0:
                            print(f"💾 保存检查点 (步骤 {global_step})...")
                        
                        checkpoint_dir = os.path.join(
                            self.training_config.checkpoint_dir,
                            f"step_{global_step}"
                        )
                        model_engine.save_checkpoint(checkpoint_dir)
                    
                    # 清理显存
                    if global_step % 10 == 0:
                        torch.cuda.empty_cache()
                
                if global_step >= self.training_config.max_steps:
                    break
        
        except Exception as e:
            if self.local_rank == 0:
                print(f"❌ 训练过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存最终检查点
        if self.local_rank == 0:
            print("💾 保存最终检查点...")
        
        final_checkpoint_dir = os.path.join(
            self.training_config.checkpoint_dir,
            "final"
        )
        model_engine.save_checkpoint(final_checkpoint_dir)
        
        if self.local_rank == 0:
            print("✅ DeepSpeed训练完成！")

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO优化训练脚本 - 修复版")
    
    # 基本参数
    parser.add_argument("--config", type=str, default="config_7b_mamba.yaml", help="配置文件路径")
    parser.add_argument("--preset", type=str, help="使用预设配置")
    parser.add_argument("--num_gpus", type=int, default=4, help="GPU数量")
    
    # DeepSpeed参数
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed配置文件")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地rank (DeepSpeed自动设置)")
    
    # 系统参数
    parser.add_argument("--dry_run", action="store_true", help="只验证配置")
    parser.add_argument("--check_memory", action="store_true", help="检查显存使用")
    parser.add_argument("--fix_config", action="store_true", help="自动修复配置问题")
    
    args = parser.parse_args()
    
    # 检查DeepSpeed
    if not DEEPSPEED_AVAILABLE:
        print("❌ 需要安装DeepSpeed:")
        print("pip install deepspeed")
        return
    
    # 显存检查
    if args.check_memory:
        print("🔍 GPU显存检查:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e9
                print(f"GPU {i}: {props.name} - {total:.1f}GB")
        return
    
    # 自动修复配置
    if args.fix_config:
        print("🔧 运行配置修复...")
        os.system(f"python fix_deepspeed_batch_size.py --num_gpus {args.num_gpus}")
    
    # 修复：优先使用用户指定的deepspeed_config
    ds_config = None
    
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        print(f"✅ 使用指定配置: {args.deepspeed_config}")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        
        # 验证配置正确性
        train_batch = ds_config.get('train_batch_size', 0)
        micro_batch = ds_config.get('train_micro_batch_size_per_gpu', 0)
        grad_acc = ds_config.get('gradient_accumulation_steps', 0)
        expected = micro_batch * grad_acc * args.num_gpus
        
        if train_batch == expected:
            print(f"✅ 配置验证通过: {train_batch} = {micro_batch} × {grad_acc} × {args.num_gpus}")
        else:
            print(f"❌ 配置验证失败: {train_batch} != {expected}")
            print(f"🔧 将自动修复批次大小为单GPU配置...")
            # 为单GPU修复配置
            ds_config['train_batch_size'] = micro_batch * grad_acc * 1
            print(f"✅ 修复完成: {ds_config['train_batch_size']} = {micro_batch} × {grad_acc} × 1")
    else:
        # 尝试使用预构建配置
        ds_config = use_prebuilt_config(args.num_gpus)
    
    if ds_config is None:
        print("🔄 fallback到动态生成配置...")
        
        # 处理预设配置
        if args.preset:
            from configs.model_presets import get_model_preset, get_training_config_for_model_size
            
            preset_config = get_model_preset(args.preset)
            model_config = preset_config['model']
            training_config = get_training_config_for_model_size(args.preset, args.num_gpus)
            
            print(f"✅ 使用预设: {preset_config['description']}")
            print(f"   参数量: {preset_config['params']}")
            
            # 创建虚拟yaml配置
            yaml_config = {
                'model_type': model_config.model_type,
                'num_gpus': args.num_gpus,
                'model': {
                    'vocab_size': model_config.vocab_size,
                    'max_seq_length': model_config.max_seq_length,
                    'd_model': model_config.d_model,
                    'n_layers': model_config.n_layers,
                    'd_state': getattr(model_config, 'd_state', 16),
                    'd_conv': getattr(model_config, 'd_conv', 4),
                    'expand': getattr(model_config, 'expand', 2),
                    'dropout': model_config.dropout
                },
                'training': {
                    'batch_size': training_config.train_batch_size,
                    'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
                    'max_length': training_config.max_length,
                    'learning_rate': training_config.learning_rate,
                    'max_steps': training_config.max_steps,
                    'warmup_steps': training_config.warmup_steps,
                    'save_steps': training_config.save_steps,
                    'logging_steps': training_config.logging_steps,
                    'fp16': training_config.fp16,
                    'output_dir': training_config.output_dir,
                    'checkpoint_dir': training_config.checkpoint_dir
                }
            }
        else:
            # 加载配置文件
            if not os.path.exists(args.config):
                print(f"❌ 配置文件不存在: {args.config}")
                return
            
            yaml_config = load_config(args.config)
            yaml_config['num_gpus'] = args.num_gpus
        
        # 创建配置对象
        model_config, training_config = create_configs_from_yaml(yaml_config)
        
        # 创建DeepSpeed配置
        ds_config = create_deepspeed_config(model_config, training_config, args.num_gpus)
    else:
        # 使用预构建配置，需要创建对应的model_config
        if args.preset:
            from configs.model_presets import get_model_preset, get_training_config_for_model_size
            preset_config = get_model_preset(args.preset)
            model_config = preset_config['model']
            training_config = get_training_config_for_model_size(args.preset, args.num_gpus)
        else:
            yaml_config = load_config(args.config)
            yaml_config['num_gpus'] = args.num_gpus
            model_config, training_config = create_configs_from_yaml(yaml_config)
    
    # 保存DeepSpeed配置文件（如果不是用户指定的）
    if not args.deepspeed_config:
        ds_config_path = "deepspeed_config.json"
        with open(ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
    else:
        ds_config_path = args.deepspeed_config
    
    # 打印配置信息
    from configs.model_presets import calculate_model_parameters
    total_params = calculate_model_parameters(model_config)
    
    # 计算批次大小详情
    micro_batch = ds_config['train_micro_batch_size_per_gpu']
    grad_acc = ds_config['gradient_accumulation_steps']
    world_size = args.num_gpus
    train_batch = ds_config['train_batch_size']
    
    print(f"\n📊 DeepSpeed训练配置:")
    print(f"模型类型: {model_config.model_type}")
    print(f"估算参数: {total_params/1e9:.2f}B")
    print(f"GPU数量: {args.num_gpus}")
    print(f"批大小/GPU: {micro_batch}")
    print(f"梯度累积: {grad_acc}")
    print(f"有效批大小: {train_batch}")
    print(f"ZeRO阶段: {ds_config['zero_optimization']['stage']}")
    print(f"DeepSpeed配置: {ds_config_path}")
    
    # 强制验证批次大小计算
    expected = micro_batch * grad_acc * world_size
    if train_batch != expected:
        print(f"\n❌ 致命错误: 批次大小不匹配")
        print(f"   train_batch_size: {train_batch}")
        print(f"   expected: {micro_batch} × {grad_acc} × {world_size} = {expected}")
        print(f"🔧 请运行: python fix_deepspeed_batch_size.py --num_gpus {args.num_gpus}")
        return
    else:
        print(f"\n✅ 批次大小验证通过: {train_batch} = {micro_batch} × {grad_acc} × {world_size}")
    
    if args.dry_run:
        print("\n✅ 配置验证完成（dry_run模式）")
        return
    
    # 启动训练
    try:
        trainer = DeepSpeedTrainer(model_config, training_config, ds_config)
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 自动运行修复建议
        print(f"\n🔧 建议运行修复命令:")
        print(f"python fix_deepspeed_batch_size.py --num_gpus {args.num_gpus}")

if __name__ == "__main__":
    main() 