#!/usr/bin/env python3
"""
显存优化版单机多GPU训练脚本
取消分布式训练，使用DataParallel，节省显存
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.nn import DataParallel
import time
import subprocess
import platform
import gc
from pathlib import Path

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 已清理所有GPU缓存")

def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def is_docker_container():
    """检查是否在Docker容器中"""
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read()
    except:
        return False

def is_root_user():
    """检查是否为root用户"""
    return os.getuid() == 0 if hasattr(os, 'getuid') else False

def auto_shutdown(delay_seconds: int = 60):
    """自动关机功能"""
    is_docker = is_docker_container()
    is_root = is_root_user()
    
    if not is_docker and not is_root:
        print(f"⚠️ 非Docker环境且非root用户，无法执行关机命令")
        return
    
    print(f"🔄 {delay_seconds}秒后自动关机...")
    
    for remaining in range(delay_seconds, 0, -1):
        print(f"⏳ 倒计时: {remaining}秒 (Ctrl+C取消)")
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n🚫 用户取消自动关机")
            return
    
    print("🔌 执行关机命令...")
    try:
        if is_docker:
            subprocess.run(['halt'], check=True)
        else:
            subprocess.run(['shutdown', '-h', 'now'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 关机命令执行失败: {e}")
    except FileNotFoundError:
        print("❌ 关机命令未找到")

def create_configs_from_yaml(yaml_config):
    """从YAML配置创建模型和训练配置"""
    from configs.base import ModelConfig, TrainingConfig
    
    # 创建模型配置
    model_params = yaml_config.get('model', {})
    model_config = ModelConfig(
        model_type=yaml_config.get('model_type', 'transformer'),
        vocab_size=model_params.get('vocab_size', 50257),
        max_seq_length=model_params.get('max_seq_length', 2048),
        d_model=model_params.get('d_model', 768),
        n_layers=model_params.get('n_layers', 12),
        n_heads=model_params.get('n_heads', 12),
        d_ff=model_params.get('d_ff', 3072),
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
        train_batch_size=training_params.get('batch_size', 4),
        eval_batch_size=training_params.get('eval_batch_size', 4),
        gradient_accumulation_steps=training_params.get('gradient_accumulation_steps', 1),
        max_length=training_params.get('max_length', 512),
        learning_rate=training_params.get('learning_rate', 5e-5),
        weight_decay=training_params.get('weight_decay', 0.01),
        max_grad_norm=training_params.get('max_grad_norm', 1.0),
        max_steps=training_params.get('max_steps', 10000),
        warmup_steps=training_params.get('warmup_steps', 1000),
        eval_steps=training_params.get('eval_steps', 500),
        save_steps=training_params.get('save_steps', 1000),
        logging_steps=training_params.get('logging_steps', 100),
        fp16=training_params.get('fp16', True),
        output_dir=training_params.get('output_dir', './outputs'),
        checkpoint_dir=training_params.get('checkpoint_dir', './checkpoints'),
        use_wandb=training_params.get('use_wandb', False),
        wandb_project=training_params.get('wandb_project', 'rag-transformer'),
        wandb_run_name=training_params.get('wandb_run_name', 'run'),
        distributed=False,  # 单机多卡不使用分布式
        world_size=1
    )
    
    return model_config, training_config

def calculate_model_size(model_config):
    """计算模型参数量"""
    if model_config.model_type == 'transformer':
        # Transformer参数估算
        embedding_params = model_config.vocab_size * model_config.d_model
        attention_params = 4 * model_config.d_model * model_config.d_model
        ffn_params = 2 * model_config.d_model * model_config.d_ff
        layer_params = attention_params + ffn_params
        total_params = embedding_params + model_config.n_layers * layer_params
    elif model_config.model_type == 'mamba':
        # Mamba参数估算（更准确）
        embedding_params = model_config.vocab_size * model_config.d_model
        
        # 每层Mamba块的参数
        d_inner = model_config.d_model * model_config.expand  # inner dimension
        
        # 输入投影层
        in_proj_params = model_config.d_model * (d_inner * 2)  # x and z projections
        
        # 卷积层
        conv_params = d_inner * model_config.d_conv
        
        # 状态空间参数
        ss_params = d_inner * model_config.d_state  # A and B matrices
        dt_params = d_inner  # dt projection
        
        # 输出投影
        out_proj_params = d_inner * model_config.d_model
        
        # 层归一化
        norm_params = model_config.d_model
        
        # 每层总参数
        layer_params = in_proj_params + conv_params + ss_params + dt_params + out_proj_params + norm_params
        
        # 最终层归一化和语言模型头
        final_norm_params = model_config.d_model
        lm_head_params = model_config.vocab_size * model_config.d_model
        
        total_params = embedding_params + (model_config.n_layers * layer_params) + final_norm_params + lm_head_params
    else:
        total_params = 0
    
    return total_params

def estimate_memory_usage(model_config, training_config):
    """估算显存使用"""
    params = calculate_model_size(model_config)
    
    # 基础模型显存 (FP16)
    model_memory = params * 2 / 1e9  # 2 bytes per parameter
    
    # 优化器状态 (Adam: 8 bytes per parameter)
    optimizer_memory = params * 8 / 1e9
    
    # 梯度 (FP16)
    gradient_memory = params * 2 / 1e9
    
    # 激活值估算 (batch_size * seq_length * d_model * layers)
    activation_memory = (training_config.train_batch_size * 
                        training_config.max_length * 
                        model_config.d_model * 
                        model_config.n_layers * 2) / 1e9
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        'model_memory_gb': model_memory,
        'optimizer_memory_gb': optimizer_memory,
        'gradient_memory_gb': gradient_memory,
        'activation_memory_gb': activation_memory,
        'total_memory_gb': total_memory
    }

def list_available_configs():
    """列出可用的模型配置"""
    return {
        'transformer': 'Transformer模型 - 标准注意力机制',
        'mamba': 'Mamba模型 - 状态空间模型'
    }

class OptimizedTrainer:
    """优化的单机多GPU训练器"""
    
    def __init__(self, model, config, device_count):
        self.model = model
        self.config = config
        self.device_count = device_count
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        print(f"💾 训练器初始化完成")
        print(f"   使用设备: {self.device}")
        print(f"   GPU数量: {self.device_count}")
        print(f"   输出目录: {os.path.abspath(self.config.output_dir)}")
        print(f"   检查点目录: {os.path.abspath(self.config.checkpoint_dir)}")
    
    def train(self):
        """训练循环"""
        print("🚀 开始训练...")
        
        # 简化的训练循环（演示）
        for step in range(1, min(self.config.max_steps + 1, 100)):  # 限制步数用于测试
            try:
                # 创建虚拟批次数据
                batch_size = self.config.train_batch_size
                seq_length = min(self.config.max_length, 1024)  # 限制序列长度
                
                # 生成随机输入数据
                input_ids = torch.randint(0, 50257, (batch_size, seq_length))
                
                # 移动到设备
                input_ids = input_ids.to(self.device)
                
                # 前向传播
                with torch.amp.autocast('cuda', enabled=self.config.fp16):
                    outputs = self.model(input_ids)
                    
                    # 计算损失
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    # 简单的语言建模损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
                
                # 记录
                if step % self.config.logging_steps == 0:
                    current_mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                    print(f"步骤 {step}: 损失 {loss.item():.4f}, 显存 {current_mem:.2f}GB")
                
                # 清理
                del input_ids, outputs, logits, loss
                if step % 5 == 0:  # 每5步清理一次（更频繁）
                    clear_gpu_memory()
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ 步骤 {step} 显存不足: {e}")
                print(f"🔧 批大小 {batch_size}, 序列长度 {seq_length} 仍然过大")
                
                # 尝试更激进的减少
                if batch_size > 1:
                    self.config.train_batch_size = max(1, batch_size // 2)
                    print(f"🔧 减小批大小至 {self.config.train_batch_size}")
                elif seq_length > 256:
                    seq_length = max(256, seq_length // 2)
                    print(f"🔧 减小序列长度至 {seq_length}")
                else:
                    print("❌ 无法进一步减小参数，模型可能太大")
                    break
                    
                # 清理后重试
                clear_gpu_memory()
                continue
                
            except Exception as e:
                print(f"❌ 步骤 {step} 训练错误: {e}")
                break
        
        # 保存最终模型
        final_model_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        
        # 如果使用了DataParallel，需要保存module
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 构建完整的配置字典
        config_dict = {
            'model_type': getattr(model_to_save, 'config', {}).get('model_type') or 
                         getattr(model_to_save, 'model_type', 'mamba'),
            'vocab_size': getattr(model_to_save.config, 'vocab_size', 50257) if hasattr(model_to_save, 'config') else 50257,
            'd_model': getattr(model_to_save.config, 'd_model', 4096) if hasattr(model_to_save, 'config') else 4096,
            'n_layers': getattr(model_to_save.config, 'n_layers', 32) if hasattr(model_to_save, 'config') else 32,
            'max_seq_length': getattr(model_to_save.config, 'max_seq_length', 4096) if hasattr(model_to_save, 'config') else 4096,
        }
        
        # 添加Mamba特有参数
        if 'mamba' in config_dict['model_type']:
            config_dict.update({
                'd_state': getattr(model_to_save.config, 'd_state', 16) if hasattr(model_to_save, 'config') else 16,
                'd_conv': getattr(model_to_save.config, 'd_conv', 4) if hasattr(model_to_save, 'config') else 4,
                'expand': getattr(model_to_save.config, 'expand', 2) if hasattr(model_to_save, 'config') else 2,
            })
        else:
            config_dict.update({
                'n_heads': getattr(model_to_save.config, 'n_heads', 32) if hasattr(model_to_save, 'config') else 32,
                'd_ff': getattr(model_to_save.config, 'd_ff', 16384) if hasattr(model_to_save, 'config') else 16384,
            })
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'config': config_dict,
            'total_params': sum(p.numel() for p in model_to_save.parameters())
        }, final_model_path)
        
        print(f"✅ 模型已保存至: {os.path.abspath(final_model_path)}")

def main():
    parser = argparse.ArgumentParser(description="显存优化的单机多GPU训练脚本")
    
    # 配置参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model_type", type=str, choices=["transformer", "mamba"], help="模型类型")
    parser.add_argument("--num_gpus", type=int, help="GPU数量")
    
    # 预设配置
    parser.add_argument("--preset", type=str, help="使用预设配置 (如: 1b_transformer, 7b_mamba)")
    parser.add_argument("--list_models", action="store_true", help="列出可用模型")
    parser.add_argument("--list_presets", action="store_true", help="列出可用预设配置")
    parser.add_argument("--list_datasets", action="store_true", help="列出可用数据集")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--max_length", type=int, help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--max_steps", type=int, help="最大训练步数")
    
    # 系统参数
    parser.add_argument("--dry_run", action="store_true", help="只验证配置")
    parser.add_argument("--no_shutdown", action="store_true", help="禁用自动关机")
    parser.add_argument("--check_memory", action="store_true", help="检查显存使用")
    parser.add_argument("--clear_cache", action="store_true", help="清理GPU缓存")
    
    args = parser.parse_args()
    
    # 清理缓存
    if args.clear_cache:
        clear_gpu_memory()
        return
    
    # 检查显存
    if args.check_memory:
        if torch.cuda.is_available():
            print("🔍 GPU显存检查:")
            total_free = 0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                free = total - reserved
                total_free += free
                
                print(f"GPU {i}: {props.name}")
                print(f"  总显存: {total:.2f}GB")
                print(f"  已分配: {allocated:.2f}GB") 
                print(f"  已保留: {reserved:.2f}GB")
                print(f"  可用: {free:.2f}GB")
                
                if free < 10:
                    print(f"  ⚠️  显存较少")
                else:
                    print(f"  ✅ 显存充足")
            
            print(f"\n总可用显存: {total_free:.2f}GB")
        return
    
    # 列出信息
    if args.list_models:
        print("\n🤖 可用模型:")
        for model_type, desc in list_available_configs().items():
            print(f"  {model_type}: {desc}")
        return
    
    if args.list_presets:
        from configs.model_presets import list_model_presets
        list_model_presets()
        return
    
    if args.list_datasets:
        from data.dataset_manager import DatasetManager
        manager = DatasetManager()
        manager.list_datasets()
        return
    
    # 处理预设配置
    if args.preset:
        from configs.model_presets import MODEL_PRESETS, get_model_preset, get_training_config_for_model_size
        
        if args.preset not in MODEL_PRESETS:
            print(f"❌ 未知预设配置: {args.preset}")
            print("可用预设:")
            for preset_id in MODEL_PRESETS.keys():
                print(f"  {preset_id}")
            return
        
        preset_config = get_model_preset(args.preset)
        model_config = preset_config['model']
        
        # 生成训练配置
        training_config = get_training_config_for_model_size(
            args.preset, 
            args.num_gpus or 1
        )
        
        print(f"✅ 使用预设配置: {preset_config['description']}")
        print(f"   参数量: {preset_config['params']}")
        print(f"   显存需求: {preset_config['memory_estimate']}")
        
        # 创建虚拟yaml配置
        yaml_config = {
            'model_type': model_config.model_type,
            'num_gpus': args.num_gpus or 1,
            'system': {'auto_shutdown': False, 'shutdown_delay': 60}
        }
        
    else:
        # 加载配置文件
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
        
        # 训练参数覆盖
        if args.batch_size:
            yaml_config.setdefault('training', {})['batch_size'] = args.batch_size
        if args.max_length:
            yaml_config.setdefault('training', {})['max_length'] = args.max_length
        if args.learning_rate:
            yaml_config.setdefault('training', {})['learning_rate'] = args.learning_rate
        if args.max_steps:
            yaml_config.setdefault('training', {})['max_steps'] = args.max_steps
        
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
    else:
        print(f"🔄 自动关机: 禁用")
    
    if args.dry_run:
        print("\n✅ 配置验证完成（dry_run模式）")
        return
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA，将使用CPU训练")
        device_count = 0
    else:
        device_count = min(yaml_config['num_gpus'], torch.cuda.device_count())
        if yaml_config['num_gpus'] > torch.cuda.device_count():
            print(f"⚠️ 请求{yaml_config['num_gpus']}个GPU，但只有{torch.cuda.device_count()}个可用")
    
    print(f"🚀 启动单机{device_count}GPU训练（使用DataParallel）...")
    
    # 清理GPU缓存
    clear_gpu_memory()
    
    try:
        # 导入模型
        from models import create_model
        
        # 创建模型
        print("📦 创建模型...")
        model = create_model(model_config.model_type, model_config)
        
        # 移动到主GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 使用DataParallel包装多GPU
        if device_count > 1:
            print(f"🔗 使用DataParallel包装{device_count}个GPU...")
            gpu_ids = list(range(device_count))
            model = DataParallel(model, device_ids=gpu_ids)
            
            # 调整批大小
            total_batch_size = training_config.train_batch_size * device_count
            training_config.train_batch_size = training_config.train_batch_size // device_count
            print(f"📏 调整批大小: 每GPU {training_config.train_batch_size}, 总计 {total_batch_size}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # 创建训练器
        trainer = OptimizedTrainer(model, training_config, device_count)
        trainer.train()
        
        print("✅ 训练完成！")
        
        # 自动测试训练的模型
        final_model_path = os.path.join(training_config.checkpoint_dir, "final_model.pt")
        if os.path.exists(final_model_path):
            print("\n🧪 开始快速测试训练的模型...")
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, 'test_after_training.py', 
                    '--checkpoint', final_model_path
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print("✅ 模型快速测试完成")
                    print("💡 如需完整基准测试，请运行: python test_benchmark.py")
                else:
                    print(f"⚠️ 模型测试遇到问题: {result.stderr}")
            except Exception as e:
                print(f"⚠️ 无法运行自动测试: {e}")
                print("💡 可手动运行: python test_after_training.py")
        
        # 自动关机功能
        if auto_shutdown_enabled and not args.no_shutdown:
            shutdown_delay = yaml_config.get('system', {}).get('shutdown_delay', 60)
            auto_shutdown(shutdown_delay)
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ 显存不足: {e}")
        print("💡 建议:")
        print("1. 减小批大小: --batch_size 2")
        print("2. 减小序列长度: --max_length 1024")
        print("3. 清理GPU进程或缓存")
        print("4. 使用更小的模型预设")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")

if __name__ == "__main__":
    main() 