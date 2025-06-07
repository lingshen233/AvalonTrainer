#!/usr/bin/env python3
"""
DeepSpeed配置调试脚本
用于诊断批次大小配置问题
"""

import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_preset_config():
    """调试预设配置"""
    print("🔍 调试预设配置...")
    
    from configs.model_presets import get_model_preset, get_training_config_for_model_size
    
    preset_id = "7b_mamba"
    num_gpus = 4
    
    print(f"\n📋 预设ID: {preset_id}")
    print(f"GPU数量: {num_gpus}")
    
    # 获取预设配置
    preset_config = get_model_preset(preset_id)
    model_config = preset_config['model']
    training_config = get_training_config_for_model_size(preset_id, num_gpus)
    
    print(f"\n🤖 模型配置:")
    print(f"  类型: {model_config.model_type}")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    
    print(f"\n🎯 训练配置:")
    print(f"  train_batch_size: {training_config.train_batch_size}")
    print(f"  gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"  world_size: {training_config.world_size}")
    
    # 计算预期的train_batch_size
    expected = training_config.train_batch_size * training_config.gradient_accumulation_steps * num_gpus
    print(f"\n🧮 批次大小计算:")
    print(f"  micro_batch_per_gpu: {training_config.train_batch_size}")
    print(f"  gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"  world_size: {num_gpus}")
    print(f"  预期train_batch_size: {expected}")
    
    return model_config, training_config

def debug_yaml_config():
    """调试YAML配置创建过程"""
    print("\n🔍 调试YAML配置创建...")
    
    from train_deepspeed import create_configs_from_yaml
    
    # 模拟预设配置创建的yaml_config
    from configs.model_presets import get_model_preset, get_training_config_for_model_size
    
    preset_config = get_model_preset("7b_mamba")
    model_config_preset = preset_config['model']
    training_config_preset = get_training_config_for_model_size("7b_mamba", 4)
    
    yaml_config = {
        'model_type': model_config_preset.model_type,
        'num_gpus': 4,
        'model': {
            'vocab_size': model_config_preset.vocab_size,
            'max_seq_length': model_config_preset.max_seq_length,
            'd_model': model_config_preset.d_model,
            'n_layers': model_config_preset.n_layers,
            'd_state': getattr(model_config_preset, 'd_state', 16),
            'd_conv': getattr(model_config_preset, 'd_conv', 4),
            'expand': getattr(model_config_preset, 'expand', 2),
            'dropout': model_config_preset.dropout
        },
        'training': {
            'batch_size': training_config_preset.train_batch_size,
            'gradient_accumulation_steps': training_config_preset.gradient_accumulation_steps,
            'max_length': training_config_preset.max_length,
            'learning_rate': training_config_preset.learning_rate,
            'max_steps': training_config_preset.max_steps,
            'warmup_steps': training_config_preset.warmup_steps,
            'save_steps': training_config_preset.save_steps,
            'logging_steps': training_config_preset.logging_steps,
            'fp16': training_config_preset.fp16,
            'output_dir': training_config_preset.output_dir,
            'checkpoint_dir': training_config_preset.checkpoint_dir
        }
    }
    
    print(f"\n📋 虚拟YAML配置:")
    print(f"  model_type: {yaml_config['model_type']}")
    print(f"  num_gpus: {yaml_config['num_gpus']}")
    print(f"  training.batch_size: {yaml_config['training']['batch_size']}")
    print(f"  training.gradient_accumulation_steps: {yaml_config['training']['gradient_accumulation_steps']}")
    
    # 通过create_configs_from_yaml创建配置
    model_config, training_config = create_configs_from_yaml(yaml_config)
    
    print(f"\n🔄 经过create_configs_from_yaml后:")
    print(f"  train_batch_size: {training_config.train_batch_size}")
    print(f"  gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"  world_size: {training_config.world_size}")
    
    return model_config, training_config, yaml_config

def debug_deepspeed_config():
    """调试DeepSpeed配置创建"""
    print("\n🔍 调试DeepSpeed配置创建...")
    
    from train_deepspeed import create_deepspeed_config
    
    # 使用前面调试得到的配置
    model_config, training_config, yaml_config = debug_yaml_config()
    num_gpus = 4
    
    # 创建DeepSpeed配置
    ds_config = create_deepspeed_config(model_config, training_config, num_gpus)
    
    print(f"\n⚙️ DeepSpeed配置结果:")
    print(f"  train_batch_size: {ds_config['train_batch_size']}")
    print(f"  train_micro_batch_size_per_gpu: {ds_config['train_micro_batch_size_per_gpu']}")
    print(f"  gradient_accumulation_steps: {ds_config['gradient_accumulation_steps']}")
    
    # 验证公式
    expected = ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps'] * num_gpus
    actual = ds_config['train_batch_size']
    
    print(f"\n✅ 公式验证:")
    print(f"  expected: {ds_config['train_micro_batch_size_per_gpu']} × {ds_config['gradient_accumulation_steps']} × {num_gpus} = {expected}")
    print(f"  actual: {actual}")
    print(f"  匹配: {'✅' if expected == actual else '❌'}")
    
    return ds_config

def debug_direct_yaml_load():
    """直接加载YAML文件调试"""
    print("\n🔍 调试直接YAML文件加载...")
    
    from train_deepspeed import load_config, create_configs_from_yaml
    
    config_file = "config_7b_mamba.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    # 加载原始YAML
    yaml_config = load_config(config_file)
    yaml_config['num_gpus'] = 4  # 模拟命令行参数
    
    print(f"\n📋 原始YAML配置:")
    print(f"  model_type: {yaml_config.get('model_type')}")
    print(f"  num_gpus: {yaml_config.get('num_gpus')}")
    
    training_section = yaml_config.get('training', {})
    print(f"  training.batch_size: {training_section.get('batch_size')}")
    print(f"  training.gradient_accumulation_steps: {training_section.get('gradient_accumulation_steps')}")
    
    # 通过create_configs_from_yaml处理
    model_config, training_config = create_configs_from_yaml(yaml_config)
    
    print(f"\n🔄 处理后的训练配置:")
    print(f"  train_batch_size: {training_config.train_batch_size}")
    print(f"  gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"  world_size: {training_config.world_size}")
    
    # 计算预期的train_batch_size
    expected = training_config.train_batch_size * training_config.gradient_accumulation_steps * 4
    print(f"\n🧮 批次大小计算:")
    print(f"  {training_config.train_batch_size} × {training_config.gradient_accumulation_steps} × 4 = {expected}")
    
    return model_config, training_config

def main():
    print("🐛 DeepSpeed配置调试工具")
    print("=" * 60)
    
    try:
        # 1. 调试预设配置
        debug_preset_config()
        
        # 2. 调试YAML配置创建
        debug_yaml_config()
        
        # 3. 调试DeepSpeed配置
        debug_deepspeed_config()
        
        # 4. 调试直接YAML加载
        debug_direct_yaml_load()
        
        print("\n🎯 问题诊断完成！")
        
    except Exception as e:
        print(f"\n❌ 调试过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 