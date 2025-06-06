#!/usr/bin/env python3
"""
模型规模预设配置
支持1B、7B等不同规模的模型配置
"""

from .base import ModelConfig, TrainingConfig

# 模型规模预设
MODEL_PRESETS = {
    # 1B模型配置
    '1b_transformer': {
        'model': ModelConfig(
            model_type='transformer',
            vocab_size=50257,  # GPT-2 vocab
            max_seq_length=2048,
            d_model=2048,      # 增大隐层维度
            n_layers=24,       # 24层
            n_heads=16,        # 16个注意力头
            d_ff=8192,         # 前馈网络维度
            dropout=0.1
        ),
        'description': 'Transformer 1B - 通用语言模型',
        'params': '1.0B',
        'memory_estimate': '12GB/GPU',
        'recommended_gpus': [1, 2, 4],
        'datasets': ['wikitext', 'bookcorpus', 'cc_news']
    },
    
    '1b_mamba': {
        'model': ModelConfig(
            model_type='mamba',
            vocab_size=50257,
            max_seq_length=2048,
            d_model=2048,
            n_layers=24,
            d_state=16,        # Mamba状态维度
            d_conv=4,          # 卷积核大小
            expand=2,          # 扩展因子
            dropout=0.1
        ),
        'description': 'Mamba 1B - 高效状态空间模型',
        'params': '1.0B',
        'memory_estimate': '9GB/GPU',
        'recommended_gpus': [1, 2, 4],
        'datasets': ['wikitext', 'bookcorpus', 'openwebtext']
    },
    
    # 7B模型配置
    '7b_transformer': {
        'model': ModelConfig(
            model_type='transformer',
            vocab_size=50257,
            max_seq_length=4096,   # 更长序列
            d_model=4096,          # 更大隐层
            n_layers=32,           # 32层
            n_heads=32,            # 32个注意力头
            d_ff=16384,            # 更大前馈网络
            dropout=0.1
        ),
        'description': 'Transformer 7B - 大规模语言模型',
        'params': '7.0B',
        'memory_estimate': '28GB/GPU',
        'recommended_gpus': [4, 8],
        'datasets': ['openwebtext', 'c4', 'the_pile']
    },
    
    '7b_mamba': {
        'model': ModelConfig(
            model_type='mamba',
            vocab_size=50257,
            max_seq_length=4096,     # 7B应有的序列长度
            d_model=4864,            # 微调到7B
            n_layers=45,             # 微调层数
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1
        ),
        'description': 'Mamba 7B - 真正的7B状态空间模型',
        'params': '7.0B',
        'memory_estimate': '35GB/GPU',
        'recommended_gpus': [8],
        'datasets': ['openwebtext', 'c4', 'the_pile'],
        'notes': [
            '⚠️ 真正的7B模型，需要40GB显存/GPU或8张24GB GPU',
            '建议使用8张A100-24GB或4张A100-40GB',
            '如显存不足，请使用3b_mamba预设'
        ]
    },
    
    # 添加诚实的3B配置
    '3b_mamba': {
        'model': ModelConfig(
            model_type='mamba',
            vocab_size=50257,
            max_seq_length=2048,
            d_model=3584,
            n_layers=32,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1
        ),
        'description': 'Mamba 3B - 适合24GB GPU的大模型',
        'params': '2.8B',
        'memory_estimate': '15GB/GPU',
        'recommended_gpus': [2, 4],
        'datasets': ['openwebtext', 'c4', 'bookcorpus']
    },
    
    # 测试配置
    'test_small': {
        'model': ModelConfig(
            model_type='transformer',
            vocab_size=50257,
            max_seq_length=512,
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            dropout=0.1
        ),
        'description': '小型测试模型 - 快速验证',
        'params': '50M',
        'memory_estimate': '2GB/GPU',
        'recommended_gpus': [1],
        'datasets': ['squad', 'wikitext']
    }
}

def get_training_config_for_model_size(model_size: str, num_gpus: int = 1):
    """根据模型规模生成训练配置"""
    
    # 基础训练参数
    base_configs = {
        '1B': {
            'train_batch_size': 4 if num_gpus == 1 else 6,
            'gradient_accumulation_steps': 4,
            'learning_rate': 3e-4,
            'max_steps': 100000,
            'warmup_steps': 2000,
            'eval_steps': 2000,
            'save_steps': 5000
        },
        '7B': {
            'train_batch_size': 2 if num_gpus <= 2 else 4,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1e-4,
            'max_steps': 300000,
            'warmup_steps': 5000,
            'eval_steps': 5000,
            'save_steps': 10000
        },
        'test': {
            'train_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'learning_rate': 5e-4,
            'max_steps': 1000,
            'warmup_steps': 100,
            'eval_steps': 200,
            'save_steps': 500
        }
    }
    
    # 确定模型规模类别
    if '1b' in model_size.lower():
        config_type = '1B'
    elif '7b' in model_size.lower():
        config_type = '7B'
    else:
        config_type = 'test'
    
    base_config = base_configs[config_type]
    
    return TrainingConfig(
        dataset_name='auto',  # 自动选择
        train_batch_size=base_config['train_batch_size'],
        eval_batch_size=base_config['train_batch_size'],
        gradient_accumulation_steps=base_config['gradient_accumulation_steps'],
        max_length=2048 if config_type != 'test' else 512,
        learning_rate=base_config['learning_rate'],
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_steps=base_config['max_steps'],
        warmup_steps=base_config['warmup_steps'],
        eval_steps=base_config['eval_steps'],
        save_steps=base_config['save_steps'],
        logging_steps=100,
        fp16=True,
        output_dir="./outputs",
        checkpoint_dir="./checkpoints",
        use_wandb=False,
        wandb_project="rag-transformer",
        wandb_run_name=f"{model_size}_{config_type}",
        distributed=(num_gpus > 1),
        world_size=num_gpus
    )

def list_model_presets():
    """列出所有可用的模型预设"""
    print("\n🤖 可用模型预设:")
    print("=" * 80)
    
    for preset_id, config in MODEL_PRESETS.items():
        print(f"\n📋 {preset_id}")
        print(f"   描述: {config['description']}")
        print(f"   参数量: {config['params']}")
        print(f"   显存需求: {config['memory_estimate']}")
        print(f"   推荐GPU: {config['recommended_gpus']}")
        print(f"   推荐数据集: {', '.join(config['datasets'])}")

def get_model_preset(preset_id: str):
    """获取指定的模型预设"""
    if preset_id not in MODEL_PRESETS:
        available = list(MODEL_PRESETS.keys())
        raise ValueError(f"未知预设 '{preset_id}'，可用预设: {available}")
    
    return MODEL_PRESETS[preset_id]

def calculate_model_parameters(config: ModelConfig):
    """准确估算模型参数量"""
    if config.model_type == 'transformer':
        # Transformer参数估算
        embedding_params = config.vocab_size * config.d_model
        
        # 每一层的参数
        attention_params = 4 * config.d_model * config.d_model  # Q,K,V,O
        ffn_params = 2 * config.d_model * config.d_ff
        layer_params = attention_params + ffn_params
        
        total_params = embedding_params + config.n_layers * layer_params
        
    elif config.model_type == 'mamba':
        # Mamba参数精确估算
        embedding_params = config.vocab_size * config.d_model
        
        # 每层Mamba块参数
        d_inner = config.d_model * config.expand  # 内部维度
        
        # 输入投影层 (x_proj, z_proj)
        in_proj_params = config.d_model * (d_inner * 2)
        
        # 卷积层
        conv_params = d_inner * config.d_conv
        
        # 状态空间参数 (A, B, C, dt)
        ss_params = d_inner * config.d_state  # A matrix
        ss_params += d_inner * config.d_state  # B projection
        ss_params += d_inner  # C projection
        ss_params += d_inner  # dt projection
        
        # 输出投影
        out_proj_params = d_inner * config.d_model
        
        # 层归一化参数
        norm_params = config.d_model
        
        # 每层总参数
        layer_params = in_proj_params + conv_params + ss_params + out_proj_params + norm_params
        
        # 最终层归一化和语言模型头
        final_norm_params = config.d_model
        lm_head_params = config.vocab_size * config.d_model
        
        total_params = embedding_params + (config.n_layers * layer_params) + final_norm_params + lm_head_params
    
    else:
        total_params = 0
    
    return total_params

if __name__ == "__main__":
    # 演示用法
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.base import ModelConfig, TrainingConfig
    
    print("🚀 模型预设配置演示")
    
    # 列出所有预设
    list_model_presets()
    
    # 获取1B Transformer配置
    print("\n" + "="*50)
    preset = get_model_preset('1b_transformer')
    print(f"1B Transformer配置: {preset['description']}")
    
    # 计算参数量
    params = calculate_model_parameters(preset['model'])
    print(f"估算参数量: {params:,} ({params/1e9:.1f}B)") 