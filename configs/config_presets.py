"""
配置预设
"""

from .base import ModelConfig

CONFIG_PRESETS = {
    '7b_mamba': ModelConfig(
        # 基础参数
        vocab_size=50257,
        d_model=4096,
        n_layers=32,
        max_seq_length=1024,
        
        # Mamba特定参数
        d_state=16,
        d_conv=4,
        expand=2,
        
        # 训练参数
        learning_rate=5e-4,
        batch_size=32,
        train_micro_batch_size_per_gpu=4,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        warmup_steps=2000,
        max_steps=100000,
        
        # 正则化
        dropout=0.1,
        
        # 优化器
        optimizer_type='adamw',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        
        # 保存和日志
        save_steps=1000,
        eval_steps=500,
        logging_steps=10,
        
        # 其他
        fp16=True,
        gradient_checkpointing=True
    ),
    
    '3b_mamba_lite': ModelConfig(
        # 基础参数（内存友好版本）
        vocab_size=50257,
        d_model=2560,          # 减少到2560
        n_layers=24,           # 减少到24层
        max_seq_length=1024,
        
        # Mamba特定参数（优化版本）
        d_state=8,             # 从16减少到8
        d_conv=4,
        expand=1.5,            # 从2减少到1.5
        
        # 训练参数
        learning_rate=3e-4,
        batch_size=32,
        train_micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=100000,
        
        # 正则化
        dropout=0.1,
        
        # 优化器
        optimizer_type='adamw',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        
        # 保存和日志
        save_steps=1000,
        eval_steps=500,
        logging_steps=10,
        
        # 其他
        fp16=True,
        gradient_checkpointing=True
    ),
} 