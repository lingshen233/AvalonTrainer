"""
简化的预设配置
只提供transformer和mamba两种基本模型配置
"""

from typing import Dict, Tuple
from .base import ModelConfig, TrainingConfig

# ========== 基础模型配置 ==========

def get_transformer_config(size: str = "1b") -> ModelConfig:
    """获取Transformer配置"""
    if size == "1b":
        return ModelConfig(
            model_type="transformer",
            vocab_size=50257,
            max_seq_length=2048,
            d_model=1536,
            n_heads=16,
            n_layers=24,
            d_ff=6144,
            dropout=0.1
        )
    elif size == "7b":
        return ModelConfig(
            model_type="transformer",
            vocab_size=50257,
            max_seq_length=4096,
            d_model=4096,
            n_heads=32,
            n_layers=32,
            d_ff=16384,
            dropout=0.1
        )
    else:
        raise ValueError(f"不支持的大小: {size}")

def get_mamba_config(size: str = "1b") -> ModelConfig:
    """获取Mamba配置"""
    if size == "1b":
        return ModelConfig(
            model_type="mamba",
            vocab_size=50257,
            max_seq_length=2048,
            d_model=1536,
            n_layers=32,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1
        )
    elif size == "7b":
        return ModelConfig(
            model_type="mamba",
            vocab_size=50257,
            max_seq_length=4096,
            d_model=2560,
            n_layers=64,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1
        )
    else:
        raise ValueError(f"不支持的大小: {size}")

# ========== 训练配置 ==========

def get_training_config(num_gpus: int = 1) -> TrainingConfig:
    """获取训练配置"""
    # 根据GPU数量调整批大小
    if num_gpus == 1:
        batch_size = 6
        grad_accum = 8
    elif num_gpus <= 4:
        batch_size = 8
        grad_accum = 4
    else:
        batch_size = 12
        grad_accum = 2
    
    return TrainingConfig(
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=2048,
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=50000,
        warmup_steps=2000,
        fp16=True,
        distributed=(num_gpus > 1),
        world_size=num_gpus,
        wandb_project="multi-gpu-training"
    )

# ========== 预设组合（简化版）==========

def get_config(model_type: str, size: str = "1b", num_gpus: int = 1) -> Tuple[ModelConfig, TrainingConfig]:
    """获取完整配置组合"""
    
    if model_type == "transformer":
        model_config = get_transformer_config(size)
    elif model_type == "mamba":
        model_config = get_mamba_config(size)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    training_config = get_training_config(num_gpus)
    
    return model_config, training_config

# ========== 便捷函数 ==========

def list_available_configs() -> Dict[str, str]:
    """列出可用的配置"""
    return {
        "transformer": "标准Transformer模型，通用性好",
        "mamba": "高效Mamba模型，显存占用更少"
    }

def calculate_model_size(config: ModelConfig) -> int:
    """计算模型参数量"""
    if config.model_type == "transformer":
        # Transformer参数估算
        embed_params = config.vocab_size * config.d_model
        attention_params = config.n_layers * 4 * config.d_model * config.d_model
        ffn_params = config.n_layers * 2 * config.d_model * config.d_ff
        output_params = config.vocab_size * config.d_model
        total_params = embed_params + attention_params + ffn_params + output_params
    elif config.model_type == "mamba":
        # Mamba参数估算
        embed_params = config.vocab_size * config.d_model
        d_inner = config.expand * config.d_model
        
        per_layer_params = (
            config.d_model * d_inner * 2 +    # in_proj
            d_inner * config.d_state * 2 +    # x_proj
            d_inner * d_inner +               # dt_proj
            d_inner * config.d_state +        # A matrix
            d_inner +                         # D vector
            d_inner * config.d_model +        # out_proj
            d_inner * config.d_conv +         # conv1d
            config.d_model                    # norm
        )
        
        mamba_params = config.n_layers * per_layer_params
        output_params = config.vocab_size * config.d_model
        total_params = embed_params + mamba_params + output_params
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")
    
    return total_params

def estimate_memory_usage(model_config: ModelConfig, training_config: TrainingConfig) -> Dict[str, float]:
    """估算显存使用（GB）"""
    total_params = calculate_model_size(model_config)
    
    # 参数内存（FP16）
    param_memory = total_params * 2 / 1024**3
    
    # 训练状态（梯度 + 优化器状态）
    training_memory = total_params * 8 / 1024**3
    
    # 激活内存
    batch_size = training_config.train_batch_size
    seq_len = training_config.max_length
    d_model = model_config.d_model
    
    if model_config.model_type == "transformer":
        # 注意力矩阵
        attention_memory = batch_size * model_config.n_heads * seq_len * seq_len * 2 / 1024**3
        activation_memory = (
            batch_size * seq_len * d_model * model_config.n_layers * 3 / 1024**3 +
            attention_memory
        )
    else:  # mamba
        activation_memory = batch_size * seq_len * d_model * model_config.n_layers * 2 / 1024**3
    
    total_memory = param_memory + training_memory + activation_memory
    
    return {
        'total_params': total_params,
        'param_memory_gb': param_memory,
        'training_memory_gb': training_memory,
        'activation_memory_gb': activation_memory,
        'total_memory_gb': total_memory
    } 