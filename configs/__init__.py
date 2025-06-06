"""
简化的配置系统
"""

from .base import ModelConfig, TrainingConfig
from .presets import (
    get_transformer_config,
    get_mamba_config,
    get_training_config,
    get_config,
    list_available_configs,
    calculate_model_size,
    estimate_memory_usage
)
from .registry import ConfigRegistry

__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "get_transformer_config",
    "get_mamba_config",
    "get_training_config",
    "get_config",
    "list_available_configs",
    "calculate_model_size",
    "estimate_memory_usage",
    "ConfigRegistry"
] 