"""
配置注册表
"""

from typing import Dict, Any
from .base import ModelConfig, TrainingConfig

class ConfigRegistry:
    """配置注册表"""
    
    _configs: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, config: Any):
        """注册配置"""
        cls._configs[name] = config
    
    @classmethod
    def get(cls, name: str) -> Any:
        """获取配置"""
        return cls._configs.get(name)
    
    @classmethod
    def list_configs(cls) -> list:
        """列出所有配置"""
        return list(cls._configs.keys()) 