"""
模型注册系统
支持动态注册和创建不同类型的模型
"""

from typing import Dict, Type, Any
import torch.nn as nn
from configs.base import ModelConfig

class ModelRegistry:
    """模型注册表"""
    
    _models: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]):
        """注册模型"""
        cls._models[name] = model_class
        print(f"已注册模型: {name}")
    
    @classmethod
    def create(cls, name: str, config: ModelConfig) -> nn.Module:
        """创建模型实例"""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"未知模型类型: {name}. 可用模型: {available}")
        
        model_class = cls._models[name]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有注册的模型"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[nn.Module]:
        """获取模型类"""
        if name not in cls._models:
            raise ValueError(f"未知模型类型: {name}")
        return cls._models[name]

# 全局实例
registry = ModelRegistry()

def register_model(name: str, model_class: Type[nn.Module]):
    """注册模型的装饰器函数"""
    registry.register(name, model_class)

def create_model(name: str, config: ModelConfig) -> nn.Module:
    """创建模型的便捷函数"""
    return registry.create(name, config)

def list_models() -> list:
    """列出所有可用模型"""
    return registry.list_models()

def model_decorator(name: str):
    """模型注册装饰器"""
    def decorator(model_class: Type[nn.Module]):
        register_model(name, model_class)
        return model_class
    return decorator 