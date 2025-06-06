"""
模型注册系统
支持动态注册和创建不同类型的模型
"""

from .transformer import TransformerModel
from .mamba import MambaModel
from .registry import ModelRegistry, register_model, create_model

# 注册内置模型
register_model("transformer", TransformerModel)
register_model("mamba", MambaModel)

__all__ = [
    "ModelRegistry",
    "register_model", 
    "create_model",
    "TransformerModel",
    "MambaModel"
] 