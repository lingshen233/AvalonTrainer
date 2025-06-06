"""
基础配置类
"""

import dataclasses
from typing import Optional, Dict, Any, List
import json
import os

@dataclasses.dataclass
class ModelConfig:
    """模型配置基类"""
    
    # 通用参数
    model_type: str = "transformer"
    vocab_size: int = 50257
    max_seq_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Transformer专用参数
    n_heads: int = 12
    d_ff: int = 3072
    
    # Mamba专用参数  
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    
    # 额外参数（用于扩展）
    extra_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

@dataclasses.dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基础训练参数
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 批次和步数
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    
    # 评估和保存
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    
    # 数据相关
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    max_length: int = 1024
    dataloader_num_workers: int = 4
    
    # 技术选项
    fp16: bool = True
    bf16: bool = False
    deepspeed_config: Optional[str] = None
    
    # 分布式训练
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # 路径
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    
    # 监控
    use_wandb: bool = True
    wandb_project: str = "multi-model-training"
    wandb_run_name: Optional[str] = None
    
    # 其他
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # 额外参数
    extra_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @property
    def effective_batch_size(self) -> int:
        """有效批大小"""
        return self.train_batch_size * self.gradient_accumulation_steps * self.world_size 