"""
GPU相关工具函数
"""

import torch
from typing import Dict
from configs.base import ModelConfig, TrainingConfig
from configs.presets import calculate_model_size, estimate_memory_usage

def check_gpu_info() -> Dict:
    """检查GPU信息"""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'memory_gb': 0,
            'devices': []
        }
    
    gpu_count = torch.cuda.device_count()
    devices = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        devices.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb,
            'compute_capability': f"{props.major}.{props.minor}"
        })
    
    # 使用第一个GPU的信息作为代表
    primary_gpu = devices[0] if devices else None
    
    info = {
        'available': True,
        'count': gpu_count,
        'memory_gb': primary_gpu['memory_gb'] if primary_gpu else 0,
        'devices': devices
    }
    
    # 打印GPU信息
    print(f"🖥️  GPU信息:")
    print(f"  数量: {gpu_count}")
    for device in devices:
        print(f"  GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f}GB)")
        
        # 特殊标注
        if "3090" in device['name']:
            print(f"    🎯 RTX 3090检测到！推荐用于1B模型训练")
        elif "4090" in device['name']:
            print(f"    🚀 RTX 4090检测到！可训练更大模型")
        elif "V100" in device['name'] or "A100" in device['name']:
            print(f"    ⚡ 数据中心GPU检测到！适合大规模训练")
    
    return info

def get_optimal_batch_size(model_config: ModelConfig, training_config: TrainingConfig, gpu_memory_gb: float) -> int:
    """根据GPU显存自动计算最优批大小"""
    
    # 预设的批大小候选
    batch_size_candidates = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
    
    for batch_size in sorted(batch_size_candidates, reverse=True):
        # 临时设置批大小
        temp_config = TrainingConfig(**training_config.to_dict())
        temp_config.train_batch_size = batch_size
        
        # 估算显存使用
        memory_info = estimate_memory_usage(model_config, temp_config)
        
        # 留出10%的安全余量
        if memory_info['total_memory_gb'] <= gpu_memory_gb * 0.9:
            return batch_size
    
    # 如果所有候选都不行，返回最小值
    return 1

def estimate_training_time(model_config: ModelConfig, training_config: TrainingConfig, gpu_info: Dict) -> Dict:
    """估算训练时间"""
    
    # 根据模型类型和GPU类型估算速度
    if gpu_info['available']:
        device_name = gpu_info['devices'][0]['name']
        
        # tokens/秒的估算（基于经验）
        if "3090" in device_name:
            if model_config.model_type == "mamba":
                tokens_per_sec = 3000  # Mamba更高效
            else:
                tokens_per_sec = 2000  # Transformer
        elif "4090" in device_name:
            if model_config.model_type == "mamba":
                tokens_per_sec = 4500
            else:
                tokens_per_sec = 3000
        elif "V100" in device_name:
            if model_config.model_type == "mamba":
                tokens_per_sec = 2500
            else:
                tokens_per_sec = 1800
        elif "A100" in device_name:
            if model_config.model_type == "mamba":
                tokens_per_sec = 6000
            else:
                tokens_per_sec = 4500
        else:
            # 其他GPU
            if model_config.model_type == "mamba":
                tokens_per_sec = 1500
            else:
                tokens_per_sec = 1000
    else:
        # CPU训练（很慢）
        tokens_per_sec = 50
    
    # 计算总tokens
    total_tokens = (
        training_config.max_steps * 
        training_config.effective_batch_size * 
        training_config.max_length
    )
    
    # 估算时间
    total_seconds = total_tokens / tokens_per_sec
    hours = total_seconds / 3600
    days = hours / 24
    
    return {
        'total_tokens': total_tokens,
        'tokens_per_sec': tokens_per_sec,
        'total_seconds': total_seconds,
        'hours': hours,
        'days': days
    }

def get_gpu_recommendations(model_config: ModelConfig) -> Dict:
    """根据模型获取GPU推荐"""
    
    total_params = calculate_model_size(model_config)
    
    if total_params < 500e6:  # <500M
        tier = "小型"
        gpus = ["GTX 1080", "RTX 2080", "RTX 3060"]
    elif total_params < 1.5e9:  # <1.5B
        tier = "中型"
        gpus = ["RTX 3080", "RTX 3090", "RTX 4080"]
    elif total_params < 7e9:  # <7B
        tier = "大型"
        gpus = ["RTX 4090", "A100-40GB", "V100-32GB"]
    else:  # >=7B
        tier = "超大型"
        gpus = ["A100-80GB", "H100", "多卡训练"]
    
    return {
        'tier': tier,
        'recommended_gpus': gpus,
        'min_memory_gb': 8 if total_params < 500e6 else 16 if total_params < 1.5e9 else 24,
        'recommended_memory_gb': 12 if total_params < 500e6 else 24 if total_params < 1.5e9 else 40
    } 