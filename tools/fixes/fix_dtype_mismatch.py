#!/usr/bin/env python3
"""
修复FP16混合精度训练中的数据类型不匹配问题
"""

import torch
import torch.nn as nn
import os
import argparse

def fix_model_dtype_consistency(model):
    """修复模型中的数据类型一致性问题"""
    print("🔧 修复模型数据类型一致性...")
    
    def ensure_dtype_consistency(module):
        """确保模块内的数据类型一致性"""
        if hasattr(module, 'weight') and module.weight is not None:
            target_dtype = module.weight.dtype
            
            # 修复bias的数据类型
            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.dtype != target_dtype:
                    module.bias.data = module.bias.data.to(target_dtype)
            
            # 修复其他参数的数据类型
            for name, param in module.named_parameters(recurse=False):
                if param is not None and param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)
    
    # 对所有模块应用数据类型修复
    for module in model.modules():
        ensure_dtype_consistency(module)
    
    print("✅ 模型数据类型一致性修复完成")

class DTypeFriendlyLinear(nn.Linear):
    """支持数据类型自动转换的Linear层"""
    
    def forward(self, input):
        # 确保input和weight的数据类型一致
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return super().forward(input)

class DTypeFriendlyEmbedding(nn.Embedding):
    """支持数据类型自动转换的Embedding层"""
    
    def forward(self, input):
        # Embedding的输入通常是long类型，输出需要匹配权重类型
        result = super().forward(input)
        return result.to(self.weight.dtype)

def replace_modules_with_dtype_friendly(model):
    """替换模型中的模块为数据类型友好版本"""
    print("🔄 替换模块为数据类型友好版本...")
    
    def replace_linear_modules(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Linear):
                # 替换为数据类型友好的Linear
                new_linear = DTypeFriendlyLinear(
                    child.in_features, 
                    child.out_features, 
                    bias=child.bias is not None
                )
                new_linear.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_linear.bias.data = child.bias.data.clone()
                
                setattr(module, name, new_linear)
                print(f"  替换: {full_name} -> DTypeFriendlyLinear")
                
            elif isinstance(child, nn.Embedding):
                # 替换为数据类型友好的Embedding
                new_embedding = DTypeFriendlyEmbedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    padding_idx=child.padding_idx,
                    max_norm=child.max_norm,
                    norm_type=child.norm_type,
                    scale_grad_by_freq=child.scale_grad_by_freq,
                    sparse=child.sparse
                )
                new_embedding.weight.data = child.weight.data.clone()
                
                setattr(module, name, new_embedding)
                print(f"  替换: {full_name} -> DTypeFriendlyEmbedding")
            else:
                # 递归处理子模块
                replace_linear_modules(child, full_name)
    
    replace_linear_modules(model)
    print("✅ 模块替换完成")

def add_dtype_hooks(model):
    """添加数据类型检查hook"""
    print("🔗 添加数据类型检查hook...")
    
    def dtype_check_hook(module, input, output):
        """检查输入输出的数据类型一致性"""
        if hasattr(module, 'weight') and module.weight is not None:
            target_dtype = module.weight.dtype
            
            # 检查输入
            if isinstance(input, (tuple, list)):
                for i, inp in enumerate(input):
                    if torch.is_tensor(inp) and inp.dtype != target_dtype and inp.dtype.is_floating_point:
                        print(f"⚠️ {module.__class__.__name__}: 输入{i} dtype不匹配 {inp.dtype} != {target_dtype}")
            
            # 检查输出
            if torch.is_tensor(output) and output.dtype != target_dtype:
                print(f"⚠️ {module.__class__.__name__}: 输出 dtype不匹配 {output.dtype} != {target_dtype}")
    
    # 为特定模块类型添加hook
    hook_modules = [nn.Linear, nn.Embedding, nn.Conv1d]
    
    for module in model.modules():
        if any(isinstance(module, cls) for cls in hook_modules):
            module.register_forward_hook(dtype_check_hook)
    
    print("✅ 数据类型检查hook添加完成")

def create_fp16_safe_config():
    """创建FP16安全的DeepSpeed配置"""
    print("📝 创建FP16安全配置...")
    
    config = {
        "train_batch_size": 48,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 8,
        
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        
        # 更安全的FP16配置
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 12,  # 降低初始loss scale
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        # 更保守的优化器设置
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,  # 降低学习率
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 2000  # 增加warmup步数
            }
        },
        
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        
        # 更激进的激活检查点
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 16,
            "synchronize_checkpoint_boundary": True,
            "profile": False
        },
        
        "comms_logger": {"enabled": False},
        "memory_breakdown": False,
        "flops_profiler": {"enabled": False}
    }
    
    # 保存配置
    import json
    config_file = "deepspeed_6gpu_fp16_safe.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ FP16安全配置已保存: {config_file}")
    return config_file

def main():
    parser = argparse.ArgumentParser(description="修复FP16数据类型不匹配问题")
    parser.add_argument("--create_safe_config", action="store_true", help="创建FP16安全配置")
    parser.add_argument("--test_model", action="store_true", help="测试模型数据类型")
    
    args = parser.parse_args()
    
    print("🔧 FP16数据类型不匹配修复工具")
    print("=" * 50)
    
    if args.create_safe_config:
        create_fp16_safe_config()
    
    if args.test_model:
        print("\n🧪 测试模型数据类型一致性...")
        # 这里可以添加模型测试代码
        
    print("\n💡 修复建议:")
    print("1. 使用更安全的FP16配置 (降低initial_scale_power)")
    print("2. 在模型中添加显式的数据类型转换")
    print("3. 使用数据类型友好的模块替换")
    print("4. 添加数据类型检查hook进行调试")
    
    print("\n🚀 使用方法:")
    print("# 创建安全配置")
    print("python fix_dtype_mismatch.py --create_safe_config")
    print("")
    print("# 使用安全配置训练")
    print("deepspeed --num_gpus=6 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_6gpu_fp16_safe.json")

if __name__ == "__main__":
    main() 