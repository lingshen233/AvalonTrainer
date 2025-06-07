#!/usr/bin/env python3
"""
训练后快速测试脚本
测试刚训练完成的模型
"""

import os
import sys
import argparse
import torch
from configs.base import ModelConfig
from models import create_model

def remove_module_prefix(state_dict):
    """移除DDP模型的module.前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除'module.'前缀
        else:
            new_state_dict[k] = v
    return new_state_dict

def test_model(checkpoint_path):
    """测试模型"""
    print("============================================================")
    print(f"🚀 快速测试模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return False
    
    # 加载检查点
    print("📥 加载模型...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return False
    
    # 获取配置
    if 'config' not in checkpoint:
        print("❌ 检查点中缺少配置信息")
        return False
    
    config_dict = checkpoint['config']
    print(f"📋 模型配置: {config_dict['model_type']}")
    
    # 创建模型配置
    model_config = ModelConfig(**config_dict)
    
    # 创建模型
    print("🔧 创建模型...")
    try:
        model = create_model(model_config.model_type, model_config)
    except Exception as e:
        print(f"❌ 创建模型失败: {e}")
        return False
    
    # 处理DDP模型的state_dict
    state_dict = checkpoint['model_state_dict']
    
    # 检查是否是DDP模型（键名有module.前缀）
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if has_module_prefix:
        print("🔄 检测到DDP模型，移除module.前缀...")
        state_dict = remove_module_prefix(state_dict)
    
    # 加载模型参数
    print("📥 加载模型参数...")
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ 模型参数加载成功")
    except Exception as e:
        print(f"❌ 加载模型参数失败: {e}")
        return False
    
    # 设置评估模式
    model.eval()
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🖥️  使用设备: {device}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 简单的前向传播测试
    print("🧪 执行前向传播测试...")
    try:
        batch_size = 2
        seq_length = 128
        
        # 创建测试输入
        test_input = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        # 处理不同类型的输出
        if isinstance(outputs, dict):
            # 字典输出（常见于Mamba模型）
            if 'logits' in outputs:
                output_shape = outputs['logits'].shape
                print(f"✅ 前向传播成功（字典输出）")
                print(f"   输入形状: {test_input.shape}")
                print(f"   输出logits形状: {output_shape}")
                logits = outputs['logits']
            else:
                # 查找第一个张量输出
                tensor_outputs = {k: v for k, v in outputs.items() if torch.is_tensor(v)}
                if tensor_outputs:
                    key, tensor = next(iter(tensor_outputs.items()))
                    output_shape = tensor.shape
                    print(f"✅ 前向传播成功（字典输出）")
                    print(f"   输入形状: {test_input.shape}")
                    print(f"   输出{key}形状: {output_shape}")
                    logits = tensor
                else:
                    print(f"⚠️  无法识别输出格式: {list(outputs.keys())}")
                    return True  # 至少模型运行了
        elif hasattr(outputs, 'logits'):
            # 命名元组输出（常见于Transformer模型）
            output_shape = outputs.logits.shape
            print(f"✅ 前向传播成功（命名元组输出）")
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出logits形状: {output_shape}")
            logits = outputs.logits
        else:
            # 直接张量输出
            output_shape = outputs.shape
            print(f"✅ 前向传播成功（张量输出）")
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output_shape}")
            logits = outputs
        
        # 简单的文本生成测试
        print("📝 执行文本生成测试...")
        prompt = torch.randint(0, 1000, (1, 10)).to(device)  # 简单的提示
        
        with torch.no_grad():
            for i in range(5):  # 生成5个token
                outputs = model(prompt)
                
                # 提取logits
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        current_logits = outputs['logits']
                    else:
                        # 获取第一个张量
                        tensor_outputs = {k: v for k, v in outputs.items() if torch.is_tensor(v)}
                        current_logits = next(iter(tensor_outputs.values()))
                elif hasattr(outputs, 'logits'):
                    current_logits = outputs.logits
                else:
                    current_logits = outputs
                
                next_token = torch.argmax(current_logits[:, -1, :], dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token], dim=1)
        
        print(f"✅ 文本生成测试成功")
        print(f"   生成序列长度: {prompt.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="训练后模型测试")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", 
                       help="模型检查点路径")
    
    args = parser.parse_args()
    
    success = test_model(args.checkpoint)
    
    success_count = 1 if success else 0
    print(f"\n🎯 测试完成: {success_count}/1 成功")
    
    if success:
        print("✅ 模型运行正常！")
        print("💡 下一步可以:")
        print("   1. 运行完整基准测试: python test_benchmark.py")
        print("   2. 尝试训练更大模型: python train.py --preset 1b_transformer")
        print("   3. 进行实际推理任务")
    else:
        print("❌ 模型测试失败，请检查模型文件")

if __name__ == "__main__":
    main() 