#!/usr/bin/env python3
"""
训练后快速测试脚本
专门用于训练完成后快速验证模型性能
"""

import os
import sys
import argparse
import torch
import time
from pathlib import Path

def quick_model_test(checkpoint_path, save_results=True):
    """快速测试训练的模型"""
    print(f"🚀 快速测试模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return False
    
    try:
        # 导入必要模块
        from models import create_model
        from configs.base import ModelConfig
        from transformers import AutoTokenizer
        import json
        
        # 加载模型
        print("📥 加载模型...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_config = ModelConfig(**checkpoint['config'])
        
        model = create_model(model_config.model_type, model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 获取设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型加载成功")
        print(f"   模型类型: {model_config.model_type}")
        print(f"   参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"   设备: {device}")
        
        # 快速文本生成测试
        print("\n🎯 快速生成测试...")
        test_prompts = [
            "人工智能",
            "The future of",
            "科技发展",
            "Machine learning"
        ]
        
        results = {
            'model_info': {
                'type': model_config.model_type,
                'params': total_params,
                'checkpoint': checkpoint_path
            },
            'generation_tests': []
        }
        
        for prompt in test_prompts:
            try:
                print(f"  测试提示: '{prompt}'")
                
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=inputs['input_ids'].size(1) + 30,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(prompt):].strip()
                
                print(f"  生成结果: {generated_text[:50]}...")
                
                results['generation_tests'].append({
                    'prompt': prompt,
                    'generated': generated_text
                })
                
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
                results['generation_tests'].append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        # 保存结果
        if save_results:
            results_dir = Path('test_results')
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f'quick_test_{int(time.time())}.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📝 测试结果已保存: {results_file}")
        
        print("\n✅ 快速测试完成！")
        print("💡 如需完整基准测试，请运行: python test_benchmark.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="训练后快速测试")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", 
                       help="模型检查点路径")
    parser.add_argument("--no-save", action="store_true", help="不保存测试结果")
    
    args = parser.parse_args()
    
    # 检查默认路径
    checkpoints_to_test = []
    
    if args.checkpoint != "checkpoints/final_model.pt":
        # 用户指定了特定路径
        checkpoints_to_test.append(args.checkpoint)
    else:
        # 自动查找检查点
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            # 优先级：final_model.pt > best_model.pt > 最新的checkpoint
            if (checkpoint_dir / "final_model.pt").exists():
                checkpoints_to_test.append("checkpoints/final_model.pt")
            elif (checkpoint_dir / "best_model.pt").exists():
                checkpoints_to_test.append("checkpoints/best_model.pt")
            else:
                # 查找最新的checkpoint
                checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    checkpoints_to_test.append(str(latest_checkpoint))
    
    if not checkpoints_to_test:
        print("❌ 未找到任何模型检查点文件")
        print("请确保训练已完成并生成了模型文件")
        return 1
    
    success_count = 0
    for checkpoint in checkpoints_to_test:
        print(f"\n{'='*60}")
        if quick_model_test(checkpoint, not args.no_save):
            success_count += 1
    
    print(f"\n🎯 测试完成: {success_count}/{len(checkpoints_to_test)} 成功")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    exit(main()) 