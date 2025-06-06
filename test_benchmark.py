#!/usr/bin/env python3
"""
综合基准测试脚本
下载标准数据集和预训练模型进行全面评估
"""

import os
import sys
import argparse
import torch
import time
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def remove_module_prefix(state_dict):
    """移除DDP模型的module.前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除'module.'前缀
        else:
            new_state_dict[k] = v
    return new_state_dict

def download_benchmarks():
    """下载基准测试数据集"""
    print("📥 下载基准数据集...")
    datasets = {}
    
    # WikiText-2 语言建模
    try:
        print("  正在下载 WikiText-2 语言建模数据集...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        datasets['wikitext'] = dataset
        print("  ✅ wikitext 下载完成")
    except Exception as e:
        print(f"  ❌ wikitext 下载失败: {e}")
    
    # LAMBADA 阅读理解
    try:
        print("  正在下载 LAMBADA 阅读理解数据集...")
        dataset = load_dataset("lambada", split="test")
        datasets['lambada'] = dataset
        print("  ✅ lambada 下载完成")
    except Exception as e:
        print(f"  ❌ lambada 下载失败: {e}")
    
    # HellaSwag 常识推理
    try:
        print("  正在下载 HellaSwag 常识推理数据集...")
        dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
        datasets['hellaswag'] = dataset
        print("  ✅ hellaswag 下载完成")
    except Exception as e:
        print(f"  ❌ hellaswag 下载失败: {e}")
    
    return datasets

def download_baseline_models():
    """下载基准对比模型"""
    print("\n🤖 下载预训练模型...")
    models = {}
    
    baseline_models = [
        ("gpt2-xl", "GPT-2 XL (1.5B参数)"),
        ("EleutherAI/gpt-neo-1.3B", "GPT-Neo 1.3B (1.3B参数)"),
        ("microsoft/DialoGPT-large", "DialoGPT Large (774M参数)"),
        ("gpt2-medium", "GPT-2 Medium (355M参数)"),
        ("distilgpt2", "DistilGPT-2 (82M参数) - 快速测试")
    ]
    
    for model_id, description in baseline_models:
        try:
            print(f"  正在下载 {description}...")
            
            # 尝试加载模型
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            param_count = sum(p.numel() for p in model.parameters())
            models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'params': param_count
            }
            
            print(f"  ✅ {model_id} 下载完成 ({param_count/1e9:.1f}B)")
            
        except Exception as e:
            print(f"  ❌ {model_id} 下载失败: {e}")
    
    return models

def test_trained_model(checkpoint_path, datasets):
    """测试训练的模型"""
    print(f"\n🧪 测试训练的模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ 模型文件不存在: {checkpoint_path}")
        return None
    
    try:
        from models import create_model
        from configs.base import ModelConfig
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_config = ModelConfig(**checkpoint['config'])
        
        model = create_model(model_config.model_type, model_config)
        
        # 处理DDP模型的state_dict
        state_dict = checkpoint['model_state_dict']
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        if has_module_prefix:
            print("  🔄 检测到DDP模型，移除module.前缀...")
            state_dict = remove_module_prefix(state_dict)
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 使用GPT-2 tokenizer（兼容性最好）
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 模型加载成功")
        print(f"     类型: {model_config.model_type}")
        print(f"     参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # 测试困惑度
        results = {}
        for dataset_name, dataset in datasets.items():
            if dataset is None:
                continue
                
            try:
                print(f"  📊 在{dataset_name}上计算困惑度...")
                perplexity = calculate_perplexity_trained_model(model, tokenizer, dataset, device)
                results[dataset_name] = {'perplexity': perplexity}
                print(f"     {dataset_name} 困惑度: {perplexity:.2f}")
            except Exception as e:
                print(f"     ❌ {dataset_name} 测试失败: {e}")
                results[dataset_name] = {'error': str(e)}
        
        return {
            'model_type': model_config.model_type,
            'params': total_params,
            'results': results
        }
        
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return None

def calculate_perplexity_trained_model(model, tokenizer, dataset, device, max_samples=50):
    """计算训练模型的困惑度"""
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
                
            if i % 10 == 0 and i > 0:
                print(f"    已处理 {i}/{max_samples} 个样本...")
            
            # 获取文本
            text = example.get('text', '') or example.get('ending', '') or str(example)
            if not text or len(text.strip()) < 10:
                continue
            
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            
            if input_ids.size(1) < 2:
                continue
            
            # 前向传播
            outputs = model(input_ids)
            
            # 处理不同类型的输出
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    # 获取第一个张量输出
                    tensor_outputs = {k: v for k, v in outputs.items() if torch.is_tensor(v)}
                    logits = next(iter(tensor_outputs.values()))
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def calculate_perplexity(model, tokenizer, dataset, device, max_samples=50):
    """计算困惑度"""
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
                
            if i % 10 == 0 and i > 0:
                print(f"  已处理 {i}/{max_samples} 个样本...")
            
            # 获取文本
            text = example.get('text', '') or example.get('ending', '') or str(example)
            if not text or len(text.strip()) < 10:
                continue
            
            # Tokenize
            try:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                
                if input_ids.size(1) < 2:
                    continue
                
                # 前向传播
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
                
            except Exception as e:
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def test_text_generation(model, tokenizer, device):
    """测试文本生成"""
    print("✍️ 测试文本生成...")
    
    test_prompts = [
        "人工智能的未来发展",
        "Once upon a time",
        "The meaning of life is",
        "科技改变了我们的生活"
    ]
    
    results = []
    
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=inputs['input_ids'].size(1) + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            print(f"\n  提示词: '{prompt}'")
            print(f"  生成: {generated_text[:100]}...")
            
            results.append({
                'prompt': prompt,
                'generated': generated_text
            })
            
        except Exception as e:
            print(f"  ❌ 生成'{prompt}'失败: {e}")
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="综合基准测试")
    parser.add_argument("--datasets-only", action="store_true", help="只下载数据集")
    parser.add_argument("--models-only", action="store_true", help="只下载模型")
    parser.add_argument("--trained-model", type=str, default="checkpoints/final_model.pt", 
                       help="训练的模型路径")
    
    args = parser.parse_args()
    
    print("🚀 开始综合测试...")
    
    # 设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU信息: {gpu_name}")
        print(f"显存: {gpu_memory:.1f}GB")
    
    results = {
        'device': str(device),
        'timestamp': time.time(),
        'models': {},
        'trained_model': None
    }
    
    # 下载数据集
    if not args.models_only:
        datasets = download_benchmarks()
    else:
        datasets = {}
    
    # 测试训练的模型
    if not args.models_only and datasets:
        trained_results = test_trained_model(args.trained_model, datasets)
        if trained_results:
            results['trained_model'] = trained_results
    
    # 下载和测试基准模型
    if not args.datasets_only:
        baseline_models = download_baseline_models()
        
        print("\n" + "="*60)
        print("开始模型测试")
        print("="*60)
        
        for model_id, model_info in baseline_models.items():
            print(f"\n🧪 测试模型: {model_id}")
            print("-" * 40)
            
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            
            model_results = {
                'params': model_info['params'],
                'perplexity': {},
                'generation': []
            }
            
            # 困惑度测试
            for dataset_name, dataset in datasets.items():
                if dataset is None:
                    continue
                    
                try:
                    print(f"📊 计算困惑度...")
                    perplexity = calculate_perplexity(model, tokenizer, dataset, device)
                    model_results['perplexity'][dataset_name] = perplexity
                    print(f"  {dataset_name} 困惑度: {perplexity:.2f}")
                except Exception as e:
                    print(f"  ❌ {dataset_name} 困惑度计算失败: {e}")
                    model_results['perplexity'][dataset_name] = None
            
            # 文本生成测试
            try:
                generation_results = test_text_generation(model, tokenizer, device)
                model_results['generation'] = generation_results
            except Exception as e:
                print(f"❌ 文本生成测试失败: {e}")
            
            results['models'][model_id] = model_results
    
    # 保存结果
    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f'benchmark_results_{int(time.time())}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 测试结果已保存至: {results_file}")
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if results['trained_model']:
        tm = results['trained_model']
        print(f"🤖 训练的模型 ({tm['model_type']}):")
        for dataset, result in tm['results'].items():
            if 'perplexity' in result:
                print(f"   {dataset}: 困惑度 {result['perplexity']:.2f}")
        print()
    
    for model_id, model_result in results['models'].items():
        print(f"✅ {model_id}: 测试完成")
        for dataset, perplexity in model_result['perplexity'].items():
            if perplexity is not None:
                print(f"   {dataset}: 困惑度 {perplexity:.2f}")

if __name__ == "__main__":
    main() 