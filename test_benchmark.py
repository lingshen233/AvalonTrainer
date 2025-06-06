#!/usr/bin/env python3
"""
自动化测试脚本
- 下载基准数据集
- 下载预训练模型
- 执行模型测试和评估
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import requests
import json
from pathlib import Path
import time

def download_benchmark_datasets():
    """下载基准测试数据集"""
    print("📥 下载基准数据集...")
    
    datasets_info = {
        'wikitext': {
            'name': 'wikitext',
            'config': 'wikitext-2-raw-v1',
            'description': 'WikiText-2 语言建模数据集'
        },
        'lambada': {
            'name': 'lambada',
            'config': None,
            'description': 'LAMBADA 阅读理解数据集'
        },
        'hellaswag': {
            'name': 'hellaswag',
            'config': None,
            'description': 'HellaSwag 常识推理数据集'
        }
    }
    
    downloaded_datasets = {}
    
    for dataset_key, info in datasets_info.items():
        try:
            print(f"  正在下载 {info['description']}...")
            if info['config']:
                dataset = load_dataset(info['name'], info['config'])
            else:
                dataset = load_dataset(info['name'])
            
            downloaded_datasets[dataset_key] = dataset
            print(f"  ✅ {dataset_key} 下载完成")
            
        except Exception as e:
            print(f"  ❌ {dataset_key} 下载失败: {e}")
            downloaded_datasets[dataset_key] = None
    
    return downloaded_datasets

def download_pretrained_models():
    """下载预训练的1B模型"""
    print("\n🤖 下载预训练模型...")
    
    models_info = {
        'gpt2-medium': {
            'name': 'gpt2-medium',
            'description': 'GPT-2 Medium (355M参数)',
            'size': '355M'
        },
        'distilgpt2': {
            'name': 'distilgpt2', 
            'description': 'DistilGPT-2 (82M参数)',
            'size': '82M'
        },
        'microsoft/DialoGPT-medium': {
            'name': 'microsoft/DialoGPT-medium',
            'description': 'DialoGPT Medium (355M参数)',
            'size': '355M'
        }
    }
    
    downloaded_models = {}
    
    for model_key, info in models_info.items():
        try:
            print(f"  正在下载 {info['description']}...")
            
            # 下载tokenizer和模型
            tokenizer = AutoTokenizer.from_pretrained(info['name'])
            model = AutoModelForCausalLM.from_pretrained(info['name'])
            
            downloaded_models[model_key] = {
                'tokenizer': tokenizer,
                'model': model,
                'info': info
            }
            
            print(f"  ✅ {model_key} 下载完成 ({info['size']})")
            
        except Exception as e:
            print(f"  ❌ {model_key} 下载失败: {e}")
            downloaded_models[model_key] = None
    
    return downloaded_models

def evaluate_perplexity(model, tokenizer, dataset, max_samples=100):
    """计算困惑度"""
    print("📊 计算困惑度...")
    
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    
    # 使用测试集的一小部分
    test_texts = dataset['test']['text'][:max_samples] if 'test' in dataset else dataset['validation']['text'][:max_samples]
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            if len(text.strip()) < 10:  # 跳过太短的文本
                continue
                
            try:
                # 编码文本
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 计算损失
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i+1}/{len(test_texts)} 个样本...")
                    
            except Exception as e:
                print(f"  跳过样本 {i}: {e}")
                continue
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
    else:
        return float('inf')

def test_text_generation(model, tokenizer, prompts=None):
    """测试文本生成"""
    print("✍️ 测试文本生成...")
    
    if prompts is None:
        prompts = [
            "人工智能的未来发展",
            "Once upon a time",
            "The meaning of life is", 
            "科技改变了我们的生活"
        ]
    
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    
    for prompt in prompts:
        try:
            print(f"\n  提示词: '{prompt}'")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=inputs['input_ids'].size(1) + 50,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            print(f"  生成: {generated_text}")
            results.append({
                'prompt': prompt,
                'generated': generated_text
            })
            
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            results.append({
                'prompt': prompt,
                'generated': None,
                'error': str(e)
            })
    
    return results

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始综合测试...")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU信息: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 下载数据集
    datasets = download_benchmark_datasets()
    
    # 下载模型
    models = download_pretrained_models()
    
    # 测试结果
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'models': {}
    }
    
    print("\n" + "="*60)
    print("开始模型测试")
    print("="*60)
    
    for model_name, model_data in models.items():
        if model_data is None:
            continue
            
        print(f"\n🧪 测试模型: {model_name}")
        print("-" * 40)
        
        try:
            model = model_data['model'].to(device)
            tokenizer = model_data['tokenizer']
            
            # 确保tokenizer有pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_results = {
                'model_info': model_data['info'],
                'perplexity': {},
                'generation_test': None
            }
            
            # 测试困惑度
            for dataset_name, dataset in datasets.items():
                if dataset is not None:
                    try:
                        ppl = evaluate_perplexity(model, tokenizer, dataset, max_samples=50)
                        model_results['perplexity'][dataset_name] = ppl
                        print(f"  {dataset_name} 困惑度: {ppl:.2f}")
                    except Exception as e:
                        print(f"  ❌ {dataset_name} 困惑度测试失败: {e}")
                        model_results['perplexity'][dataset_name] = None
            
            # 测试文本生成
            try:
                generation_results = test_text_generation(model, tokenizer)
                model_results['generation_test'] = generation_results
            except Exception as e:
                print(f"  ❌ 文本生成测试失败: {e}")
                model_results['generation_test'] = None
            
            test_results['models'][model_name] = model_results
            
            # 清理GPU内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 测试失败: {e}")
            test_results['models'][model_name] = {'error': str(e)}
    
    # 保存测试结果
    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f'benchmark_results_{int(time.time())}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 测试结果已保存至: {results_file}")
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for model_name, results in test_results['models'].items():
        if 'error' in results:
            print(f"❌ {model_name}: 测试失败")
        else:
            print(f"✅ {model_name}: 测试完成")
            if results.get('perplexity'):
                for dataset, ppl in results['perplexity'].items():
                    if ppl is not None:
                        print(f"   {dataset}: 困惑度 {ppl:.2f}")

def main():
    parser = argparse.ArgumentParser(description="自动化基准测试脚本")
    parser.add_argument("--datasets-only", action="store_true", help="只下载数据集")
    parser.add_argument("--models-only", action="store_true", help="只下载模型")
    parser.add_argument("--quick-test", action="store_true", help="快速测试模式")
    
    args = parser.parse_args()
    
    if args.datasets_only:
        download_benchmark_datasets()
    elif args.models_only:
        download_pretrained_models()
    else:
        run_comprehensive_test()

if __name__ == "__main__":
    main() 