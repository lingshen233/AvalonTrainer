#!/usr/bin/env python3
"""
综合改进版测试脚本
支持DDP模型加载、完整基准测试、性能分析
"""

import os
import sys
import argparse
import torch
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def remove_module_prefix(state_dict):
    """移除DDP模型的module.前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_trained_model(checkpoint_path):
    """加载训练的模型（支持DDP）"""
    from models import create_model
    from configs.base import ModelConfig
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config_dict = checkpoint['config']
    model_config = ModelConfig(**config_dict)
    
    model = create_model(model_config.model_type, model_config)
    
    # 处理DDP模型
    state_dict = checkpoint['model_state_dict']
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if has_module_prefix:
        print("🔄 检测到DDP模型，移除module.前缀...")
        state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model, model_config

def comprehensive_model_test(model, model_config, device):
    """综合模型测试"""
    results = {
        'model_type': model_config.model_type,
        'params': sum(p.numel() for p in model.parameters()),
        'tests': {}
    }
    
    print(f"📊 模型信息:")
    print(f"   类型: {model_config.model_type}")
    print(f"   参数量: {results['params']:,} ({results['params']/1e6:.1f}M)")
    
    # 1. 基础前向传播测试
    print("\n🧪 基础前向传播测试...")
    try:
        batch_sizes = [1, 2, 4]
        seq_lengths = [128, 256, 512, 1024]
        
        forward_results = {}
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                if seq_length > model_config.max_seq_length:
                    continue
                    
                test_input = torch.randint(0, model_config.vocab_size, (batch_size, seq_length)).to(device)
                
                # 测量时间和显存
                torch.cuda.synchronize()
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() / 1e9
                
                with torch.no_grad():
                    outputs = model(test_input)
                
                torch.cuda.synchronize()
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() / 1e9
                
                forward_results[f'batch{batch_size}_seq{seq_length}'] = {
                    'time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'tokens_per_sec': (batch_size * seq_length) / (end_time - start_time)
                }
                
                print(f"   批大小{batch_size}, 序列{seq_length}: {forward_results[f'batch{batch_size}_seq{seq_length}']['tokens_per_sec']:.0f} tokens/sec")
        
        results['tests']['forward_pass'] = forward_results
        
    except Exception as e:
        print(f"   ❌ 前向传播测试失败: {e}")
        results['tests']['forward_pass'] = {'error': str(e)}
    
    # 2. 生成测试
    print("\n📝 文本生成测试...")
    try:
        generation_tests = []
        test_prompts = [
            "人工智能的发展",
            "The future of technology",
            "机器学习算法",
            "Once upon a time in"
        ]
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # 生成10个token
                current_input = inputs['input_ids']
                for _ in range(10):
                    outputs = model(current_input)
                    
                    # 处理不同输出格式
                    if isinstance(outputs, dict):
                        if 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            tensor_outputs = {k: v for k, v in outputs.items() if torch.is_tensor(v)}
                            logits = next(iter(tensor_outputs.values()))
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    current_input = torch.cat([current_input, next_token], dim=1)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            generated = tokenizer.decode(current_input[0], skip_special_tokens=True)
            generated_text = generated[len(prompt):].strip()
            
            generation_tests.append({
                'prompt': prompt,
                'generated': generated_text,
                'time': end_time - start_time,
                'tokens_per_sec': 10 / (end_time - start_time)
            })
            
            print(f"   '{prompt}' -> '{generated_text[:30]}...' ({generation_tests[-1]['tokens_per_sec']:.1f} t/s)")
        
        results['tests']['generation'] = generation_tests
        
    except Exception as e:
        print(f"   ❌ 生成测试失败: {e}")
        results['tests']['generation'] = {'error': str(e)}
    
    # 3. 显存使用分析
    print("\n💾 显存使用分析...")
    try:
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated() / 1e9
        
        # 测试不同批大小的显存使用
        memory_usage = {}
        for batch_size in [1, 2, 4, 8]:
            try:
                test_input = torch.randint(0, model_config.vocab_size, (batch_size, 512)).to(device)
                
                with torch.no_grad():
                    outputs = model(test_input)
                
                current_memory = torch.cuda.memory_allocated() / 1e9
                memory_usage[f'batch_{batch_size}'] = current_memory - baseline_memory
                
                print(f"   批大小{batch_size}: {memory_usage[f'batch_{batch_size}']:.2f}GB")
                
                # 清理
                del test_input, outputs
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"   批大小{batch_size}: OOM")
                memory_usage[f'batch_{batch_size}'] = None
                break
        
        results['tests']['memory_usage'] = memory_usage
        
    except Exception as e:
        print(f"   ❌ 显存分析失败: {e}")
        results['tests']['memory_usage'] = {'error': str(e)}
    
    return results

def benchmark_comparison(model, model_config, device):
    """与基准模型对比"""
    print("\n📊 基准对比测试...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 选择合适的基准模型
        if model_config.model_type == 'mamba':
            baseline_models = ['distilgpt2', 'gpt2']  # 轻量级对比
        else:
            baseline_models = ['distilgpt2', 'gpt2', 'gpt2-medium']
        
        comparison_results = {}
        
        for baseline_id in baseline_models:
            try:
                print(f"   加载基准模型: {baseline_id}")
                
                tokenizer = AutoTokenizer.from_pretrained(baseline_id)
                baseline_model = AutoModelForCausalLM.from_pretrained(
                    baseline_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # 简单性能对比
                test_input = torch.randint(0, min(model_config.vocab_size, tokenizer.vocab_size), (2, 256)).to(device)
                
                # 测试我们的模型
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    our_outputs = model(test_input)
                torch.cuda.synchronize()
                our_time = time.time() - start_time
                
                # 测试基准模型
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    baseline_outputs = baseline_model(test_input)
                torch.cuda.synchronize()
                baseline_time = time.time() - start_time
                
                comparison_results[baseline_id] = {
                    'our_time': our_time,
                    'baseline_time': baseline_time,
                    'speedup': baseline_time / our_time,
                    'params_baseline': sum(p.numel() for p in baseline_model.parameters())
                }
                
                print(f"   {baseline_id}: 速度比 {comparison_results[baseline_id]['speedup']:.2f}x")
                
                # 清理
                del baseline_model, tokenizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ {baseline_id} 对比失败: {e}")
                comparison_results[baseline_id] = {'error': str(e)}
        
        return comparison_results
        
    except Exception as e:
        print(f"   ❌ 基准对比失败: {e}")
        return {'error': str(e)}

def save_test_report(results, output_file):
    """保存详细测试报告"""
    
    # JSON报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成可视化报告
    try:
        create_visualization_report(results, output_file.replace('.json', '_visual.png'))
    except Exception as e:
        print(f"⚠️ 可视化报告生成失败: {e}")

def create_visualization_report(results, output_file):
    """创建可视化报告"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'模型性能测试报告 - {results["model_type"]}', fontsize=16)
    
    # 1. 前向传播性能
    if 'forward_pass' in results['tests'] and 'error' not in results['tests']['forward_pass']:
        forward_data = results['tests']['forward_pass']
        batch_sizes = []
        tokens_per_sec = []
        
        for key, data in forward_data.items():
            if 'batch1' in key:  # 只显示batch_size=1的结果
                seq_len = int(key.split('_seq')[1])
                batch_sizes.append(seq_len)
                tokens_per_sec.append(data['tokens_per_sec'])
        
        if batch_sizes:
            axes[0, 0].plot(batch_sizes, tokens_per_sec, marker='o')
            axes[0, 0].set_title('前向传播性能')
            axes[0, 0].set_xlabel('序列长度')
            axes[0, 0].set_ylabel('Tokens/秒')
            axes[0, 0].grid(True)
    
    # 2. 显存使用
    if 'memory_usage' in results['tests'] and 'error' not in results['tests']['memory_usage']:
        memory_data = results['tests']['memory_usage']
        batch_sizes = []
        memory_usage = []
        
        for key, usage in memory_data.items():
            if usage is not None:
                batch_size = int(key.split('_')[1])
                batch_sizes.append(batch_size)
                memory_usage.append(usage)
        
        if batch_sizes:
            axes[0, 1].bar(range(len(batch_sizes)), memory_usage)
            axes[0, 1].set_title('显存使用')
            axes[0, 1].set_xlabel('批大小')
            axes[0, 1].set_ylabel('显存使用 (GB)')
            axes[0, 1].set_xticks(range(len(batch_sizes)))
            axes[0, 1].set_xticklabels(batch_sizes)
    
    # 3. 生成性能
    if 'generation' in results['tests'] and 'error' not in results['tests']['generation']:
        gen_data = results['tests']['generation']
        prompts = [item['prompt'][:10] + '...' for item in gen_data]
        gen_speeds = [item['tokens_per_sec'] for item in gen_data]
        
        axes[1, 0].bar(range(len(prompts)), gen_speeds)
        axes[1, 0].set_title('文本生成速度')
        axes[1, 0].set_xlabel('提示词')
        axes[1, 0].set_ylabel('Tokens/秒')
        axes[1, 0].set_xticks(range(len(prompts)))
        axes[1, 0].set_xticklabels(prompts, rotation=45)
    
    # 4. 模型参数分布（如果有对比数据）
    if 'benchmark_comparison' in results:
        comp_data = results['benchmark_comparison']
        models = list(comp_data.keys())
        speedups = [comp_data[model].get('speedup', 0) for model in models]
        
        axes[1, 1].bar(range(len(models)), speedups)
        axes[1, 1].set_title('相对基准模型速度')
        axes[1, 1].set_xlabel('基准模型')
        axes[1, 1].set_ylabel('速度倍数')
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 可视化报告已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="综合改进版模型测试")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt",
                       help="模型检查点路径")
    parser.add_argument("--output", type=str, default="test_results",
                       help="输出目录")
    parser.add_argument("--benchmark", action="store_true",
                       help="执行基准对比测试")
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式")
    
    args = parser.parse_args()
    
    print("🚀 综合模型测试开始...")
    print("=" * 60)
    
    # 检查环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 加载模型
    print(f"\n📥 加载模型: {args.checkpoint}")
    try:
        model, model_config = load_trained_model(args.checkpoint)
        model = model.to(device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1
    
    # 执行测试
    results = {
        'timestamp': time.time(),
        'model_checkpoint': args.checkpoint,
        'device': str(device),
        'model_type': model_config.model_type,
        'params': sum(p.numel() for p in model.parameters())
    }
    
    # 综合测试
    print("\n" + "=" * 60)
    print("开始综合性能测试")
    print("=" * 60)
    
    test_results = comprehensive_model_test(model, model_config, device)
    results.update(test_results)
    
    # 基准对比
    if args.benchmark and not args.quick:
        print("\n" + "=" * 60)
        print("基准模型对比测试")
        print("=" * 60)
        
        benchmark_results = benchmark_comparison(model, model_config, device)
        results['benchmark_comparison'] = benchmark_results
    
    # 保存报告
    timestamp = int(time.time())
    output_file = output_dir / f'comprehensive_test_{timestamp}.json'
    save_test_report(results, str(output_file))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    print(f"✅ 测试完成")
    print(f"   模型类型: {results['model_type']}")
    print(f"   参数量: {results['params']:,} ({results['params']/1e6:.1f}M)")
    print(f"   详细报告: {output_file}")
    
    if 'tests' in results:
        if 'forward_pass' in results['tests'] and 'error' not in results['tests']['forward_pass']:
            # 显示最佳性能
            forward_data = results['tests']['forward_pass']
            best_perf = max(data['tokens_per_sec'] for data in forward_data.values() if isinstance(data, dict))
            print(f"   最佳吞吐量: {best_perf:.0f} tokens/sec")
        
        if 'generation' in results['tests'] and 'error' not in results['tests']['generation']:
            gen_data = results['tests']['generation']
            avg_speed = np.mean([item['tokens_per_sec'] for item in gen_data])
            print(f"   平均生成速度: {avg_speed:.1f} tokens/sec")
    
    print(f"\n💡 下一步建议:")
    print(f"   1. 查看详细报告: cat {output_file}")
    print(f"   2. 如果性能满意，可尝试更大模型")
    print(f"   3. 运行基准测试: --benchmark")
    
    return 0

if __name__ == "__main__":
    exit(main()) 