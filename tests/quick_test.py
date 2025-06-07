#!/usr/bin/env python3
"""
快速测试脚本 - 验证训练框架基本功能
"""

import torch
import yaml
import subprocess
import sys
import os
from pathlib import Path

def test_dependencies():
    """测试依赖包是否正确安装"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 
        'yaml', 'numpy', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def test_cuda():
    """测试CUDA环境"""
    print("\n🔍 检查CUDA环境...")
    
    if torch.cuda.is_available():
        print(f"  ✅ CUDA 可用")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
        return True
    else:
        print("  ⚠️ CUDA 不可用，将使用CPU")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n🔍 测试配置文件...")
    
    config_files = ['config.yaml', 'config_transformer_4gpu.yaml']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"  ✅ {config_file} 加载成功")
            except Exception as e:
                print(f"  ❌ {config_file} 加载失败: {e}")
                return False
        else:
            print(f"  ❌ {config_file} 不存在")
            return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        # 测试导入
        from configs.base import ModelConfig
        from models import create_model
        
        # 创建小型测试配置
        test_config = ModelConfig(
            model_type='transformer',
            vocab_size=1000,
            max_seq_length=128,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1
        )
        
        # 创建模型
        model = create_model('transformer', test_config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✅ Transformer模型创建成功")
        print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 测试Mamba模型
        test_config.model_type = 'mamba'
        test_config.d_state = 16
        test_config.d_conv = 4
        test_config.expand = 2
        
        model = create_model('mamba', test_config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✅ Mamba模型创建成功")
        print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return False

def test_training_script():
    """测试训练脚本"""
    print("\n🔍 测试训练脚本...")
    
    # 测试列出模型
    try:
        result = subprocess.run([sys.executable, 'train.py', '--list_models'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("  ✅ --list_models 工作正常")
        else:
            print(f"  ❌ --list_models 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ --list_models 测试失败: {e}")
        return False
    
    # 测试dry run
    try:
        result = subprocess.run([sys.executable, 'train.py', '--dry_run'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("  ✅ --dry_run 工作正常")
        else:
            print(f"  ❌ --dry_run 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ --dry_run 测试失败: {e}")
        return False
    
    return True

def main():
    print("🚀 开始快速测试...")
    print("=" * 50)
    
    tests = [
        ("依赖包检查", test_dependencies),
        ("CUDA环境", test_cuda),
        ("配置文件", test_config_loading),
        ("模型创建", test_model_creation),
        ("训练脚本", test_training_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！框架可以正常使用")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关问题")
        return 1

if __name__ == "__main__":
    exit(main()) 