#!/usr/bin/env python3
"""
数据集管理器
支持多种高质量训练数据集的下载和处理
"""

import os
from datasets import load_dataset
from typing import Dict, List, Optional
import torch

class DatasetManager:
    """数据集管理器"""
    
    # 可用的数据集配置
    AVAILABLE_DATASETS = {
        'openwebtext': {
            'name': 'openwebtext',
            'description': 'OpenWebText - 高质量网页文本（40GB+）',
            'size': '40GB+',
            'quality': '⭐⭐⭐⭐⭐',
            'lang': 'en',
            'recommended_for': ['1B', '7B']
        },
        'c4': {
            'name': 'c4',
            'config': 'en',
            'description': 'C4 (Colossal Clean Crawled Corpus) - T5训练数据',
            'size': '750GB+', 
            'quality': '⭐⭐⭐⭐⭐',
            'lang': 'en',
            'recommended_for': ['7B']
        },
        'the_pile': {
            'name': 'the_pile',
            'description': 'The Pile - 多领域大规模文本集合（800GB）',
            'size': '800GB',
            'quality': '⭐⭐⭐⭐⭐',
            'lang': 'en',
            'recommended_for': ['7B']
        },
        'wikitext': {
            'name': 'wikitext',
            'config': 'wikitext-103-raw-v1',
            'description': 'WikiText-103 - 维基百科文章（500MB）',
            'size': '500MB',
            'quality': '⭐⭐⭐⭐',
            'lang': 'en',
            'recommended_for': ['1B']
        },
        'bookcorpus': {
            'name': 'bookcorpus',
            'description': 'BookCorpus - 11000+本书籍文本（5GB）',
            'size': '5GB',
            'quality': '⭐⭐⭐⭐',
            'lang': 'en', 
            'recommended_for': ['1B']
        },
        'cc_news': {
            'name': 'cc_news',
            'description': 'CC-News - Common Crawl新闻文章（76GB）',
            'size': '76GB',
            'quality': '⭐⭐⭐⭐',
            'lang': 'en',
            'recommended_for': ['1B', '7B']
        },
        'squad': {
            'name': 'squad',
            'description': 'SQuAD - 问答数据集（35MB）- 快速测试',
            'size': '35MB',
            'quality': '⭐⭐⭐',
            'lang': 'en',
            'recommended_for': ['test']
        },
        'chinese_webtext': {
            'name': 'BAAI/chinese_webtext',
            'description': '中文网页文本 - 清洗后的中文语料（20GB+）',
            'size': '20GB+',
            'quality': '⭐⭐⭐⭐',
            'lang': 'zh',
            'recommended_for': ['1B', '7B']
        },
        'chinese_poetry': {
            'name': 'chinese_poetry_collection',
            'description': '中国古诗词集合 - 传统文化语料（100MB）',
            'size': '100MB', 
            'quality': '⭐⭐⭐',
            'lang': 'zh',
            'recommended_for': ['1B']
        }
    }
    
    def __init__(self):
        self.cache_dir = "./data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def list_datasets(self, model_size: Optional[str] = None, language: Optional[str] = None):
        """列出可用的数据集"""
        print("\n📚 可用训练数据集:")
        print("=" * 80)
        
        for dataset_id, info in self.AVAILABLE_DATASETS.items():
            # 过滤条件
            if model_size and model_size not in info['recommended_for']:
                continue
            if language and info['lang'] != language:
                continue
                
            print(f"\n🗂️  {dataset_id}")
            print(f"   描述: {info['description']}")
            print(f"   大小: {info['size']}")
            print(f"   质量: {info['quality']}")
            print(f"   语言: {info['lang']}")
            print(f"   推荐: {', '.join(info['recommended_for'])}")
    
    def get_dataset_recommendations(self, model_size: str, language: str = 'en'):
        """根据模型大小推荐数据集"""
        recommendations = []
        
        for dataset_id, info in self.AVAILABLE_DATASETS.items():
            if (model_size in info['recommended_for'] and 
                info['lang'] == language):
                recommendations.append(dataset_id)
        
        return recommendations
    
    def download_dataset(self, dataset_id: str, streaming: bool = True, subset_size: Optional[int] = None):
        """下载指定数据集"""
        if dataset_id not in self.AVAILABLE_DATASETS:
            raise ValueError(f"未知数据集: {dataset_id}")
        
        info = self.AVAILABLE_DATASETS[dataset_id]
        print(f"\n📥 下载数据集: {info['description']}")
        print(f"预计大小: {info['size']}")
        
        try:
            if 'config' in info:
                dataset = load_dataset(
                    info['name'], 
                    info['config'],
                    streaming=streaming,
                    cache_dir=self.cache_dir
                )
            else:
                dataset = load_dataset(
                    info['name'],
                    streaming=streaming, 
                    cache_dir=self.cache_dir
                )
            
            print(f"✅ {dataset_id} 下载成功")
            
            # 如果指定了子集大小，进行采样
            if subset_size and not streaming:
                if 'train' in dataset:
                    total_size = len(dataset['train'])
                    if total_size > subset_size:
                        dataset['train'] = dataset['train'].select(range(subset_size))
                        print(f"📊 已采样 {subset_size:,} 条数据（原始: {total_size:,}）")
            
            return dataset
            
        except Exception as e:
            print(f"❌ {dataset_id} 下载失败: {e}")
            return None
    
    def prepare_mixed_dataset(self, dataset_ids: List[str], weights: Optional[List[float]] = None):
        """准备混合数据集"""
        print(f"\n🔀 准备混合数据集: {', '.join(dataset_ids)}")
        
        datasets = []
        successful_ids = []
        
        for dataset_id in dataset_ids:
            dataset = self.download_dataset(dataset_id, streaming=True)
            if dataset is not None:
                datasets.append(dataset)
                successful_ids.append(dataset_id)
        
        if not datasets:
            print("❌ 没有成功下载任何数据集")
            return None
        
        print(f"✅ 成功准备 {len(datasets)} 个数据集")
        return datasets, successful_ids

def get_dataset_config_for_model_size(model_size: str):
    """为不同模型大小推荐数据集配置"""
    manager = DatasetManager()
    
    configs = {
        '1B': {
            'primary': ['wikitext', 'bookcorpus'],
            'secondary': ['cc_news'],
            'chinese': ['chinese_webtext', 'chinese_poetry'],
            'mix_ratio': [0.4, 0.3, 0.3]
        },
        '7B': {
            'primary': ['openwebtext', 'c4'],
            'secondary': ['the_pile', 'cc_news'],
            'chinese': ['chinese_webtext'],
            'mix_ratio': [0.5, 0.3, 0.2]
        },
        'test': {
            'primary': ['squad', 'wikitext'],
            'secondary': [],
            'chinese': ['chinese_poetry'],
            'mix_ratio': [0.8, 0.2]
        }
    }
    
    return configs.get(model_size, configs['1B'])

if __name__ == "__main__":
    # 演示用法
    manager = DatasetManager()
    
    print("🚀 数据集管理器演示")
    
    # 列出所有数据集
    manager.list_datasets()
    
    # 为1B模型推荐数据集
    print("\n" + "="*50)
    recommendations = manager.get_dataset_recommendations('1B', 'en')
    print(f"1B模型推荐数据集: {recommendations}")
    
    # 为7B模型推荐数据集
    recommendations = manager.get_dataset_recommendations('7B', 'en')
    print(f"7B模型推荐数据集: {recommendations}") 