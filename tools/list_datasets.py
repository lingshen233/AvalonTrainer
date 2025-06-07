#!/usr/bin/env python3
"""
数据集浏览和选择工具
帮助用户选择合适的训练数据集
"""

import argparse
from data.dataset_manager import DatasetManager

def main():
    parser = argparse.ArgumentParser(description="数据集浏览和选择工具")
    parser.add_argument("--model-size", type=str, choices=['1B', '7B', 'test'], 
                       help="根据模型大小过滤数据集")
    parser.add_argument("--language", type=str, choices=['en', 'zh'], 
                       help="根据语言过滤数据集")
    parser.add_argument("--download", type=str, help="下载指定数据集")
    parser.add_argument("--recommend", type=str, choices=['1B', '7B'], 
                       help="为指定模型大小推荐数据集")
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.download:
        # 下载指定数据集
        print(f"📥 下载数据集: {args.download}")
        dataset = manager.download_dataset(args.download, streaming=False)
        if dataset:
            print("✅ 下载完成")
    
    elif args.recommend:
        # 推荐数据集
        print(f"\n💡 为 {args.recommend} 模型推荐的数据集:")
        recommendations = manager.get_dataset_recommendations(args.recommend, 'en')
        for i, dataset_id in enumerate(recommendations, 1):
            info = manager.AVAILABLE_DATASETS[dataset_id]
            print(f"{i}. {dataset_id}: {info['description']}")
        
        # 中文数据集推荐
        zh_recommendations = manager.get_dataset_recommendations(args.recommend, 'zh')
        if zh_recommendations:
            print(f"\n🈳 中文数据集推荐:")
            for i, dataset_id in enumerate(zh_recommendations, 1):
                info = manager.AVAILABLE_DATASETS[dataset_id]
                print(f"{i}. {dataset_id}: {info['description']}")
    
    else:
        # 列出数据集
        manager.list_datasets(args.model_size, args.language)
        
        print(f"\n💡 使用提示:")
        print(f"  查看1B模型推荐: python {__file__} --recommend 1B")
        print(f"  查看7B模型推荐: python {__file__} --recommend 7B")
        print(f"  下载数据集: python {__file__} --download wikitext")
        print(f"  只看中文数据集: python {__file__} --language zh")

if __name__ == "__main__":
    main() 