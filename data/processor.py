"""
数据处理器
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from configs.base import TrainingConfig

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.data_collator = None
        
    def setup_tokenizer(self):
        """设置分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        return self.tokenizer
    
    def load_dataset(self):
        """加载数据集"""
        if self.config.dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
            train_texts = [item['text'] for item in dataset['train'] if len(item['text'].strip()) > 0]
            val_texts = [item['text'] for item in dataset['validation'] if len(item['text'].strip()) > 0]
        else:
            # 简化：使用dummy数据
            train_texts = ["This is a dummy text for training."] * 1000
            val_texts = ["This is a dummy text for validation."] * 100
        
        return train_texts, val_texts
    
    def create_dataloaders(self):
        """创建数据加载器"""
        # 设置分词器
        tokenizer = self.setup_tokenizer()
        
        # 加载数据
        train_texts, val_texts = self.load_dataset()
        
        # 创建数据集
        train_dataset = TextDataset(train_texts, tokenizer, self.config.max_length)
        val_dataset = TextDataset(val_texts, tokenizer, self.config.max_length)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.data_collator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.data_collator
        )
        
        return train_loader, val_loader, tokenizer

# 推荐的数据集
RECOMMENDED_DATASETS = {
    "wikitext": {
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "description": "维基百科文本，适合语言建模任务",
        "size": "~500MB",
        "difficulty": "初级"
    },
    "openwebtext": {
        "name": "openwebtext", 
        "config": None,
        "description": "开放网页文本，类似GPT-2训练数据",
        "size": "~12GB",
        "difficulty": "中级"
    },
    "the_pile": {
        "name": "the_pile",
        "config": "all",
        "description": "大规模多样化文本数据集",
        "size": "~800GB",
        "difficulty": "高级"
    },
    "c4": {
        "name": "c4",
        "config": "en",
        "description": "Common Crawl清洁版本，T5使用的数据集",
        "size": "~300GB", 
        "difficulty": "高级"
    }
}

def print_dataset_recommendations():
    """打印数据集推荐"""
    print("\n=== 推荐数据集 ===")
    for key, info in RECOMMENDED_DATASETS.items():
        print(f"\n【{info['difficulty']}】{key}:")
        print(f"  描述: {info['description']}")
        print(f"  大小: {info['size']}")
        print(f"  配置: {info.get('config', 'None')}")
    
    print(f"\n建议:")
    print("- 初学者建议从wikitext开始")
    print("- 有经验的用户可以尝试openwebtext")
    print("- A100显卡可以处理the_pile这样的大型数据集")
    print("- 根据你的存储空间和训练时间来选择") 