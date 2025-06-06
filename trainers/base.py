"""
基础训练器类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import math
import json
from tqdm import tqdm
import wandb
from datetime import datetime
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import numpy as np

from models import create_model
from configs.base import ModelConfig, TrainingConfig
from data import DataProcessor

class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_rank = training_config.local_rank
        
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        
        self._setup_directories()
        self._setup_model()
        self._setup_data()
        self._setup_training_state()
        
        print(f"训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"模型类型: {model_config.model_type}")
        print(f"参数量: {self.total_params:,} ({self.total_params/1e6:.1f}M)")
    
    def _setup_directories(self):
        """设置输出目录"""
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training_config.logs_dir, exist_ok=True)
    
    def _setup_model(self):
        """设置模型"""
        # 创建模型
        self.model = create_model(self.model_config.model_type, self.model_config)
        self.model = self.model.to(self.device)
        
        # 计算参数量
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 打印模型信息
        print(f"总参数量: {self.total_params:,}")
        print(f"可训练参数量: {self.trainable_params:,}")
    
    def _setup_data(self):
        """设置数据处理器"""
        self.data_processor = DataProcessor(self.training_config)
    
    def _setup_training_state(self):
        """设置训练状态"""
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 混合精度训练
        self.scaler = GradScaler() if self.training_config.fp16 else None
        
        # 初始化W&B
        if self.training_config.use_wandb and (self.local_rank <= 0):
            self._setup_wandb()
    
    def _setup_wandb(self):
        """设置W&B"""
        run_name = self.training_config.wandb_run_name or f"{self.model_config.model_type}-{self.total_params//1e6:.0f}M"
        wandb.init(
            project=self.training_config.wandb_project,
            name=run_name,
            config={
                **self.model_config.to_dict(), 
                **self.training_config.to_dict(),
                'total_params': self.total_params,
                'trainable_params': self.trainable_params
            }
        )
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """设置优化器和学习率调度器"""
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
            eps=self.training_config.adam_eps
        )
        
        # 学习率调度器
        total_steps = self.training_config.max_steps
        if self.training_config.lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif self.training_config.lr_scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=0.1, 
                total_iters=self.training_config.warmup_steps
            )
        else:
            self.scheduler = None
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 数据移到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 前向传播
        if self.training_config.fp16:
            with autocast():
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
        else:
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
        
        # 梯度累积
        loss = loss / self.training_config.gradient_accumulation_steps
        
        # 反向传播
        if self.training_config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.training_config.gradient_accumulation_steps
    
    def validation_step(self, val_loader):
        """验证步骤"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中", disable=(self.local_rank > 0)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.training_config.fp16:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask, labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        if self.local_rank > 0:  # 只在主进程保存
            return
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir, 
            f'checkpoint_step_{self.global_step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.training_config.checkpoint_dir, 
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"从步骤 {self.global_step} 恢复训练")
    
    def train(self):
        """主训练循环"""
        if self.local_rank <= 0:
            print("开始准备数据...")
        
        train_loader, val_loader, tokenizer = self.data_processor.create_dataloaders()
        
        # 设置优化器和调度器
        self.setup_optimizer_and_scheduler(train_loader)
        
        if self.local_rank <= 0:
            print(f"开始训练，目标步数: {self.training_config.max_steps}")
        
        # 训练循环
        accumulation_count = 0
        total_loss = 0
        
        while self.global_step < self.training_config.max_steps:
            epoch_iterator = tqdm(
                train_loader, 
                desc=f"Epoch {self.epoch}",
                disable=(self.local_rank > 0)
            )
            
            for batch in epoch_iterator:
                # 训练步骤
                loss = self.train_step(batch)
                total_loss += loss
                accumulation_count += 1
                
                # 梯度累积
                if accumulation_count >= self.training_config.gradient_accumulation_steps:
                    self._optimizer_step()
                    accumulation_count = 0
                    
                    # 记录日志
                    if self.global_step % self.training_config.logging_steps == 0:
                        self._log_training_metrics(total_loss)
                        total_loss = 0
                    
                    # 验证
                    if self.global_step % self.training_config.eval_steps == 0:
                        self._run_validation(val_loader)
                    
                    # 检查是否达到最大步数
                    if self.global_step >= self.training_config.max_steps:
                        break
                
                # 更新进度条
                if self.local_rank <= 0:
                    epoch_iterator.set_postfix({
                        'loss': f'{loss:.4f}',
                        'step': self.global_step
                    })
            
            self.epoch += 1
            
            if self.global_step >= self.training_config.max_steps:
                break
        
        if self.local_rank <= 0:
            print("训练完成！")
            self.save_checkpoint()
            
            if self.training_config.use_wandb:
                wandb.finish()
    
    def _optimizer_step(self):
        """优化器步骤"""
        # 梯度裁剪
        if self.training_config.fp16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.training_config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        self.optimizer.zero_grad()
        self.global_step += 1
    
    def _log_training_metrics(self, total_loss):
        """记录训练指标"""
        if self.local_rank > 0:
            return
            
        avg_loss = total_loss / self.training_config.logging_steps
        lr = self.optimizer.param_groups[0]['lr']
        
        print(f"步骤 {self.global_step}: 损失={avg_loss:.4f}, 学习率={lr:.2e}")
        
        if self.training_config.use_wandb:
            wandb.log({
                'train_loss': avg_loss,
                'learning_rate': lr,
                'global_step': self.global_step
            })
    
    def _run_validation(self, val_loader):
        """运行验证"""
        if self.local_rank > 0:
            return
            
        val_loss, perplexity = self.validation_step(val_loader)
        print(f"验证损失: {val_loss:.4f}, 困惑度: {perplexity:.2f}")
        
        if self.training_config.use_wandb:
            wandb.log({
                'val_loss': val_loss,
                'perplexity': perplexity,
                'global_step': self.global_step
            })
        
        # 保存最佳模型
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        if self.global_step % self.training_config.save_steps == 0:
            self.save_checkpoint(is_best)
    
    def generate_sample(self, tokenizer, prompt="Hello", max_length=50):
        """生成样本文本"""
        self.model.eval()
        
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids, 
                max_length=max_length, 
                temperature=0.8,
                top_k=50
            )
        
        # 解码
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text 