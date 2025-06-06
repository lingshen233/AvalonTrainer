"""
Transformer模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from configs.base import ModelConfig

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用注意力掩码
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(output)

class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        # Pre-norm架构
        attn_output = self.attention(self.ln1(x), attention_mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() *
                           -(math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = PositionalEncoding(config)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 最终层归一化
        self.ln_final = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # Token嵌入
        x = self.token_embedding(input_ids)
        
        # 位置编码
        x = self.position_encoding(x)
        
        # 通过所有Transformer层
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # 最终层归一化
        x = self.ln_final(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """文本生成"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k采样
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 检查是否生成结束符
                if next_token.item() == 50256:  # GPT-2的结束符
                    break
        
        return input_ids 

# 兼容性别名
Transformer = TransformerModel 