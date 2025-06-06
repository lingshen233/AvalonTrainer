"""
Mamba模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from configs.base import ModelConfig

class RMSNorm(nn.Module):
    """RMS归一化"""
    
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SelectiveSSM(nn.Module):
    """选择性状态空间模型（Mamba核心）"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # x_proj, dt_proj, A_log, D 参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A参数（初始化为负数以确保稳定性）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.register_buffer("A", -torch.exp(self.A_log.float()))
        
        # D参数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        x: (batch, seqlen, dim)
        Returns: (batch, seqlen, dim)
        """
        batch, seqlen, dim = x.shape
        
        # 输入投影 -> (batch, seqlen, d_inner * 2)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (batch, seqlen, d_inner)
        
        # 卷积 (需要转置到 (batch, d_inner, seqlen))
        x = x.transpose(1, 2)  # (batch, d_inner, seqlen)
        x = self.conv1d(x)[..., :seqlen]  # 截断padding
        x = x.transpose(1, 2)  # (batch, seqlen, d_inner)
        
        # SiLU激活
        x = F.silu(x)
        
        # SSM参数
        B_C = self.x_proj(x)  # (batch, seqlen, d_state * 2)
        B, C = B_C.chunk(2, dim=-1)  # (batch, seqlen, d_state)
        
        # 时间步长
        dt = self.dt_proj(x)  # (batch, seqlen, d_inner)
        dt = F.softplus(dt)
        
        # 状态空间计算（简化版本）
        y = self.selective_scan(x, dt, self.A, B, C, self.D)
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return self.dropout(output)
    
    def selective_scan(self, u, dt, A, B, C, D):
        """
        选择性扫描算法（简化版本）
        """
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[1]
        
        # 重新计算A以确保维度匹配
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 离散化
        dt = dt.unsqueeze(-1)  # (batch, seqlen, d_inner, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        
        # 计算离散化参数
        dA = torch.exp(dt * A)  # (batch, seqlen, d_inner, d_state)
        dB = dt * B.unsqueeze(2)  # (batch, seqlen, d_inner, d_state)
        
        # 初始化状态
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        # 扫描
        ys = []
        for i in range(seqlen):
            # 更新状态 - 修复维度匹配问题
            u_i = u[:, i].unsqueeze(-1)  # (batch, d_inner, 1)
            x = dA[:, i] * x + dB[:, i] * u_i  # 正确的维度匹配
            
            # 输出
            C_i = C[:, i].unsqueeze(1)  # (batch, 1, d_state)
            y = torch.sum(x * C_i, dim=-1)  # (batch, d_inner)
            y = y + D * u[:, i]
            ys.append(y)
        
        return torch.stack(ys, dim=1)  # (batch, seqlen, d_inner)

class MambaBlock(nn.Module):
    """Mamba块"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mixer = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model)
        
    def forward(self, x):
        """
        x: (batch, seqlen, d_model)
        """
        # 残差连接
        return x + self.mixer(self.norm(x))

class MambaModel(nn.Module):
    """完整的Mamba模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_model,
                d_state=getattr(config, 'd_state', 16),
                d_conv=getattr(config, 'd_conv', 4),
                expand=getattr(config, 'expand', 2)
            ) for _ in range(config.n_layers)
        ])
        
        # 最终归一化
        self.norm_f = RMSNorm(config.d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重绑定（可选）
        # self.lm_head.weight = self.embedding.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embedding(input_ids)
        
        # 通过所有Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 最终归一化
        x = self.norm_f(x)
        
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

# 添加RMSNorm到torch.nn模块（如果不存在）
if not hasattr(nn, 'RMSNorm'):
    nn.RMSNorm = RMSNorm 