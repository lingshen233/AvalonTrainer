# RAG Transformer - 多GPU训练框架

一个简洁高效的多GPU深度学习训练框架，专注于Transformer和Mamba模型训练。

## 🚀 特性

- **双模型支持**: Transformer和Mamba状态空间模型
- **多GPU训练**: 支持1-8个GPU并行训练（PyTorch DDP）
- **智能批大小**: 自动根据GPU数量和模型类型优化批大小
- **YAML配置**: 简单直观的配置文件系统
- **显存优化**: 自动估算显存需求，避免OOM错误
- **即开即用**: 预配置的训练脚本，快速上手
- **自动关机**: 训练完成后可自动关机（可选）

## 📋 环境要求

- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+
- 支持的GPU: RTX 3090/4090, A100等

## 🛠️ 安装

```bash
# 克隆项目
git clone <repository-url>
cd "RAG Transformer"

# 安装依赖
pip install -r requirements.txt
```

## 🎯 快速开始

### 1. 环境检查
```bash
# 快速验证环境和依赖
python quick_test.py
```

### 2. 查看可用配置
```bash
# 查看预设模型配置
python train.py --list_presets

# 查看可用数据集
python train.py --list_datasets

# 查看可用模型
python train.py --list_models
```

### 3. 选择训练规模
```bash
# 1B模型训练 (适合单卡RTX 4090)
python train.py --preset 1b_transformer

# 7B模型训练 (需要4卡以上)
python train.py --preset 7b_transformer --num_gpus 4

# Mamba模型 (显存效率更高)
python train.py --preset 1b_mamba
```

### 4. 数据集管理
```bash
# 浏览所有可用数据集
python list_datasets.py

# 查看1B模型推荐数据集
python list_datasets.py --recommend 1B

# 查看7B模型推荐数据集
python list_datasets.py --recommend 7B

# 下载指定数据集
python list_datasets.py --download wikitext
```

### 5. 测试GPU设置
```bash
python test_multi_gpu.py
```

### 6. 验证配置
```bash
python train.py --dry_run
```

### 7. 开始训练

**单GPU训练（默认）：**
```bash
python train.py
```

**多GPU训练：**
```bash
python train.py --config config_transformer_4gpu.yaml --num_gpus 4
```

**命令行覆盖配置：**
```bash
python train.py --model_type transformer --num_gpus 2
```

**启用自动关机训练：**
```bash
# 修改config.yaml中 system.auto_shutdown: true
python train.py
```

### 8. 基准测试
```bash
# 下载标准数据集和预训练模型进行测试
python test_benchmark.py

# 仅下载数据集
python test_benchmark.py --datasets-only

# 仅下载预训练模型
python test_benchmark.py --models-only

# 快速测试训练完成的模型
python test_after_training.py

# 测试指定的模型检查点
python test_after_training.py --checkpoint checkpoints/best_model.pt
```

## 📊 模型对比

| 预设配置 | 模型类型 | 参数量 | 估算显存 | 推荐GPU | 适用场景 |
|----------|----------|--------|----------|---------|----------|
| **1b_transformer** | Transformer | 1.0B | 12GB/GPU | RTX 4090×1 | 通用语言建模 |
| **1b_mamba** | Mamba | 1.0B | 9GB/GPU | RTX 3090×1 | 高效长序列处理 |
| **7b_transformer** | Transformer | 7.0B | 28GB/GPU | RTX 4090×4 | 大规模语言建模 |
| **7b_mamba** | Mamba | 7.0B | 20GB/GPU | RTX 4090×2 | 高效大模型训练 |
| **test_small** | Transformer | 50M | 2GB/GPU | 任意GPU | 快速测试验证 |

### 数据集推荐

| 模型规模 | 英文数据集 | 中文数据集 | 总大小 | 训练时间估算 |
|----------|------------|------------|--------|--------------|
| **1B** | WikiText + BookCorpus + CC-News | 中文网页文本 | ~80GB | 3-5天 |
| **7B** | OpenWebText + C4 + The Pile | 中文网页文本 | ~1TB+ | 2-3周 |

## ⚙️ 配置文件

### 默认配置 (config.yaml)
```yaml
# 基础设置
model_type: "mamba"  # transformer 或 mamba
num_gpus: 1          # GPU数量

# 模型配置
model:
  d_model: 1536      # 模型维度
  n_layers: 24       # 层数
  dropout: 0.1

# 训练配置
training:
  batch_size: null   # null=自动计算
  max_steps: 50000
  learning_rate: 3e-4
  fp16: true
  output_dir: "./outputs"
  checkpoint_dir: "./checkpoints"

# 系统配置
system:
  auto_shutdown: false  # 训练完成后自动关机
  shutdown_delay: 60    # 关机前等待时间（秒）
```

### 多GPU配置示例
```yaml
model_type: "transformer"
num_gpus: 4

model:
  d_model: 1536
  n_layers: 24

training:
  batch_size: 8      # 每个GPU的批大小
  gradient_accumulation_steps: 2

system:
  auto_shutdown: true  # 长时间训练后自动关机
  shutdown_delay: 60
```

## 💾 模型文件保存

### 保存位置
训练完成后，模型文件将自动保存到：

```
RAG Transformer/
├── checkpoints/           # 主要模型文件
│   ├── final_model.pt    # 最终训练完成的模型
│   ├── best_model.pt     # 验证集上表现最好的模型
│   └── checkpoint_step_*.pt  # 定期检查点
└── outputs/              # 日志和输出文件
```

### 模型文件说明

1. **final_model.pt**: 训练完成时的最终模型状态
2. **best_model.pt**: 验证集上损失最低的模型（推荐使用）
3. **checkpoint_step_*.pt**: 每5000步保存的检查点（用于断点续训）

详细说明请参考：[MODEL_FILES.md](MODEL_FILES.md)

### 加载模型
```python
import torch
from models import create_model
from configs.base import ModelConfig

# 加载最终模型
checkpoint = torch.load('checkpoints/final_model.pt')
model_config = ModelConfig(**checkpoint['config'])
model = create_model(model_config.model_type, model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🔧 配置选项

### 模型类型
- **transformer**: 标准Transformer架构，通用性好
- **mamba**: 高效Mamba状态空间模型，显存占用更少

### GPU配置
- `num_gpus: 1`: 单GPU训练
- `num_gpus: 2-8`: 多GPU并行训练

### 批大小策略
- `batch_size: null`: 自动优化（推荐）
- `batch_size: 8`: 手动指定

### 自动关机
- `auto_shutdown: false`: 禁用自动关机（默认）
- `auto_shutdown: true`: 训练完成后自动关机
- `shutdown_delay: 60`: 关机前等待时间

## 📁 项目结构

```
RAG Transformer/
├── train.py                    # 主训练脚本
├── test_multi_gpu.py           # GPU测试工具
├── quick_test.py               # 快速环境测试
├── test_benchmark.py           # 基准测试脚本
├── test_after_training.py      # 训练后快速测试
├── list_datasets.py            # 数据集浏览工具
├── config.yaml                 # 默认配置
├── config_transformer_4gpu.yaml # 多GPU示例配置
├── config_1b_transformer.yaml  # 1B模型配置
├── config_7b_transformer.yaml  # 7B模型配置
├── requirements.txt            # 依赖包列表
├── configs/                    # 配置系统
│   ├── base.py                 # 基础配置类
│   ├── presets.py              # 预设配置
│   ├── model_presets.py        # 模型规模预设
│   └── registry.py             # 配置注册
├── models/                     # 模型实现
│   ├── transformer.py          # Transformer模型
│   ├── mamba.py                # Mamba模型
│   └── registry.py             # 模型注册
├── trainers/                   # 训练器
│   ├── base.py                 # 基础训练器
│   └── multi_gpu.py            # 多GPU训练器
├── data/                       # 数据处理
│   ├── processor.py            # 数据处理器
│   └── dataset_manager.py      # 数据集管理器
├── utils/                      # 工具函数
│   └── logging.py              # 日志工具
├── test_results/               # 测试结果目录（自动创建）
└── data_cache/                 # 数据集缓存目录（自动创建）
```

## 📈 性能优化

### 自动优化
- 根据GPU数量自动调整批大小
- 智能梯度累积设置
- 混合精度训练（FP16）

### 显存管理
- 训练前显存需求估算
- 自动批大小计算
- OOM错误预防

### 多GPU加速
- PyTorch DistributedDataParallel (DDP)
- 自动设备分配
- 高效进程间通信

## ⚡ 自动关机功能

### 启用方式
1. 修改 `config.yaml`：
   ```yaml
   system:
     auto_shutdown: true
     shutdown_delay: 60
   ```

2. 或使用命令行：
   ```bash
   python train.py --no_shutdown  # 禁用自动关机
   ```

### 训练完成流程
1. ✅ 训练完成
2. 💾 自动保存模型到 `checkpoints/final_model.pt`
3. 📊 显示完整的模型保存路径
4. ⏰ 开始倒计时（默认60秒）
5. 💤 执行关机命令

### 取消关机
- **按 Ctrl+C**: 在倒计时期间取消自动关机
- **命令行**: 使用 `--no_shutdown` 参数

## 🛠️ 故障排除

### 显存不足
```yaml
training:
  batch_size: 2                    # 减少批大小
  gradient_accumulation_steps: 16  # 增加梯度累积
```

### 多GPU问题
- 确保所有GPU型号一致
- 检查CUDA和NCCL版本
- 验证GPU间通信带宽

### 训练速度慢
- 使用更大的批大小
- 启用FP16混合精度
- 优化数据加载器worker数量

### 自动关机失败
- **Windows**: 确保运行在管理员权限
- **Linux/macOS**: 确保sudo权限或配置免密sudo

## 📊 使用示例

### 适合3090单卡的配置
```bash
# Mamba模型，显存友好
python train.py --model_type mamba
```

### 适合4090多卡的配置  
```bash
# Transformer模型，4GPU并行，训练完成后自动关机
python train.py --config config_transformer_4gpu.yaml
```

### 自定义大模型训练
```yaml
model_type: "transformer"
num_gpus: 8

model:
  d_model: 2048
  n_layers: 32

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  max_steps: 100000

system:
  auto_shutdown: true
  shutdown_delay: 120  # 2分钟倒计时
```

## 🚀 命令行参数

```bash
python train.py [OPTIONS]

选项:
  --config PATH          配置文件路径 (默认: config.yaml)
  --model_type TEXT      模型类型 (transformer/mamba)
  --num_gpus INTEGER     GPU数量
  --list_models          列出可用模型
  --dry_run              验证配置但不训练
  --no_shutdown          禁用自动关机
  --help                 显示帮助信息
```

## 🔍 GPU测试工具

```bash
# 全面测试GPU设置和配置
python test_multi_gpu.py

# 只检查GPU信息
python test_multi_gpu.py --skip_config
```

输出示例：
```
🚀 RAG Transformer 多GPU训练测试
==================================================
🔍 GPU信息检测:
CUDA可用: True
GPU数量: 4
  GPU 0: NVIDIA GeForce RTX 4090 (24.0GB)
  GPU 1: NVIDIA GeForce RTX 4090 (24.0GB)
  GPU 2: NVIDIA GeForce RTX 4090 (24.0GB)
  GPU 3: NVIDIA GeForce RTX 4090 (24.0GB)

💡 使用建议:
- 可以使用4GPU并行训练
- 修改配置文件中的 num_gpus 参数
- 批大小会自动调整以适应多GPU
```

## 📝 开发指南

### 添加新模型
1. 在 `models/` 目录下实现模型类
2. 在 `models/registry.py` 中注册模型
3. 更新配置系统支持新模型参数

### 自定义训练逻辑
1. 继承 `trainers/base.py` 中的 `BaseTrainer`
2. 重写 `train_step()` 和 `evaluate()` 方法
3. 在训练脚本中使用自定义训练器

## 🤝 贡献

欢迎提交Issue和Pull Request！

## �� 许可证

MIT License 