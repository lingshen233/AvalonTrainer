# 🚀 RAG Transformer - 大模型训练框架

一个专为大规模语言模型训练设计的高效框架，支持Transformer和Mamba架构，集成DeepSpeed ZeRO优化技术。

## ✨ 核心特性

### 🎯 模型架构支持
- **Transformer**: 经典注意力机制，支持7B参数规模
- **Mamba**: 状态空间模型，内存效率更高，训练速度更快

### ⚡ 训练优化
- **DeepSpeed ZeRO**: 支持ZeRO-1/2/3优化策略
- **混合精度**: FP16训练，显著降低显存占用
- **梯度累积**: 支持大批次训练
- **激活检查点**: 进一步节省显存

### 🔧 硬件适配
- **多GPU支持**: 1-8卡分布式训练
- **A800优化**: 专门针对80GB显存优化
- **内存管理**: 智能显存清理和批次大小调整

### 🛠️ 开发工具
- **诊断工具**: 自动检测硬件配置和内存使用
- **修复工具**: 自动修复配置错误
- **预设配置**: 开箱即用的模型配置

## 📁 项目结构

```
RAG Transformer/
├── 🐍 train.py                     # 主训练脚本 (DeepSpeed优化)
├── 📄 README.md                    # 项目文档
├── 📄 requirements.txt              # 依赖包列表
├── 📄 .gitignore                   # Git忽略文件
│
├── 📁 configs/                     # 配置文件
│   ├── 📁 deepspeed/              # DeepSpeed配置 (11个预设)
│   │   ├── deepspeed_1gpu.json    # 单卡配置
│   │   ├── deepspeed_single_a800_80g.json  # A800专用
│   │   └── deepspeed_*gpu.json    # 多卡配置
│   ├── 🐍 *.py                    # Python配置模块
│   └── 📄 *.yaml                  # 模型配置文件
│
├── 📁 scripts/                     # 脚本工具
│   ├── 📁 launch/                 # 启动脚本
│   │   ├── launch_single_a800_80g.sh    # A800启动脚本
│   │   └── launch_*gpu*.sh        # 多GPU启动脚本
│   └── 📄 install_*.sh            # 安装脚本
│
├── 📁 tools/                       # 工具集
│   ├── 📁 diagnostic/             # 诊断工具
│   │   ├── diagnose_a800_80g.py   # A800诊断
│   │   ├── gpu_tflop_calculator.py # 性能计算
│   │   └── memory_calculator.py   # 内存估算
│   ├── 📁 fixes/                  # 修复工具
│   │   ├── fix_deepspeed_batch_size.py  # 批次修复
│   │   └── fix_dtype_mismatch.py  # 数据类型修复
│   └── 🐍 list_datasets.py        # 数据集工具
│
├── 📁 tests/                       # 测试文件
├── 📁 models/                      # 模型定义
├── 📁 utils/                       # 工具函数
├── 📁 trainers/                    # 训练器
└── 📁 data/                        # 数据处理
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd "RAG Transformer"

# 安装依赖
pip install -r requirements.txt

# 安装DeepSpeed (推荐)
bash scripts/install_deepspeed.sh
```

### 2. 硬件检测

```bash
# 检测GPU配置
python tools/diagnostic/diagnose_a800_80g.py

# 计算模型内存需求
python tools/diagnostic/memory_calculator.py
```

### 3. 开始训练

#### 🎯 A800单卡训练 (推荐)
```bash
# 使用A800专用配置
deepspeed --num_gpus=1 train.py \
    --preset 7b_mamba \
    --deepspeed_config configs/deepspeed/deepspeed_single_a800_80g.json

# 或使用启动脚本
bash scripts/launch/launch_single_a800_80g.sh
```

#### 🔥 多GPU训练
```bash
# 4卡训练
deepspeed --num_gpus=4 train.py \
    --preset 7b_mamba \
    --deepspeed_config configs/deepspeed/deepspeed_4gpu.json

# 6卡极限优化
bash scripts/launch/launch_6gpu_extreme_safe.sh
```

#### ⚙️ 自定义配置
```bash
# 使用YAML配置文件
python train.py --config configs/config_7b_mamba.yaml --num_gpus 1
```

## 📊 模型配置

### 🦣 Mamba模型 (推荐)
- **7B参数**: `7b_mamba` - 生产级配置
- **3B轻量**: `3b_mamba_lite` - 内存友好版本

### 🤖 Transformer模型
- **7B参数**: 经典Transformer架构
- **1B-4B**: 多种规模可选

## 🔧 故障排除

### 常见问题

#### 1. 批次大小错误
```bash
# 自动修复
python tools/fixes/fix_deepspeed_batch_size.py --num_gpus 1

# 手动检查
python train.py --dry_run --preset 7b_mamba
```

#### 2. 显存不足
```bash
# 诊断内存使用
python tools/diagnostic/diagnose_7b_memory.py

# 使用轻量配置
python train.py --preset 3b_mamba_lite
```

#### 3. 数据类型错误
```bash
# 修复FP16不匹配
python tools/fixes/fix_dtype_mismatch.py
```

### 🆘 紧急修复

如果遇到配置问题，运行诊断工具：
```bash
# 全面诊断
python tools/diagnostic/diagnose_a800_80g.py --verbose

# 检查GPU状态
python tools/diagnostic/check_gpu_memory.py
```

## 📈 性能优化

### 🎯 A800-80GB优化建议
- **批次大小**: 4-8 (根据序列长度调整)
- **梯度累积**: 4-8步
- **ZeRO阶段**: ZeRO-2 (平衡性能和内存)
- **激活检查点**: 4个检查点

### 🔥 多GPU优化
- **6×32GB**: 使用极限内存优化配置
- **4×40GB**: 标准ZeRO-2配置
- **8×80GB**: 可使用ZeRO-1获得最佳性能

## 🛡️ 最佳实践

### 1. 训练前检查
```bash
# 验证配置
python train.py --dry_run --check_memory

# 清理GPU缓存
bash scripts/clear_gpu_processes.sh
```

### 2. 监控训练
- 使用 `--use_wandb` 启用WandB监控
- 定期检查显存使用情况
- 设置合理的保存间隔

### 3. 错误恢复
- 启用自动检查点保存
- 使用 `--resume_from_checkpoint` 恢复训练
- 定期备份重要检查点

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境
```bash
# 安装开发依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black . && isort .
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练优化
- [Mamba](https://github.com/state-spaces/mamba) - 状态空间模型
- [Transformers](https://github.com/huggingface/transformers) - 模型架构参考

---

**🚀 开始你的大模型训练之旅！**

如有问题，请查看 [故障排除](#-故障排除) 部分或提交Issue。 