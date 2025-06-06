# 🎮 单卡A800-80GB训练指南

## 🎯 A800优势

相比6×32GB GPU环境，单卡A800-80GB有显著优势：

- ✅ **80GB超大显存**：足够容纳7B模型 + 梯度 + 激活值
- ✅ **无通信开销**：单GPU无需分布式通信
- ✅ **配置简单**：无需复杂的ZeRO-3设置
- ✅ **稳定性高**：避免多GPU同步问题
- ✅ **调试方便**：单进程更容易调试

## 🚀 快速开始

### 1. 运行诊断工具
```bash
python diagnose_a800_80g.py
```

### 2. 使用专用启动脚本
```bash
./launch_single_a800_80g.sh
```

### 3. 手动启动（如果需要自定义参数）
```bash
python train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_single_a800_80g.json \
    --max_seq_length 1024 \
    --save_steps 500 \
    --eval_steps 250
```

## ⚙️ A800专用配置说明

### 核心配置 (`deepspeed_single_a800_80g.json`)

```json
{
    "train_batch_size": 16,           # 较大的batch size
    "train_micro_batch_size_per_gpu": 4,  # 单GPU可以承受更大批次
    "gradient_accumulation_steps": 4,  # 适度的梯度累积
    
    "zero_optimization": {
        "stage": 2,                   # ZeRO-2足够，避免ZeRO-3开销
        "offload_optimizer": {        # 仅优化器CPU卸载
            "device": "cpu"
        }
        # 不卸载参数，保持在GPU获得最佳性能
    },
    
    "activation_checkpointing": {
        "partition_activations": false,  # 关闭激活分区
        "cpu_checkpointing": false,     # 关闭CPU检查点
        "number_checkpoints": 4         # 轻度检查点
    }
}
```

## 📊 内存使用对比

| 组件 | 6×32GB (极端优化) | A800-80GB (推荐) | A800-80GB (激进) |
|------|------------------|------------------|------------------|
| Batch Size | 6 (每GPU 1) | 16 (单GPU 4) | 32 (单GPU 8) |
| 序列长度 | 512 | 1024 | 1024 |
| ZeRO Stage | 3 + 全卸载 | 2 + 优化器卸载 | 2 + 优化器卸载 |
| 激活检查点 | 32个 | 4个 | 2个 |
| 预计显存 | ~28GB/GPU | ~45GB | ~65GB |
| 训练速度 | 慢 | 快 | 最快 |

## 🔧 配置建议

### 保守配置（稳定优先）
```bash
python train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_single_a800_80g.json \
    --max_seq_length 1024
```

### 激进配置（性能优先）
修改 `deepspeed_single_a800_80g.json`：
```json
{
    "train_micro_batch_size_per_gpu": 8,  # 增加到8
    "gradient_accumulation_steps": 2,     # 减少累积步数
    "activation_checkpointing": {
        "number_checkpoints": 2           # 更少检查点
    }
}
```

### 超激进配置（充分利用80GB）
```json
{
    "train_micro_batch_size_per_gpu": 12, # 更大批次
    "max_seq_length": 2048,               # 更长序列
    "zero_optimization": {
        "stage": 1,                       # 甚至可以用ZeRO-1
        "offload_optimizer": {"device": "cpu"}
    }
}
```

## 💡 优化技巧

### 1. 充分利用大显存
- 使用更大的 `micro_batch_size_per_gpu` (4-12)
- 启用完整的序列长度 (1024-2048)
- 减少不必要的激活检查点

### 2. 提高训练效率
- 使用ZeRO-2而非ZeRO-3（减少开销）
- 仅优化器CPU卸载，参数保留GPU
- 增加CPU线程数 (`OMP_NUM_THREADS=8`)

### 3. 如果显存还有余量
- 尝试增加模型复杂度（`d_state=32`, `expand=3`）
- 使用更大的学习率
- 启用更复杂的优化器设置

## 🔍 问题排查

### 如果出现OOM
1. 降低 `micro_batch_size_per_gpu`: 4 → 2 → 1
2. 减少序列长度: 1024 → 512 → 256
3. 启用更多激活检查点: 4 → 8 → 16
4. 升级到ZeRO-3配置

### 如果训练速度慢
1. 增加 `micro_batch_size_per_gpu`
2. 减少激活检查点
3. 关闭不必要的CPU卸载
4. 检查是否有CPU瓶颈

## 📈 性能预期

使用A800-80GB训练7B Mamba模型：

- **内存使用**: 40-60GB (根据配置)
- **训练速度**: 比6×32GB快2-3倍
- **稳定性**: 极高（无分布式问题）
- **调试便利性**: 优秀（单进程）

## 🎉 总结

A800-80GB是训练7B Mamba模型的理想选择：
- 无需复杂的分布式配置
- 可以使用更大的batch size和序列长度
- 训练速度快且稳定
- 充分利用大显存优势

立即开始：`./launch_single_a800_80g.sh` 🚀 