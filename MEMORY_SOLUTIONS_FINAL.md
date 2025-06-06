# 🔧 6×32GB GPU 内存溢出终极解决方案

## 📊 问题分析

您的环境：6张32GB显卡，训练7B Mamba模型时出现显存溢出。

**根本原因**：
1. 7B参数模型在FP16下需要~26GB显存（仅模型参数）
2. Mamba的`selective_scan`函数创建大量中间张量
3. 即使使用ZeRO-3，激活值仍占用大量显存

## 🚀 解决方案（按推荐优先级）

### 方案1：极端内存优化（推荐） ⭐⭐⭐⭐⭐

```bash
# 使用我们创建的极端优化配置
./launch_6gpu_extreme_safe.sh
```

**关键优化**：
- ✅ Batch size: 6 (每GPU仅1个样本)
- ✅ ZeRO-3 + 全CPU卸载
- ✅ 分块式`selective_scan`算法
- ✅ 32个激活检查点
- ✅ FP16 initial_scale_power: 8
- ✅ 序列长度: 512

### 方案2：3B轻量版模型 ⭐⭐⭐⭐

```bash
# 使用内存友好的3B模型
deepspeed --num_gpus=6 train_deepspeed.py --preset 3b_mamba_lite --deepspeed_config deepspeed_6gpu_extreme.json
```

**模型优化**：
- d_model: 2560 (vs 4096)
- n_layers: 24 (vs 32) 
- d_state: 8 (vs 16)
- expand: 1.5 (vs 2.0)

### 方案3：环境级内存优化 ⭐⭐⭐

```bash
# 设置最大内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:16"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# 使用原配置但减少序列长度
deepspeed --num_gpus=6 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_6gpu.json --max_seq_length 256
```

## 📁 重要文件说明

### 🔧 优化后的模型文件
- `models/mamba.py`: 实现分块式`selective_scan`算法
- `fix_dtype_mismatch.py`: FP16数据类型修复工具

### ⚙️ DeepSpeed配置文件
- `deepspeed_6gpu_extreme.json`: 极端内存优化配置
- `deepspeed_6gpu_fp16_safe.json`: FP16安全配置
- `deepspeed_6gpu.json`: 标准6GPU配置

### 🚀 启动脚本
- `launch_6gpu_extreme_safe.sh`: 极端安全模式启动脚本
- `launch_6gpu_memory_safe.sh`: 标准内存安全启动脚本

### 🔍 诊断工具
- `diagnose_7b_memory.py`: 专门的7B模型内存诊断工具
- `diagnose_memory_issue.py`: 通用内存诊断工具

## 📋 配置对比表

| 配置类型 | Batch Size | micro_batch | ZeRO Stage | CPU Offload | 内存需求 |
|----------|------------|-------------|------------|-------------|----------|
| 极端优化 | 6          | 1           | 3          | 全部        | 最低     |
| 标准配置 | 48         | 1           | 3          | 全部        | 中等     |
| 原始配置 | 48         | 4           | 3          | 部分        | 最高     |

## 🔍 实时诊断

运行诊断工具检查当前状态：
```bash
python diagnose_7b_memory.py
```

## 🎯 成功指标

**训练成功的标志**：
- ✅ DeepSpeed初始化成功
- ✅ 第一个forward pass完成
- ✅ 梯度计算和反向传播完成
- ✅ 优化器步骤完成

## 🆘 如果仍然失败

### Plan B: 进一步减小模型
```bash
# 使用1B参数版本（需要创建配置）
deepspeed --num_gpus=6 train_deepspeed.py --preset 1b_mamba --deepspeed_config deepspeed_6gpu_extreme.json
```

### Plan C: 减少GPU数量
```bash
# 使用4张GPU
deepspeed --num_gpus=4 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_4gpu.json
```

### Plan D: 8bit量化
```bash
# 启用8bit训练（需要修改配置）
pip install bitsandbytes
```

## 📈 内存使用预估

| 模型大小 | 参数量 | FP16显存 | +梯度 | +优化器 | +激活值 | 总计 |
|----------|--------|----------|-------|---------|---------|------|
| 7B       | 7B     | 13GB     | 26GB  | 52GB    | 8GB     | 60GB |
| 3B       | 3B     | 6GB      | 12GB  | 24GB    | 4GB     | 30GB |
| 1B       | 1B     | 2GB      | 4GB   | 8GB     | 2GB     | 12GB |

## 💡 关键经验

1. **显存不够时优先级**：减小batch size > 减小模型 > 减少GPU
2. **Mamba特殊性**：`selective_scan`函数是显存瓶颈
3. **ZeRO-3必须**：对于7B模型，ZeRO-3 + CPU卸载是必需的
4. **环境变量重要**：`PYTORCH_CUDA_ALLOC_CONF`显著影响内存使用

## 🎉 预期结果

使用极端优化配置后，您应该能够：
- ✅ 成功启动7B Mamba模型训练
- ✅ 每个GPU使用约25-28GB显存
- ✅ 稳定进行训练而无OOM错误
- ✅ 获得合理的训练速度（虽然比理想情况慢）

---

**立即行动**: 运行 `./launch_6gpu_extreme_safe.sh` 开始训练！ 