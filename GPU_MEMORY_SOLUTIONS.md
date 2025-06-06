# GPU显存问题解决方案指南

## 🚨 问题现象
即使使用6张32GB显卡，在DeepSpeed初始化阶段仍然出现显存溢出：
```
CUDA out of memory. Tried to allocate 6.93 GiB. GPU 5 has a total capacity of 31.50 GiB of which 6.80 GiB is free.
```

## 🔍 问题根本原因

### 1. 批次大小自适应问题
- 训练脚本根据GPU数量自动调整批次大小
- `micro_batch_per_gpu` 设置过大(通常为4)
- 7B参数模型需要大量显存，micro_batch=4会导致溢出

### 2. ZeRO优化不够激进
- 使用ZeRO-2而不是ZeRO-3
- 没有启用CPU参数卸载
- 激活检查点设置不够激进

### 3. 显存碎片化
- PyTorch显存分配器的碎片化问题
- 24.26GB已分配，但只有6.8GB可用

## ⚡ 解决方案

### 方案1: 使用极限优化配置 (推荐)

```bash
# 使用专门为6GPU优化的配置
./launch_6gpu_memory_safe.sh
```

**配置特点：**
- `micro_batch_per_gpu = 1` (最小值)
- `gradient_accumulation_steps = 8` 
- ZeRO-3阶段(参数分片)
- CPU卸载优化器和参数
- 激进的激活检查点(16个)

### 方案2: 环境变量优化

```bash
# 启用显存段扩展，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启用显存垃圾回收
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
```

### 方案3: 手动清理显存

```bash
# 训练前清理所有GPU显存
python -c "
import torch
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()
print('显存已清理')
"
```

### 方案4: 诊断并自动修复

```bash
# 诊断显存问题并自动生成配置
python diagnose_memory_issue.py --fix --model_size 7

# 验证配置
python train_deepspeed.py --preset 7b_mamba --num_gpus 6 --dry_run
```

## 📊 不同GPU数量的推荐配置

| GPU数量 | micro_batch | grad_acc | train_batch | ZeRO阶段 | CPU卸载 | 配置文件 |
|---------|-------------|----------|-------------|----------|---------|----------|
| 1       | 1           | 32       | 32          | 2        | 是      | deepspeed_1gpu.json |
| 2       | 1           | 16       | 32          | 2        | 是      | deepspeed_2gpu.json |
| 4       | 1           | 8        | 32          | 2        | 是      | deepspeed_4gpu.json |
| **6**   | **1**       | **8**    | **48**      | **3**    | **是**  | **deepspeed_6gpu.json** |
| 8       | 1           | 4        | 32          | 3        | 是      | deepspeed_8gpu.json |

## 🔧 关键优化技术

### ZeRO-3配置
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
    },
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9
  }
}
```

### 激进激活检查点
```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 16,
    "synchronize_checkpoint_boundary": true
  }
}
```

## ✅ 验证配置正确性

### 1. 批次大小公式验证
```
train_batch_size = micro_batch_per_gpu × gradient_accumulation_steps × world_size
48 = 1 × 8 × 6 ✓
```

### 2. 显存需求估算
```
7B模型 ZeRO-3 每GPU需求: ~3-4GB
激活值: ~2GB
总需求: ~6GB per GPU (安全范围)
```

### 3. 配置验证命令
```bash
# 验证配置不启动实际训练
python train_deepspeed.py --preset 7b_mamba --num_gpus 6 --dry_run --deepspeed_config deepspeed_6gpu.json
```

## 🚀 推荐启动流程

```bash
# 1. 清理显存
python clear_gpu_processes.sh

# 2. 诊断问题
python diagnose_memory_issue.py --model_size 7

# 3. 生成优化配置
python fix_deepspeed_batch_size.py --num_gpus 6 --generate

# 4. 验证配置
python train_deepspeed.py --preset 7b_mamba --num_gpus 6 --dry_run

# 5. 启动训练
./launch_6gpu_memory_safe.sh
```

## 🔍 问题排查检查清单

- [ ] micro_batch_per_gpu = 1
- [ ] ZeRO阶段 = 3  
- [ ] CPU卸载已启用
- [ ] 激活检查点数量 >= 8
- [ ] 环境变量已设置
- [ ] GPU显存已清理
- [ ] 批次大小公式验证通过
- [ ] dry_run测试通过

## 💡 性能权衡说明

**优点：**
- 解决显存溢出问题
- 支持更大模型训练
- 训练过程稳定

**缺点：**
- 训练速度略慢(CPU卸载开销)
- 通信开销增加(ZeRO-3)
- micro_batch=1可能影响收敛速度

**建议：**
- 显存充足后可逐步增加micro_batch_size
- 监控训练损失，确保收敛正常
- 根据实际情况调整gradient_accumulation_steps 