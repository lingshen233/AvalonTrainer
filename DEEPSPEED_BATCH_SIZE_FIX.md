# DeepSpeed批次大小问题快速修复指南

## 🎯 问题现象
```bash
AssertionError: Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 16 != 4 * 8 * 4
```

## 🔧 问题原因
DeepSpeed要求 `train_batch_size` 必须等于：
```
train_batch_size = micro_batch_per_gpu × gradient_accumulation_steps × world_size
```

用户的错误中：
- `train_batch_size = 16` (配置文件中错误的值)
- `micro_batch_per_gpu = 4`
- `gradient_accumulation_steps = 8` 
- `world_size = 4` (4个GPU)
- **正确值应该是**: `4 × 8 × 4 = 128`

## ⚡ 快速修复方法

### 方法1: 使用自动修复脚本 (推荐)
```bash
# 修复4GPU配置
python fix_deepspeed_batch_size.py --num_gpus 4

# 验证修复结果
python train_deepspeed.py --preset 7b_mamba --num_gpus 4 --dry_run
```

### 方法2: 使用预生成配置
脚本已生成了正确的配置文件：
```bash
# 4GPU训练使用
deepspeed --num_gpus=4 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_4gpu.json

# 或者直接复制配置
cp deepspeed_4gpu.json deepspeed_config.json
```

### 方法3: 手动修复现有配置
如果有现有的 `deepspeed_config.json`，手动修改：
```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8
}
```

## 📊 不同GPU数量的正确配置

| GPU数量 | micro_batch | grad_acc | train_batch_size | 配置文件 |
|---------|-------------|----------|------------------|----------|
| 1       | 2           | 16       | 32               | deepspeed_1gpu.json |
| 2       | 2           | 16       | 64               | deepspeed_2gpu.json |
| 4       | 4           | 8        | 128              | deepspeed_4gpu.json |
| 8       | 4           | 4        | 128              | deepspeed_8gpu.json |

## ✅ 验证修复成功

运行以下命令应该显示：
```bash
python train_deepspeed.py --preset 7b_mamba --num_gpus 4 --dry_run
```

成功的输出：
```
✅ 使用预构建配置: deepspeed_4gpu.json
✅ 配置验证通过: 128 = 4 × 8 × 4

📊 DeepSpeed训练配置:
模型类型: mamba
估算参数: 6.89B
GPU数量: 4
批大小/GPU: 4
梯度累积: 8
有效批大小: 128
ZeRO阶段: 2

✅ 批次大小验证通过: 128 = 4 × 8 × 4
✅ 配置验证完成（dry_run模式）
```

## 🚀 启动训练

修复后，使用以下命令启动训练：
```bash
# 使用启动脚本（推荐）
./launch_deepspeed.sh --num_gpus 4 --preset 7b_mamba

# 或直接使用deepspeed
deepspeed --num_gpus=4 train_deepspeed.py --preset 7b_mamba
```

## 🔍 问题排查

如果仍有问题，检查以下内容：

### 1. 检查配置文件
```bash
# 查看当前配置
cat deepspeed_config.json | grep -E "(train_batch_size|train_micro_batch_size_per_gpu|gradient_accumulation_steps)"
```

### 2. 清理旧配置
```bash
# 删除可能的旧配置文件
rm -f deepspeed_config.json ds_config.json

# 重新生成正确配置
python fix_deepspeed_batch_size.py --num_gpus 4
```

### 3. 检查GPU数量匹配
确保命令中的 `--num_gpus` 参数与实际GPU数量一致：
```bash
nvidia-smi --query-gpu=count --format=csv,noheader,nounits
```

## 🔧 防止再次出现

1. **始终使用修复脚本**：训练前运行 `fix_deepspeed_batch_size.py`
2. **使用预设配置**：优先使用 `--preset` 参数而不是手动配置
3. **验证配置**：训练前总是运行 `--dry_run` 检查

## 📁 相关文件

- `fix_deepspeed_batch_size.py` - 自动修复脚本
- `train_deepspeed.py` - 主训练脚本（已修复）
- `deepspeed_*gpu.json` - 预生成的正确配置
- `debug_deepspeed_config.py` - 调试工具

## 💡 技术原理

DeepSpeed的批次大小公式确保：
- **有效批大小**：所有GPU处理的总样本数
- **显存效率**：每GPU只处理micro_batch样本  
- **梯度累积**：在更新前累积多个mini-batch的梯度
- **多GPU同步**：确保所有GPU的梯度更新同步

正确的配置让DeepSpeed能够：
- 高效地分片优化器状态（ZeRO-2）
- 正确同步梯度更新
- 最大化GPU利用率
- 保持训练稳定性 