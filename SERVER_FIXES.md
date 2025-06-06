# 🔧 服务器端A800训练修复指南

由于网络问题无法git pull，请手动应用以下修复：

## 🚨 紧急修复：批次大小错误

**当前错误**: `train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 32 != 1 * 8 * 1`

**立即解决方案**:

```bash
cd ~/autodl-tmp/AvalonTrainer

# 运行批次大小修复工具
python fix_deepspeed_batch_size.py --num_gpus 1 --config deepspeed_single_a800_80g.json

# 或者手动修复配置文件
nano deepspeed_single_a800_80g.json
```

**确保deepspeed_single_a800_80g.json内容为**:
```json
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    ...
}
```

**验证修复**: `4 × 4 × 1 = 16` ✅

## 1. 修复 configs/config_presets.py

**问题**: SyntaxError: illegal target for annotation

**解决方案**: 完全替换文件内容为：

```python
"""
配置预设
"""

from .base import ModelConfig

CONFIG_PRESETS = {
    '7b_mamba': ModelConfig(
        # 基础参数
        vocab_size=50257,
        d_model=4096,
        n_layers=32,
        max_seq_length=1024,
        
        # Mamba特定参数
        d_state=16,
        d_conv=4,
        expand=2,
        
        # 训练参数
        learning_rate=5e-4,
        batch_size=32,
        train_micro_batch_size_per_gpu=4,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        warmup_steps=2000,
        max_steps=100000,
        
        # 正则化
        dropout=0.1,
        
        # 优化器
        optimizer_type='adamw',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        
        # 保存和日志
        save_steps=1000,
        eval_steps=500,
        logging_steps=10,
        
        # 其他
        fp16=True,
        gradient_checkpointing=True
    ),
    
    '3b_mamba_lite': ModelConfig(
        # 基础参数（内存友好版本）
        vocab_size=50257,
        d_model=2560,          # 减少到2560
        n_layers=24,           # 减少到24层
        max_seq_length=1024,
        
        # Mamba特定参数（优化版本）
        d_state=8,             # 从16减少到8
        d_conv=4,
        expand=1.5,            # 从2减少到1.5
        
        # 训练参数
        learning_rate=3e-4,
        batch_size=32,
        train_micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=100000,
        
        # 正则化
        dropout=0.1,
        
        # 优化器
        optimizer_type='adamw',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        
        # 保存和日志
        save_steps=1000,
        eval_steps=500,
        logging_steps=10,
        
        # 其他
        fp16=True,
        gradient_checkpointing=True
    ),
}
```

## 2. 修复 launch_single_a800_80g.sh

**问题**: unrecognized arguments

**解决方案**: 将启动命令改为：

```bash
# 原来的错误命令
python train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_single_a800_80g.json \
    --max_seq_length 1024 \
    --save_steps 500 \
    --eval_steps 250 \
    --logging_steps 50

# 修改为正确的命令
deepspeed --num_gpus=1 train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_single_a800_80g.json
```

## 3. 应用修复的完整步骤

```bash
cd ~/autodl-tmp/AvalonTrainer

# 1. 修复批次大小问题 (最重要!)
python fix_deepspeed_batch_size.py --num_gpus 1

# 2. 修复配置文件
nano configs/config_presets.py
# 将上面的Python代码完全替换原文件内容

# 3. 修复启动脚本
nano launch_single_a800_80g.sh
# 找到最后的启动命令，按照上面的方式修改

# 4. 设置权限
chmod +x launch_single_a800_80g.sh

# 5. 测试诊断工具
python diagnose_a800_80g.py

# 6. 启动训练
./launch_single_a800_80g.sh
```

## 4. 验证修复成功

修复后应该看到：
- ✅ 批次大小验证通过: `16 = 4 × 4 × 1` 
- ✅ configs/config_presets.py 无语法错误
- ✅ diagnose_a800_80g.py 正常运行
- ✅ launch_single_a800_80g.sh 无参数错误
- ✅ 训练正常开始

## 5. 快速测试命令

```bash
# 测试配置文件
python -c "from configs.config_presets import CONFIG_PRESETS; print('✅ 配置文件正常')"

# 修复批次大小
python fix_deepspeed_batch_size.py --num_gpus 1

# 测试A800诊断
python diagnose_a800_80g.py

# 直接启动训练
deepspeed --num_gpus=1 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_single_a800_80g.json
```

## 6. 如果还有问题

备用方案：
```bash
# 使用标准1GPU配置
deepspeed --num_gpus=1 train_deepspeed.py --preset 7b_mamba --deepspeed_config deepspeed_1gpu.json
```

## 7. 手动批次大小修复

如果修复工具不起作用，手动编辑 `deepspeed_single_a800_80g.json`:

```json
{
    "train_batch_size": 16,              // 确保这个等于下面的乘积
    "train_micro_batch_size_per_gpu": 4, // A800可以承受4
    "gradient_accumulation_steps": 4,    // 4 × 4 × 1 = 16
    ...
}
```

现在您的A800应该可以正常开始训练了！ 