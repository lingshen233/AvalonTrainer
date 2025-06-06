#!/bin/bash

# 6GPU显存安全启动脚本
# 使用极限优化配置避免显存溢出

echo "🚀 启动6GPU显存安全训练"
echo "=========================="

# 设置环境变量优化显存使用
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# 清理GPU显存
echo "🧹 清理GPU显存..."
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print(f'✅ 已清理 {torch.cuda.device_count()} 张GPU显存')
else:
    print('❌ CUDA不可用')
"

# 检查显存状态
echo "🔍 检查显存状态..."
python diagnose_memory_issue.py --model_size 7

echo ""
echo "🚀 使用6GPU极限优化配置启动训练..."
echo "配置文件: deepspeed_6gpu.json"
echo "ZeRO阶段: 3 (参数分片)"
echo "CPU卸载: 优化器+参数"
echo "micro_batch_per_gpu: 1"
echo "gradient_accumulation_steps: 8"
echo "有效批大小: 48"
echo ""

# 启动训练
deepspeed --num_gpus=6 train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_6gpu.json \
    --num_gpus 6

echo "✅ 训练完成！" 