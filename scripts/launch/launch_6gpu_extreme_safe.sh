#!/bin/bash

echo "🚀 极端安全模式：6GPU训练启动脚本"
echo "================================================"

# 设置最大内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:16"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# 设置CPU使用
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache(); print('✅ GPU缓存已清理')"

echo "📊 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🔧 使用极端优化配置..."
echo "  - Batch Size: 6 (每GPU 1个)"
echo "  - ZeRO-3 + 全CPU卸载"
echo "  - 32个激活检查点"
echo "  - FP16 初始scale: 8"
echo ""

# 启动训练
deepspeed --num_gpus=6 train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_6gpu_extreme.json \
    --max_seq_length 512 \
    --save_steps 100 \
    --eval_steps 50 \
    --logging_steps 10

echo "✅ 训练完成！" 