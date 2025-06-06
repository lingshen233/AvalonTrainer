#!/bin/bash

echo "🚀 单卡A800-80GB训练启动脚本"
echo "================================================"

# 检查是否为A800/A100
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | head -n1)
echo "🎮 检测到GPU: $GPU_NAME"

if [[ ! "$GPU_NAME" =~ "A800" ]] && [[ ! "$GPU_NAME" =~ "A100" ]]; then
    echo "⚠️  警告: 检测到的GPU不是A800/A100，可能需要调整配置"
fi

# 设置内存优化环境变量（相对温和，利用大显存优势）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

# 设置CPU使用
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache(); print('✅ GPU缓存已清理')"

echo "📊 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🔧 使用单卡A800-80G优化配置..."
echo "  - Batch Size: 16 (每GPU 4个样本)"
echo "  - ZeRO-2 + 优化器CPU卸载"
echo "  - 适度激活检查点"
echo "  - FP16标准设置"
echo "  - 序列长度: 1024 (充分利用大显存)"
echo ""

# 检查配置文件是否存在
if [ ! -f "deepspeed_single_a800_80g.json" ]; then
    echo "❌ 配置文件 deepspeed_single_a800_80g.json 不存在！"
    exit 1
fi

# 启动训练 - 单GPU不需要deepspeed启动器
python train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_single_a800_80g.json \
    --max_seq_length 1024 \
    --save_steps 500 \
    --eval_steps 250 \
    --logging_steps 50

echo "✅ 训练完成！" 