#!/bin/bash

# DeepSpeed ZeRO优化训练启动脚本
# 支持多GPU训练真正的7B模型

set -e

echo "🚀 DeepSpeed ZeRO训练启动脚本"
echo "=================================="

# 默认参数
NUM_GPUS=4
PRESET="7b_mamba"
CONFIG_FILE="config_7b_mamba.yaml"
MASTER_PORT=29500

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --num_gpus NUM    GPU数量 (默认: 4)"
            echo "  --preset PRESET   预设配置 (默认: 7b_mamba)"
            echo "  --config CONFIG   配置文件 (默认: config_7b_mamba.yaml)"
            echo "  --port PORT       通信端口 (默认: 29500)"
            echo "  --help           显示帮助"
            echo ""
            echo "示例:"
            echo "  $0 --num_gpus 4 --preset 7b_mamba"
            echo "  $0 --num_gpus 8 --config config_7b_mamba.yaml"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "❌ 请求 $NUM_GPUS 个GPU，但只有 $AVAILABLE_GPUS 个可用"
    exit 1
fi

echo "📊 训练配置:"
echo "   GPU数量: $NUM_GPUS"
echo "   预设配置: $PRESET"
echo "   配置文件: $CONFIG_FILE"
echo "   通信端口: $MASTER_PORT"
 
# 检查DeepSpeed安装
if ! python -c "import deepspeed" 2>/dev/null; then
    echo "❌ DeepSpeed未安装，正在安装..."
    pip install deepspeed
    if [ $? -ne 0 ]; then
        echo "❌ DeepSpeed安装失败"
        exit 1
    fi
    echo "✅ DeepSpeed安装完成"
fi

# 检查配置文件（如果使用配置文件模式）
if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 显存检查
echo ""
echo "🔍 GPU显存检查:"
python -c "
import torch
if torch.cuda.is_available():
    for i in range($NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1e9
        print(f'  GPU {i}: {props.name} - {total_gb:.1f}GB')
        if total_gb < 20:
            print(f'    ⚠️  显存可能不足')
        else:
            print(f'    ✅ 显存充足')
else:
    print('❌ 未检测到CUDA')
    exit(1)
"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT

echo ""
echo "🌍 环境变量:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"

# 清理之前的进程
echo ""
echo "🧹 清理GPU进程..."
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print('✅ GPU缓存已清理')
"

# 构建DeepSpeed命令
if [ -n "$PRESET" ]; then
    TRAINING_CMD="train_deepspeed.py --preset $PRESET --num_gpus $NUM_GPUS"
else
    TRAINING_CMD="train_deepspeed.py --config $CONFIG_FILE --num_gpus $NUM_GPUS"
fi

echo ""
echo "🚀 启动DeepSpeed训练..."
echo "命令: deepspeed --num_gpus=$NUM_GPUS $TRAINING_CMD"
echo ""

# 启动训练
deepspeed --num_gpus=$NUM_GPUS $TRAINING_CMD

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DeepSpeed训练完成！"
    echo "📁 检查点保存在: ./checkpoints/"
    echo "📊 输出保存在: ./outputs/"
else
    echo ""
    echo "❌ DeepSpeed训练失败"
    echo "💡 建议:"
    echo "   1. 检查GPU显存是否足够"
    echo "   2. 减少批大小或GPU数量"
    echo "   3. 查看错误日志"
    exit 1
fi 