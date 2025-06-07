#!/bin/bash

# DeepSpeed安装脚本
# 针对不同操作系统优化安装

set -e

echo "🚀 DeepSpeed安装脚本"
echo "===================="

# 检测操作系统
OS_TYPE=$(uname -s)
echo "🌍 检测到操作系统: $OS_TYPE"

# 检查Python环境
echo "🐍 检查Python环境..."
python --version

# 检查CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "🔥 检测到CUDA: $CUDA_VERSION"
else
    echo "⚠️  未检测到CUDA，将安装CPU版本"
fi

# 检查是否已安装DeepSpeed
if python -c "import deepspeed" 2>/dev/null; then
    echo "✅ DeepSpeed已安装"
    python -c "import deepspeed; print(f'版本: {deepspeed.__version__}')"
    exit 0
fi

echo "📦 开始安装DeepSpeed..."

# 更新pip
echo "📋 更新pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装依赖包..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装DeepSpeed
echo "🚀 安装DeepSpeed..."
case $OS_TYPE in
    "Linux")
        # Linux系统
        echo "🐧 Linux环境安装..."
        
        # 安装编译依赖
        if command -v apt-get &> /dev/null; then
            echo "📦 安装编译依赖 (Ubuntu/Debian)..."
            sudo apt-get update
            sudo apt-get install -y build-essential python3-dev
        elif command -v yum &> /dev/null; then
            echo "📦 安装编译依赖 (CentOS/RHEL)..."
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y python3-devel
        fi
        
        # 直接从PyPI安装
        pip install deepspeed
        ;;
        
    "Darwin")
        # macOS系统
        echo "🍎 macOS环境安装..."
        echo "⚠️  注意: macOS不支持CUDA，将安装CPU版本"
        
        # 安装Xcode命令行工具
        if ! command -v gcc &> /dev/null; then
            echo "📦 安装Xcode命令行工具..."
            xcode-select --install
        fi
        
        # CPU版本安装
        pip install deepspeed
        ;;
        
    *)
        echo "❓ 未知操作系统，尝试通用安装..."
        pip install deepspeed
        ;;
esac

# 验证安装
echo ""
echo "🧪 验证DeepSpeed安装..."
if python -c "import deepspeed; print(f'✅ DeepSpeed {deepspeed.__version__} 安装成功!')" 2>/dev/null; then
    echo "🎉 安装完成！"
    
    # 显示支持的功能
    echo ""
    echo "📊 支持的功能:"
    python -c "
import deepspeed
import torch

print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)')

# 检查DeepSpeed ops
try:
    from deepspeed.ops.adam import FusedAdam
    print('✅ FusedAdam支持')
except:
    print('❌ FusedAdam不支持')

try:
    from deepspeed.ops.transformer import DeepSpeedTransformerLayer
    print('✅ Transformer层优化支持')
except:
    print('❌ Transformer层优化不支持')
"

else
    echo "❌ DeepSpeed安装失败"
    echo "💡 常见解决方案:"
    echo "1. 检查CUDA版本兼容性"
    echo "2. 更新PyTorch版本"
    echo "3. 安装编译工具链"
    exit 1
fi

echo ""
echo "🎯 下一步:"
echo "1. 运行 ./launch_deepspeed.sh --help 查看训练选项"
echo "2. 使用 ./launch_deepspeed.sh --num_gpus 4 --preset 7b_mamba 开始训练" 