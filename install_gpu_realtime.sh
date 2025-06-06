#!/bin/bash
# 实时GPU TFLOP检测器自动安装脚本

echo "🚀 实时GPU TFLOP检测器安装脚本"
echo "=================================="

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="Windows"
else
    OS="Unknown"
fi

echo "检测到操作系统: $OS"

# 检查Python
echo ""
echo "🐍 检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON="python3"
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python3已安装: $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON="python"
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python已安装: $PYTHON_VERSION"
else
    echo "❌ 未找到Python，请先安装Python 3.7+"
    exit 1
fi

# 检查pip
echo ""
echo "📦 检查pip..."
if command -v pip3 &> /dev/null; then
    PIP="pip3"
elif command -v pip &> /dev/null; then
    PIP="pip"
else
    echo "❌ 未找到pip，请先安装pip"
    exit 1
fi

echo "✅ pip已安装"

# 检查NVIDIA驱动
echo ""
echo "🔍 检查NVIDIA驱动..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
else
    echo "⚠️ 未检测到nvidia-smi，可能需要安装NVIDIA驱动"
    echo "   但脚本仍可在没有GPU的环境下运行"
fi

# 检查CUDA
echo ""
echo "🔧 检查CUDA环境..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ CUDA已安装: $CUDA_VERSION"
    
    # 根据CUDA版本选择PyTorch
    if [[ "$CUDA_VERSION" == "11.8"* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    elif [[ "$CUDA_VERSION" == "12.1"* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [[ "$CUDA_VERSION" == "12.0"* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        echo "⚠️ CUDA版本 $CUDA_VERSION 可能需要手动选择PyTorch版本"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
else
    echo "⚠️ 未检测到CUDA，将安装CPU版本的PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# 安装依赖
echo ""
echo "📥 开始安装依赖包..."

# 创建虚拟环境 (可选)
read -p "是否创建虚拟环境? (推荐) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "🏗️ 创建虚拟环境..."
    $PYTHON -m venv gpu_tflop_env
    
    if [[ "$OS" == "Windows" ]]; then
        source gpu_tflop_env/Scripts/activate
    else
        source gpu_tflop_env/bin/activate
    fi
    
    echo "✅ 虚拟环境已激活"
fi

# 升级pip
echo "🔄 升级pip..."
$PIP install --upgrade pip

# 安装PyTorch
echo "🔥 安装PyTorch..."
if [[ "$TORCH_INDEX" == *"cpu"* ]]; then
    $PIP install torch torchvision --index-url $TORCH_INDEX
else
    $PIP install torch torchvision --index-url $TORCH_INDEX
fi

# 安装其他依赖
echo "📚 安装其他依赖..."
$PIP install nvidia-ml-py3 GPUtil psutil

# 验证安装
echo ""
echo "🧪 验证安装..."
$PYTHON -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
$PYTHON -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

if $PYTHON -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    $PYTHON -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
    $PYTHON -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
fi

# 测试脚本
echo ""
echo "🎮 测试GPU检测脚本..."
if [[ -f "gpu_realtime_tflop.py" ]]; then
    $PYTHON gpu_realtime_tflop.py --list
    echo ""
    echo "✅ 安装完成！"
    echo ""
    echo "📖 使用指南:"
    echo "  列出GPU信息:    python gpu_realtime_tflop.py --list"
    echo "  运行性能测试:    python gpu_realtime_tflop.py --benchmark"
    echo "  实时监控:       python gpu_realtime_tflop.py --monitor"
    echo "  查看帮助:       python gpu_realtime_tflop.py --help"
else
    echo "⚠️ 未找到gpu_realtime_tflop.py文件"
    echo "请确保脚本文件与安装脚本在同一目录"
fi

echo ""
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "🔧 激活虚拟环境命令:"
    if [[ "$OS" == "Windows" ]]; then
        echo "    gpu_tflop_env\\Scripts\\activate"
    else
        echo "    source gpu_tflop_env/bin/activate"
    fi
fi

echo ""
echo "🎉 安装脚本执行完成！" 