#!/bin/bash
# GPU进程清理脚本

echo "🔍 检查GPU进程..."

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi命令未找到"
    exit 1
fi

# 显示当前GPU状态
echo "当前GPU状态:"
nvidia-smi

echo ""
echo "正在运行的GPU进程:"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits

# 获取GPU进程PID
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

if [ -z "$PIDS" ]; then
    echo "✅ 没有发现GPU进程"
else
    echo ""
    echo "发现以下GPU进程:"
    for PID in $PIDS; do
        PROCESS_INFO=$(ps -p $PID -o pid,ppid,user,comm --no-headers 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "  PID $PID: $PROCESS_INFO"
        else
            echo "  PID $PID: (进程已退出)"
        fi
    done
    
    echo ""
    read -p "是否终止这些GPU进程？(y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔪 终止GPU进程..."
        for PID in $PIDS; do
            echo "终止进程 $PID..."
            kill -9 $PID 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✅ 进程 $PID 已终止"
            else
                echo "  ❌ 无法终止进程 $PID (可能需要sudo权限)"
            fi
        done
        
        echo ""
        echo "等待2秒后检查..."
        sleep 2
        
        echo "清理后的GPU状态:"
        nvidia-smi
    else
        echo "❌ 取消操作"
    fi
fi

echo ""
echo "🧹 清理PyTorch缓存..."
python -c "import torch; torch.cuda.empty_cache(); print('✅ PyTorch GPU缓存已清理')" 2>/dev/null || echo "❌ 无法清理PyTorch缓存"

echo ""
echo "💡 使用提示:"
echo "  手动终止进程: kill -9 <PID>"
echo "  查看GPU使用: watch -n 1 nvidia-smi"
echo "  清理缓存: python -c 'import torch; torch.cuda.empty_cache()'" 