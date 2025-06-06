# 实时GPU TFLOP性能检测器

🚀 **真实检测当前系统GPU并运行实际基准测试的TFLOP性能计算器**

与静态数据库不同，这个工具会：
- 🔍 **实时检测** 当前系统的真实GPU硬件
- ⚡ **实际测试** 运行真实的矩阵运算基准测试
- 📊 **准确计算** 基于实际运算的TFLOP性能
- 🔄 **实时监控** GPU状态、温度、功耗等

## ✨ 核心特性

### 🎯 真实性能测试
- **FP32基准测试**: 实际矩阵乘法运算测试
- **FP16高精度测试**: 半精度浮点性能
- **Tensor Core测试**: AI加速器性能测试
- **多精度对比**: 全面的性能分析

### 🔍 全面GPU检测
- **多种检测方式**: nvidia-smi、PyTorch、pynvml
- **实时状态**: 温度、功耗、利用率、频率
- **硬件信息**: 显存、CUDA核心、计算能力
- **驱动信息**: 驱动版本、CUDA版本

### 📈 智能分析
- **性能效率**: TFLOPS/W能效比
- **显存效率**: GB/TFLOP比率
- **实时对比**: 理论值vs实测值
- **多GPU支持**: 并行检测多张显卡

## 🚀 快速开始

### 自动安装 (推荐)
```bash
# 下载脚本并运行自动安装
bash install_gpu_realtime.sh
```

### 手动安装
```bash
# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装监控库
pip install nvidia-ml-py3 GPUtil psutil
```

## 📖 使用指南

### 基本检测
```bash
# 列出当前系统所有GPU
python gpu_realtime_tflop.py --list

# 检测GPU并运行完整性能测试
python gpu_realtime_tflop.py --benchmark

# 只测试特定精度
python gpu_realtime_tflop.py --benchmark --precision fp32
python gpu_realtime_tflop.py --benchmark --precision tensor
```

### 高级功能
```bash
# 指定测试时长(秒)
python gpu_realtime_tflop.py --benchmark --duration 10

# 只测试特定GPU
python gpu_realtime_tflop.py --benchmark --device 0

# 导出测试结果
python gpu_realtime_tflop.py --benchmark --export results.json

# 实时监控模式
python gpu_realtime_tflop.py --monitor
```

## 📊 使用示例

### 1. GPU检测输出
```
🔍 检测当前系统GPU...
✅ 通过nvidia-smi检测到 1 个GPU

🎮 GPU 0: NVIDIA GeForce RTX 4090
================================================================================
显存: 24.0 GB (使用: 0.5 GB, 空闲: 23.5 GB)
温度: 35°C
功耗: 65.2 W
利用率: 0%
核心频率: 2520 MHz
显存频率: 10501 MHz
CUDA核心: 16,384
计算能力: 8.9
驱动版本: 525.60.11
CUDA版本: 12.0
```

### 2. 性能测试输出
```
🏃 开始测试GPU 0: NVIDIA GeForce RTX 4090
🔥 GPU 0 FP32预热中...
⚡ GPU 0 FP32性能测试中 (5秒)...
🔥 GPU 0 FP16预热中...
⚡ GPU 0 FP16性能测试中 (5秒)...
🔥 GPU 0 Tensor Core预热中...
⚡ GPU 0 Tensor Core性能测试中 (5秒)...

📊 GPU 0 性能测试结果:
------------------------------------------------------------
FP32性能:      85.43 TFLOPS
FP16性能:      167.21 TFLOPS
Tensor性能:    1342.67 TFLOPS
FP32效率:      1.310 TFLOPS/W
显存效率:      0.3 GB/TFLOP
```

### 3. 实时监控模式
```
🔄 实时GPU状态监控
==================================================
🎮 GPU 0: NVIDIA GeForce RTX 4090
温度: 45°C → 47°C
功耗: 120.5W → 125.3W
利用率: 85% → 89%
显存使用: 18.2GB / 24.0GB
```

## 🔧 工作原理

### 检测机制
1. **nvidia-smi**: 获取实时GPU状态和硬件信息
2. **PyTorch**: 检测CUDA可用性和计算能力
3. **pynvml**: 深度监控GPU状态参数
4. **信息融合**: 综合多种来源提供完整信息

### 性能测试算法
```python
# FP32测试: 大规模矩阵乘法
matrix_size = 2048
operations = 2 * (matrix_size ** 3)  # 乘法+加法
tflops = operations / duration / 1e12

# FP16测试: 更大矩阵利用半精度优势
matrix_size = 4096
# 使用torch.float16进行计算

# Tensor Core测试: 使用autocast优化
with torch.cuda.amp.autocast():
    result = torch.matmul(a, b)
```

## 🎯 支持的硬件

### ✅ 完全支持
- **RTX 40系列**: RTX 4090, 4080, 4070等
- **RTX 30系列**: RTX 3090, 3080, 3070等  
- **数据中心GPU**: A100, H100, V100等
- **专业卡**: RTX A6000, A5000等

### ⚠️ 部分支持
- **GTX系列**: 检测和FP32测试，无Tensor Core
- **较老GPU**: 基本检测，性能测试可能受限

### ❌ 不支持
- **AMD GPU**: 目前仅支持NVIDIA GPU
- **集成显卡**: Intel/AMD集成GPU

## 🔍 依赖要求

### 必需依赖
- **Python 3.7+**
- **NVIDIA驱动** (包含nvidia-smi)

### 推荐依赖
- **PyTorch** (性能测试)
- **CUDA工具包** (完整功能)
- **nvidia-ml-py3** (深度监控)

### 最小运行
即使没有PyTorch，脚本也能：
- 检测GPU基本信息
- 显示实时状态
- 导出硬件规格

## 🆚 与静态工具对比

| 特性 | 实时检测器 | 静态数据库 |
|------|-----------|-----------|
| 数据来源 | 实际硬件检测 | 预设数据库 |
| 性能数据 | 实测基准结果 | 官方理论值 |
| 实时状态 | ✅ 温度/功耗/频率 | ❌ 无实时信息 |
| 自定义硬件 | ✅ 自动检测 | ❌ 需手动添加 |
| 准确性 | 🎯 实际环境 | 📊 理论最优 |

## 🚨 注意事项

### 性能测试警告
- **高功耗**: 性能测试会让GPU满载运行
- **高温度**: 确保散热良好，监控温度
- **显存占用**: 测试期间会占用大量显存
- **时间消耗**: 完整测试需要15-30秒

### 系统要求
- **足够散热**: 避免过热保护
- **稳定电源**: 高功耗测试需要充足供电
- **最新驱动**: 建议使用最新NVIDIA驱动

## 📈 高级用法

### 脚本集成
```python
from gpu_realtime_tflop import RealTimeGPUDetector, RealTimeTFLOPBenchmark

# 检测GPU
detector = RealTimeGPUDetector()
gpus = detector.detect_gpus()

# 性能测试
benchmark = RealTimeTFLOPBenchmark()
fp32_performance = benchmark.run_fp32_benchmark(0, duration=10)
```

### 自动化测试
```bash
# 批量测试脚本
for precision in fp32 fp16 tensor; do
    python gpu_realtime_tflop.py --benchmark --precision $precision --export "results_$precision.json"
done
```

### CI/CD集成
```yaml
# GitHub Actions示例
- name: GPU Performance Test
  run: |
    python gpu_realtime_tflop.py --list
    python gpu_realtime_tflop.py --benchmark --duration 3 --export gpu_results.json
```

## 🤝 贡献指南

### 添加新GPU支持
在`_estimate_cuda_cores`方法中添加：
```python
'YOUR_GPU_NAME': CUDA_CORES_COUNT
```

### 改进检测算法
- 支持AMD GPU检测
- 优化性能测试算法
- 增加新的精度支持

### 报告问题
- 提供完整的错误信息
- 包含GPU型号和驱动版本
- 附上`--list`命令的输出

## 📄 许可证

MIT License

## ⭐ 支持项目

如果这个工具帮助到了你，请：
- 给项目一个Star ⭐
- 分享给其他开发者 📢
- 提交改进建议 💡
- 贡献代码优化 🔧

---

**💡 提示**: 这是一个真实的性能测试工具，结果基于实际运算，可能与理论值有差异，这反映了真实的GPU性能表现。 