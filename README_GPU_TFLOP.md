# GPU TFLOP计算器

一个全面的GPU性能计算和比较工具，支持主流显卡的TFLOP、效率和成本分析。

## ✨ 功能特性

- 🎯 **精确计算**: 支持FP32/FP16/BF16/Tensor多种精度的TFLOP计算
- 📊 **性能对比**: 横向比较多款GPU的性能指标
- ⚡ **效率分析**: 计算每瓦性能、显存效率等关键指标
- 🏆 **智能推荐**: 根据不同需求推荐最佳GPU
- 📈 **数据导出**: 支持JSON格式导出比较结果
- 🎮 **全面覆盖**: 包含RTX 40/30系列、A100、H100等主流显卡

## 🚀 快速开始

### 基本用法

```bash
# 显示所有支持的GPU
python gpu_tflop_calculator.py --list

# 查看特定GPU详细信息
python gpu_tflop_calculator.py --info "RTX 4090"

# 比较多款GPU
python gpu_tflop_calculator.py --compare "RTX 4090" "RTX 4080" "RTX 3090"

# 指定精度比较
python gpu_tflop_calculator.py --compare "A100-80GB" "H100" --precision fp16

# 寻找最佳GPU
python gpu_tflop_calculator.py --best tflops --precision tensor
python gpu_tflop_calculator.py --best efficiency --max-power 300
```

## 📊 使用示例

### 1. 列出所有GPU
```bash
python gpu_tflop_calculator.py --list
```
输出：
```
🎮 支持的GPU列表:
========================================
 1. RTX 4090      (Ada Lovelace)
 2. RTX 4080      (Ada Lovelace)
 3. RTX 4070      (Ada Lovelace)
 4. RTX 3090      (Ampere)
 5. RTX 3080      (Ampere)
 ...
```

### 2. 详细GPU信息
```bash
python gpu_tflop_calculator.py --info "RTX 4090"
```
输出：
```
🔍 RTX 4090 详细信息
============================================================
架构: Ada Lovelace
制程: 4nm
CUDA核心: 16,384
Tensor核心: 128
基础频率: 2230 MHz
加速频率: 2520 MHz
显存: 24 GB
显存带宽: 1008 GB/s
TDP: 450 W

🚀 性能数据:
FP32: 83.0 TFLOPS
FP16: 165.0 TFLOPS
BF16: 165.0 TFLOPS
Tensor: 1320.0 TFLOPS

⚡ 效率指标:
FP32效率: 0.18 TFLOPS/W
显存/性能比: 0.3 GB/TFLOP
```

### 3. GPU性能对比
```bash
python gpu_tflop_calculator.py --compare "RTX 4090" "RTX 4080" "A100-80GB" --precision fp16
```
输出：
```
📊 GPU性能比较 (FP16)
========================================================
GPU             架构          TFLOP    显存   带宽     功耗   效率     内存/T  
--------------------------------------------------------
RTX 4090        Ada Lovelace  165.0    24GB   1008     450W   0.37     0.1
A100-80GB       Ampere        78.0     80GB   1935     400W   0.20     1.0
RTX 4080        Ada Lovelace  97.4     16GB   717      320W   0.30     0.2
```

### 4. 寻找最佳GPU
```bash
# 寻找Tensor性能最强的GPU
python gpu_tflop_calculator.py --best tflops --precision tensor

# 寻找功耗限制下最高效的GPU
python gpu_tflop_calculator.py --best efficiency --max-power 250
```

### 5. 导出比较结果
```bash
python gpu_tflop_calculator.py --compare "RTX 4090" "A100-80GB" "H100" --export gpu_comparison.json
```

## 🎯 支持的GPU列表

### 🎮 消费级显卡
- **RTX 40系列**: RTX 4090, RTX 4080, RTX 4070
- **RTX 30系列**: RTX 3090, RTX 3080, RTX 3070

### 🏢 数据中心GPU
- **NVIDIA A系列**: A100-40GB, A100-80GB
- **NVIDIA H系列**: H100
- **NVIDIA V系列**: V100

### 💼 专业卡
- **RTX A系列**: RTX A6000, RTX A5000

## 📈 性能指标说明

### TFLOP类型
- **FP32**: 单精度浮点运算性能
- **FP16**: 半精度浮点运算性能
- **BF16**: Brain Float 16性能
- **Tensor**: AI加速器性能(Tensor Core)

### 效率指标
- **TFLOPS/W**: 每瓦性能，衡量能效
- **GB/TFLOP**: 每TFLOP显存容量，衡量显存效率
- **理论vs实际**: 比较理论计算值与官方数据

## 🔧 扩展使用

### 在Python脚本中使用
```python
from gpu_tflop_calculator import TFLOPCalculator

calc = TFLOPCalculator()

# 获取GPU信息
gpu_info = calc.get_gpu_info("RTX 4090")
print(f"RTX 4090 FP32性能: {gpu_info.fp32_tflops} TFLOPS")

# 比较GPU
comparison = calc.compare_gpus(["RTX 4090", "A100-80GB"], "fp16")
print(comparison)

# 寻找最佳GPU
best_gpu = calc.find_best_gpu("efficiency", "fp32", max_power=300)
print(f"最高效GPU: {best_gpu}")
```

### 添加新GPU
在`GPU_DATABASE`中添加新的GPU规格：
```python
"NEW_GPU": GPUSpec(
    name="NEW_GPU",
    cuda_cores=8192,
    tensor_cores=64,
    base_clock_mhz=1500,
    boost_clock_mhz=1800,
    memory_gb=16,
    memory_bandwidth_gbps=512,
    memory_bus_width=256,
    architecture="New Arch",
    process_node="5nm",
    tdp_watts=250,
    fp32_tflops=30.0,
    fp16_tflops=60.0
)
```

## 🛠 依赖要求

- Python 3.7+
- 无外部依赖，使用标准库

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交PR来添加新的GPU数据或改进功能！

### 贡献指南
1. Fork仓库
2. 创建功能分支
3. 添加GPU数据或新功能
4. 提交PR

## 📞 支持

如有问题或建议，请提交Issue。

---

**⭐ 如果这个工具对你有帮助，请给个Star支持！** 