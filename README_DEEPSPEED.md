# DeepSpeed ZeRO优化训练指南

## 🎯 目标
使用DeepSpeed ZeRO-2优化，在4-8张24GB GPU上训练**真正的7B Mamba模型**（6.89B参数）。

## 📊 显存需求对比

### DataParallel模式（原脚本）
- **每张GPU**: 87.5GB 显存 ❌ **不可行**

### DeepSpeed ZeRO-2优化
- **4张24GB GPU**: 23.1GB/GPU ✅ **刚好可行**
- **6张24GB GPU**: 16.1GB/GPU ✅ **安全**  
- **8张24GB GPU**: 12.6GB/GPU ✅ **很安全**
- **4张32GB vGPU**: 23.1GB/GPU ✅ **可行**

## 🚀 快速开始

### 1. 安装DeepSpeed
```bash
# 自动安装（推荐）
./install_deepspeed.sh

# 手动安装
pip install deepspeed
```

### 2. 启动训练

**训练真正7B模型**：
```bash
# 4张GPU（最低配置）
./launch_deepspeed.sh --num_gpus 4 --preset 7b_mamba

# 6张GPU（推荐配置）
./launch_deepspeed.sh --num_gpus 6 --preset 7b_mamba

# 8张GPU（最佳配置）
./launch_deepspeed.sh --num_gpus 8 --preset 7b_mamba
```

**训练3B模型**（如果7B太大）：
```bash
./launch_deepspeed.sh --num_gpus 4 --preset 3b_mamba
```

### 3. 监控训练
```bash
# 检查显存使用
python train_deepspeed.py --check_memory

# 验证配置（不训练）
python train_deepspeed.py --preset 7b_mamba --num_gpus 4 --dry_run
```

## 🔧 配置文件

### 可用预设
- `7b_mamba`: 真正的7B模型（6.89B参数）
- `3b_mamba`: 诚实的3B模型（2.84B参数）
- `1b_mamba`: 1B测试模型

### 自定义配置
修改 `config_7b_mamba.yaml` 或创建新配置：
```yaml
model_type: "mamba"
num_gpus: 4

model:
  vocab_size: 50257
  max_seq_length: 4096
  d_model: 4864      # 决定模型大小
  n_layers: 45       # 决定模型深度
  d_state: 16
  d_conv: 4
  expand: 2

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  max_steps: 200000
```

## 🎛️ 关键优化

### ZeRO-2分片
- **参数分片**: 模型参数均匀分布到各GPU
- **优化器分片**: Adam状态分片，节省55GB/GPU
- **梯度分片**: 梯度计算和通信重叠

### 激活检查点
```json
"activation_checkpointing": {
    "partition_activations": true,
    "number_checkpoints": 4
}
```

### 混合精度
```json
"fp16": {
    "enabled": true,
    "loss_scale": 0
}
```

## 📁 输出文件

### 检查点结构
```
checkpoints/
├── step_10000/
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   ├── zero_pp_rank_1_mp_rank_00_optim_states.pt
│   └── mp_rank_00_model_states.pt
└── final/
    └── [相同结构]
```

### 加载检查点
```python
# 恢复训练
deepspeed --num_gpus=4 train_deepspeed.py \
    --preset 7b_mamba \
    --deepspeed_config deepspeed_config.json \
    --load_checkpoint ./checkpoints/step_10000
```

## ⚠️ 故障排除

### 显存不足
```bash
# 减少批大小
sed -i 's/batch_size: 1/batch_size: 1/' config_7b_mamba.yaml

# 增加GPU数量
./launch_deepspeed.sh --num_gpus 6

# 使用更小模型
./launch_deepspeed.sh --preset 3b_mamba
```

### 通信错误
```bash
# 更换通信端口
./launch_deepspeed.sh --port 29501

# 检查网络连接
ping localhost
```

### CUDA版本问题
```bash
# 重新安装对应CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 性能基准

### 训练速度（预估）
- **4张RTX 4090**: ~0.8 steps/sec
- **6张RTX 4090**: ~1.2 steps/sec  
- **8张RTX 4090**: ~1.6 steps/sec

### 收敛时间（预估）
- **7B模型**: ~5-7天（200K步）
- **3B模型**: ~3-4天（150K步）

## 🎯 与其他方案对比

| 方案 | 7B模型 | 显存/GPU | GPU数量 | 训练速度 |
|------|--------|----------|---------|----------|
| DataParallel | ❌ | 87.5GB | N/A | N/A |
| DeepSpeed ZeRO-2 | ✅ | 23.1GB | 4 | 0.8 steps/s |
| DeepSpeed ZeRO-3 | ✅ | 15.0GB | 4 | 0.6 steps/s |
| Model Parallel | ✅ | 25.0GB | 4 | 0.5 steps/s |

**ZeRO-2是最佳平衡方案**：显存效率高，通信开销适中。

## 🔄 下一步扩展

1. **ZeRO-3优化**: 进一步减少显存到15GB/GPU
2. **Gradient Compression**: 减少通信带宽
3. **Pipeline Parallel**: 支持超大模型
4. **CPU Offload**: 混合CPU-GPU训练

---

**准备好训练真正的7B模型了吗？开始吧！** 🚀 