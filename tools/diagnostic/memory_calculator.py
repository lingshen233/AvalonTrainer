#!/usr/bin/env python3
"""
7B模型显存需求计算器
"""

def calculate_memory_requirements():
    """计算真正7B Mamba模型的显存需求"""
    
    # 真正7B Mamba参数
    params = 6.89e9
    
    print('=== 真正7B Mamba (6.89B参数) 显存需求 ===')
    
    # DataParallel模式：每张卡完整模型
    print('\n📊 DataParallel模式（每张GPU完整模型）:')
    model_memory = params * 2 / 1e9  # FP16模型参数
    optimizer_memory = params * 8 / 1e9  # Adam优化器状态
    gradient_memory = params * 2 / 1e9  # FP16梯度
    
    print(f'模型参数(FP16): {model_memory:.2f}GB')
    print(f'优化器状态(Adam): {optimizer_memory:.2f}GB') 
    print(f'梯度存储(FP16): {gradient_memory:.2f}GB')
    
    base_memory = model_memory + optimizer_memory + gradient_memory
    print(f'基础显存需求: {base_memory:.2f}GB/GPU')
    
    # 激活值和总需求
    activation = 1 * 4096 * 4864 * 45 * 2 / 1e9  # 简化激活值
    total_dp = base_memory + activation + 3
    print(f'总显存需求: {total_dp:.1f}GB/GPU ❌ 太高了！')
    
    # ZeRO优化模式
    print('\n🔧 DeepSpeed ZeRO-2优化（参数和梯度分片）:')
    configs = [
        ("2张3090/4090", 2, 24), ("4张3090/4090", 4, 24),
        ("6张3090/4090", 6, 24), ("8张3090/4090", 8, 24),
        ("2张vGPU", 2, 32), ("4张vGPU", 4, 32)
    ]
    
    for name, num_gpus, gpu_memory in configs:
        # ZeRO-2：优化器状态分片，模型参数分片
        model_per_gpu = model_memory / num_gpus
        optimizer_per_gpu = optimizer_memory / num_gpus
        gradient_per_gpu = gradient_memory / num_gpus
        
        # 激活值（每GPU处理1/num_gpus的数据）
        activation_per_gpu = activation / num_gpus
        
        total_zero = model_per_gpu + optimizer_per_gpu + gradient_per_gpu + activation_per_gpu + 2
        
        print(f'{name}: {total_zero:.1f}GB/GPU')
        print(f'  - 模型: {model_per_gpu:.1f}GB')
        print(f'  - 优化器: {optimizer_per_gpu:.1f}GB')
        print(f'  - 梯度: {gradient_per_gpu:.1f}GB')
        print(f'  - 激活: {activation_per_gpu:.1f}GB')
        
        if total_zero <= gpu_memory:
            print(f'  ✅ 可行！剩余 {gpu_memory - total_zero:.1f}GB')
        else:
            print(f'  ❌ 超出 {total_zero - gpu_memory:.1f}GB')
        print()

def calculate_3b_requirements():
    """计算3B模型需求（对比）"""
    print('\n=== 对比：3B Mamba (2.84B参数) ===')
    params = 2.84e9
    
    model_memory = params * 2 / 1e9
    optimizer_memory = params * 8 / 1e9
    gradient_memory = params * 2 / 1e9
    activation = 1 * 2048 * 3584 * 32 * 2 / 1e9
    
    total = model_memory + optimizer_memory + gradient_memory + activation + 2
    
    print(f'DataParallel模式: {total:.1f}GB/GPU')
    if total <= 24:
        print('✅ 单张24GB GPU即可运行!')
    else:
        print('❌ 仍需多张GPU')

if __name__ == "__main__":
    calculate_memory_requirements()
    calculate_3b_requirements() 