#!/usr/bin/env python3
"""
GPU TFLOP计算器
支持主流显卡的FP32/FP16/BF16/INT8性能计算和比较
"""

import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

@dataclass
class GPUSpec:
    """GPU规格数据类"""
    name: str
    cuda_cores: int
    tensor_cores: Optional[int]
    base_clock_mhz: int
    boost_clock_mhz: int
    memory_gb: int
    memory_bandwidth_gbps: float
    memory_bus_width: int
    architecture: str
    process_node: str
    tdp_watts: int
    fp32_tflops: Optional[float] = None
    fp16_tflops: Optional[float] = None
    bf16_tflops: Optional[float] = None
    int8_tops: Optional[float] = None
    tensor_tflops: Optional[float] = None

# 显卡数据库
GPU_DATABASE = {
    # RTX 40系列
    "RTX 4090": GPUSpec(
        name="RTX 4090",
        cuda_cores=16384,
        tensor_cores=128,
        base_clock_mhz=2230,
        boost_clock_mhz=2520,
        memory_gb=24,
        memory_bandwidth_gbps=1008,
        memory_bus_width=384,
        architecture="Ada Lovelace",
        process_node="4nm",
        tdp_watts=450,
        fp32_tflops=83.0,
        fp16_tflops=165.0,
        bf16_tflops=165.0,
        tensor_tflops=1320.0
    ),
    
    "RTX 4080": GPUSpec(
        name="RTX 4080",
        cuda_cores=9728,
        tensor_cores=76,
        base_clock_mhz=2205,
        boost_clock_mhz=2505,
        memory_gb=16,
        memory_bandwidth_gbps=717,
        memory_bus_width=256,
        architecture="Ada Lovelace",
        process_node="4nm",
        tdp_watts=320,
        fp32_tflops=48.7,
        fp16_tflops=97.4,
        bf16_tflops=97.4,
        tensor_tflops=1560.0
    ),
    
    "RTX 4070": GPUSpec(
        name="RTX 4070",
        cuda_cores=5888,
        tensor_cores=46,
        base_clock_mhz=1920,
        boost_clock_mhz=2475,
        memory_gb=12,
        memory_bandwidth_gbps=504,
        memory_bus_width=192,
        architecture="Ada Lovelace",
        process_node="4nm",
        tdp_watts=200,
        fp32_tflops=29.1,
        fp16_tflops=58.2,
        bf16_tflops=58.2,
        tensor_tflops=929.0
    ),
    
    # RTX 30系列
    "RTX 3090": GPUSpec(
        name="RTX 3090",
        cuda_cores=10496,
        tensor_cores=82,
        base_clock_mhz=1395,
        boost_clock_mhz=1695,
        memory_gb=24,
        memory_bandwidth_gbps=936,
        memory_bus_width=384,
        architecture="Ampere",
        process_node="8nm",
        tdp_watts=350,
        fp32_tflops=35.6,
        fp16_tflops=71.2,
        bf16_tflops=71.2,
        tensor_tflops=568.0
    ),
    
    "RTX 3080": GPUSpec(
        name="RTX 3080",
        cuda_cores=8704,
        tensor_cores=68,
        base_clock_mhz=1440,
        boost_clock_mhz=1710,
        memory_gb=10,
        memory_bandwidth_gbps=760,
        memory_bus_width=320,
        architecture="Ampere",
        process_node="8nm",
        tdp_watts=320,
        fp32_tflops=29.8,
        fp16_tflops=59.6,
        bf16_tflops=59.6,
        tensor_tflops=476.0
    ),
    
    "RTX 3070": GPUSpec(
        name="RTX 3070",
        cuda_cores=5888,
        tensor_cores=46,
        base_clock_mhz=1500,
        boost_clock_mhz=1725,
        memory_gb=8,
        memory_bandwidth_gbps=448,
        memory_bus_width=256,
        architecture="Ampere",
        process_node="8nm",
        tdp_watts=220,
        fp32_tflops=20.3,
        fp16_tflops=40.6,
        bf16_tflops=40.6,
        tensor_tflops=325.0
    ),
    
    # 数据中心GPU
    "A100-40GB": GPUSpec(
        name="A100-40GB",
        cuda_cores=6912,
        tensor_cores=108,
        base_clock_mhz=765,
        boost_clock_mhz=1410,
        memory_gb=40,
        memory_bandwidth_gbps=1555,
        memory_bus_width=5120,
        architecture="Ampere",
        process_node="7nm",
        tdp_watts=400,
        fp32_tflops=19.5,
        fp16_tflops=78.0,
        bf16_tflops=78.0,
        tensor_tflops=1248.0
    ),
    
    "A100-80GB": GPUSpec(
        name="A100-80GB",
        cuda_cores=6912,
        tensor_cores=108,
        base_clock_mhz=765,
        boost_clock_mhz=1410,
        memory_gb=80,
        memory_bandwidth_gbps=1935,
        memory_bus_width=5120,
        architecture="Ampere",
        process_node="7nm",
        tdp_watts=400,
        fp32_tflops=19.5,
        fp16_tflops=78.0,
        bf16_tflops=78.0,
        tensor_tflops=1248.0
    ),
    
    "H100": GPUSpec(
        name="H100",
        cuda_cores=14592,
        tensor_cores=132,
        base_clock_mhz=1230,
        boost_clock_mhz=1830,
        memory_gb=80,
        memory_bandwidth_gbps=3350,
        memory_bus_width=5120,
        architecture="Hopper",
        process_node="4nm",
        tdp_watts=700,
        fp32_tflops=51.2,
        fp16_tflops=102.4,
        bf16_tflops=102.4,
        tensor_tflops=3958.0
    ),
    
    "V100": GPUSpec(
        name="V100",
        cuda_cores=5120,
        tensor_cores=80,
        base_clock_mhz=1245,
        boost_clock_mhz=1380,
        memory_gb=32,
        memory_bandwidth_gbps=900,
        memory_bus_width=4096,
        architecture="Volta",
        process_node="12nm",
        tdp_watts=300,
        fp32_tflops=14.1,
        fp16_tflops=28.2,
        tensor_tflops=112.0
    ),
    
    # 专业卡
    "RTX A6000": GPUSpec(
        name="RTX A6000",
        cuda_cores=10752,
        tensor_cores=84,
        base_clock_mhz=1410,
        boost_clock_mhz=1800,
        memory_gb=48,
        memory_bandwidth_gbps=768,
        memory_bus_width=384,
        architecture="Ampere",
        process_node="8nm",
        tdp_watts=300,
        fp32_tflops=38.7,
        fp16_tflops=77.4,
        bf16_tflops=77.4,
        tensor_tflops=619.0
    ),
    
    "RTX A5000": GPUSpec(
        name="RTX A5000",
        cuda_cores=8192,
        tensor_cores=64,
        base_clock_mhz=1170,
        boost_clock_mhz=1695,
        memory_gb=24,
        memory_bandwidth_gbps=768,
        memory_bus_width=384,
        architecture="Ampere",
        process_node="8nm",
        tdp_watts=230,
        fp32_tflops=27.8,
        fp16_tflops=55.6,
        bf16_tflops=55.6,
        tensor_tflops=444.0
    ),
}

class TFLOPCalculator:
    """TFLOP计算器"""
    
    def __init__(self):
        self.gpus = GPU_DATABASE
    
    def calculate_theoretical_tflops(self, gpu: GPUSpec, precision: str = "fp32") -> float:
        """计算理论TFLOP值"""
        if precision == "fp32":
            # FP32: cores × boost_clock × 2 / 1e12
            return (gpu.cuda_cores * gpu.boost_clock_mhz * 1e6 * 2) / 1e12
        elif precision == "fp16":
            # FP16: 通常是FP32的2倍
            fp32_tflops = self.calculate_theoretical_tflops(gpu, "fp32")
            return fp32_tflops * 2
        elif precision == "bf16":
            # BF16: 与FP16类似
            return self.calculate_theoretical_tflops(gpu, "fp16")
        else:
            return 0.0
    
    def get_official_tflops(self, gpu: GPUSpec, precision: str = "fp32") -> Optional[float]:
        """获取官方TFLOP数据"""
        if precision == "fp32":
            return gpu.fp32_tflops
        elif precision == "fp16":
            return gpu.fp16_tflops
        elif precision == "bf16":
            return gpu.bf16_tflops
        elif precision == "tensor":
            return gpu.tensor_tflops
        return None
    
    def calculate_performance_per_watt(self, gpu: GPUSpec, precision: str = "fp32") -> float:
        """计算每瓦性能"""
        tflops = self.get_official_tflops(gpu, precision)
        if tflops and gpu.tdp_watts:
            return tflops / gpu.tdp_watts
        return 0.0
    
    def list_all_gpus(self) -> List[str]:
        """列出所有GPU"""
        return list(self.gpus.keys())
    
    def get_gpu_info(self, gpu_name: str) -> Optional[GPUSpec]:
        """获取GPU信息"""
        return self.gpus.get(gpu_name)
    
    def compare_gpus(self, gpu_names: List[str], precision: str = "fp32") -> Dict:
        """比较多个GPU性能"""
        results = {}
        
        for name in gpu_names:
            if name in self.gpus:
                gpu = self.gpus[name]
                theoretical = self.calculate_theoretical_tflops(gpu, precision)
                official = self.get_official_tflops(gpu, precision)
                perf_per_watt = self.calculate_performance_per_watt(gpu, precision)
                
                results[name] = {
                    'theoretical_tflops': theoretical,
                    'official_tflops': official,
                    'performance_per_watt': perf_per_watt,
                    'memory_gb': gpu.memory_gb,
                    'memory_bandwidth_gbps': gpu.memory_bandwidth_gbps,
                    'tdp_watts': gpu.tdp_watts,
                    'architecture': gpu.architecture
                }
        
        return results
    
    def find_best_gpu(self, criteria: str = "tflops", precision: str = "fp32", max_power: Optional[int] = None) -> str:
        """寻找最佳GPU"""
        best_gpu = None
        best_value = 0
        
        for name, gpu in self.gpus.items():
            if max_power and gpu.tdp_watts > max_power:
                continue
                
            if criteria == "tflops":
                value = self.get_official_tflops(gpu, precision) or 0
            elif criteria == "efficiency":
                value = self.calculate_performance_per_watt(gpu, precision)
            elif criteria == "memory":
                value = gpu.memory_gb
            else:
                continue
            
            if value > best_value:
                best_value = value
                best_gpu = name
        
        return best_gpu or ""

def print_gpu_table(calculator: TFLOPCalculator, gpu_names: List[str], precision: str = "fp32"):
    """打印GPU比较表格"""
    print(f"\n📊 GPU性能比较 ({precision.upper()})")
    print("=" * 120)
    print(f"{'GPU':<15} {'架构':<12} {'TFLOP':<8} {'显存':<6} {'带宽':<8} {'功耗':<6} {'效率':<8} {'内存/T':<8}")
    print("-" * 120)
    
    comparison = calculator.compare_gpus(gpu_names, precision)
    
    # 按TFLOP降序排序
    sorted_gpus = sorted(comparison.items(), 
                        key=lambda x: x[1]['official_tflops'] or 0, 
                        reverse=True)
    
    for name, data in sorted_gpus:
        tflops = data['official_tflops']
        memory_per_tflop = data['memory_gb'] / tflops if tflops else 0
        
        print(f"{name:<15} "
              f"{data['architecture']:<12} "
              f"{tflops:<8.1f} "
              f"{data['memory_gb']:<6}GB "
              f"{data['memory_bandwidth_gbps']:<8.0f} "
              f"{data['tdp_watts']:<6}W "
              f"{data['performance_per_watt']:<8.2f} "
              f"{memory_per_tflop:<8.1f}")

def print_detailed_info(calculator: TFLOPCalculator, gpu_name: str):
    """打印详细GPU信息"""
    gpu = calculator.get_gpu_info(gpu_name)
    if not gpu:
        print(f"❌ 未找到GPU: {gpu_name}")
        return
    
    print(f"\n🔍 {gpu.name} 详细信息")
    print("=" * 60)
    print(f"架构: {gpu.architecture}")
    print(f"制程: {gpu.process_node}")
    print(f"CUDA核心: {gpu.cuda_cores:,}")
    if gpu.tensor_cores:
        print(f"Tensor核心: {gpu.tensor_cores}")
    print(f"基础频率: {gpu.base_clock_mhz} MHz")
    print(f"加速频率: {gpu.boost_clock_mhz} MHz")
    print(f"显存: {gpu.memory_gb} GB")
    print(f"显存带宽: {gpu.memory_bandwidth_gbps} GB/s")
    print(f"显存位宽: {gpu.memory_bus_width} bit")
    print(f"TDP: {gpu.tdp_watts} W")
    
    print(f"\n🚀 性能数据:")
    if gpu.fp32_tflops:
        print(f"FP32: {gpu.fp32_tflops} TFLOPS")
    if gpu.fp16_tflops:
        print(f"FP16: {gpu.fp16_tflops} TFLOPS")
    if gpu.bf16_tflops:
        print(f"BF16: {gpu.bf16_tflops} TFLOPS")
    if gpu.tensor_tflops:
        print(f"Tensor: {gpu.tensor_tflops} TFLOPS")
    
    # 计算理论值
    theoretical_fp32 = calculator.calculate_theoretical_tflops(gpu, "fp32")
    print(f"\n🧮 理论计算:")
    print(f"理论FP32: {theoretical_fp32:.1f} TFLOPS")
    
    # 效率指标
    if gpu.fp32_tflops:
        print(f"\n⚡ 效率指标:")
        print(f"FP32效率: {calculator.calculate_performance_per_watt(gpu, 'fp32'):.2f} TFLOPS/W")
        print(f"显存/性能比: {gpu.memory_gb/gpu.fp32_tflops:.1f} GB/TFLOP")

def main():
    parser = argparse.ArgumentParser(description="GPU TFLOP计算器")
    parser.add_argument("--list", action="store_true", help="列出所有GPU")
    parser.add_argument("--info", type=str, help="显示特定GPU的详细信息")
    parser.add_argument("--compare", nargs="+", help="比较多个GPU")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16", "tensor"], 
                       default="fp32", help="计算精度")
    parser.add_argument("--best", choices=["tflops", "efficiency", "memory"], 
                       help="找出最佳GPU")
    parser.add_argument("--max-power", type=int, help="最大功耗限制(W)")
    parser.add_argument("--export", type=str, help="导出结果到JSON文件")
    
    args = parser.parse_args()
    
    calculator = TFLOPCalculator()
    
    if args.list:
        print("🎮 支持的GPU列表:")
        print("=" * 40)
        for i, gpu_name in enumerate(calculator.list_all_gpus(), 1):
            gpu = calculator.get_gpu_info(gpu_name)
            print(f"{i:2d}. {gpu_name:<15} ({gpu.architecture})")
    
    elif args.info:
        print_detailed_info(calculator, args.info)
    
    elif args.compare:
        print_gpu_table(calculator, args.compare, args.precision)
    
    elif args.best:
        best_gpu = calculator.find_best_gpu(args.best, args.precision, args.max_power)
        print(f"🏆 最佳GPU ({args.best}): {best_gpu}")
        if best_gpu:
            print_detailed_info(calculator, best_gpu)
    
    else:
        # 默认显示热门GPU比较
        popular_gpus = ["RTX 4090", "RTX 4080", "RTX 3090", "A100-80GB", "H100"]
        print("🔥 热门GPU性能对比")
        print_gpu_table(calculator, popular_gpus, args.precision)
    
    # 导出功能
    if args.export:
        if args.compare:
            results = calculator.compare_gpus(args.compare, args.precision)
        else:
            popular_gpus = ["RTX 4090", "RTX 4080", "RTX 3090", "A100-80GB", "H100"]
            results = calculator.compare_gpus(popular_gpus, args.precision)
        
        with open(args.export, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📁 结果已导出到: {args.export}")

if __name__ == "__main__":
    main() 