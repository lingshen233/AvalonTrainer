#!/usr/bin/env python3
"""
实时GPU TFLOP性能检测器
检测当前系统GPU并运行实际基准测试计算TFLOP性能
"""

import os
import sys
import time
import json
import argparse
import subprocess
import platform
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

# GPU检测相关导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

@dataclass
class RealTimeGPUInfo:
    """实时GPU信息"""
    device_id: int
    name: str
    total_memory: int  # 字节
    free_memory: int   # 字节
    used_memory: int   # 字节
    temperature: Optional[int]
    power_usage: Optional[float]
    utilization: Optional[int]
    clock_graphics: Optional[int]  # MHz
    clock_memory: Optional[int]    # MHz
    cuda_cores: Optional[int]
    compute_capability: Optional[Tuple[int, int]]
    driver_version: Optional[str]
    cuda_version: Optional[str]

class RealTimeGPUDetector:
    """实时GPU检测器"""
    
    def __init__(self):
        self.torch_available = TORCH_AVAILABLE
        self.nvidia_ml_available = NVIDIA_ML_AVAILABLE
        self.gputil_available = GPUTIL_AVAILABLE
        
        # 初始化NVIDIA ML
        if self.nvidia_ml_available:
            try:
                pynvml.nvmlInit()
                self.nvidia_ml_initialized = True
            except:
                self.nvidia_ml_initialized = False
        else:
            self.nvidia_ml_initialized = False
    
    def get_nvidia_smi_info(self) -> Dict:
        """通过nvidia-smi获取GPU信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,utilization.gpu,clocks.current.graphics,clocks.current.memory,driver_version',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 11:
                            gpus.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                                'memory_free': int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                                'memory_used': int(parts[4]) if parts[4] != '[Not Supported]' else 0,
                                'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else None,
                                'power_draw': float(parts[6]) if parts[6] != '[Not Supported]' else None,
                                'utilization': int(parts[7]) if parts[7] != '[Not Supported]' else None,
                                'clock_graphics': int(parts[8]) if parts[8] != '[Not Supported]' else None,
                                'clock_memory': int(parts[9]) if parts[9] != '[Not Supported]' else None,
                                'driver_version': parts[10] if parts[10] != '[Not Supported]' else None
                            })
                return {'gpus': gpus, 'method': 'nvidia-smi'}
        except Exception as e:
            print(f"nvidia-smi检测失败: {e}")
        
        return {'gpus': [], 'method': 'nvidia-smi'}
    
    def get_torch_gpu_info(self) -> Dict:
        """通过PyTorch获取GPU信息"""
        if not self.torch_available:
            return {'gpus': [], 'method': 'torch'}
        
        try:
            if not torch.cuda.is_available():
                return {'gpus': [], 'method': 'torch'}
            
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device = torch.device(f'cuda:{i}')
                
                # 获取显存信息
                torch.cuda.set_device(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_free = torch.cuda.memory_reserved(i)
                memory_used = torch.cuda.memory_allocated(i)
                
                gpus.append({
                    'index': i,
                    'name': props.name,
                    'memory_total': memory_total // (1024**2),  # MB
                    'memory_free': (memory_total - memory_used) // (1024**2),
                    'memory_used': memory_used // (1024**2),
                    'compute_capability': (props.major, props.minor),
                    'multiprocessor_count': props.multi_processor_count,
                    'max_threads_per_multiprocessor': props.max_threads_per_multi_processor,
                    'max_threads_per_block': props.max_threads_per_block,
                    'cuda_version': torch.version.cuda
                })
            
            return {'gpus': gpus, 'method': 'torch'}
        except Exception as e:
            print(f"PyTorch GPU检测失败: {e}")
            return {'gpus': [], 'method': 'torch'}
    
    def get_pynvml_info(self) -> Dict:
        """通过pynvml获取GPU信息"""
        if not self.nvidia_ml_initialized:
            return {'gpus': [], 'method': 'pynvml'}
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 基本信息
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # 显存信息
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # 温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # 功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_usage = None
                
                # 利用率
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                except:
                    gpu_util = None
                
                # 时钟频率
                try:
                    clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    clock_graphics = clock_memory = None
                
                gpus.append({
                    'index': i,
                    'name': name,
                    'memory_total': memory_info.total // (1024**2),  # MB
                    'memory_free': memory_info.free // (1024**2),
                    'memory_used': memory_info.used // (1024**2),
                    'temperature': temperature,
                    'power_usage': power_usage,
                    'utilization': gpu_util,
                    'clock_graphics': clock_graphics,
                    'clock_memory': clock_memory
                })
            
            return {'gpus': gpus, 'method': 'pynvml'}
        except Exception as e:
            print(f"pynvml GPU检测失败: {e}")
            return {'gpus': [], 'method': 'pynvml'}
    
    def detect_gpus(self) -> List[RealTimeGPUInfo]:
        """综合检测GPU信息"""
        print("🔍 检测当前系统GPU...")
        
        # 优先使用nvidia-smi
        nvidia_smi_info = self.get_nvidia_smi_info()
        torch_info = self.get_torch_gpu_info()
        pynvml_info = self.get_pynvml_info()
        
        # 合并信息
        gpu_list = []
        
        if nvidia_smi_info['gpus']:
            print(f"✅ 通过nvidia-smi检测到 {len(nvidia_smi_info['gpus'])} 个GPU")
            
            for smi_gpu in nvidia_smi_info['gpus']:
                # 查找对应的torch信息
                torch_gpu = None
                if torch_info['gpus']:
                    for tg in torch_info['gpus']:
                        if tg['index'] == smi_gpu['index']:
                            torch_gpu = tg
                            break
                
                # 查找对应的pynvml信息
                pynvml_gpu = None
                if pynvml_info['gpus']:
                    for pg in pynvml_info['gpus']:
                        if pg['index'] == smi_gpu['index']:
                            pynvml_gpu = pg
                            break
                
                # 合并信息
                gpu_info = RealTimeGPUInfo(
                    device_id=smi_gpu['index'],
                    name=smi_gpu['name'],
                    total_memory=smi_gpu['memory_total'] * 1024 * 1024,  # MB to bytes
                    free_memory=smi_gpu['memory_free'] * 1024 * 1024,
                    used_memory=smi_gpu['memory_used'] * 1024 * 1024,
                    temperature=smi_gpu['temperature'],
                    power_usage=smi_gpu['power_draw'],
                    utilization=smi_gpu['utilization'],
                    clock_graphics=smi_gpu['clock_graphics'],
                    clock_memory=smi_gpu['clock_memory'],
                    cuda_cores=self._estimate_cuda_cores(smi_gpu['name']),
                    compute_capability=torch_gpu['compute_capability'] if torch_gpu else None,
                    driver_version=smi_gpu['driver_version'],
                    cuda_version=torch_gpu['cuda_version'] if torch_gpu else None
                )
                
                gpu_list.append(gpu_info)
        
        elif torch_info['gpus']:
            print(f"✅ 通过PyTorch检测到 {len(torch_info['gpus'])} 个GPU")
            
            for torch_gpu in torch_info['gpus']:
                gpu_info = RealTimeGPUInfo(
                    device_id=torch_gpu['index'],
                    name=torch_gpu['name'],
                    total_memory=torch_gpu['memory_total'] * 1024 * 1024,
                    free_memory=torch_gpu['memory_free'] * 1024 * 1024,
                    used_memory=torch_gpu['memory_used'] * 1024 * 1024,
                    temperature=None,
                    power_usage=None,
                    utilization=None,
                    clock_graphics=None,
                    clock_memory=None,
                    cuda_cores=self._estimate_cuda_cores(torch_gpu['name']),
                    compute_capability=torch_gpu['compute_capability'],
                    driver_version=None,
                    cuda_version=torch_gpu['cuda_version']
                )
                gpu_list.append(gpu_info)
        
        else:
            print("❌ 未检测到NVIDIA GPU")
        
        return gpu_list
    
    def _estimate_cuda_cores(self, gpu_name: str) -> Optional[int]:
        """根据GPU名称估算CUDA核心数"""
        core_mapping = {
            'RTX 4090': 16384,
            'RTX 4080': 9728,
            'RTX 4070': 5888,
            'RTX 3090': 10496,
            'RTX 3080': 8704,
            'RTX 3070': 5888,
            'A100': 6912,
            'V100': 5120,
            'RTX A6000': 10752,
            'RTX A5000': 8192,
            'GTX 1080': 2560,
            'GTX 1070': 1920,
            'GTX 1060': 1280,
        }
        
        for key, cores in core_mapping.items():
            if key in gpu_name:
                return cores
        
        return None

class RealTimeTFLOPBenchmark:
    """实时TFLOP性能测试"""
    
    def __init__(self):
        self.torch_available = TORCH_AVAILABLE
    
    def run_fp32_benchmark(self, device_id: int, duration: int = 5) -> float:
        """运行FP32性能测试"""
        if not self.torch_available:
            return 0.0
        
        try:
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)
            
            # 预热
            print(f"🔥 GPU {device_id} FP32预热中...")
            for _ in range(10):
                a = torch.randn(1024, 1024, device=device)
                b = torch.randn(1024, 1024, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            # 实际测试
            print(f"⚡ GPU {device_id} FP32性能测试中 ({duration}秒)...")
            
            matrix_size = 2048  # 可根据显存调整
            operations = 0
            
            torch.cuda.synchronize()
            start_time = time.time()
            end_time = start_time + duration
            
            while time.time() < end_time:
                a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
                b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # 矩阵乘法运算量: 2 * n^3 (n^3次乘法 + n^3次加法)
                operations += 2 * (matrix_size ** 3)
            
            actual_duration = time.time() - start_time
            tflops = operations / actual_duration / 1e12
            
            return tflops
            
        except Exception as e:
            print(f"FP32测试失败: {e}")
            return 0.0
    
    def run_fp16_benchmark(self, device_id: int, duration: int = 5) -> float:
        """运行FP16性能测试"""
        if not self.torch_available:
            return 0.0
        
        try:
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)
            
            # 检查FP16支持
            if not torch.cuda.get_device_capability(device_id)[0] >= 6:
                print(f"GPU {device_id} 不支持FP16")
                return 0.0
            
            # 预热
            print(f"🔥 GPU {device_id} FP16预热中...")
            for _ in range(10):
                a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
                b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            # 实际测试
            print(f"⚡ GPU {device_id} FP16性能测试中 ({duration}秒)...")
            
            matrix_size = 4096  # FP16可以使用更大矩阵
            operations = 0
            
            torch.cuda.synchronize()
            start_time = time.time()
            end_time = start_time + duration
            
            while time.time() < end_time:
                a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
                b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                operations += 2 * (matrix_size ** 3)
            
            actual_duration = time.time() - start_time
            tflops = operations / actual_duration / 1e12
            
            return tflops
            
        except Exception as e:
            print(f"FP16测试失败: {e}")
            return 0.0
    
    def run_tensor_benchmark(self, device_id: int, duration: int = 5) -> float:
        """运行Tensor Core性能测试"""
        if not self.torch_available:
            return 0.0
        
        try:
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)
            
            # 检查Tensor Core支持
            major, minor = torch.cuda.get_device_capability(device_id)
            if major < 7:  # Volta架构开始支持Tensor Core
                print(f"GPU {device_id} 不支持Tensor Core")
                return 0.0
            
            # 预热
            print(f"🔥 GPU {device_id} Tensor Core预热中...")
            for _ in range(10):
                a = torch.randn(512, 512, device=device, dtype=torch.float16)
                b = torch.randn(512, 512, device=device, dtype=torch.float16)
                with torch.cuda.amp.autocast():
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            # 实际测试
            print(f"⚡ GPU {device_id} Tensor Core性能测试中 ({duration}秒)...")
            
            matrix_size = 8192  # Tensor Core优化的矩阵大小
            operations = 0
            
            torch.cuda.synchronize()
            start_time = time.time()
            end_time = start_time + duration
            
            while time.time() < end_time:
                a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
                b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
                
                with torch.cuda.amp.autocast():
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                operations += 2 * (matrix_size ** 3)
            
            actual_duration = time.time() - start_time
            tflops = operations / actual_duration / 1e12
            
            return tflops
            
        except Exception as e:
            print(f"Tensor Core测试失败: {e}")
            return 0.0

def print_gpu_status(gpu_info: RealTimeGPUInfo):
    """打印GPU状态信息"""
    print(f"\n🎮 GPU {gpu_info.device_id}: {gpu_info.name}")
    print("=" * 80)
    
    # 基本信息
    print(f"显存: {gpu_info.total_memory // (1024**3):.1f} GB "
          f"(使用: {gpu_info.used_memory // (1024**3):.1f} GB, "
          f"空闲: {gpu_info.free_memory // (1024**3):.1f} GB)")
    
    if gpu_info.temperature:
        print(f"温度: {gpu_info.temperature}°C")
    
    if gpu_info.power_usage:
        print(f"功耗: {gpu_info.power_usage:.1f} W")
    
    if gpu_info.utilization is not None:
        print(f"利用率: {gpu_info.utilization}%")
    
    if gpu_info.clock_graphics:
        print(f"核心频率: {gpu_info.clock_graphics} MHz")
    
    if gpu_info.clock_memory:
        print(f"显存频率: {gpu_info.clock_memory} MHz")
    
    if gpu_info.cuda_cores:
        print(f"CUDA核心: {gpu_info.cuda_cores:,}")
    
    if gpu_info.compute_capability:
        print(f"计算能力: {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}")
    
    if gpu_info.driver_version:
        print(f"驱动版本: {gpu_info.driver_version}")
    
    if gpu_info.cuda_version:
        print(f"CUDA版本: {gpu_info.cuda_version}")

def print_benchmark_results(gpu_info: RealTimeGPUInfo, fp32_tflops: float, fp16_tflops: float, tensor_tflops: float):
    """打印性能测试结果"""
    print(f"\n📊 GPU {gpu_info.device_id} 性能测试结果:")
    print("-" * 60)
    print(f"FP32性能:      {fp32_tflops:.2f} TFLOPS")
    print(f"FP16性能:      {fp16_tflops:.2f} TFLOPS")
    print(f"Tensor性能:    {tensor_tflops:.2f} TFLOPS")
    
    # 效率计算
    if gpu_info.power_usage and fp32_tflops > 0:
        efficiency = fp32_tflops / gpu_info.power_usage
        print(f"FP32效率:      {efficiency:.3f} TFLOPS/W")
    
    # 显存效率
    if fp32_tflops > 0:
        memory_efficiency = (gpu_info.total_memory // (1024**3)) / fp32_tflops
        print(f"显存效率:      {memory_efficiency:.1f} GB/TFLOP")

def main():
    parser = argparse.ArgumentParser(description="实时GPU TFLOP性能检测器")
    parser.add_argument("--list", action="store_true", help="仅列出GPU信息")
    parser.add_argument("--benchmark", action="store_true", help="运行性能测试")
    parser.add_argument("--duration", type=int, default=5, help="每项测试持续时间(秒)")
    parser.add_argument("--device", type=int, help="指定测试的GPU设备ID")
    parser.add_argument("--precision", choices=["fp32", "fp16", "tensor", "all"], 
                       default="all", help="测试精度")
    parser.add_argument("--export", type=str, help="导出结果到JSON文件")
    parser.add_argument("--monitor", action="store_true", help="实时监控模式")
    
    args = parser.parse_args()
    
    print("🚀 实时GPU TFLOP性能检测器")
    print("=" * 50)
    
    # 检查依赖
    print("\n📋 依赖检查:")
    print(f"PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"pynvml: {'✅' if NVIDIA_ML_AVAILABLE else '❌'}")
    print(f"GPUtil: {'✅' if GPUTIL_AVAILABLE else '❌'}")
    
    if not TORCH_AVAILABLE:
        print("\n⚠️ 警告: 未安装PyTorch，无法运行性能测试")
        print("安装命令: pip install torch")
    
    # 检测GPU
    detector = RealTimeGPUDetector()
    gpus = detector.detect_gpus()
    
    if not gpus:
        print("\n❌ 未检测到GPU或GPU不可用")
        sys.exit(1)
    
    print(f"\n✅ 检测到 {len(gpus)} 个GPU")
    
    # 显示GPU信息
    for gpu in gpus:
        print_gpu_status(gpu)
    
    if args.list:
        return
    
    # 运行性能测试
    if args.benchmark and TORCH_AVAILABLE:
        benchmark = RealTimeTFLOPBenchmark()
        results = {}
        
        target_gpus = [gpus[args.device]] if args.device is not None else gpus
        
        for gpu in target_gpus:
            print(f"\n🏃 开始测试GPU {gpu.device_id}: {gpu.name}")
            
            fp32_tflops = fp16_tflops = tensor_tflops = 0.0
            
            if args.precision in ["fp32", "all"]:
                fp32_tflops = benchmark.run_fp32_benchmark(gpu.device_id, args.duration)
            
            if args.precision in ["fp16", "all"]:
                fp16_tflops = benchmark.run_fp16_benchmark(gpu.device_id, args.duration)
            
            if args.precision in ["tensor", "all"]:
                tensor_tflops = benchmark.run_tensor_benchmark(gpu.device_id, args.duration)
            
            print_benchmark_results(gpu, fp32_tflops, fp16_tflops, tensor_tflops)
            
            results[f"gpu_{gpu.device_id}"] = {
                "name": gpu.name,
                "fp32_tflops": fp32_tflops,
                "fp16_tflops": fp16_tflops,
                "tensor_tflops": tensor_tflops,
                "memory_gb": gpu.total_memory // (1024**3),
                "power_usage": gpu.power_usage,
                "temperature": gpu.temperature
            }
        
        # 导出结果
        if args.export:
            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n📁 结果已导出到: {args.export}")
    
    # 实时监控模式
    if args.monitor:
        print("\n🔄 进入实时监控模式 (Ctrl+C退出)")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print("🔄 实时GPU状态监控")
                print("=" * 50)
                
                # 重新检测GPU状态
                gpus = detector.detect_gpus()
                for gpu in gpus:
                    print_gpu_status(gpu)
                
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n👋 监控结束")

if __name__ == "__main__":
    main() 