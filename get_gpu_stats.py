#!/usr/bin/env python3
"""
Skrypt do zbierania szczegółowych statystyk GPU
Wspiera różne metody: nvidia-smi, pynvml, torch, GPUtil
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_gpu_stats_nvidia_smi():
    """Zbiera statystyki GPU używając nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,driver_version,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,compute_mode,pstate',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 14:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'driver_version': parts[2],
                        'pci_bus_id': parts[3],
                        'memory_total_mb': float(parts[4]),
                        'memory_used_mb': float(parts[5]),
                        'memory_free_mb': float(parts[6]),
                        'gpu_utilization_percent': float(parts[7]) if parts[7] != '[N/A]' else None,
                        'memory_utilization_percent': float(parts[8]) if parts[8] != '[N/A]' else None,
                        'temperature_c': float(parts[9]) if parts[9] != '[N/A]' else None,
                        'power_draw_w': float(parts[10]) if parts[10] != '[N/A]' else None,
                        'power_limit_w': float(parts[11]) if parts[11] != '[N/A]' else None,
                        'compute_mode': parts[12],
                        'performance_state': parts[13]
                    })
        
        return {'method': 'nvidia-smi', 'gpus': gpus, 'success': True}
    
    except FileNotFoundError:
        return {'method': 'nvidia-smi', 'error': 'nvidia-smi not found', 'success': False}
    except Exception as e:
        return {'method': 'nvidia-smi', 'error': str(e), 'success': False}


def get_gpu_stats_pynvml():
    """Zbiera statystyki GPU używając pynvml (NVIDIA Management Library)"""
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Podstawowe informacje
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Pamięć
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Wykorzystanie
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                mem_util = utilization.memory
            except:
                gpu_util = None
                mem_util = None
            
            # Temperatura
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None
            
            # Moc
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except:
                power = None
                power_limit = None
            
            # Wersja sterownika
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_version, bytes):
                    driver_version = driver_version.decode('utf-8')
            except:
                driver_version = None
            
            # CUDA version
            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_major = cuda_version // 1000
                cuda_minor = (cuda_version % 1000) // 10
                cuda_version_str = f"{cuda_major}.{cuda_minor}"
            except:
                cuda_version_str = None
            
            # Performance state
            try:
                pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
                pstate_str = f"P{pstate}"
            except:
                pstate_str = None
            
            # Compute mode
            try:
                compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                compute_mode_map = {
                    0: 'Default',
                    1: 'Exclusive_Thread',
                    2: 'Prohibited',
                    3: 'Exclusive_Process'
                }
                compute_mode_str = compute_mode_map.get(compute_mode, f"Unknown({compute_mode})")
            except:
                compute_mode_str = None
            
            gpus.append({
                'index': i,
                'name': name,
                'driver_version': driver_version,
                'cuda_version': cuda_version_str,
                'memory_total_mb': mem_info.total / (1024**2),
                'memory_used_mb': mem_info.used / (1024**2),
                'memory_free_mb': mem_info.free / (1024**2),
                'gpu_utilization_percent': gpu_util,
                'memory_utilization_percent': mem_util,
                'temperature_c': temp,
                'power_draw_w': power,
                'power_limit_w': power_limit,
                'performance_state': pstate_str,
                'compute_mode': compute_mode_str
            })
        
        pynvml.nvmlShutdown()
        
        return {'method': 'pynvml', 'gpus': gpus, 'success': True}
    
    except ImportError:
        return {'method': 'pynvml', 'error': 'pynvml not installed (pip install nvidia-ml-py3)', 'success': False}
    except Exception as e:
        return {'method': 'pynvml', 'error': str(e), 'success': False}


def get_gpu_stats_torch():
    """Zbiera statystyki GPU używając PyTorch"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'method': 'pytorch', 'error': 'CUDA not available', 'success': False}
        
        gpus = []
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            
            # Pamięć
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**2)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**2)
            mem_total = props.total_memory / (1024**2)
            
            gpus.append({
                'index': i,
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
                'memory_total_mb': mem_total,
                'memory_allocated_mb': mem_allocated,
                'memory_reserved_mb': mem_reserved,
                'memory_free_mb': mem_total - mem_reserved,
                'max_threads_per_multi_processor': props.max_threads_per_multi_processor,
                'warp_size': props.warp_size
            })
        
        return {
            'method': 'pytorch',
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'gpus': gpus,
            'success': True
        }
    
    except ImportError:
        return {'method': 'pytorch', 'error': 'PyTorch not installed', 'success': False}
    except Exception as e:
        return {'method': 'pytorch', 'error': str(e), 'success': False}


def get_gpu_stats_gputil():
    """Zbiera statystyki GPU używając GPUtil"""
    try:
        import GPUtil
        
        gpus_list = GPUtil.getGPUs()
        
        if not gpus_list:
            return {'method': 'GPUtil', 'error': 'No GPUs found', 'success': False}
        
        gpus = []
        for gpu in gpus_list:
            gpus.append({
                'index': gpu.id,
                'name': gpu.name,
                'uuid': gpu.uuid,
                'memory_total_mb': gpu.memoryTotal,
                'memory_used_mb': gpu.memoryUsed,
                'memory_free_mb': gpu.memoryFree,
                'memory_utilization_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                'gpu_utilization_percent': gpu.load * 100,
                'temperature_c': gpu.temperature
            })
        
        return {'method': 'GPUtil', 'gpus': gpus, 'success': True}
    
    except ImportError:
        return {'method': 'GPUtil', 'error': 'GPUtil not installed (pip install gputil)', 'success': False}
    except Exception as e:
        return {'method': 'GPUtil', 'error': str(e), 'success': False}


def format_gpu_stats(stats_data):
    """Formatuje statystyki GPU do czytelnej formy tekstowej"""
    output = []
    output.append("=" * 80)
    output.append("GPU STATISTICS")
    output.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    
    for method_name, method_result in stats_data.items():
        output.append(f"\n{'='*80}")
        output.append(f"Method: {method_name.upper()}")
        output.append(f"{'='*80}")
        
        if not method_result.get('success', False):
            output.append(f"❌ Error: {method_result.get('error', 'Unknown error')}")
            continue
        
        # Dodaj informacje o wersji CUDA jeśli dostępne
        if 'cuda_version' in method_result:
            output.append(f"\n🔧 CUDA Version: {method_result['cuda_version']}")
        if 'cudnn_version' in method_result:
            output.append(f"🔧 cuDNN Version: {method_result['cudnn_version']}")
        
        gpus = method_result.get('gpus', [])
        if not gpus:
            output.append("No GPU information available")
            continue
        
        for gpu in gpus:
            output.append(f"\n--- GPU {gpu.get('index', 'N/A')} ---")
            output.append(f"Name: {gpu.get('name', 'N/A')}")
            
            if 'driver_version' in gpu:
                output.append(f"Driver Version: {gpu.get('driver_version', 'N/A')}")
            
            if 'cuda_version' in gpu:
                output.append(f"CUDA Version: {gpu.get('cuda_version', 'N/A')}")
            
            if 'compute_capability' in gpu:
                output.append(f"Compute Capability: {gpu.get('compute_capability', 'N/A')}")
            
            if 'pci_bus_id' in gpu:
                output.append(f"PCI Bus ID: {gpu.get('pci_bus_id', 'N/A')}")
            
            if 'uuid' in gpu:
                output.append(f"UUID: {gpu.get('uuid', 'N/A')}")
            
            # Pamięć
            mem_total = gpu.get('memory_total_mb')
            mem_used = gpu.get('memory_used_mb')
            mem_free = gpu.get('memory_free_mb')
            
            if mem_total is not None:
                output.append(f"\n💾 Memory:")
                output.append(f"  Total: {mem_total:.2f} MB ({mem_total/1024:.2f} GB)")
                if mem_used is not None:
                    output.append(f"  Used:  {mem_used:.2f} MB ({mem_used/1024:.2f} GB) - {(mem_used/mem_total*100):.1f}%")
                if mem_free is not None:
                    output.append(f"  Free:  {mem_free:.2f} MB ({mem_free/1024:.2f} GB) - {(mem_free/mem_total*100):.1f}%")
            
            if 'memory_allocated_mb' in gpu:
                output.append(f"  Allocated: {gpu['memory_allocated_mb']:.2f} MB ({gpu['memory_allocated_mb']/1024:.2f} GB)")
            
            if 'memory_reserved_mb' in gpu:
                output.append(f"  Reserved:  {gpu['memory_reserved_mb']:.2f} MB ({gpu['memory_reserved_mb']/1024:.2f} GB)")
            
            # Wykorzystanie
            gpu_util = gpu.get('gpu_utilization_percent')
            mem_util = gpu.get('memory_utilization_percent')
            
            if gpu_util is not None or mem_util is not None:
                output.append(f"\n📊 Utilization:")
                if gpu_util is not None:
                    output.append(f"  GPU:    {gpu_util:.1f}%")
                if mem_util is not None:
                    output.append(f"  Memory: {mem_util:.1f}%")
            
            # Temperatura
            temp = gpu.get('temperature_c')
            if temp is not None:
                output.append(f"\n🌡️  Temperature: {temp}°C")
            
            # Moc
            power_draw = gpu.get('power_draw_w')
            power_limit = gpu.get('power_limit_w')
            
            if power_draw is not None or power_limit is not None:
                output.append(f"\n⚡ Power:")
                if power_draw is not None:
                    output.append(f"  Draw:  {power_draw:.2f} W")
                if power_limit is not None:
                    output.append(f"  Limit: {power_limit:.2f} W")
                    if power_draw is not None:
                        output.append(f"  Usage: {(power_draw/power_limit*100):.1f}%")
            
            # Performance state
            pstate = gpu.get('performance_state')
            if pstate is not None:
                output.append(f"\n🎯 Performance State: {pstate}")
            
            # Compute mode
            compute_mode = gpu.get('compute_mode')
            if compute_mode is not None:
                output.append(f"🔒 Compute Mode: {compute_mode}")
            
            # Inne informacje specyficzne dla PyTorch
            if 'multi_processor_count' in gpu:
                output.append(f"\n🔢 Hardware Details:")
                output.append(f"  Multi-Processor Count: {gpu['multi_processor_count']}")
                if 'max_threads_per_multi_processor' in gpu:
                    output.append(f"  Max Threads per MP: {gpu['max_threads_per_multi_processor']}")
                if 'warp_size' in gpu:
                    output.append(f"  Warp Size: {gpu['warp_size']}")
    
    output.append("\n" + "="*80)
    
    return "\n".join(output)


def main():
    """Główna funkcja zbierająca wszystkie dostępne statystyki GPU"""
    
    print("Collecting GPU statistics...\n")
    
    stats = {}
    
    # Próbuj wszystkie metody
    methods = [
        ('nvidia-smi', get_gpu_stats_nvidia_smi),
        ('pynvml', get_gpu_stats_pynvml),
        ('pytorch', get_gpu_stats_torch),
        ('gputil', get_gpu_stats_gputil)
    ]
    
    for method_name, method_func in methods:
        print(f"Trying {method_name}...", end=" ")
        result = method_func()
        stats[method_name] = result
        if result.get('success'):
            print("✅")
        else:
            print(f"❌ ({result.get('error', 'Unknown error')})")
    
    # Formatuj i wyświetl wyniki
    formatted_output = format_gpu_stats(stats)
    print("\n" + formatted_output)
    
    # Zapisz do pliku JSON
    output_dir = Path("reports/system_info")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = output_dir / f"gpu_stats_{timestamp}.json"
    txt_file = output_dir / f"gpu_stats_{timestamp}.txt"
    
    # Zapisz JSON
    stats_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'stats': stats
    }
    
    with open(json_file, 'w') as f:
        json.dump(stats_with_timestamp, f, indent=2)
    
    # Zapisz TXT
    with open(txt_file, 'w') as f:
        f.write(formatted_output)
    
    print(f"\n📁 Results saved:")
    print(f"   JSON: {json_file}")
    print(f"   TXT:  {txt_file}")
    
    # Sprawdź czy są jakieś udane wyniki
    successful_methods = [name for name, result in stats.items() if result.get('success')]
    
    if not successful_methods:
        print("\n⚠️  WARNING: No GPU statistics could be collected!")
        print("   This might mean:")
        print("   - No NVIDIA GPU is available")
        print("   - NVIDIA drivers are not installed")
        print("   - Required Python packages are not installed")
        return 1
    else:
        print(f"\n✅ Successfully collected GPU stats using: {', '.join(successful_methods)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

