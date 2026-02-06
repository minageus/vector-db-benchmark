"""
Resource Monitor for Benchmark Tracking

Monitors system resources during benchmark execution:
- CPU usage (per-core and total)
- Memory usage (RSS, VMS, available)
- Disk I/O (read/write bytes, IOPS)
- GPU utilization (if available)
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ResourceSnapshot:
    """Single snapshot of resource usage"""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    memory_available_mb: float
    disk_read_mb: float
    disk_write_mb: float
    disk_read_count: int
    disk_write_count: int
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None


class ResourceMonitor:
    """Monitor system resources during operations"""
    
    def __init__(self, interval: float = 0.5, enable_gpu: bool = True):
        """
        Initialize resource monitor
        
        Args:
            interval: Sampling interval in seconds
            enable_gpu: Try to monitor GPU if available
        """
        self.interval = interval
        self.enable_gpu = enable_gpu
        self.snapshots: List[ResourceSnapshot] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        
        self._initial_disk_io = psutil.disk_io_counters()
        
        self.gpu_available = False
        if enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                self._pynvml = pynvml
            except:
                self.gpu_available = False
    
    def start(self):
        if self._monitoring:
            return
        
        self._monitoring = True
        self.snapshots = []
        self._initial_disk_io = psutil.disk_io_counters()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        while self._monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        
        cpu_percent = self._process.cpu_percent()
        cpu_per_core = psutil.cpu_percent(percpu=True)
        
        mem_info = self._process.memory_info()
        mem_rss_mb = mem_info.rss / (1024 * 1024)
        mem_vms_mb = mem_info.vms / (1024 * 1024)
        
        sys_mem = psutil.virtual_memory()
        mem_percent = sys_mem.percent
        mem_available_mb = sys_mem.available / (1024 * 1024)
        
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes - self._initial_disk_io.read_bytes) / (1024 * 1024)
        disk_write_mb = (disk_io.write_bytes - self._initial_disk_io.write_bytes) / (1024 * 1024)
        disk_read_count = disk_io.read_count - self._initial_disk_io.read_count
        disk_write_count = disk_io.write_count - self._initial_disk_io.write_count
        
        gpu_util = None
        gpu_mem_mb = None
        if self.gpu_available:
            try:
                gpu_util = self._pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_mb = mem_info.used / (1024 * 1024)
            except:
                pass
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_rss_mb=mem_rss_mb,
            memory_vms_mb=mem_vms_mb,
            memory_percent=mem_percent,
            memory_available_mb=mem_available_mb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            disk_read_count=disk_read_count,
            disk_write_count=disk_write_count,
            gpu_utilization=gpu_util,
            gpu_memory_used_mb=gpu_mem_mb
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all snapshots
        
        Returns:
            Dictionary with min, max, mean, p50, p95, p99 for each metric
        """
        if not self.snapshots:
            return {}
        
        cpu_percents = [s.cpu_percent for s in self.snapshots]
        mem_rss = [s.memory_rss_mb for s in self.snapshots]
        mem_percent = [s.memory_percent for s in self.snapshots]
        disk_read = [s.disk_read_mb for s in self.snapshots]
        disk_write = [s.disk_write_mb for s in self.snapshots]
        
        stats = {
            'cpu': self._compute_stats(cpu_percents),
            'memory_rss_mb': self._compute_stats(mem_rss),
            'memory_percent': self._compute_stats(mem_percent),
            'disk_read_mb': {
                'total': disk_read[-1] if disk_read else 0,
                'rate_mb_s': (disk_read[-1] - disk_read[0]) / (len(self.snapshots) * self.interval) if len(disk_read) > 1 else 0
            },
            'disk_write_mb': {
                'total': disk_write[-1] if disk_write else 0,
                'rate_mb_s': (disk_write[-1] - disk_write[0]) / (len(self.snapshots) * self.interval) if len(disk_write) > 1 else 0
            },
            'duration_seconds': len(self.snapshots) * self.interval
        }
        
        if self.gpu_available and self.snapshots[0].gpu_utilization is not None:
            gpu_utils = [s.gpu_utilization for s in self.snapshots if s.gpu_utilization is not None]
            gpu_mems = [s.gpu_memory_used_mb for s in self.snapshots if s.gpu_memory_used_mb is not None]
            
            if gpu_utils:
                stats['gpu_utilization'] = self._compute_stats(gpu_utils)
            if gpu_mems:
                stats['gpu_memory_mb'] = self._compute_stats(gpu_mems)
        
        return stats
    
    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute statistics for a list of values"""
        if not values:
            return {}
        
        arr = np.array(values)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'std': float(np.std(arr))
        }
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage"""
        if not self.snapshots:
            return {}
        
        return {
            'cpu_peak_percent': max(s.cpu_percent for s in self.snapshots),
            'memory_peak_mb': max(s.memory_rss_mb for s in self.snapshots),
            'disk_read_total_mb': self.snapshots[-1].disk_read_mb if self.snapshots else 0,
            'disk_write_total_mb': self.snapshots[-1].disk_write_mb if self.snapshots else 0,
        }
    
    def print_summary(self):
        """Print summary of resource usage"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("RESOURCE USAGE SUMMARY")
        print("="*60)
        
        if 'cpu' in stats:
            print(f"\nCPU Usage:")
            print(f"  Mean: {stats['cpu']['mean']:.1f}%")
            print(f"  Peak: {stats['cpu']['max']:.1f}%")
            print(f"  P95:  {stats['cpu']['p95']:.1f}%")
        
        if 'memory_rss_mb' in stats:
            print(f"\nMemory Usage (RSS):")
            print(f"  Mean: {stats['memory_rss_mb']['mean']:.1f} MB")
            print(f"  Peak: {stats['memory_rss_mb']['max']:.1f} MB")
        
        if 'disk_read_mb' in stats:
            print(f"\nDisk I/O:")
            print(f"  Read:  {stats['disk_read_mb']['total']:.1f} MB ({stats['disk_read_mb']['rate_mb_s']:.2f} MB/s)")
            print(f"  Write: {stats['disk_write_mb']['total']:.1f} MB ({stats['disk_write_mb']['rate_mb_s']:.2f} MB/s)")
        
        if 'gpu_utilization' in stats:
            print(f"\nGPU Usage:")
            print(f"  Utilization Mean: {stats['gpu_utilization']['mean']:.1f}%")
            print(f"  Memory Mean: {stats['gpu_memory_mb']['mean']:.1f} MB")
        
        print(f"\nDuration: {stats.get('duration_seconds', 0):.1f} seconds")
        print("="*60)
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

if __name__ == "__main__":
    import time
    
    print("Testing Resource Monitor...")
    
    with ResourceMonitor(interval=0.1) as monitor:
        print("Doing some CPU-intensive work...")
        for i in range(5):
            _ = sum(range(10000000))
            time.sleep(0.5)
    
    monitor.print_summary()
    
    stats = monitor.get_stats()
    print("\nDetailed Stats:")
    import json
    print(json.dumps(stats, indent=2))
