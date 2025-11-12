import time
import threading
import queue
import numpy as np
from typing import List, Callable, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LoadTestResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    qps: float
    latencies: List[float]
    p50: float
    p95: float
    p99: float
    mean: float

class ConcurrentTester:
    """Run concurrent load tests"""
    
    def __init__(self, n_clients: int = 10):
        self.n_clients = n_clients
        self.results_queue = queue.Queue()
    
    def worker(
        self, 
        worker_id: int, 
        search_func: Callable,
        queries: np.ndarray,
        stop_event: threading.Event
    ):
        """Worker thread for executing queries"""
        query_idx = 0
        n_queries = len(queries)
        
        while not stop_event.is_set():
            query = queries[query_idx % n_queries]
            
            start_time = time.time()
            try:
                search_func(query)
                latency = time.time() - start_time
                self.results_queue.put(('success', latency))
            except Exception as e:
                self.results_queue.put(('failure', str(e)))
            
            query_idx += 1
    
    def run_load_test(
        self,
        search_func: Callable,
        queries: np.ndarray,
        duration_seconds: int = 60
    ) -> LoadTestResult:
        """Run load test for specified duration"""
        
        print(f"Starting load test with {self.n_clients} clients for {duration_seconds}s...")
        
        stop_event = threading.Event()
        threads = []
        
        # Start worker threads
        start_time = time.time()
        for i in range(self.n_clients):
            t = threading.Thread(
                target=self.worker,
                args=(i, search_func, queries, stop_event)
            )
            t.start()
            threads.append(t)
        
        # Run for specified duration
        time.sleep(duration_seconds)
        stop_event.set()
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        latencies = []
        failures = []
        
        while not self.results_queue.empty():
            result_type, value = self.results_queue.get()
            if result_type == 'success':
                latencies.append(value)
            else:
                failures.append(value)
        
        latencies = np.array(latencies)
        
        result = LoadTestResult(
            total_requests=len(latencies) + len(failures),
            successful_requests=len(latencies),
            failed_requests=len(failures),
            total_time=total_time,
            qps=len(latencies) / total_time,
            latencies=latencies.tolist(),
            p50=np.percentile(latencies, 50),
            p95=np.percentile(latencies, 95),
            p99=np.percentile(latencies, 99),
            mean=np.mean(latencies)
        )
        
        print(f"Load test complete:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Successful: {result.successful_requests}")
        print(f"  Failed: {result.failed_requests}")
        print(f"  QPS: {result.qps:.2f}")
        print(f"  P50 latency: {result.p50*1000:.2f}ms")
        print(f"  P95 latency: {result.p95*1000:.2f}ms")
        
        return result