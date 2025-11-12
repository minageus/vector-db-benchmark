from pymilvus import Collection
import time
import numpy as np
from typing import List, Dict

class MilvusQueryExecutor:
    """Execute queries on Milvus and measure performance"""
    
    def __init__(self, collection: Collection):
        self.collection = collection
    
    def search(
        self, 
        query_vectors: np.ndarray,
        top_k: int = 10,
        metric_type: str = 'L2',
        search_params: Dict = None,
        filters: List[Dict] = None
    ) -> Dict:
        """Execute search queries and return metrics"""
        if search_params is None:
            search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        
        latencies = []
        results_list = []
        
        for i, query_vec in enumerate(query_vectors):
            expr = None
            if filters and i < len(filters):
                # Build filter expression
                filter_dict = filters[i]
                if 'category' in filter_dict:
                    cats = filter_dict['category']['$in']
                    expr = f"category in {cats}"
            
            start_time = time.time()
            
            results = self.collection.search(
                data=[query_vec.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr
            )
            
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Convert to ms
            results_list.append(results)
        
        latencies = np.array(latencies)
        
        return {
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'qps': 1000 / np.mean(latencies),
            'all_latencies': latencies,
            'results': results_list
        }