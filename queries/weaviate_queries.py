import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
import time
import numpy as np
from typing import List, Dict

class WeaviateQueryExecutor:
    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.collection = client.collections.get(collection_name)
    
    def search(
        self, 
        query_vectors: np.ndarray,
        top_k: int = 10,
        metric_type: str = 'IP',
        filters: List[Dict] = None
    ) -> Dict:
        latencies = []
        results_list = []
        
        for i, query_vec in enumerate(query_vectors):
            where_filter = None
            if filters and i < len(filters):
                filter_dict = filters[i]
                if 'category' in filter_dict:
                    where_filter = Filter.by_property("category").equal(filter_dict['category']['$in'][0])
            
            start_time = time.time()
            
            if where_filter:
                results = self.collection.query.near_vector(
                    near_vector=query_vec.tolist(),
                    limit=top_k,
                    filters=where_filter,
                    return_properties=["vectorId", "category", "price"]
                )
            else:
                results = self.collection.query.near_vector(
                    near_vector=query_vec.tolist(),
                    limit=top_k,
                    return_properties=["vectorId", "category", "price"]
                )
            
            latency = time.time() - start_time
            latencies.append(latency * 1000)
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
