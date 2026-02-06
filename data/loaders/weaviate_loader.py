import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.config import VectorDistances
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout
import time
import psutil
from typing import Dict, List
import numpy as np
from tqdm import tqdm

class WeaviateLoader:

    INDEX_CONFIG = {
        'type': 'HNSW',
        'ef': 200,
        'efConstruction': 128,
        'maxConnections': 64
    }
    
    def __init__(self, host='localhost', port=8080, grpc_port=50051):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.client = None
        self.collection_name = "BenchmarkVector"
        self.collection = None
        self.distance_metric = None
        self.metrics = {
            'load_time': 0,
            'vectors_per_second': 0,
            'memory_used_mb': 0,
            'storage_size_mb': 0,
            'failed_inserts': 0
        }
    
    def connect(self):

        self.client = weaviate.connect_to_local(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    init=30,
                    query=60,
                    insert=300  
                )
            )
        )
        print(f"Connected to Weaviate at {self.host}:{self.port} (gRPC: {self.grpc_port})")
    
    def create_schema(self, dimension: int, drop_existing=True, metric_type: str = 'L2'):

        if drop_existing:
            try:
                self.client.collections.delete(self.collection_name)
            except:
                pass
        
        metric_map = {
            'L2': VectorDistances.L2_SQUARED,
            'l2': VectorDistances.L2_SQUARED,
            'IP': VectorDistances.COSINE,  
            'ip': VectorDistances.COSINE,
            'cosine': VectorDistances.COSINE,
            'angular': VectorDistances.COSINE,
        }
        distance = metric_map.get(metric_type, VectorDistances.L2_SQUARED)
        self.distance_metric = metric_type
        
        self.collection = self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=distance,
                ef=self.INDEX_CONFIG['ef'],
                ef_construction=self.INDEX_CONFIG['efConstruction'],
                max_connections=self.INDEX_CONFIG['maxConnections']
            ),
            properties=[
                Property(name="vectorId", data_type=DataType.INT),
                Property(name="category", data_type=DataType.INT),
                Property(name="price", data_type=DataType.NUMBER)
            ]
        )
        print(f"Created collection: {self.collection_name}")
        print(f"  Index: HNSW (ef={self.INDEX_CONFIG['ef']}, efConstruction={self.INDEX_CONFIG['efConstruction']}, M={self.INDEX_CONFIG['maxConnections']})")
        print(f"  Distance metric: {metric_type} -> {distance}")
    
    def load_data(
        self, 
        ids: np.ndarray, 
        vectors: np.ndarray,
        metadata: Dict = None,
        batch_size: int = 100
    ):

        n_vectors = len(ids)
        dimension = vectors.shape[1] if len(vectors.shape) > 1 else 1
        failed = 0
        
        if dimension > 500:
            batch_size = min(batch_size, 50)
        elif dimension > 200:
            batch_size = min(batch_size, 100)
        else:
            batch_size = min(batch_size, 200)
        
        print(f"Using batch size: {batch_size} (dimension: {dimension})")
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        if self.collection is None:
            self.collection = self.client.collections.get(self.collection_name)
        
        with self.collection.batch.fixed_size(batch_size=batch_size, concurrent_requests=2) as batch:
            for i in tqdm(range(n_vectors), desc="Loading vectors"):
                properties = {
                    "vectorId": int(ids[i]),
                    "category": int(metadata['category'][i]) if metadata is not None else 0,
                    "price": float(metadata['price'][i]) if metadata is not None else 0.0
                }
                
                try:
                    batch.add_object(
                        properties=properties,
                        vector=vectors[i].tolist()
                    )
                except Exception as e:
                    failed += 1
                    if failed < 10:
                        print(f"Failed to insert vector {i}: {e}")
        
        load_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024
        
        self.metrics['load_time'] = load_time
        self.metrics['vectors_per_second'] = n_vectors / load_time
        self.metrics['memory_used_mb'] = mem_after - mem_before
        self.metrics['failed_inserts'] = failed
        
        print(f"\nLoad complete:")
        print(f"  Time: {load_time:.2f}s")
        print(f"  Throughput: {self.metrics['vectors_per_second']:.2f} vectors/s")
        print(f"  Memory used: {self.metrics['memory_used_mb']:.2f} MB")
        print(f"  Failed inserts: {failed}")