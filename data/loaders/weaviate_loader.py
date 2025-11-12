import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
import time
import psutil
from typing import Dict, List
import numpy as np
from tqdm import tqdm

class WeaviateLoader:
    """Load data into Weaviate and track metrics"""
    
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.client = None
        self.collection_name = "BenchmarkVector"
        self.collection = None
        self.metrics = {
            'load_time': 0,
            'vectors_per_second': 0,
            'memory_used_mb': 0,
            'storage_size_mb': 0,
            'failed_inserts': 0
        }
    
    def connect(self):
        """Connect to Weaviate server"""
        self.client = weaviate.connect_to_local(
            host=self.host,
            port=self.port
        )
        print(f"Connected to Weaviate at {self.host}:{self.port}")
    
    def create_schema(self, dimension: int, drop_existing=True):
        """Create schema for vectors"""
        if drop_existing:
            try:
                self.client.collections.delete(self.collection_name)
            except:
                pass
        
        self.collection = self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                ef=200,
                ef_construction=128,
                max_connections=64
            ),
            properties=[
                Property(name="vectorId", data_type=DataType.INT),
                Property(name="category", data_type=DataType.INT),
                Property(name="price", data_type=DataType.NUMBER)
            ]
        )
        print(f"Created collection: {self.collection_name}")
    
    def load_data(
        self, 
        ids: np.ndarray, 
        vectors: np.ndarray,
        metadata: Dict = None,
        batch_size: int = 100
    ):
        """Load vectors into Weaviate in batches"""
        n_vectors = len(ids)
        failed = 0
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        # Get or create collection reference
        if self.collection is None:
            self.collection = self.client.collections.get(self.collection_name)
        
        with self.collection.batch.dynamic() as batch:
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