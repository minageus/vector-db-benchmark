from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time
import psutil
from typing import Dict, List
import numpy as np

class MilvusLoader:
    """Load data into Milvus and track metrics"""
    
    def __init__(self, host='localhost', port=19530):
        self.host = host
        self.port = port
        self.collection = None
        self.metrics = {
            'load_time': 0,
            'vectors_per_second': 0,
            'memory_used_mb': 0,
            'storage_size_mb': 0,
            'index_build_time': 0
        }
    
    def connect(self):
        """Connect to Milvus server"""
        connections.connect(host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")
    
    def create_collection(self, collection_name: str, dimension: int, drop_existing=True):
        """Create a collection with specified dimension"""
        if drop_existing and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="category", dtype=DataType.INT64),
            FieldSchema(name="price", dtype=DataType.FLOAT),
        ]
        
        schema = CollectionSchema(fields=fields, description="Benchmark collection")
        self.collection = Collection(name=collection_name, schema=schema)
        print(f"Created collection: {collection_name}")
    
    def load_data(
        self, 
        ids: np.ndarray, 
        vectors: np.ndarray,
        metadata: Dict = None,
        batch_size: int = 10000
    ):
        """Load vectors into Milvus in batches"""
        n_vectors = len(ids)
        n_batches = (n_vectors + batch_size - 1) // batch_size
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_vectors)
            
            batch_data = [
                ids[start_idx:end_idx].tolist(),
                vectors[start_idx:end_idx].tolist(),
                metadata['category'][start_idx:end_idx].tolist() if metadata is not None else [0] * (end_idx - start_idx),
                metadata['price'][start_idx:end_idx].tolist() if metadata is not None else [0.0] * (end_idx - start_idx),
            ]
            
            self.collection.insert(batch_data)
            
            if (i + 1) % 10 == 0:
                print(f"Loaded {end_idx}/{n_vectors} vectors ({(end_idx/n_vectors)*100:.1f}%)")
        
        self.collection.flush()
        
        load_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024
        
        self.metrics['load_time'] = load_time
        self.metrics['vectors_per_second'] = n_vectors / load_time
        self.metrics['memory_used_mb'] = mem_after - mem_before
        
        print(f"\nLoad complete:")
        print(f"  Time: {load_time:.2f}s")
        print(f"  Throughput: {self.metrics['vectors_per_second']:.2f} vectors/s")
        print(f"  Memory used: {self.metrics['memory_used_mb']:.2f} MB")
    
    def create_index(self, index_type='HNSW', metric_type='L2', index_params=None, wait_timeout=600):
        """Create index on the collection and wait for it to complete
        
        Args:
            index_type: Type of index (HNSW, IVF_FLAT, etc.)
            metric_type: Distance metric (L2, IP)
            index_params: Index parameters
            wait_timeout: Max seconds to wait for index to build (default 10 minutes)
        """
        if index_params is None:
            index_params = {
                "M": 16,
                "efConstruction": 200
            }
        
        index_config = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": index_params
        }
        
        start_time = time.time()
        print(f"Creating {index_type} index (this may take several minutes for large datasets)...")
        self.collection.create_index(field_name="embedding", index_params=index_config)
        
        # Wait for index to be fully built
        print("Waiting for index to finish building...")
        wait_start = time.time()
        last_progress = -1
        
        while time.time() - wait_start < wait_timeout:
            # Check index building progress
            index_info = self.collection.index()
            index_progress = utility.index_building_progress(self.collection.name)
            
            # Progress is a dict with 'total_rows' and 'indexed_rows'
            total = index_progress.get('total_rows', 0)
            indexed = index_progress.get('indexed_rows', 0)
            
            if total > 0:
                progress_pct = (indexed / total) * 100
                if int(progress_pct) != last_progress and int(progress_pct) % 10 == 0:
                    print(f"  Index progress: {progress_pct:.0f}% ({indexed:,}/{total:,} rows)")
                    last_progress = int(progress_pct)
                
                if indexed >= total:
                    index_time = time.time() - start_time
                    self.metrics['index_build_time'] = index_time
                    print(f"Index built successfully in {index_time:.1f}s")
                    return
            
            time.sleep(2)
        
        # If we reach here, timeout occurred
        index_time = time.time() - start_time
        self.metrics['index_build_time'] = index_time
        print(f"WARNING: Index build timed out after {wait_timeout}s, but will try to continue")
        print(f"  (Index may still be building in background)")
    
    def load_collection(self, timeout=300):
        """Load collection into memory and wait for it to be ready"""
        from pymilvus import utility
        
        self.collection.load()
        print("Loading collection into memory...")
        
        # Wait for collection to be loaded
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = utility.load_state(self.collection.name)
            if state.name == 'Loaded':
                print("Collection loaded into memory")
                return
            time.sleep(1)
        
        raise TimeoutError(f"Collection failed to load within {timeout} seconds")
    
    def get_storage_size(self) -> float:
        """Get storage size in MB"""
        stats = self.collection.get_compaction_state()
        # This is a simplified version; actual implementation may vary
        return 0.0  # Placeholder
