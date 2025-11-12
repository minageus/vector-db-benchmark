import numpy as np
import pandas as pd
from typing import Tuple, List
import h5py
from tqdm import tqdm

class VectorDataGenerator:
    """Generate synthetic vector data for benchmarking"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_random_vectors(
        self, 
        n_vectors: int, 
        dimension: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random normalized vectors with IDs"""
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-8)
        
        ids = np.arange(n_vectors)
        return ids, vectors
    
    def generate_with_metadata(
        self, 
        n_vectors: int, 
        dimension: int,
        n_categories: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate vectors with metadata for filtered search"""
        ids, vectors = self.generate_random_vectors(n_vectors, dimension)
        
        metadata = pd.DataFrame({
            'id': ids,
            'category': np.random.randint(0, n_categories, n_vectors),
            'price': np.random.uniform(10, 1000, n_vectors),
            'timestamp': pd.date_range('2024-01-01', periods=n_vectors, freq='s'),
            'rating': np.random.uniform(1, 5, n_vectors)
        })
        
        return ids, vectors, metadata
    
    def save_to_hdf5(self, filepath: str, ids: np.ndarray, vectors: np.ndarray):
        """Save vectors to HDF5 format for efficient loading"""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('ids', data=ids, compression='gzip')
            f.create_dataset('vectors', data=vectors, compression='gzip')
    
    def load_from_hdf5(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load vectors from HDF5 file"""
        with h5py.File(filepath, 'r') as f:
            ids = f['ids'][:]
            vectors = f['vectors'][:]
        return ids, vectors

# Example usage
if __name__ == "__main__":
    generator = VectorDataGenerator()
    
    # Generate small dataset
    print("Generating small dataset (100K vectors, 128D)...")
    ids_small, vecs_small, meta_small = generator.generate_with_metadata(100000, 128)
    generator.save_to_hdf5('data/small_dataset.h5', ids_small, vecs_small)
    meta_small.to_csv('data/small_metadata.csv', index=False)
    
    # Generate medium dataset
    print("Generating medium dataset (1M vectors, 384D)...")
    ids_medium, vecs_medium, meta_medium = generator.generate_with_metadata(1000000, 384)
    generator.save_to_hdf5('data/medium_dataset.h5', ids_medium, vecs_medium)
    meta_medium.to_csv('data/medium_metadata.csv', index=False)
    
    # Large dataset generated in batches to avoid memory issues
    print("Generating large dataset (10M vectors, 768D)...")
    batch_size = 100000
    n_batches = 100
    
    with h5py.File('data/large_dataset.h5', 'w') as f:
        ids_dataset = f.create_dataset('ids', (10000000,), dtype=np.int64)
        vecs_dataset = f.create_dataset('vectors', (10000000, 768), dtype=np.float32)
        
        for i in tqdm(range(n_batches)):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            ids_batch, vecs_batch = generator.generate_random_vectors(batch_size, 768)
            ids_dataset[start_idx:end_idx] = ids_batch + start_idx
            vecs_dataset[start_idx:end_idx] = vecs_batch