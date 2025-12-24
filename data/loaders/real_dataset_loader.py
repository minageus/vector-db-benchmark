"""
Real Dataset Loader for ANN Benchmarks

Loads standard ANN benchmark datasets with pre-computed ground truth:
- SIFT1M, GIST1M: Image descriptors
- GloVe: Word embeddings
- Provides unified interface for all datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from utils.dataset_downloader import DatasetDownloader, read_fvecs, read_ivecs, read_bvecs


class RealDatasetLoader:
    """Load real-world ANN benchmark datasets"""
    
    def __init__(self, cache_dir: str = 'data/datasets'):
        """Initialize loader with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.downloader = DatasetDownloader(cache_dir=str(cache_dir))
    
    def load_sift1m(self, download: bool = True) -> Dict[str, Any]:
        """
        Load SIFT1M dataset
        
        Returns:
            dict with keys:
                - base: (1M, 128) base vectors
                - query: (10K, 128) query vectors
                - groundtruth: (10K, 100) ground truth neighbors
                - learn: (100K, 128) learning/training vectors
                - metadata: dataset information
        """
        dataset_name = 'sift1m'
        dataset_dir = self.downloader.get_dataset_path(dataset_name)
        
        if dataset_dir is None:
            if download:
                print(f"Dataset not found. Downloading {dataset_name}...")
                dataset_dir = self.downloader.download_dataset(dataset_name)
            else:
                raise FileNotFoundError(f"Dataset {dataset_name} not found. Use download=True")
        
        # Find the actual directory containing the files (may be nested)
        sift_dir = self._find_dataset_files(dataset_dir, 'sift_base.fvecs')
        
        print(f"\nLoading SIFT1M from {sift_dir}...")
        
        # Load vectors
        base = read_fvecs(str(sift_dir / 'sift_base.fvecs'))
        query = read_fvecs(str(sift_dir / 'sift_query.fvecs'))
        groundtruth = read_ivecs(str(sift_dir / 'sift_groundtruth.ivecs'))
        
        # Learn set (optional, for training)
        learn_file = sift_dir / 'sift_learn.fvecs'
        learn = read_fvecs(str(learn_file)) if learn_file.exists() else None
        
        print(f"OK Loaded SIFT1M:")
        print(f"  Base vectors: {base.shape}")
        print(f"  Query vectors: {query.shape}")
        print(f"  Ground truth: {groundtruth.shape}")
        if learn is not None:
            print(f"  Learn vectors: {learn.shape}")
        
        # Generate metadata
        metadata = self._generate_metadata(base.shape[0])
        
        return {
            'base': base,
            'query': query,
            'groundtruth': groundtruth,
            'learn': learn,
            'metadata': metadata,
            'info': {
                'name': 'SIFT1M',
                'dimension': 128,
                'n_base': base.shape[0],
                'n_query': query.shape[0],
                'metric': 'L2',
                'dtype': 'float32'
            }
        }
    
    def load_gist1m(self, download: bool = True) -> Dict[str, Any]:
        """
        Load GIST1M dataset
        
        Returns:
            dict with keys similar to load_sift1m
        """
        dataset_name = 'gist1m'
        dataset_dir = self.downloader.get_dataset_path(dataset_name)
        
        if dataset_dir is None:
            if download:
                print(f"Dataset not found. Downloading {dataset_name}...")
                dataset_dir = self.downloader.download_dataset(dataset_name)
            else:
                raise FileNotFoundError(f"Dataset {dataset_name} not found. Use download=True")
        
        gist_dir = self._find_dataset_files(dataset_dir, 'gist_base.fvecs')
        
        print(f"\nLoading GIST1M from {gist_dir}...")
        
        base = read_fvecs(str(gist_dir / 'gist_base.fvecs'))
        query = read_fvecs(str(gist_dir / 'gist_query.fvecs'))
        groundtruth = read_ivecs(str(gist_dir / 'gist_groundtruth.ivecs'))
        
        learn_file = gist_dir / 'gist_learn.fvecs'
        learn = read_fvecs(str(learn_file)) if learn_file.exists() else None
        
        print(f"OK Loaded GIST1M:")
        print(f"  Base vectors: {base.shape}")
        print(f"  Query vectors: {query.shape}")
        print(f"  Ground truth: {groundtruth.shape}")
        if learn is not None:
            print(f"  Learn vectors: {learn.shape}")
        
        metadata = self._generate_metadata(base.shape[0])
        
        return {
            'base': base,
            'query': query,
            'groundtruth': groundtruth,
            'learn': learn,
            'metadata': metadata,
            'info': {
                'name': 'GIST1M',
                'dimension': 960,
                'n_base': base.shape[0],
                'n_query': query.shape[0],
                'metric': 'L2',
                'dtype': 'float32'
            }
        }
    
    def load_sift10m(self, download: bool = True) -> Dict[str, Any]:
        """
        Load SIFT10M dataset (BigANN 10M subset)
        
        Returns:
            dict with keys:
                - base: (10M, 128) base vectors
                - query: (10K, 128) query vectors
                - groundtruth: (10K, 100) ground truth neighbors
                - metadata: dataset information
        """
        dataset_name = 'sift10m'
        dataset_dir = self.downloader.get_dataset_path(dataset_name)
        
        if dataset_dir is None:
            if download:
                print(f"Dataset not found. Downloading {dataset_name}...")
                print("WARNING: This is a large dataset (~12GB download). This may take a while.")
                dataset_dir = self.downloader.download_dataset(dataset_name)
            else:
                raise FileNotFoundError(f"Dataset {dataset_name} not found. Use download=True")
        
        # The processed files are in fvecs format
        sift_dir = dataset_dir
        
        print(f"\nLoading SIFT10M from {sift_dir}...")
        print("This may take a few minutes due to dataset size...")
        
        # Load vectors
        base = read_fvecs(str(sift_dir / 'sift10m_base.fvecs'))
        query = read_fvecs(str(sift_dir / 'sift10m_query.fvecs'))
        groundtruth = read_ivecs(str(sift_dir / 'idx_10M.ivecs'))
        
        print(f"OK Loaded SIFT10M:")
        print(f"  Base vectors: {base.shape}")
        print(f"  Query vectors: {query.shape}")
        print(f"  Ground truth: {groundtruth.shape}")
        
        # Generate metadata
        metadata = self._generate_metadata(base.shape[0])
        
        return {
            'base': base,
            'query': query,
            'groundtruth': groundtruth,
            'learn': None,
            'metadata': metadata,
            'info': {
                'name': 'SIFT10M',
                'dimension': 128,
                'n_base': base.shape[0],
                'n_query': query.shape[0],
                'metric': 'L2',
                'dtype': 'float32'
            }
        }
    
    def load_glove(self, dimension: int = 100, download: bool = True, max_vectors: Optional[int] = None) -> Dict[str, Any]:
        """
        Load GloVe word embeddings
        
        Args:
            dimension: Embedding dimension (25, 50, 100, 200, 300)
            download: Download if not cached
            max_vectors: Limit number of vectors (for testing)
            
        Returns:
            dict with base vectors and metadata
        """
        dataset_name = 'glove-100'
        dataset_dir = self.downloader.get_dataset_path(dataset_name)
        
        if dataset_dir is None:
            if download:
                print(f"Dataset not found. Downloading {dataset_name}...")
                dataset_dir = self.downloader.download_dataset(dataset_name)
            else:
                raise FileNotFoundError(f"Dataset {dataset_name} not found. Use download=True")
        
        glove_file = dataset_dir / f'glove.6B.{dimension}d.txt'
        if not glove_file.exists():
            # Try to find in subdirectory
            glove_file = self._find_dataset_files(dataset_dir, f'glove.6B.{dimension}d.txt') / f'glove.6B.{dimension}d.txt'
        
        print(f"\nLoading GloVe-{dimension}D from {glove_file}...")
        
        vectors = []
        words = []
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_vectors and i >= max_vectors:
                    break
                
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                words.append(word)
                vectors.append(vector)
        
        base = np.array(vectors, dtype=np.float32)
        
        print(f"OK Loaded GloVe-{dimension}D:")
        print(f"  Vectors: {base.shape}")
        print(f"  Vocabulary: {len(words)} words")
        
        # Create metadata with words
        metadata = pd.DataFrame({
            'id': np.arange(len(words)),
            'word': words,
            'category': np.random.randint(0, 10, len(words)),  # Dummy categories
            'price': np.random.uniform(10, 1000, len(words)),
            'timestamp': pd.date_range('2024-01-01', periods=len(words), freq='s'),
            'rating': np.random.uniform(1, 5, len(words))
        })
        
        # Generate query vectors (random subset)
        n_queries = min(1000, len(base) // 10)
        query_indices = np.random.choice(len(base), n_queries, replace=False)
        query = base[query_indices]
        
        return {
            'base': base,
            'query': query,
            'groundtruth': None,  # No pre-computed ground truth
            'metadata': metadata,
            'words': words,
            'info': {
                'name': f'GloVe-{dimension}D',
                'dimension': dimension,
                'n_base': base.shape[0],
                'n_query': query.shape[0],
                'metric': 'cosine',
                'dtype': 'float32'
            }
        }
    
    def load_hdf5_dataset(self, dataset_name: str, download: bool = True) -> Dict[str, Any]:
        """
        Load HDF5 format datasets from ann-benchmarks.com
        
        Works with: mnist-784, fashion-mnist-784, nytimes-256, lastfm-64, 
                    kosarak-27983, deep-image-96, random-xs-20
        """
        import h5py
        
        dataset_dir = self.downloader.get_dataset_path(dataset_name)
        
        if dataset_dir is None:
            if download:
                print(f"Dataset not found. Downloading {dataset_name}...")
                dataset_dir = self.downloader.download_dataset(dataset_name)
            else:
                raise FileNotFoundError(f"Dataset {dataset_name} not found. Use download=True")
        
        # Find HDF5 file in directory
        hdf5_files = list(dataset_dir.glob('*.hdf5'))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")
        
        hdf5_file = hdf5_files[0]
        print(f"\nLoading {dataset_name} from {hdf5_file}...")
        
        with h5py.File(hdf5_file, 'r') as f:
            # Standard ann-benchmarks HDF5 structure
            base = np.array(f['train']).astype(np.float32)
            query = np.array(f['test']).astype(np.float32)
            
            # Ground truth neighbors (may have different keys)
            if 'neighbors' in f:
                groundtruth = np.array(f['neighbors'])
            else:
                groundtruth = None
            
            # Get distance metric from filename or attributes
            metric = 'L2'  # default
            if 'angular' in str(hdf5_file) or 'cosine' in str(hdf5_file):
                metric = 'cosine'
            elif 'dot' in str(hdf5_file):
                metric = 'IP'
            elif 'jaccard' in str(hdf5_file):
                metric = 'jaccard'
        
        n_vectors = base.shape[0]
        dimension = base.shape[1]
        
        print(f"OK Loaded {dataset_name}:")
        print(f"  Base vectors: {base.shape}")
        print(f"  Query vectors: {query.shape}")
        if groundtruth is not None:
            print(f"  Ground truth: {groundtruth.shape}")
        print(f"  Metric: {metric}")
        
        metadata = self._generate_metadata(n_vectors)
        
        return {
            'base': base,
            'query': query,
            'groundtruth': groundtruth,
            'metadata': metadata,
            'info': {
                'name': dataset_name,
                'dimension': dimension,
                'n_base': n_vectors,
                'n_query': query.shape[0],
                'metric': metric,
                'dtype': 'float32'
            }
        }
    
    def load_dataset(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Load any dataset by name
        
        Args:
            name: Dataset name (sift1m, sift10m, gist1m, glove-100, mnist-784, etc.)
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Dataset dictionary
        """
        # Datasets with specific loaders
        specific_loaders = {
            'sift1m': self.load_sift1m,
            'sift10m': self.load_sift10m,
            'gist1m': self.load_gist1m,
            'glove-100': lambda **kw: self.load_glove(dimension=100, **kw),
            'glove-200': lambda **kw: self.load_glove(dimension=200, **kw),
            'glove-300': lambda **kw: self.load_glove(dimension=300, **kw),
        }
        
        # HDF5 datasets from ann-benchmarks.com
        hdf5_datasets = [
            'mnist-784', 'fashion-mnist-784', 'nytimes-256', 
            'lastfm-64', 'kosarak-27983', 'deep-image-96', 'random-xs-20',
            'glove-25', 'glove-200'
        ]
        
        if name in specific_loaders:
            return specific_loaders[name](**kwargs)
        elif name in hdf5_datasets:
            return self.load_hdf5_dataset(name, **kwargs)
        else:
            available = list(specific_loaders.keys()) + hdf5_datasets
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    def _find_dataset_files(self, base_dir: Path, target_file: str) -> Path:
        """Find directory containing target file (handles nested archives)"""
        # Check base directory
        if (base_dir / target_file).exists():
            return base_dir
        
        # Search subdirectories
        for subdir in base_dir.rglob('*'):
            if subdir.is_dir() and (subdir / target_file).exists():
                return subdir
        
        raise FileNotFoundError(f"Could not find {target_file} in {base_dir}")
    
    def _generate_metadata(self, n_vectors: int) -> pd.DataFrame:
        """Generate dummy metadata for datasets without metadata"""
        return pd.DataFrame({
            'id': np.arange(n_vectors),
            'category': np.random.randint(0, 10, n_vectors),
            'price': np.random.uniform(10, 1000, n_vectors),
            'timestamp': pd.date_range('2024-01-01', periods=n_vectors, freq='s'),
            'rating': np.random.uniform(1, 5, n_vectors)
        })
    
    def get_subset(self, data: Dict[str, Any], n_vectors: int) -> Dict[str, Any]:
        """
        Get a subset of a dataset (for testing with smaller data)
        
        Args:
            data: Dataset dictionary
            n_vectors: Number of vectors to keep
            
        Returns:
            Subset dataset dictionary
        """
        n_vectors = min(n_vectors, len(data['base']))
        
        subset = {
            'base': data['base'][:n_vectors],
            'query': data['query'],  # Keep all queries
            'groundtruth': data['groundtruth'],  # Keep all ground truth
            'metadata': data['metadata'].iloc[:n_vectors] if data['metadata'] is not None else None,
            'info': data['info'].copy()
        }
        
        subset['info']['n_base'] = n_vectors
        subset['info']['name'] = f"{data['info']['name']}_subset_{n_vectors}"
        
        return subset


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load ANN benchmark datasets')
    parser.add_argument('--dataset', type=str, default='sift1m', 
                       help='Dataset name (sift1m, gist1m, glove-100)')
    parser.add_argument('--no-download', action='store_true', 
                       help='Do not download if missing')
    parser.add_argument('--subset', type=int, 
                       help='Use subset of N vectors')
    
    args = parser.parse_args()
    
    loader = RealDatasetLoader()
    
    # Load dataset
    data = loader.load_dataset(args.dataset, download=not args.no_download)
    
    # Get subset if requested
    if args.subset:
        data = loader.get_subset(data, args.subset)
    
    print(f"\n{'='*60}")
    print("Dataset Information:")
    print(f"{'='*60}")
    for key, value in data['info'].items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
