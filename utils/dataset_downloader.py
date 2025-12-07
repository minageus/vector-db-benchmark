"""
Dataset Downloader for ANN Benchmark Datasets

Downloads and caches standard ANN benchmark datasets:
- SIFT1M: 1M SIFT image descriptors (128D)
- GIST1M: 1M GIST image features (960D)
- GloVe: Word embeddings (300D)
- Deep1B subsets: Deep learning features (96D)
"""

import os
import requests
import hashlib
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple
import numpy as np

class DatasetDownloader:
    """Download and cache ANN benchmark datasets"""
    
    # Dataset URLs and metadata
    DATASETS = {
        'sift1m': {
            'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz',
            'size_mb': 161,
            'description': 'SIFT1M - 1M SIFT image descriptors (128D)',
            'files': ['sift_base.fvecs', 'sift_query.fvecs', 'sift_groundtruth.ivecs', 'sift_learn.fvecs']
        },
        'sift10k': {
            'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz',
            'size_mb': 16,
            'description': 'SIFT10K - 10K SIFT descriptors (128D) - Small test set',
            'files': ['siftsmall_base.fvecs', 'siftsmall_query.fvecs', 'siftsmall_groundtruth.ivecs']
        },
        'gist1m': {
            'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz',
            'size_mb': 3600,
            'description': 'GIST1M - 1M GIST image features (960D)',
            'files': ['gist_base.fvecs', 'gist_query.fvecs', 'gist_groundtruth.ivecs', 'gist_learn.fvecs']
        },
        'glove-100': {
            'url': 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip',
            'size_mb': 822,
            'description': 'GloVe - Word embeddings (100D)',
            'files': ['glove.6B.100d.txt']
        },
        'mnist-784': {
            'url': 'https://ann-benchmarks.com/mnist-784-euclidean.hdf5',
            'size_mb': 217,
            'description': 'MNIST - 60K handwritten digits (784D)',
            'files': ['mnist-784-euclidean.hdf5']
        },
        'fashion-mnist-784': {
            'url': 'https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5',
            'size_mb': 217,
            'description': 'Fashion-MNIST - 60K fashion items (784D)',
            'files': ['fashion-mnist-784-euclidean.hdf5']
        },
        'nytimes-256': {
            'url': 'https://ann-benchmarks.com/nytimes-256-angular.hdf5',
            'size_mb': 301,
            'description': 'NYTimes - 290K article embeddings (256D)',
            'files': ['nytimes-256-angular.hdf5']
        },
        'lastfm-64': {
            'url': 'https://ann-benchmarks.com/lastfm-64-dot.hdf5',
            'size_mb': 95,
            'description': 'Last.fm - 292K music embeddings (64D)',
            'files': ['lastfm-64-dot.hdf5']
        },
        'kosarak-27983': {
            'url': 'https://ann-benchmarks.com/kosarak-jaccard.hdf5',
            'size_mb': 112,
            'description': 'Kosarak - 75K click-stream data (27983D sparse)',
            'files': ['kosarak-jaccard.hdf5']
        },
        'deep-image-96': {
            'url': 'https://ann-benchmarks.com/deep-image-96-angular.hdf5',
            'size_mb': 3800,
            'description': 'Deep1M - 10M deep learning image features (96D)',
            'files': ['deep-image-96-angular.hdf5']
        },
        'random-xs-20': {
            'url': 'https://ann-benchmarks.com/random-xs-20-euclidean.hdf5',
            'size_mb': 50,
            'description': 'Random - 10K random vectors (20D) - For testing',
            'files': ['random-xs-20-euclidean.hdf5']
        }
    }
    
    def __init__(self, cache_dir: str = 'data/datasets'):
        """Initialize downloader with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, name: str, force: bool = False) -> Path:
        """
        Download dataset if not cached
        
        Args:
            name: Dataset name (sift1m, gist1m, glove-100)
            force: Force re-download even if cached
            
        Returns:
            Path to dataset directory
        """
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASETS.keys())}")
        
        dataset_info = self.DATASETS[name]
        dataset_dir = self.cache_dir / name
        
        # Check if already downloaded
        if dataset_dir.exists() and not force:
            print(f"OK Dataset '{name}' already cached at {dataset_dir}")
            return dataset_dir
        
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_info['description']}")
        print(f"Size: ~{dataset_info['size_mb']} MB")
        print(f"{'='*60}\n")
        
        # Download file
        url = dataset_info['url']
        filename = url.split('/')[-1]
        filepath = dataset_dir / filename
        
        self._download_file(url, filepath)
        
        # Extract if compressed
        if filename.endswith('.tar.gz'):
            print(f"\nExtracting {filename}...")
            self._extract_tar_gz(filepath, dataset_dir)
            filepath.unlink()  # Remove archive after extraction
        elif filename.endswith('.zip'):
            print(f"\nExtracting {filename}...")
            self._extract_zip(filepath, dataset_dir)
            filepath.unlink()
        
        print(f"\nOK Dataset downloaded to: {dataset_dir}")
        return dataset_dir
    
    def _download_file(self, url: str, filepath: Path) -> None:
        """Download file with progress bar"""
        # Handle FTP URLs
        if url.startswith('ftp://'):
            import urllib.request
            
            def reporthook(block_num, block_size, total_size):
                if not hasattr(reporthook, 'pbar'):
                    reporthook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
                reporthook.pbar.update(block_size)
            
            try:
                urllib.request.urlretrieve(url, filepath, reporthook)
                if hasattr(reporthook, 'pbar'):
                    reporthook.pbar.close()
            except Exception as e:
                print(f"FTP download failed: {e}")
                print("Please download manually from: http://corpus-texmex.irisa.fr/")
                raise
        else:
            # HTTP download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filepath.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _extract_tar_gz(self, filepath: Path, extract_dir: Path) -> None:
        """Extract tar.gz archive"""
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(extract_dir)
    
    def _extract_zip(self, filepath: Path, extract_dir: Path) -> None:
        """Extract zip archive"""
        import zipfile
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def list_datasets(self) -> None:
        """Print available datasets"""
        print("\nAvailable Datasets:")
        print("=" * 80)
        for name, info in self.DATASETS.items():
            cached = "[OK]" if (self.cache_dir / name).exists() else "    "
            print(f"{cached} {name:15} - {info['description']}")
            print(f"   {'':15}   Size: ~{info['size_mb']} MB")
        print("=" * 80)
    
    def get_dataset_path(self, name: str) -> Optional[Path]:
        """Get path to cached dataset, or None if not downloaded"""
        dataset_dir = self.cache_dir / name
        return dataset_dir if dataset_dir.exists() else None


# Utility functions for reading dataset formats

def read_fvecs(filepath: str) -> np.ndarray:
    """
    Read .fvecs file format (used by SIFT, GIST datasets)
    
    Format: [dim, vec1_val1, vec1_val2, ..., dim, vec2_val1, ...]
    Each vector is preceded by its dimension (int32)
    """
    with open(filepath, 'rb') as f:
        # Read first dimension to get vector size
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        
        # Calculate number of vectors
        file_size = os.path.getsize(filepath)
        vec_size = 4 + dim * 4  # 4 bytes for dim + dim * 4 bytes for floats
        n_vecs = file_size // vec_size
        
        # Read all data
        data = np.fromfile(f, dtype=np.int32, count=n_vecs * (dim + 1))
        data = data.reshape(n_vecs, dim + 1)
        
        # Extract vectors (skip dimension column)
        vectors = data[:, 1:].view(np.float32)
        
    return vectors


def read_ivecs(filepath: str) -> np.ndarray:
    """
    Read .ivecs file format (used for ground truth)
    
    Format: [k, id1, id2, ..., idk, k, ...]
    Each row is preceded by k (number of neighbors)
    """
    with open(filepath, 'rb') as f:
        # Read first k to get number of neighbors
        k = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        
        # Calculate number of queries
        file_size = os.path.getsize(filepath)
        row_size = 4 + k * 4  # 4 bytes for k + k * 4 bytes for ints
        n_queries = file_size // row_size
        
        # Read all data
        data = np.fromfile(f, dtype=np.int32, count=n_queries * (k + 1))
        data = data.reshape(n_queries, k + 1)
        
        # Extract IDs (skip k column)
        ids = data[:, 1:]
        
    return ids


def read_bvecs(filepath: str) -> np.ndarray:
    """
    Read .bvecs file format (used for some datasets)
    
    Similar to fvecs but with uint8 values
    """
    with open(filepath, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        
        file_size = os.path.getsize(filepath)
        vec_size = 4 + dim  # 4 bytes for dim + dim bytes for uint8
        n_vecs = file_size // vec_size
        
        vectors = np.zeros((n_vecs, dim), dtype=np.uint8)
        
        for i in range(n_vecs):
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            vec = np.fromfile(f, dtype=np.uint8, count=d)
            vectors[i] = vec
            
    return vectors


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ANN benchmark datasets')
    parser.add_argument('--dataset', type=str, help='Dataset name (sift1m, gist1m, glove-100)')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--cache-dir', type=str, default='data/datasets', help='Cache directory')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(cache_dir=args.cache_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.dataset:
        dataset_path = downloader.download_dataset(args.dataset, force=args.force)
        print(f"\nDataset ready at: {dataset_path}")
    else:
        print("Use --list to see available datasets or --dataset <name> to download")
        downloader.list_datasets()
