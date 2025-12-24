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
            'url': 'http://corpus-texmex.irisa.fr/sift.tar.gz',
            'mirror_urls': [
                'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz',
            ],
            'size_mb': 161,
            'description': 'SIFT1M - 1M SIFT image descriptors (128D)',
            'files': ['sift_base.fvecs', 'sift_query.fvecs', 'sift_groundtruth.ivecs', 'sift_learn.fvecs']
        },
        'sift10k': {
            'url': 'http://corpus-texmex.irisa.fr/siftsmall.tar.gz',
            'mirror_urls': [
                'ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz',
            ],
            'size_mb': 16,
            'description': 'SIFT10K - 10K SIFT descriptors (128D) - Small test set',
            'files': ['siftsmall_base.fvecs', 'siftsmall_query.fvecs', 'siftsmall_groundtruth.ivecs']
        },
        'sift10m': {
            'base_url': 'ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz',
            'query_url': 'ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz',
            'groundtruth_url': 'ftp://ftp.irisa.fr/local/texmex/corpus/gnd/idx_10M.ivecs',
            'learn_url': 'ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz',
            'size_mb': 12000,
            'description': 'SIFT10M - 10M SIFT descriptors (128D) - BigANN subset',
            'files': ['bigann_base.bvecs', 'bigann_query.bvecs', 'idx_10M.ivecs'],
            'n_vectors': 10000000,
            'is_bigann_subset': True
        },
        'gist1m': {
            'url': 'http://corpus-texmex.irisa.fr/gist.tar.gz',
            'mirror_urls': [
                'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz',
            ],
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
        'glove-25': {
            'url': 'https://ann-benchmarks.com/glove-25-angular.hdf5',
            'size_mb': 120,
            'description': 'GloVe-25 - 1.2M word embeddings (25D) - Good for 500K subset',
            'files': ['glove-25-angular.hdf5']
        },
        'glove-200': {
            'url': 'https://ann-benchmarks.com/glove-200-angular.hdf5',
            'size_mb': 950,
            'description': 'GloVe-200 - 1.2M word embeddings (200D) - Medium-large scale',
            'files': ['glove-200-angular.hdf5']
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
            name: Dataset name (sift1m, gist1m, glove-100, sift10m)
            force: Force re-download even if cached
            
        Returns:
            Path to dataset directory
        """
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASETS.keys())}")
        
        # Handle special case for sift10m (BigANN subset)
        if name == 'sift10m':
            return self.download_sift10m(force=force)
        
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
        
        # Try primary URL first, then mirrors
        urls_to_try = [dataset_info['url']]
        if 'mirror_urls' in dataset_info:
            urls_to_try.extend(dataset_info['mirror_urls'])
        
        filename = urls_to_try[0].split('/')[-1]
        filepath = dataset_dir / filename
        
        # Try each URL until one works
        last_error = None
        for i, url in enumerate(urls_to_try):
            try:
                if i > 0:
                    print(f"\nTrying mirror {i}: {url}")
                self._download_file(url, filepath)
                break  # Success!
            except Exception as e:
                last_error = e
                print(f"Failed: {e}")
                if i < len(urls_to_try) - 1:
                    continue
                else:
                    # All URLs failed
                    print(f"\n{'='*60}")
                    print("ERROR: All download sources failed!")
                    print(f"{'='*60}")
                    print("\nManual download instructions:")
                    print(f"1. Visit: http://corpus-texmex.irisa.fr/")
                    print(f"2. Download: {filename}")
                    print(f"3. Place in: {dataset_dir}")
                    print(f"4. Re-run the benchmark")
                    raise last_error
        
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
    
    def download_sift10m(self, force: bool = False) -> Path:
        """
        Download SIFT10M (BigANN 10M subset) dataset
        
        This is a special case because BigANN is split into multiple files
        and we only need the first 10M vectors.
        """
        dataset_info = self.DATASETS['sift10m']
        dataset_dir = self.cache_dir / 'sift10m'
        
        if dataset_dir.exists() and not force:
            # Check if we have the necessary processed files
            if (dataset_dir / 'sift10m_base.fvecs').exists():
                print(f"OK Dataset 'sift10m' already cached at {dataset_dir}")
                return dataset_dir
        
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_info['description']}")
        print(f"Size: ~{dataset_info['size_mb']} MB (downloading only first 10M vectors)")
        print(f"{'='*60}\n")
        print("WARNING: This is a large download (~12GB for base vectors).")
        print("The download will fetch the BigANN base file and extract the first 10M vectors.\n")
        
        # Download base vectors (this is the big one ~12GB compressed)
        base_gz = dataset_dir / 'bigann_base.bvecs.gz'
        if not base_gz.exists() and not (dataset_dir / 'sift10m_base.fvecs').exists():
            print("Downloading base vectors (this will take a while)...")
            try:
                self._download_file(dataset_info['base_url'], base_gz)
            except Exception as e:
                print(f"\nFailed to download base vectors: {e}")
                print("\nManual download instructions:")
                print(f"1. Download: {dataset_info['base_url']}")
                print(f"2. Place in: {dataset_dir}")
                raise
        
        # Download query vectors
        query_gz = dataset_dir / 'bigann_query.bvecs.gz'
        if not query_gz.exists() and not (dataset_dir / 'sift10m_query.fvecs').exists():
            print("Downloading query vectors...")
            self._download_file(dataset_info['query_url'], query_gz)
        
        # Download groundtruth for 10M subset
        gt_file = dataset_dir / 'idx_10M.ivecs'
        if not gt_file.exists():
            print("Downloading groundtruth for 10M subset...")
            self._download_file(dataset_info['groundtruth_url'], gt_file)
        
        # Extract and convert to fvecs format (only first 10M vectors)
        if not (dataset_dir / 'sift10m_base.fvecs').exists():
            print("\nExtracting and converting base vectors (first 10M only)...")
            print("This may take several minutes...")
            self._extract_bigann_subset(base_gz, dataset_dir / 'sift10m_base.fvecs', 
                                        n_vectors=10000000, dimension=128)
            # Optionally remove the compressed file to save space
            # base_gz.unlink()
        
        if not (dataset_dir / 'sift10m_query.fvecs').exists():
            print("Extracting and converting query vectors...")
            self._extract_bigann_all(query_gz, dataset_dir / 'sift10m_query.fvecs', dimension=128)
        
        print(f"\nOK Dataset downloaded and processed to: {dataset_dir}")
        return dataset_dir
    
    def _extract_bigann_subset(self, gz_file: Path, output_file: Path, n_vectors: int, dimension: int):
        """
        Extract first n_vectors from a BigANN .bvecs.gz file and convert to .fvecs format
        """
        import gzip
        
        vec_size = 4 + dimension  # 4 bytes for dim + dim bytes for uint8 values
        total_bytes = n_vectors * vec_size
        
        vectors = []
        bytes_read = 0
        
        with gzip.open(gz_file, 'rb') as f:
            with tqdm(total=n_vectors, unit='vectors', desc='Converting') as pbar:
                while bytes_read < total_bytes and len(vectors) < n_vectors:
                    # Read dimension
                    dim_bytes = f.read(4)
                    if len(dim_bytes) < 4:
                        break
                    dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
                    
                    # Read vector
                    vec_bytes = f.read(dim)
                    if len(vec_bytes) < dim:
                        break
                    vec = np.frombuffer(vec_bytes, dtype=np.uint8).astype(np.float32)
                    vectors.append(vec)
                    
                    bytes_read += 4 + dim
                    pbar.update(1)
        
        # Convert to numpy array and save as fvecs
        vectors = np.array(vectors, dtype=np.float32)
        self._write_fvecs(output_file, vectors)
        print(f"  Saved {len(vectors):,} vectors to {output_file}")
    
    def _extract_bigann_all(self, gz_file: Path, output_file: Path, dimension: int):
        """
        Extract all vectors from a BigANN .bvecs.gz file and convert to .fvecs format
        """
        import gzip
        
        vectors = []
        
        with gzip.open(gz_file, 'rb') as f:
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    break
                dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
                
                # Read vector
                vec_bytes = f.read(dim)
                if len(vec_bytes) < dim:
                    break
                vec = np.frombuffer(vec_bytes, dtype=np.uint8).astype(np.float32)
                vectors.append(vec)
        
        vectors = np.array(vectors, dtype=np.float32)
        self._write_fvecs(output_file, vectors)
        print(f"  Saved {len(vectors):,} vectors to {output_file}")
    
    def _write_fvecs(self, filepath: Path, vectors: np.ndarray):
        """
        Write vectors to .fvecs format
        """
        n_vectors, dim = vectors.shape
        with open(filepath, 'wb') as f:
            for i in range(n_vectors):
                # Write dimension as int32
                np.array([dim], dtype=np.int32).tofile(f)
                # Write vector as float32
                vectors[i].astype(np.float32).tofile(f)

    def _download_file(self, url: str, filepath: Path) -> None:
        """Download file with progress bar"""
        # Handle FTP URLs
        if url.startswith('ftp://'):
            import urllib.request
            import socket
            
            # Set socket timeout to avoid hanging
            socket.setdefaulttimeout(30)
            
            def reporthook(block_num, block_size, total_size):
                if not hasattr(reporthook, 'pbar'):
                    reporthook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
                reporthook.pbar.update(block_size)
            
            try:
                urllib.request.urlretrieve(url, filepath, reporthook)
                if hasattr(reporthook, 'pbar'):
                    reporthook.pbar.close()
            except Exception as e:
                if hasattr(reporthook, 'pbar'):
                    reporthook.pbar.close()
                raise
        else:
            # HTTP download with timeout
            response = requests.get(url, stream=True, timeout=30)
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
