"""
Storage Analyzer for Vector Databases

Analyzes storage efficiency and index sizes:
- Raw data size vs stored size (compression ratio)
- Index overhead
- Metadata storage
- Per-collection breakdown
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import json


class StorageAnalyzer:
    """Analyze storage efficiency of vector databases"""
    
    def __init__(self):
        """Initialize storage analyzer"""
        self.results = {}
    
    def analyze_milvus_storage(self, 
                               data_dir: str = 'docker/volumes/milvus',
                               collection_name: str = 'benchmark_collection',
                               raw_data_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze Milvus storage
        
        Args:
            data_dir: Milvus data directory
            collection_name: Collection name to analyze
            raw_data_size_mb: Raw vector data size for compression ratio
            
        Returns:
            Storage analysis results
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {'error': f'Milvus data directory not found: {data_dir}'}
        
        # Get total size
        total_size = self._get_directory_size(data_path)
        
        # Try to break down by component
        breakdown = {}
        
        # MinIO storage (vector data)
        minio_path = data_path / 'rdb_data'
        if minio_path.exists():
            breakdown['vector_data'] = self._get_directory_size(minio_path)
        
        # etcd storage (metadata)
        etcd_path = Path('docker/volumes/etcd')
        if etcd_path.exists():
            breakdown['metadata'] = self._get_directory_size(etcd_path)
        
        # Calculate compression ratio
        compression_ratio = None
        if raw_data_size_mb and breakdown.get('vector_data', 0) > 0:
            stored_size_mb = breakdown['vector_data']
            compression_ratio = raw_data_size_mb / stored_size_mb
        
        result = {
            'database': 'Milvus',
            'collection': collection_name,
            'total_size_mb': total_size,
            'breakdown': breakdown,
            'raw_data_size_mb': raw_data_size_mb,
            'compression_ratio': compression_ratio,
            'storage_path': str(data_path)
        }
        
        self.results['milvus'] = result
        return result
    
    def analyze_weaviate_storage(self,
                                  data_dir: str = 'docker/volumes/weaviate',
                                  class_name: str = 'BenchmarkVector',
                                  raw_data_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze Weaviate storage
        
        Args:
            data_dir: Weaviate data directory
            class_name: Class name to analyze
            raw_data_size_mb: Raw vector data size for compression ratio
            
        Returns:
            Storage analysis results
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {'error': f'Weaviate data directory not found: {data_dir}'}
        
        # Get total size
        total_size = self._get_directory_size(data_path)
        
        # Try to break down by component
        breakdown = {}
        
        # Look for index files
        for item in data_path.rglob('*'):
            if item.is_file():
                if 'hnsw' in item.name.lower():
                    breakdown['hnsw_index'] = breakdown.get('hnsw_index', 0) + os.path.getsize(item) / (1024 * 1024)
                elif 'vector' in item.name.lower():
                    breakdown['vector_data'] = breakdown.get('vector_data', 0) + os.path.getsize(item) / (1024 * 1024)
                elif 'object' in item.name.lower() or 'property' in item.name.lower():
                    breakdown['metadata'] = breakdown.get('metadata', 0) + os.path.getsize(item) / (1024 * 1024)
        
        # Calculate compression ratio
        compression_ratio = None
        if raw_data_size_mb and breakdown.get('vector_data', 0) > 0:
            stored_size_mb = breakdown['vector_data']
            compression_ratio = raw_data_size_mb / stored_size_mb
        
        result = {
            'database': 'Weaviate',
            'class': class_name,
            'total_size_mb': total_size,
            'breakdown': breakdown,
            'raw_data_size_mb': raw_data_size_mb,
            'compression_ratio': compression_ratio,
            'storage_path': str(data_path)
        }
        
        self.results['weaviate'] = result
        return result
    
    def _get_directory_size(self, path: Path) -> float:
        """
        Get total size of directory in MB
        
        Args:
            path: Directory path
            
        Returns:
            Size in MB
        """
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except PermissionError:
            pass
        
        return total / (1024 * 1024)  # Convert to MB
    
    def compare_storage(self) -> Dict[str, Any]:
        """
        Compare storage efficiency between databases
        
        Returns:
            Comparison results
        """
        if 'milvus' not in self.results or 'weaviate' not in self.results:
            return {'error': 'Need to analyze both databases first'}
        
        milvus = self.results['milvus']
        weaviate = self.results['weaviate']
        
        comparison = {
            'total_size': {
                'milvus_mb': milvus['total_size_mb'],
                'weaviate_mb': weaviate['total_size_mb'],
                'difference_mb': milvus['total_size_mb'] - weaviate['total_size_mb'],
                'ratio': milvus['total_size_mb'] / weaviate['total_size_mb'] if weaviate['total_size_mb'] > 0 else None
            },
            'compression': {
                'milvus_ratio': milvus.get('compression_ratio'),
                'weaviate_ratio': weaviate.get('compression_ratio'),
                'winner': 'Milvus' if (milvus.get('compression_ratio') or 0) > (weaviate.get('compression_ratio') or 0) else 'Weaviate'
            },
            'efficiency_winner': 'Milvus' if milvus['total_size_mb'] < weaviate['total_size_mb'] else 'Weaviate'
        }
        
        return comparison
    
    def print_analysis(self, db_name: str):
        """Print storage analysis for a database"""
        if db_name not in self.results:
            print(f"No analysis results for {db_name}")
            return
        
        result = self.results[db_name]
        
        print(f"\n{'='*60}")
        print(f"{result['database'].upper()} STORAGE ANALYSIS")
        print(f"{'='*60}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nTotal Storage: {result['total_size_mb']:.2f} MB")
        
        if result.get('raw_data_size_mb'):
            print(f"Raw Data Size: {result['raw_data_size_mb']:.2f} MB")
            if result.get('compression_ratio'):
                print(f"Compression Ratio: {result['compression_ratio']:.2f}x")
        
        if result.get('breakdown'):
            print(f"\nStorage Breakdown:")
            for component, size in result['breakdown'].items():
                print(f"  {component:20s}: {size:8.2f} MB")
        
        print(f"\nStorage Path: {result['storage_path']}")
        print(f"{'='*60}")
    
    def print_comparison(self):
        """Print storage comparison"""
        comparison = self.compare_storage()
        
        if 'error' in comparison:
            print(f"Error: {comparison['error']}")
            return
        
        print(f"\n{'='*60}")
        print("STORAGE COMPARISON")
        print(f"{'='*60}")
        
        print(f"\nTotal Storage:")
        print(f"  Milvus:   {comparison['total_size']['milvus_mb']:8.2f} MB")
        print(f"  Weaviate: {comparison['total_size']['weaviate_mb']:8.2f} MB")
        print(f"  Difference: {abs(comparison['total_size']['difference_mb']):8.2f} MB")
        print(f"  Winner: {comparison['efficiency_winner']} (smaller is better)")
        
        if comparison['compression']['milvus_ratio'] and comparison['compression']['weaviate_ratio']:
            print(f"\nCompression Ratio:")
            print(f"  Milvus:   {comparison['compression']['milvus_ratio']:.2f}x")
            print(f"  Weaviate: {comparison['compression']['weaviate_ratio']:.2f}x")
            print(f"  Winner: {comparison['compression']['winner']} (higher is better)")
        
        print(f"{'='*60}")
    
    def save_results(self, filepath: str):
        """Save analysis results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {filepath}")


def calculate_raw_data_size(n_vectors: int, dimension: int, dtype_bytes: int = 4) -> float:
    """
    Calculate raw vector data size
    
    Args:
        n_vectors: Number of vectors
        dimension: Vector dimension
        dtype_bytes: Bytes per element (4 for float32, 2 for float16, 1 for int8)
        
    Returns:
        Size in MB
    """
    total_bytes = n_vectors * dimension * dtype_bytes
    return total_bytes / (1024 * 1024)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze vector database storage')
    parser.add_argument('--milvus-dir', type=str, default='docker/volumes/milvus',
                       help='Milvus data directory')
    parser.add_argument('--weaviate-dir', type=str, default='docker/volumes/weaviate',
                       help='Weaviate data directory')
    parser.add_argument('--n-vectors', type=int, help='Number of vectors (for compression ratio)')
    parser.add_argument('--dimension', type=int, help='Vector dimension (for compression ratio)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    analyzer = StorageAnalyzer()
    
    # Calculate raw data size if provided
    raw_size = None
    if args.n_vectors and args.dimension:
        raw_size = calculate_raw_data_size(args.n_vectors, args.dimension)
        print(f"Raw data size: {raw_size:.2f} MB")
    
    # Analyze both databases
    analyzer.analyze_milvus_storage(args.milvus_dir, raw_data_size_mb=raw_size)
    analyzer.print_analysis('milvus')
    
    analyzer.analyze_weaviate_storage(args.weaviate_dir, raw_data_size_mb=raw_size)
    analyzer.print_analysis('weaviate')
    
    # Compare
    analyzer.print_comparison()
    
    # Save results
    if args.output:
        analyzer.save_results(args.output)
