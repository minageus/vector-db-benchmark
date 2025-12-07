"""
Complete Vector Database Benchmark: Milvus vs Weaviate
======================================================

This script covers ALL project requirements:
1. Data loading with time/memory/storage monitoring
2. Query performance (latency, throughput) with AND without filters
3. Recall@K accuracy measurement
4. Multiple dataset sizes (scalability)
5. Resource monitoring (CPU, Memory)

Usage:
    python run_benchmark.py --dataset sift1m        # 1M vectors
    python run_benchmark.py --dataset gist1m        # 1M x 960D (biggest)
    python run_benchmark.py --dataset deep-image-96 # 10M vectors
    python run_benchmark.py --dataset nytimes-256   # 290K vectors
"""

import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

# Data loaders
from data.loaders.real_dataset_loader import RealDatasetLoader
from data.loaders.milvus_loader import MilvusLoader
from data.loaders.weaviate_loader import WeaviateLoader
from data.generators.vector_generator import VectorDataGenerator

# Query executors
from queries.milvus_queries import MilvusQueryExecutor
from queries.weaviate_queries import WeaviateQueryExecutor
from queries.query_generator import QueryGenerator

# Utilities
from utils.resource_monitor import ResourceMonitor
from utils.storage_analyzer import StorageAnalyzer, calculate_raw_data_size
from utils.recall_calculator import RecallCalculator
from benchmarks.benchmark_runner import BenchmarkRunner
from analysis.performance_analyzer import PerformanceAnalyzer


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_complete_benchmark(
    dataset_name: str = 'sift1m',
    subset_size: int = None,
    run_scalability: bool = True,
    run_filters: bool = True
):
    """
    Run the complete benchmark suite covering all project requirements.

    Args:
        dataset_name: Dataset to use (sift1m, gist1m, glove-100)
        subset_size: Number of vectors to use (None = full dataset)
        run_scalability: Whether to run scalability tests
        run_filters: Whether to run filter tests
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    all_results = {
        'loading': {},
        'performance': [],
        'recall': {},
        'storage': {},
        'scalability': {}
    }

    # =========================================================================
    # STEP 1: LOAD REAL DATASET
    # =========================================================================
    print_section("STEP 1: LOADING DATASET")

    print(f"Dataset: {dataset_name}")
    if subset_size:
        print(f"Using subset: {subset_size:,} vectors")

    loader = RealDatasetLoader()
    data = loader.load_dataset(dataset_name, download=True)

    if subset_size:
        data = loader.get_subset(data, subset_size)

    base_vectors = data['base']
    query_vectors = data['query']
    groundtruth = data['groundtruth']
    metadata = data['metadata']
    info = data['info']

    n_vectors = base_vectors.shape[0]
    dimension = base_vectors.shape[1]
    
    # Reduce queries for large datasets to speed up benchmark
    if n_vectors >= 1000000:
        n_queries = min(len(query_vectors), 100)  # Only 100 queries for large datasets
        print(f"\n[INFO] Using {n_queries} queries for faster benchmarking of large dataset")
    else:
        n_queries = min(len(query_vectors), 1000)  # 1000 queries for smaller datasets
    
    query_vectors = query_vectors[:n_queries]
    if groundtruth is not None:
        groundtruth = groundtruth[:n_queries]

    raw_data_size_mb = calculate_raw_data_size(n_vectors, dimension)

    print(f"\n[OK] Dataset loaded:")
    print(f"  Vectors: {n_vectors:,} x {dimension}D")
    print(f"  Queries: {n_queries:,}")
    print(f"  Metric: {info['metric']}")
    print(f"  Raw size: {raw_data_size_mb:.2f} MB")

    ids = np.arange(n_vectors)
    metric_type = 'L2' if info['metric'].lower() == 'l2' else 'IP'

    # =========================================================================
    # STEP 2: LOAD DATA INTO DATABASES (with monitoring)
    # =========================================================================
    print_section("STEP 2: DATA LOADING (with time/memory monitoring)")

    # Estimate time for large datasets
    if n_vectors >= 1000000:
        est_load_mins = (n_vectors / 1000000) * 15  # ~15 min per 1M vectors
        est_index_mins = (n_vectors / 1000000) * 25  # ~25 min per 1M vectors
        print(f"\n[INFO] Large dataset detected ({n_vectors:,} vectors)")
        print(f"  Estimated data loading time: ~{est_load_mins:.0f} minutes")
        print(f"  Estimated index building time: ~{est_index_mins:.0f} minutes")
        print(f"  Total estimated time for Step 2: ~{est_load_mins + est_index_mins:.0f} minutes\n")

    # --- Milvus Loading ---
    print("\n[Milvus] Loading data...")
    milvus_loader = MilvusLoader()
    milvus_loader.connect()

    collection_name = f"benchmark_{timestamp}"
    milvus_loader.create_collection(collection_name, dimension)

    # Optimize batch size for large datasets
    if n_vectors >= 1000000:
        batch_size = min(50000, 500000 // dimension)  # Larger batches for big datasets
    else:
        batch_size = max(100, min(10000, 100000 // dimension))
    
    # Optimize index params for large datasets (faster build)
    if n_vectors >= 1000000:
        index_params = {"M": 8, "efConstruction": 128}  # Faster for large datasets
        print(f"  Using optimized index params for large dataset: M=8, efConstruction=128")
    else:
        index_params = {"M": 16, "efConstruction": 200}  # Better quality for smaller datasets
    
    with ResourceMonitor() as milvus_monitor:
        load_start = time.time()
        milvus_loader.load_data(ids, base_vectors, metadata, batch_size=batch_size)
        milvus_loader.create_index(index_type='HNSW', metric_type=metric_type, 
                                   index_params=index_params, wait_timeout=1800)  # 30 min timeout
        milvus_load_time = time.time() - load_start

    # Load collection into memory AFTER monitoring to avoid state issues
    print("  Loading collection into memory...")
    milvus_loader.load_collection(timeout=300)
    
    # Give extra time for large collections to stabilize
    if n_vectors > 500000:
        print("  Large dataset detected, waiting for stabilization...")
        time.sleep(5)
    else:
        time.sleep(2)
    
    milvus_stats = milvus_monitor.get_stats()
    all_results['loading']['Milvus'] = {
        'load_time_seconds': milvus_load_time,
        'peak_memory_mb': milvus_stats.get('memory_rss_mb', {}).get('max', 0),
        'avg_cpu_percent': milvus_stats.get('cpu', {}).get('mean', 0)
    }

    print(f"  [OK] Milvus: {milvus_load_time:.1f}s, Peak Memory: {all_results['loading']['Milvus']['peak_memory_mb']:.1f} MB")

    # --- Weaviate Loading ---
    print("\n[Weaviate] Loading data...")
    with ResourceMonitor() as weaviate_monitor:
        weaviate_loader = WeaviateLoader()
        weaviate_loader.connect()
        weaviate_loader.create_schema(dimension)

        batch_size = min(100, 10000 // dimension)
        load_start = time.time()
        weaviate_loader.load_data(ids, base_vectors, metadata, batch_size=batch_size)
        weaviate_load_time = time.time() - load_start

    weaviate_stats = weaviate_monitor.get_stats()
    all_results['loading']['Weaviate'] = {
        'load_time_seconds': weaviate_load_time,
        'peak_memory_mb': weaviate_stats.get('memory_rss_mb', {}).get('max', 0),
        'avg_cpu_percent': weaviate_stats.get('cpu', {}).get('mean', 0)
    }

    print(f"  [OK] Weaviate: {weaviate_load_time:.1f}s, Peak Memory: {all_results['loading']['Weaviate']['peak_memory_mb']:.1f} MB")

    # =========================================================================
    # STEP 3: STORAGE ANALYSIS
    # =========================================================================
    print_section("STEP 3: STORAGE EFFICIENCY ANALYSIS")

    storage_analyzer = StorageAnalyzer()
    storage_analyzer.analyze_milvus_storage(collection_name=collection_name, raw_data_size_mb=raw_data_size_mb)
    storage_analyzer.analyze_weaviate_storage(raw_data_size_mb=raw_data_size_mb)
    storage_analyzer.print_comparison()

    all_results['storage'] = {
        'raw_data_mb': raw_data_size_mb,
        'milvus': storage_analyzer.results.get('milvus', {}),
        'weaviate': storage_analyzer.results.get('weaviate', {})
    }

    # =========================================================================
    # STEP 4: QUERY PERFORMANCE (with and without filters)
    # =========================================================================
    print_section("STEP 4: QUERY PERFORMANCE BENCHMARKS")

    milvus_executor = MilvusQueryExecutor(milvus_loader.collection)
    weaviate_executor = WeaviateQueryExecutor(weaviate_loader.client, 'BenchmarkVector')

    # Generate filters if needed
    query_gen = QueryGenerator()
    filters = query_gen.generate_filter_conditions(n_queries, selectivity=0.1) if run_filters else None

    # Define test configurations
    test_configs = [
        {'name': 'k10_nofilter', 'k': 10, 'use_filters': False},
        {'name': 'k100_nofilter', 'k': 100, 'use_filters': False},
        {'name': 'k1000_nofilter', 'k': 1000, 'use_filters': False},
    ]

    if run_filters and filters:
        test_configs.extend([
            {'name': 'k10_filter', 'k': 10, 'use_filters': True, 'filters': filters},
            {'name': 'k100_filter', 'k': 100, 'use_filters': True, 'filters': filters},
        ])

    runner = BenchmarkRunner(output_dir='results')

    print("\n[Milvus] Running query benchmarks...")
    with ResourceMonitor() as milvus_query_monitor:
        runner.run_benchmark('Milvus', milvus_executor, query_vectors, test_configs, metric_type=metric_type)

    print("\n[Weaviate] Running query benchmarks...")
    with ResourceMonitor() as weaviate_query_monitor:
        runner.run_benchmark('Weaviate', weaviate_executor, query_vectors, test_configs)

    comparison_df = runner.generate_comparison_report()
    all_results['performance'] = comparison_df.to_dict('records')

    print("\n[OK] Performance Results:")
    print(comparison_df.to_string())

    # =========================================================================
    # STEP 5: RECALL@K ACCURACY
    # =========================================================================
    print_section("STEP 5: RECALL@K ACCURACY (Search Quality)")

    if groundtruth is not None:
        recall_calc = RecallCalculator(metric=info['metric'].lower())
        k_values = [1, 5, 10, 20, 50, 100]
        recall_results = {'K': [], 'Milvus': [], 'Weaviate': []}

        for k in k_values:
            if k > groundtruth.shape[1]:
                continue

            # Milvus recall
            # ef must be >= k for HNSW search
            ef_value = max(k, 64)
            milvus_retrieved = []
            for query in query_vectors:
                results = milvus_loader.collection.search(
                    data=[query.tolist()],
                    anns_field="embedding",
                    param={"metric_type": metric_type, "params": {"ef": ef_value}},
                    limit=k
                )
                milvus_retrieved.append([hit.id for hit in results[0]])

            milvus_recall, _ = recall_calc.calculate_recall(milvus_retrieved, groundtruth, k=k)

            # Weaviate recall
            weaviate_retrieved = []
            for query in query_vectors:
                results = weaviate_loader.collection.query.near_vector(
                    near_vector=query.tolist(),
                    limit=k,
                    return_properties=["vectorId"]
                )
                weaviate_retrieved.append([obj.properties.get('vectorId', 0) for obj in results.objects])

            weaviate_recall, _ = recall_calc.calculate_recall(weaviate_retrieved, groundtruth, k=k)

            recall_results['K'].append(k)
            recall_results['Milvus'].append(milvus_recall)
            recall_results['Weaviate'].append(weaviate_recall)

            winner = "Milvus" if milvus_recall > weaviate_recall else "Weaviate" if weaviate_recall > milvus_recall else "Tie"
            print(f"  Recall@{k:3d}: Milvus={milvus_recall:.4f}, Weaviate={weaviate_recall:.4f} -> {winner}")

        all_results['recall'] = recall_results

        # Save recall results
        recall_df = pd.DataFrame(recall_results)
        recall_df.to_csv(results_dir / f'recall_{timestamp}.csv', index=False)
    else:
        print("  [SKIP] No ground truth available for recall calculation")

    # =========================================================================
    # STEP 6: SCALABILITY ANALYSIS (optional)
    # =========================================================================
    if run_scalability and n_vectors < 1000000:
        print_section("STEP 6: SCALABILITY ANALYSIS")

        scale_configs = [
            {'size': 10000, 'name': '10K'},
            {'size': 50000, 'name': '50K'},
            {'size': 100000, 'name': '100K'},
        ]

        # Only add larger sizes if we have enough data
        if n_vectors >= 500000:
            scale_configs.append({'size': 500000, 'name': '500K'})

        generator = VectorDataGenerator()

        for config in scale_configs:
            size = config['size']
            if size > n_vectors:
                continue

            print(f"\n--- Testing {config['name']} vectors ---")

            # Use subset of data
            test_ids = ids[:size]
            test_vectors = base_vectors[:size]
            test_metadata = metadata[:size] if metadata is not None else None

            # Generate query vectors
            test_queries = query_vectors[:100]

            # Test Milvus
            milvus_loader2 = MilvusLoader()
            milvus_loader2.connect()
            milvus_loader2.create_collection(f"scale_{size}", dimension)
            milvus_loader2.load_data(test_ids, test_vectors, test_metadata, batch_size=1000)
            milvus_loader2.create_index(index_type='HNSW', metric_type=metric_type)
            milvus_loader2.load_collection()

            milvus_exec2 = MilvusQueryExecutor(milvus_loader2.collection)
            milvus_result = milvus_exec2.search(test_queries, top_k=10)

            # Test Weaviate
            weaviate_loader2 = WeaviateLoader()
            weaviate_loader2.connect()
            weaviate_loader2.create_schema(dimension)
            weaviate_loader2.load_data(test_ids, test_vectors, test_metadata, batch_size=100)

            weaviate_exec2 = WeaviateQueryExecutor(weaviate_loader2.client, 'BenchmarkVector')
            weaviate_result = weaviate_exec2.search(test_queries, top_k=10)

            all_results['scalability'][config['name']] = {
                'size': size,
                'Milvus': {
                    'p50_ms': milvus_result['p50_latency_ms'],
                    'p95_ms': milvus_result['p95_latency_ms'],
                    'qps': milvus_result['qps']
                },
                'Weaviate': {
                    'p50_ms': weaviate_result['p50_latency_ms'],
                    'p95_ms': weaviate_result['p95_latency_ms'],
                    'qps': weaviate_result['qps']
                }
            }

            print(f"  Milvus:   P50={milvus_result['p50_latency_ms']:.2f}ms, QPS={milvus_result['qps']:.1f}")
            print(f"  Weaviate: P50={weaviate_result['p50_latency_ms']:.2f}ms, QPS={weaviate_result['qps']:.1f}")
    elif n_vectors >= 1000000:
        print_section("STEP 6: SCALABILITY ANALYSIS")
        print("  [SKIP] Skipped for large datasets (>1M vectors) to save time")
        print("  Use --skip-scalability flag or test with smaller datasets")

    # =========================================================================
    # STEP 7: GENERATE FINAL REPORT
    # =========================================================================
    print_section("STEP 7: FINAL REPORT")

    # Save all results
    report_file = results_dir / f'complete_benchmark_{timestamp}.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VECTOR DATABASE BENCHMARK: MILVUS vs WEAVIATE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name} ({n_vectors:,} vectors, {dimension}D)\n\n")

        # Loading Performance
        f.write("-" * 80 + "\n")
        f.write("DATA LOADING PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        for db, stats in all_results['loading'].items():
            f.write(f"\n{db}:\n")
            f.write(f"  Load Time: {stats['load_time_seconds']:.1f} seconds\n")
            f.write(f"  Peak Memory: {stats['peak_memory_mb']:.1f} MB\n")

        # Storage
        f.write("\n" + "-" * 80 + "\n")
        f.write("STORAGE EFFICIENCY\n")
        f.write("-" * 80 + "\n")
        f.write(f"\nRaw Data Size: {all_results['storage']['raw_data_mb']:.2f} MB\n")

        # Query Performance
        f.write("\n" + "-" * 80 + "\n")
        f.write("QUERY PERFORMANCE\n")
        f.write("-" * 80 + "\n\n")
        f.write(comparison_df.to_string() + "\n")

        # Recall
        if all_results['recall']:
            f.write("\n" + "-" * 80 + "\n")
            f.write("RECALL@K ACCURACY\n")
            f.write("-" * 80 + "\n\n")
            for i, k in enumerate(all_results['recall']['K']):
                m = all_results['recall']['Milvus'][i]
                w = all_results['recall']['Weaviate'][i]
                f.write(f"Recall@{k}: Milvus={m:.4f}, Weaviate={w:.4f}\n")

        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Count wins
        milvus_wins = 0
        weaviate_wins = 0
        for record in all_results['performance']:
            if record['Database'] == 'Milvus':
                milvus_wins += 1  # Simplified - Milvus typically wins on latency

        f.write("Choose Milvus if:\n")
        f.write("  - Maximum query performance is critical\n")
        f.write("  - You need fine-grained index control\n")
        f.write("  - GPU acceleration is needed\n\n")

        f.write("Choose Weaviate if:\n")
        f.write("  - You need GraphQL API support\n")
        f.write("  - Hybrid search (vector + keyword) is important\n")
        f.write("  - Easier setup and management is preferred\n")

    print(f"\n[OK] Complete report saved to: {report_file}")
    print(f"[OK] Performance CSV saved to: results/comparison_{timestamp}.csv")
    if all_results['recall']:
        print(f"[OK] Recall CSV saved to: results/recall_{timestamp}.csv")

    # Print summary
    print_section("BENCHMARK COMPLETE - SUMMARY")

    print("\nLoading Performance:")
    print(f"  Milvus:   {all_results['loading']['Milvus']['load_time_seconds']:.1f}s")
    print(f"  Weaviate: {all_results['loading']['Weaviate']['load_time_seconds']:.1f}s")

    print("\nQuery Performance (see detailed results above)")

    if all_results['recall']:
        print("\nRecall@10:")
        idx = all_results['recall']['K'].index(10) if 10 in all_results['recall']['K'] else 0
        print(f"  Milvus:   {all_results['recall']['Milvus'][idx]:.4f}")
        print(f"  Weaviate: {all_results['recall']['Weaviate'][idx]:.4f}")

    print(f"\nAll results saved in: results/")

    # Clean up connections
    try:
        if weaviate_loader and weaviate_loader.client:
            weaviate_loader.client.close()
            print("\n[OK] Weaviate connection closed")
    except Exception as e:
        print(f"\n[WARNING] Error closing Weaviate connection: {e}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Complete Milvus vs Weaviate Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                           # Full benchmark with SIFT1M
  python run_benchmark.py --subset 100000           # Quick test with 100K vectors
  python run_benchmark.py --dataset gist1m          # Use GIST1M (960D vectors)
  python run_benchmark.py --skip-scalability        # Skip scalability tests
  python run_benchmark.py --skip-filters            # Skip filter tests
        """
    )

    parser.add_argument('--dataset', type=str, default='sift1m',
                       choices=[
                           # Standard ANN datasets
                           'sift1m', 'gist1m', 
                           # Word embeddings
                           'glove-100', 'glove-200', 'glove-300',
                           # Image datasets (HDF5 from ann-benchmarks.com)
                           'mnist-784', 'fashion-mnist-784',
                           # Large-scale datasets
                           'nytimes-256', 'lastfm-64', 'deep-image-96',
                           # Sparse/test datasets
                           'kosarak-27983', 'random-xs-20'
                       ],
                       help='Dataset to use (default: sift1m)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of N vectors (for faster testing)')
    parser.add_argument('--skip-scalability', action='store_true',
                       help='Skip scalability analysis')
    parser.add_argument('--skip-filters', action='store_true',
                       help='Skip filter tests')

    args = parser.parse_args()

    print("=" * 80)
    print("MILVUS vs WEAVIATE - COMPLETE BENCHMARK SUITE")
    print("=" * 80)
    print("\nThis benchmark covers:")
    print("  1. Data loading (time, memory, CPU)")
    print("  2. Storage efficiency")
    print("  3. Query performance (with/without filters)")
    print("  4. Recall@K accuracy")
    print("  5. Scalability analysis")
    print("")

    run_complete_benchmark(
        dataset_name=args.dataset,
        subset_size=args.subset,
        run_scalability=not args.skip_scalability,
        run_filters=not args.skip_filters
    )
