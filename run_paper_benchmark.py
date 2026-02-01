"""
Paper-Quality Vector Database Benchmark: Milvus vs Weaviate
============================================================

This script is designed for academic paper publication with:
1. FAIR index parameters (identical HNSW config for both databases)
2. Warm-up queries before measurement
3. Multiple runs with statistical reporting (mean ± std)
4. Latency-recall tradeoff sweep
5. Configurable parameters for reproducibility

Usage:
    python run_paper_benchmark.py --dataset sift1m
    python run_paper_benchmark.py --dataset sift1m --runs 5
    python run_paper_benchmark.py --dataset sift1m --subset 500000
    python run_paper_benchmark.py --sweep-ef  # Run latency-recall tradeoff

Author: Milvus vs Weaviate Benchmark Project
"""

import argparse
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Data loaders
from data.loaders.real_dataset_loader import RealDatasetLoader
from data.loaders.milvus_loader import MilvusLoader
from data.loaders.weaviate_loader import WeaviateLoader

# Query executors
from queries.milvus_queries import MilvusQueryExecutor
from queries.weaviate_queries import WeaviateQueryExecutor
from queries.query_generator import QueryGenerator

# Utilities
from utils.resource_monitor import ResourceMonitor
from utils.storage_analyzer import StorageAnalyzer, calculate_raw_data_size
from utils.recall_calculator import RecallCalculator
from utils.concurrent_tester import ConcurrentTester


# =============================================================================
# FAIR INDEX CONFIGURATION (SAME FOR BOTH DATABASES)
# =============================================================================

@dataclass
class IndexConfig:
    """Fair HNSW index configuration for both databases"""
    M: int = 16                # Number of bi-directional links per node
    efConstruction: int = 200  # Size of dynamic candidate list for construction
    ef: int = 200              # Size of dynamic candidate list for search
    
    def __str__(self):
        return f"HNSW(M={self.M}, efConstruction={self.efConstruction}, ef={self.ef})"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    dataset: str = 'sift1m'
    subset: int = None
    num_runs: int = 3
    num_warmup_queries: int = 100
    num_queries: int = 1000
    k_values: List[int] = None
    index_config: IndexConfig = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [10, 100]
        if self.index_config is None:
            self.index_config = IndexConfig()


# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

def get_deployment_config(mode: str) -> dict:
    """Return connection params based on deployment mode."""
    if mode == 'standalone':
        return {
            'milvus_host': 'localhost', 'milvus_port': 19530,
            'weaviate_host': 'localhost', 'weaviate_port': 8080,
            'weaviate_grpc_port': 50051,
        }
    else:  # cluster
        return {
            'milvus_host': 'localhost', 'milvus_port': 19530,
            'weaviate_host': 'localhost', 'weaviate_port': 8080,
            'weaviate_grpc_port': 50051,
        }


# =============================================================================
# SEARCH WRAPPERS FOR CONCURRENT TESTING
# =============================================================================

def make_milvus_search_func(collection, metric_type: str, ef: int, top_k: int = 10):
    """Create a single-vector search callable for ConcurrentTester."""
    search_params = {"metric_type": metric_type, "params": {"ef": ef}}
    def search_func(query_vector: np.ndarray):
        return collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )
    return search_func


def make_weaviate_search_func(collection, top_k: int = 10):
    """Create a single-vector search callable for ConcurrentTester."""
    def search_func(query_vector: np.ndarray):
        return collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=top_k,
            return_properties=["vectorId"]
        )
    return search_func


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class PaperBenchmarkRunner:
    """Run fair benchmarks for academic paper"""
    
    def __init__(self, config: BenchmarkConfig, output_dir: str = 'results/paper',
                 deploy_config: dict = None, concurrent_clients: List[int] = None,
                 skip_concurrent: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.deploy_config = deploy_config or get_deployment_config('standalone')
        self.concurrent_clients = concurrent_clients or [1, 2, 4, 8, 16]
        self.skip_concurrent = skip_concurrent

        self.results = {
            'config': asdict(config),
            'loading': {},
            'query_performance': [],
            'recall': {},
            'concurrent': {},
            'runs': []
        }
    
    def print_section(self, title: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    
    def load_dataset(self) -> Dict:
        """Load and prepare dataset"""
        self.print_section("STEP 1: LOADING DATASET")
        
        print(f"Dataset: {self.config.dataset}")
        if self.config.subset:
            print(f"Using subset: {self.config.subset:,} vectors")
        
        loader = RealDatasetLoader()
        data = loader.load_dataset(self.config.dataset, download=True)
        
        if self.config.subset:
            data = loader.get_subset(data, self.config.subset)
        
        self.base_vectors = data['base']
        self.query_vectors = data['query'][:self.config.num_queries]
        self.groundtruth = data['groundtruth'][:self.config.num_queries] if data['groundtruth'] is not None else None
        self.metadata = data['metadata']
        self.info = data['info']
        
        self.n_vectors = self.base_vectors.shape[0]
        self.dimension = self.base_vectors.shape[1]
        
        # Handle normalization for cosine metric
        dataset_metric = self.info['metric'].lower()
        if dataset_metric in ['cosine', 'angular']:
            print("  Normalizing vectors for cosine similarity...")
            self.base_vectors = self._normalize(self.base_vectors)
            self.query_vectors = self._normalize(self.query_vectors)
            self.metric_type = 'IP'
        elif dataset_metric in ['l2', 'euclidean']:
            self.metric_type = 'L2'
        else:
            self.metric_type = 'L2'
        
        print(f"\n[OK] Dataset loaded:")
        print(f"  Vectors: {self.n_vectors:,} x {self.dimension}D")
        print(f"  Queries: {len(self.query_vectors):,} (+ {self.config.num_warmup_queries} warm-up)")
        print(f"  Metric: {self.info['metric']} -> {self.metric_type}")
        print(f"  Raw size: {calculate_raw_data_size(self.n_vectors, self.dimension):.2f} MB")
        
        return data
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    
    def setup_milvus(self, collection_name: str) -> Tuple[MilvusLoader, MilvusQueryExecutor]:
        """Setup Milvus with FAIR index parameters"""
        print("\n[Milvus] Setting up with fair index parameters...")
        
        loader = MilvusLoader(
            host=self.deploy_config['milvus_host'],
            port=self.deploy_config['milvus_port']
        )
        loader.connect()
        loader.create_collection(collection_name, self.dimension)
        
        # Use fair index parameters
        index_params = {
            "M": self.config.index_config.M,
            "efConstruction": self.config.index_config.efConstruction
        }
        
        print(f"  Index config: {self.config.index_config}")
        
        # Load data
        batch_size = min(50000, 500000 // self.dimension)
        with ResourceMonitor() as monitor:
            load_start = time.time()
            loader.load_data(
                np.arange(self.n_vectors),
                self.base_vectors,
                self.metadata,
                batch_size=batch_size
            )
            loader.create_index(
                index_type='HNSW',
                metric_type=self.metric_type,
                index_params=index_params,
                wait_timeout=1800
            )
            load_time = time.time() - load_start
        
        loader.load_collection(timeout=300)
        time.sleep(2)
        
        stats = monitor.get_stats()
        self.results['loading']['Milvus'] = {
            'load_time_seconds': load_time,
            'peak_memory_mb': stats.get('memory_rss_mb', {}).get('max', 0),
            'avg_cpu_percent': stats.get('cpu', {}).get('mean', 0),
            'peak_cpu_percent': stats.get('cpu', {}).get('max', 0),
            'disk_read_mb': stats.get('disk_read_mb', {}).get('total', 0),
            'disk_write_mb': stats.get('disk_write_mb', {}).get('total', 0),
            'index_config': str(self.config.index_config)
        }

        print(f"  [OK] Milvus: {load_time:.1f}s, Peak Memory: {self.results['loading']['Milvus']['peak_memory_mb']:.1f} MB, Avg CPU: {self.results['loading']['Milvus']['avg_cpu_percent']:.1f}%")
        
        executor = MilvusQueryExecutor(loader.collection)
        return loader, executor
    
    def setup_weaviate(self) -> Tuple[WeaviateLoader, WeaviateQueryExecutor]:
        """Setup Weaviate with FAIR index parameters - requires modifying the loader"""
        print("\n[Weaviate] Setting up with fair index parameters...")
        
        loader = WeaviateLoader(
            host=self.deploy_config['weaviate_host'],
            port=self.deploy_config['weaviate_port'],
            grpc_port=self.deploy_config['weaviate_grpc_port']
        )

        # Override the INDEX_CONFIG to match Milvus (FAIR comparison)
        loader.INDEX_CONFIG = {
            'type': 'HNSW',
            'ef': self.config.index_config.ef,
            'efConstruction': self.config.index_config.efConstruction,
            'maxConnections': self.config.index_config.M  # M in Weaviate is maxConnections
        }
        
        loader.connect()
        
        print(f"  Index config: {self.config.index_config}")
        
        # Create schema with fair parameters
        loader.create_schema(self.dimension, metric_type=self.info['metric'])
        
        # Load data
        batch_size = min(100, 10000 // self.dimension)
        with ResourceMonitor() as monitor:
            load_start = time.time()
            loader.load_data(
                np.arange(self.n_vectors),
                self.base_vectors,
                self.metadata,
                batch_size=batch_size
            )
            load_time = time.time() - load_start
        
        stats = monitor.get_stats()
        self.results['loading']['Weaviate'] = {
            'load_time_seconds': load_time,
            'peak_memory_mb': stats.get('memory_rss_mb', {}).get('max', 0),
            'avg_cpu_percent': stats.get('cpu', {}).get('mean', 0),
            'peak_cpu_percent': stats.get('cpu', {}).get('max', 0),
            'disk_read_mb': stats.get('disk_read_mb', {}).get('total', 0),
            'disk_write_mb': stats.get('disk_write_mb', {}).get('total', 0),
            'index_config': str(self.config.index_config)
        }

        print(f"  [OK] Weaviate: {load_time:.1f}s, Peak Memory: {self.results['loading']['Weaviate']['peak_memory_mb']:.1f} MB, Avg CPU: {self.results['loading']['Weaviate']['avg_cpu_percent']:.1f}%")
        
        executor = WeaviateQueryExecutor(loader.client, 'BenchmarkVector')
        return loader, executor
    
    def run_warmup(self, milvus_exec: MilvusQueryExecutor, weaviate_exec: WeaviateQueryExecutor):
        """Run warm-up queries before measurement"""
        print(f"\n[WARMUP] Running {self.config.num_warmup_queries} warm-up queries on each database...")
        
        warmup_queries = self.query_vectors[:self.config.num_warmup_queries]
        
        # Milvus warmup
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": self.config.index_config.ef}
        }
        for q in warmup_queries:
            milvus_exec.collection.search(
                data=[q.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=10
            )
        
        # Weaviate warmup
        for q in warmup_queries:
            weaviate_exec.collection.query.near_vector(
                near_vector=q.tolist(),
                limit=10,
                return_properties=["vectorId"]
            )
        
        print("  [OK] Warm-up complete")
    
    def run_single_benchmark(
        self,
        milvus_exec: MilvusQueryExecutor,
        weaviate_exec: WeaviateQueryExecutor,
        run_id: int
    ) -> Dict:
        """Run a single benchmark iteration (with and without filters)"""
        print(f"\n--- Run {run_id + 1}/{self.config.num_runs} ---")

        run_results = {'milvus': {}, 'weaviate': {}}

        # Generate filters for filtered queries
        qgen = QueryGenerator()
        filters = qgen.generate_filter_conditions(len(self.query_vectors), selectivity=0.1)

        for k in self.config.k_values:
            # --- WITHOUT FILTERS ---
            print(f"\n  Testing k={k} (no filter)...")

            search_params = {
                "metric_type": self.metric_type,
                "params": {"ef": max(k, self.config.index_config.ef)}
            }

            milvus_latencies = []
            milvus_retrieved = []
            for query in self.query_vectors:
                start = time.time()
                results = milvus_exec.collection.search(
                    data=[query.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=k
                )
                milvus_latencies.append((time.time() - start) * 1000)
                milvus_retrieved.append([hit.id for hit in results[0]])

            milvus_latencies = np.array(milvus_latencies)
            run_results['milvus'][f'k{k}'] = {
                'p50_ms': np.percentile(milvus_latencies, 50),
                'p95_ms': np.percentile(milvus_latencies, 95),
                'p99_ms': np.percentile(milvus_latencies, 99),
                'mean_ms': np.mean(milvus_latencies),
                'std_ms': np.std(milvus_latencies),
                'qps': 1000 / np.mean(milvus_latencies),
                'retrieved': milvus_retrieved
            }

            weaviate_latencies = []
            weaviate_retrieved = []
            for query in self.query_vectors:
                start = time.time()
                results = weaviate_exec.collection.query.near_vector(
                    near_vector=query.tolist(),
                    limit=k,
                    return_properties=["vectorId"]
                )
                weaviate_latencies.append((time.time() - start) * 1000)
                weaviate_retrieved.append([obj.properties.get('vectorId', 0) for obj in results.objects])

            weaviate_latencies = np.array(weaviate_latencies)
            run_results['weaviate'][f'k{k}'] = {
                'p50_ms': np.percentile(weaviate_latencies, 50),
                'p95_ms': np.percentile(weaviate_latencies, 95),
                'p99_ms': np.percentile(weaviate_latencies, 99),
                'mean_ms': np.mean(weaviate_latencies),
                'std_ms': np.std(weaviate_latencies),
                'qps': 1000 / np.mean(weaviate_latencies),
                'retrieved': weaviate_retrieved
            }

            print(f"    Milvus:   P50={run_results['milvus'][f'k{k}']['p50_ms']:.2f}ms, QPS={run_results['milvus'][f'k{k}']['qps']:.1f}")
            print(f"    Weaviate: P50={run_results['weaviate'][f'k{k}']['p50_ms']:.2f}ms, QPS={run_results['weaviate'][f'k{k}']['qps']:.1f}")

            # --- WITH FILTERS ---
            print(f"  Testing k={k} (with filter)...")

            # Milvus filtered search
            milvus_filter_latencies = []
            for i, query in enumerate(self.query_vectors):
                expr = None
                if i < len(filters) and 'category' in filters[i]:
                    cats = filters[i]['category']['$in']
                    expr = f"category in {cats}"

                start = time.time()
                milvus_exec.collection.search(
                    data=[query.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=k,
                    expr=expr
                )
                milvus_filter_latencies.append((time.time() - start) * 1000)

            milvus_filter_latencies = np.array(milvus_filter_latencies)
            run_results['milvus'][f'k{k}_filter'] = {
                'p50_ms': np.percentile(milvus_filter_latencies, 50),
                'p95_ms': np.percentile(milvus_filter_latencies, 95),
                'p99_ms': np.percentile(milvus_filter_latencies, 99),
                'mean_ms': np.mean(milvus_filter_latencies),
                'std_ms': np.std(milvus_filter_latencies),
                'qps': 1000 / np.mean(milvus_filter_latencies),
            }

            # Weaviate filtered search
            from weaviate.classes.query import Filter
            weaviate_filter_latencies = []
            for i, query in enumerate(self.query_vectors):
                where_filter = None
                if i < len(filters) and 'category' in filters[i]:
                    where_filter = Filter.by_property("category").equal(filters[i]['category']['$in'][0])

                start = time.time()
                if where_filter:
                    weaviate_exec.collection.query.near_vector(
                        near_vector=query.tolist(),
                        limit=k,
                        filters=where_filter,
                        return_properties=["vectorId"]
                    )
                else:
                    weaviate_exec.collection.query.near_vector(
                        near_vector=query.tolist(),
                        limit=k,
                        return_properties=["vectorId"]
                    )
                weaviate_filter_latencies.append((time.time() - start) * 1000)

            weaviate_filter_latencies = np.array(weaviate_filter_latencies)
            run_results['weaviate'][f'k{k}_filter'] = {
                'p50_ms': np.percentile(weaviate_filter_latencies, 50),
                'p95_ms': np.percentile(weaviate_filter_latencies, 95),
                'p99_ms': np.percentile(weaviate_filter_latencies, 99),
                'mean_ms': np.mean(weaviate_filter_latencies),
                'std_ms': np.std(weaviate_filter_latencies),
                'qps': 1000 / np.mean(weaviate_filter_latencies),
            }

            print(f"    Milvus   (filtered): P50={run_results['milvus'][f'k{k}_filter']['p50_ms']:.2f}ms, QPS={run_results['milvus'][f'k{k}_filter']['qps']:.1f}")
            print(f"    Weaviate (filtered): P50={run_results['weaviate'][f'k{k}_filter']['p50_ms']:.2f}ms, QPS={run_results['weaviate'][f'k{k}_filter']['qps']:.1f}")

        return run_results
    
    def calculate_recall(self, run_results: Dict) -> Dict:
        """Calculate recall@K for both databases"""
        if self.groundtruth is None:
            return {}
        
        recall_results = {}
        recall_calc = RecallCalculator(metric=self.info['metric'].lower())
        
        for k in self.config.k_values:
            milvus_retrieved = run_results['milvus'][f'k{k}']['retrieved']
            weaviate_retrieved = run_results['weaviate'][f'k{k}']['retrieved']
            
            milvus_recall, _ = recall_calc.calculate_recall(milvus_retrieved, self.groundtruth, k=k)
            weaviate_recall, _ = recall_calc.calculate_recall(weaviate_retrieved, self.groundtruth, k=k)
            
            recall_results[f'k{k}'] = {
                'Milvus': milvus_recall,
                'Weaviate': weaviate_recall
            }
        
        return recall_results
    
    def aggregate_results(self) -> pd.DataFrame:
        """Aggregate results from multiple runs with mean ± std"""
        if not self.results['runs']:
            return pd.DataFrame()

        aggregated = []

        # Collect all test keys (k10, k100, k10_filter, k100_filter, etc.)
        all_keys = list(self.results['runs'][0]['milvus'].keys())

        for test_key in all_keys:
            for db in ['milvus', 'weaviate']:
                try:
                    p50_values = [r[db][test_key]['p50_ms'] for r in self.results['runs']]
                    p95_values = [r[db][test_key]['p95_ms'] for r in self.results['runs']]
                    qps_values = [r[db][test_key]['qps'] for r in self.results['runs']]
                except KeyError:
                    continue

                aggregated.append({
                    'Database': db.capitalize(),
                    'K': test_key,
                    'P50 (ms)': f"{np.mean(p50_values):.2f} ± {np.std(p50_values):.2f}",
                    'P50_mean': np.mean(p50_values),
                    'P50_std': np.std(p50_values),
                    'P95 (ms)': f"{np.mean(p95_values):.2f} ± {np.std(p95_values):.2f}",
                    'P95_mean': np.mean(p95_values),
                    'P95_std': np.std(p95_values),
                    'QPS': f"{np.mean(qps_values):.1f} ± {np.std(qps_values):.1f}",
                    'QPS_mean': np.mean(qps_values),
                    'QPS_std': np.std(qps_values)
                })

        return pd.DataFrame(aggregated)
    
    def run_ef_sweep(
        self,
        milvus_exec: MilvusQueryExecutor,
        weaviate_loader: WeaviateLoader,
        ef_values: List[int] = None
    ) -> pd.DataFrame:
        """Run latency-recall tradeoff sweep by varying ef parameter"""
        if ef_values is None:
            ef_values = [16, 32, 64, 128, 200, 256, 512]
        
        self.print_section("LATENCY-RECALL TRADEOFF SWEEP")
        print(f"Testing ef values: {ef_values}")
        
        sweep_results = []
        k = 10  # Standard k for tradeoff analysis
        
        for ef in ef_values:
            print(f"\n  Testing ef={ef}...")
            
            # Milvus with varying ef
            search_params = {
                "metric_type": self.metric_type,
                "params": {"ef": ef}
            }
            
            milvus_latencies = []
            milvus_retrieved = []
            for query in self.query_vectors[:100]:  # Use fewer queries for sweep
                start = time.time()
                results = milvus_exec.collection.search(
                    data=[query.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=k
                )
                milvus_latencies.append((time.time() - start) * 1000)
                milvus_retrieved.append([hit.id for hit in results[0]])
            
            # Calculate recall
            if self.groundtruth is not None:
                recall_calc = RecallCalculator(metric=self.info['metric'].lower())
                milvus_recall, _ = recall_calc.calculate_recall(milvus_retrieved, self.groundtruth[:100], k=k)
            else:
                milvus_recall = None
            
            sweep_results.append({
                'Database': 'Milvus',
                'ef': ef,
                'P50_ms': np.percentile(milvus_latencies, 50),
                'QPS': 1000 / np.mean(milvus_latencies),
                'Recall@10': milvus_recall
            })
            
            print(f"    Milvus:   P50={sweep_results[-1]['P50_ms']:.2f}ms, Recall@10={milvus_recall:.4f if milvus_recall else 'N/A'}")
        
        # Save sweep results
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(self.output_dir / f'ef_sweep_{self.timestamp}.csv', index=False)
        
        return sweep_df
    
    def run_concurrent_benchmark(
        self,
        milvus_exec: MilvusQueryExecutor,
        weaviate_exec: WeaviateQueryExecutor,
    ) -> Dict:
        """Run concurrent load tests with varying client counts."""
        self.print_section("STEP 6: CONCURRENT LOAD TEST")
        print(f"Client counts: {self.concurrent_clients}")

        milvus_search = make_milvus_search_func(
            milvus_exec.collection, self.metric_type,
            self.config.index_config.ef, top_k=10
        )
        weaviate_search = make_weaviate_search_func(
            weaviate_exec.collection, top_k=10
        )

        concurrent_results = {'milvus': [], 'weaviate': []}

        for n_clients in self.concurrent_clients:
            print(f"\n  Testing {n_clients} concurrent client(s)...")

            tester = ConcurrentTester(n_clients=n_clients)
            milvus_result = tester.run_load_test(
                milvus_search, self.query_vectors,
                duration_seconds=30, warmup_seconds=2
            )
            concurrent_results['milvus'].append({
                'n_clients': n_clients,
                'qps': milvus_result.qps,
                'p50_ms': milvus_result.p50 * 1000,
                'p95_ms': milvus_result.p95 * 1000,
                'p99_ms': milvus_result.p99 * 1000,
                'mean_ms': milvus_result.mean * 1000,
                'failed': milvus_result.failed_requests,
            })

            tester2 = ConcurrentTester(n_clients=n_clients)
            weaviate_result = tester2.run_load_test(
                weaviate_search, self.query_vectors,
                duration_seconds=30, warmup_seconds=2
            )
            concurrent_results['weaviate'].append({
                'n_clients': n_clients,
                'qps': weaviate_result.qps,
                'p50_ms': weaviate_result.p50 * 1000,
                'p95_ms': weaviate_result.p95 * 1000,
                'p99_ms': weaviate_result.p99 * 1000,
                'mean_ms': weaviate_result.mean * 1000,
                'failed': weaviate_result.failed_requests,
            })

            print(f"    Milvus:   QPS={milvus_result.qps:.1f}, P50={milvus_result.p50*1000:.2f}ms")
            print(f"    Weaviate: QPS={weaviate_result.qps:.1f}, P50={weaviate_result.p50*1000:.2f}ms")

        self.results['concurrent'] = concurrent_results

        # Save concurrent CSV
        rows = []
        for db_name in ['milvus', 'weaviate']:
            for entry in concurrent_results[db_name]:
                rows.append({'database': db_name.capitalize(), **entry})
        pd.DataFrame(rows).to_csv(
            self.output_dir / f'concurrent_load_test_{self.timestamp}.csv', index=False
        )

        return concurrent_results

    def generate_report(self):
        """Generate comprehensive report"""
        self.print_section("FINAL REPORT")
        
        report_file = self.output_dir / f'paper_benchmark_{self.timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PAPER-QUALITY VECTOR DATABASE BENCHMARK: MILVUS vs WEAVIATE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.config.dataset} ({self.n_vectors:,} vectors, {self.dimension}D)\n")
            f.write(f"Number of runs: {self.config.num_runs}\n")
            f.write(f"Index configuration: {self.config.index_config}\n")
            f.write(f"Warm-up queries: {self.config.num_warmup_queries}\n\n")
            
            # Loading Performance
            f.write("-" * 80 + "\n")
            f.write("DATA LOADING PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for db, stats in self.results['loading'].items():
                f.write(f"\n{db}:\n")
                f.write(f"  Load Time: {stats['load_time_seconds']:.1f} seconds\n")
                f.write(f"  Peak Memory: {stats['peak_memory_mb']:.1f} MB\n")
                f.write(f"  Avg CPU: {stats.get('avg_cpu_percent', 0):.1f}%\n")
                f.write(f"  Peak CPU: {stats.get('peak_cpu_percent', 0):.1f}%\n")
                f.write(f"  Disk Read: {stats.get('disk_read_mb', 0):.1f} MB\n")
                f.write(f"  Disk Write: {stats.get('disk_write_mb', 0):.1f} MB\n")
                f.write(f"  Index Config: {stats['index_config']}\n")

            # Storage efficiency
            if self.results.get('storage'):
                f.write("\n" + "-" * 80 + "\n")
                f.write("STORAGE EFFICIENCY\n")
                f.write("-" * 80 + "\n\n")
                st = self.results['storage']
                f.write(f"Raw data size: {st['raw_data_size_mb']:.2f} MB\n")
                for db_key in ['milvus', 'weaviate']:
                    db_st = st.get(db_key, {})
                    if 'error' not in db_st:
                        f.write(f"\n{db_key.capitalize()}:\n")
                        f.write(f"  Total disk usage: {db_st.get('total_size_mb', 0):.2f} MB\n")
                        if db_st.get('compression_ratio'):
                            f.write(f"  Compression ratio: {db_st['compression_ratio']:.2f}x\n")
                        if db_st.get('breakdown'):
                            for comp, size in db_st['breakdown'].items():
                                f.write(f"  {comp}: {size:.2f} MB\n")
            
            # Query Performance with statistics
            f.write("\n" + "-" * 80 + "\n")
            f.write("QUERY PERFORMANCE (mean ± std over {} runs)\n".format(self.config.num_runs))
            f.write("-" * 80 + "\n\n")
            
            agg_df = self.aggregate_results()
            if not agg_df.empty:
                f.write(agg_df[['Database', 'K', 'P50 (ms)', 'P95 (ms)', 'QPS']].to_string(index=False))
                f.write("\n")

            # Filtered query performance
            if self.results['runs']:
                last_run = self.results['runs'][-1]
                filter_keys = [k for k in last_run['milvus'] if k.endswith('_filter')]
                if filter_keys:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("FILTERED QUERY PERFORMANCE (last run)\n")
                    f.write("-" * 80 + "\n\n")
                    f.write(f"{'Test':>15} | {'Milvus P50ms':>14} {'QPS':>8} | {'Weaviate P50ms':>16} {'QPS':>8}\n")
                    f.write("-" * 70 + "\n")
                    for fk in filter_keys:
                        m = last_run['milvus'][fk]
                        w = last_run['weaviate'][fk]
                        f.write(f"{fk:>15} | {m['p50_ms']:>14.2f} {m['qps']:>8.1f} | {w['p50_ms']:>16.2f} {w['qps']:>8.1f}\n")

            # Query-phase resource usage
            if self.results.get('query_resources'):
                qr = self.results['query_resources']
                f.write("\n" + "-" * 80 + "\n")
                f.write("RESOURCE USAGE DURING QUERIES\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"Avg CPU: {qr['avg_cpu_percent']:.1f}%\n")
                f.write(f"Peak CPU: {qr['peak_cpu_percent']:.1f}%\n")
                f.write(f"Peak Memory: {qr['peak_memory_mb']:.1f} MB\n")
                f.write(f"Disk Read: {qr['disk_read_mb']:.1f} MB\n")
                f.write(f"Disk Write: {qr['disk_write_mb']:.1f} MB\n")

            # Recall
            if self.results['recall']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("RECALL@K ACCURACY\n")
                f.write("-" * 80 + "\n\n")
                for k_str, recalls in self.results['recall'].items():
                    k = k_str.replace('k', '')
                    f.write(f"Recall@{k}: Milvus={recalls['Milvus']:.4f}, Weaviate={recalls['Weaviate']:.4f}\n")
            
            # Concurrent load test
            if self.results.get('concurrent') and self.results['concurrent'].get('milvus'):
                f.write("\n" + "-" * 80 + "\n")
                f.write("CONCURRENT LOAD TEST (QPS Scaling)\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"{'Clients':>8} | {'Milvus QPS':>12} {'P50ms':>8} {'P95ms':>8} | {'Weaviate QPS':>14} {'P50ms':>8} {'P95ms':>8}\n")
                f.write("-" * 78 + "\n")
                for m, w in zip(self.results['concurrent']['milvus'],
                                self.results['concurrent']['weaviate']):
                    f.write(f"{m['n_clients']:>8} | {m['qps']:>12.1f} {m['p50_ms']:>8.2f} {m['p95_ms']:>8.2f} | {w['qps']:>14.1f} {w['p50_ms']:>8.2f} {w['p95_ms']:>8.2f}\n")

            # Fair comparison note
            f.write("\n" + "=" * 80 + "\n")
            f.write("METHODOLOGY NOTE\n")
            f.write("=" * 80 + "\n\n")
            f.write("This benchmark uses IDENTICAL index parameters for both databases:\n")
            f.write(f"  - HNSW M: {self.config.index_config.M}\n")
            f.write(f"  - efConstruction: {self.config.index_config.efConstruction}\n")
            f.write(f"  - Search ef: {self.config.index_config.ef}\n")
            f.write(f"  - Warm-up queries: {self.config.num_warmup_queries}\n")
            f.write(f"  - Test queries: {len(self.query_vectors)}\n")
            f.write(f"  - Runs: {self.config.num_runs} (mean ± std reported)\n")
        
        print(f"\n[OK] Report saved to: {report_file}")
        
        # Save raw results as JSON for further analysis
        json_file = self.output_dir / f'paper_benchmark_{self.timestamp}.json'
        
        # Convert to JSON-serializable format
        json_results = {
            'config': asdict(self.config),
            'loading': self.results['loading'],
            'aggregated': agg_df.to_dict('records') if not agg_df.empty else [],
            'recall': self.results['recall'],
            'storage': self.results.get('storage', {}),
            'query_resources': self.results.get('query_resources', {}),
            'concurrent': self.results.get('concurrent', {})
        }
        json_results['config']['index_config'] = asdict(self.config.index_config)
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"[OK] JSON results saved to: {json_file}")
        
        # Save CSV for easy import
        if not agg_df.empty:
            csv_file = self.output_dir / f'paper_benchmark_{self.timestamp}.csv'
            agg_df.to_csv(csv_file, index=False)
            print(f"[OK] CSV results saved to: {csv_file}")
    
    def run(self):
        """Run the complete benchmark"""
        self.print_section("PAPER-QUALITY BENCHMARK")
        print(f"Configuration:")
        print(f"  Dataset: {self.config.dataset}")
        print(f"  Subset: {self.config.subset or 'Full'}")
        print(f"  Runs: {self.config.num_runs}")
        print(f"  Index: {self.config.index_config}")
        print(f"  K values: {self.config.k_values}")
        
        # Load dataset
        self.load_dataset()
        
        # Setup databases with fair parameters
        self.print_section("STEP 2: SETUP DATABASES (Fair Index Parameters)")
        collection_name = f"paper_benchmark_{self.timestamp}"
        milvus_loader, milvus_exec = self.setup_milvus(collection_name)
        weaviate_loader, weaviate_exec = self.setup_weaviate()
        
        # Storage analysis
        self.print_section("STEP 2b: STORAGE ANALYSIS")
        raw_size_mb = calculate_raw_data_size(self.n_vectors, self.dimension)
        storage_analyzer = StorageAnalyzer()
        milvus_storage = storage_analyzer.analyze_milvus_storage(raw_data_size_mb=raw_size_mb)
        weaviate_storage = storage_analyzer.analyze_weaviate_storage(raw_data_size_mb=raw_size_mb)
        self.results['storage'] = {
            'raw_data_size_mb': raw_size_mb,
            'milvus': milvus_storage,
            'weaviate': weaviate_storage,
        }
        storage_analyzer.print_comparison()

        # Warm-up
        self.print_section("STEP 3: WARM-UP")
        self.run_warmup(milvus_exec, weaviate_exec)
        
        # Run multiple iterations
        self.print_section(f"STEP 4: RUNNING {self.config.num_runs} BENCHMARK ITERATIONS")

        with ResourceMonitor() as query_monitor:
            for run_id in range(self.config.num_runs):
                run_results = self.run_single_benchmark(milvus_exec, weaviate_exec, run_id)
                self.results['runs'].append(run_results)

        query_stats = query_monitor.get_stats()
        self.results['query_resources'] = {
            'avg_cpu_percent': query_stats.get('cpu', {}).get('mean', 0),
            'peak_cpu_percent': query_stats.get('cpu', {}).get('max', 0),
            'peak_memory_mb': query_stats.get('memory_rss_mb', {}).get('max', 0),
            'disk_read_mb': query_stats.get('disk_read_mb', {}).get('total', 0),
            'disk_write_mb': query_stats.get('disk_write_mb', {}).get('total', 0),
        }
        print(f"\n  Query phase resources: Avg CPU={self.results['query_resources']['avg_cpu_percent']:.1f}%, Peak Mem={self.results['query_resources']['peak_memory_mb']:.1f} MB")
        
        # Calculate recall (from last run)
        self.print_section("STEP 5: RECALL CALCULATION")
        self.results['recall'] = self.calculate_recall(self.results['runs'][-1])
        
        if self.results['recall']:
            for k_str, recalls in self.results['recall'].items():
                print(f"  {k_str}: Milvus={recalls['Milvus']:.4f}, Weaviate={recalls['Weaviate']:.4f}")
        else:
            print("  [SKIP] No ground truth available")

        # Concurrent load test
        if not self.skip_concurrent:
            self.run_concurrent_benchmark(milvus_exec, weaviate_exec)
        else:
            print("\n[SKIP] Concurrent load test (--skip-concurrent)")

        # Generate report
        self.generate_report()
        
        # Cleanup
        try:
            if weaviate_loader and weaviate_loader.client:
                weaviate_loader.client.close()
        except:
            pass
        
        return self.results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper-Quality Milvus vs Weaviate Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_paper_benchmark.py --dataset sift1m
  python run_paper_benchmark.py --dataset sift1m --runs 5
  python run_paper_benchmark.py --dataset sift1m --subset 500000
  python run_paper_benchmark.py --dataset gist1m --M 16 --ef 200
        """
    )
    
    parser.add_argument('--dataset', type=str, default='sift1m',
                       help='Dataset to benchmark')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of N vectors')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of benchmark runs (default: 3)')
    parser.add_argument('--queries', type=int, default=1000,
                       help='Number of test queries (default: 1000)')
    parser.add_argument('--warmup', type=int, default=100,
                       help='Number of warm-up queries (default: 100)')
    
    # Index parameters
    parser.add_argument('--M', type=int, default=16,
                       help='HNSW M parameter (default: 16)')
    parser.add_argument('--ef-construction', type=int, default=200,
                       help='HNSW efConstruction (default: 200)')
    parser.add_argument('--ef', type=int, default=200,
                       help='HNSW ef search parameter (default: 200)')
    
    # Sweep mode
    parser.add_argument('--sweep-ef', action='store_true',
                       help='Run ef parameter sweep for latency-recall tradeoff')

    # Deployment mode
    parser.add_argument('--mode', type=str, default='standalone',
                       choices=['standalone', 'cluster'],
                       help='Deployment mode: standalone or cluster (default: standalone)')

    # Concurrent testing
    parser.add_argument('--concurrent-clients', type=str, default='1,2,4,8,16',
                       help='Comma-separated client counts for concurrent test (default: 1,2,4,8,16)')
    parser.add_argument('--skip-concurrent', action='store_true',
                       help='Skip concurrent load test')

    args = parser.parse_args()
    
    # Create config
    index_config = IndexConfig(
        M=args.M,
        efConstruction=args.ef_construction,
        ef=args.ef
    )
    
    config = BenchmarkConfig(
        dataset=args.dataset,
        subset=args.subset,
        num_runs=args.runs,
        num_queries=args.queries,
        num_warmup_queries=args.warmup,
        index_config=index_config
    )
    
    # Deployment and concurrent config
    deploy_config = get_deployment_config(args.mode)
    concurrent_clients = [int(x) for x in args.concurrent_clients.split(',')]

    # Run benchmark
    runner = PaperBenchmarkRunner(
        config,
        deploy_config=deploy_config,
        concurrent_clients=concurrent_clients,
        skip_concurrent=args.skip_concurrent
    )
    runner.run()
    
    # Optional: ef sweep
    if args.sweep_ef:
        print("\n\n[INFO] Running ef sweep - this requires re-using the loaded data")


if __name__ == "__main__":
    main()
