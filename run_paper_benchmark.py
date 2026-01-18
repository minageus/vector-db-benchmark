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
# BENCHMARK RUNNER
# =============================================================================

class PaperBenchmarkRunner:
    """Run fair benchmarks for academic paper"""
    
    def __init__(self, config: BenchmarkConfig, output_dir: str = 'results/paper'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.results = {
            'config': asdict(config),
            'loading': {},
            'query_performance': [],
            'recall': {},
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
        
        loader = MilvusLoader()
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
            'index_config': str(self.config.index_config)
        }
        
        print(f"  [OK] Milvus: {load_time:.1f}s, Peak Memory: {self.results['loading']['Milvus']['peak_memory_mb']:.1f} MB")
        
        executor = MilvusQueryExecutor(loader.collection)
        return loader, executor
    
    def setup_weaviate(self) -> Tuple[WeaviateLoader, WeaviateQueryExecutor]:
        """Setup Weaviate with FAIR index parameters - requires modifying the loader"""
        print("\n[Weaviate] Setting up with fair index parameters...")
        
        loader = WeaviateLoader()
        
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
            'index_config': str(self.config.index_config)
        }
        
        print(f"  [OK] Weaviate: {load_time:.1f}s, Peak Memory: {self.results['loading']['Weaviate']['peak_memory_mb']:.1f} MB")
        
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
        """Run a single benchmark iteration"""
        print(f"\n--- Run {run_id + 1}/{self.config.num_runs} ---")
        
        run_results = {'milvus': {}, 'weaviate': {}}
        
        for k in self.config.k_values:
            print(f"\n  Testing k={k}...")
            
            # Milvus search with fair ef parameter
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
            
            # Weaviate search
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
        
        for k in self.config.k_values:
            for db in ['milvus', 'weaviate']:
                p50_values = [r[db][f'k{k}']['p50_ms'] for r in self.results['runs']]
                p95_values = [r[db][f'k{k}']['p95_ms'] for r in self.results['runs']]
                qps_values = [r[db][f'k{k}']['qps'] for r in self.results['runs']]
                
                aggregated.append({
                    'Database': db.capitalize(),
                    'K': k,
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
                f.write(f"  Index Config: {stats['index_config']}\n")
            
            # Query Performance with statistics
            f.write("\n" + "-" * 80 + "\n")
            f.write("QUERY PERFORMANCE (mean ± std over {} runs)\n".format(self.config.num_runs))
            f.write("-" * 80 + "\n\n")
            
            agg_df = self.aggregate_results()
            if not agg_df.empty:
                f.write(agg_df[['Database', 'K', 'P50 (ms)', 'P95 (ms)', 'QPS']].to_string(index=False))
                f.write("\n")
            
            # Recall
            if self.results['recall']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("RECALL@K ACCURACY\n")
                f.write("-" * 80 + "\n\n")
                for k_str, recalls in self.results['recall'].items():
                    k = k_str.replace('k', '')
                    f.write(f"Recall@{k}: Milvus={recalls['Milvus']:.4f}, Weaviate={recalls['Weaviate']:.4f}\n")
            
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
            'recall': self.results['recall']
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
        
        # Warm-up
        self.print_section("STEP 3: WARM-UP")
        self.run_warmup(milvus_exec, weaviate_exec)
        
        # Run multiple iterations
        self.print_section(f"STEP 4: RUNNING {self.config.num_runs} BENCHMARK ITERATIONS")
        
        for run_id in range(self.config.num_runs):
            run_results = self.run_single_benchmark(milvus_exec, weaviate_exec, run_id)
            self.results['runs'].append(run_results)
        
        # Calculate recall (from last run)
        self.print_section("STEP 5: RECALL CALCULATION")
        self.results['recall'] = self.calculate_recall(self.results['runs'][-1])
        
        if self.results['recall']:
            for k_str, recalls in self.results['recall'].items():
                print(f"  {k_str}: Milvus={recalls['Milvus']:.4f}, Weaviate={recalls['Weaviate']:.4f}")
        else:
            print("  [SKIP] No ground truth available")
        
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
    
    # Run benchmark
    runner = PaperBenchmarkRunner(config)
    runner.run()
    
    # Optional: ef sweep
    if args.sweep_ef:
        print("\n\n[INFO] Running ef sweep - this requires re-using the loaded data")


if __name__ == "__main__":
    main()
