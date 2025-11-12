import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BenchmarkRunner:
    """Run complete benchmark suite and generate reports"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.results = {}
    
    def run_benchmark(self, db_name: str, executor, queries, test_configs):
        """Run benchmark for a specific database"""
        print(f"\n{'='*60}")
        print(f"Running {db_name} Benchmark")
        print(f"{'='*60}")
        
        db_results = {}
        
        for config in test_configs:
            print(f"\nTest: {config['name']}")
            print(f"  k={config['k']}, queries={len(queries)}, filters={config.get('use_filters', False)}")
            
            result = executor.search(
                query_vectors=queries,
                top_k=config['k'],
                filters=config.get('filters')
            )
            
            db_results[config['name']] = result
            
            print(f"  P50: {result['p50_latency_ms']:.2f}ms")
            print(f"  P95: {result['p95_latency_ms']:.2f}ms")
            print(f"  P99: {result['p99_latency_ms']:.2f}ms")
            print(f"  QPS: {result['qps']:.2f}")
        
        self.results[db_name] = db_results
        return db_results
    
    def generate_comparison_report(self):
        """Generate comparison charts and tables"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        comparison_data = []
        for db_name, tests in self.results.items():
            for test_name, metrics in tests.items():
                comparison_data.append({
                    'Database': db_name,
                    'Test': test_name,
                    'P50 (ms)': metrics['p50_latency_ms'],
                    'P95 (ms)': metrics['p95_latency_ms'],
                    'P99 (ms)': metrics['p99_latency_ms'],
                    'Mean (ms)': metrics['mean_latency_ms'],
                    'QPS': metrics['qps']
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = f"{self.output_dir}/comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Generate plots
        self._plot_latency_comparison(df, timestamp)
        self._plot_qps_comparison(df, timestamp)
        
        return df
    
    def _plot_latency_comparison(self, df, timestamp):
        """Plot latency comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(['P50 (ms)', 'P95 (ms)', 'P99 (ms)']):
            pivot = df.pivot(index='Test', columns='Database', values=metric)
            pivot.plot(kind='bar', ax=axes[idx], title=metric)
            axes[idx].set_ylabel('Latency (ms)')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/latency_comparison_{timestamp}.png", dpi=300)
        print(f"Latency plot saved")
    
    def _plot_qps_comparison(self, df, timestamp):
        """Plot QPS comparison"""
        plt.figure(figsize=(10, 6))
        pivot = df.pivot(index='Test', columns='Database', values='QPS')
        pivot.plot(kind='bar')
        plt.title('Queries Per Second Comparison')
        plt.ylabel('QPS')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/qps_comparison_{timestamp}.png", dpi=300)
        print(f"QPS plot saved")