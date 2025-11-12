import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

class PerformanceAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        sns.set_style('whitegrid')
    
    def analyze_latency_distribution(
        self,
        latencies_dict: Dict[str, np.ndarray],
        output_path: str
    ):
        """Create latency distribution plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        for db_name, latencies in latencies_dict.items():
            axes[0, 0].hist(latencies, bins=50, alpha=0.6, label=db_name)
        axes[0, 0].set_xlabel('Latency (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Latency Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        data_for_box = []
        labels_for_box = []
        for db_name, latencies in latencies_dict.items():
            data_for_box.append(latencies)
            labels_for_box.append(db_name)
        axes[0, 1].boxplot(data_for_box, labels=labels_for_box)
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Latency Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # CDF
        for db_name, latencies in latencies_dict.items():
            sorted_lat = np.sort(latencies)
            cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
            axes[1, 0].plot(sorted_lat, cdf, label=db_name, linewidth=2)
        axes[1, 0].set_xlabel('Latency (ms)')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Percentile comparison
        percentiles = [50, 75, 90, 95, 99]
        x = np.arange(len(percentiles))
        width = 0.35
        
        for i, (db_name, latencies) in enumerate(latencies_dict.items()):
            values = [np.percentile(latencies, p) for p in percentiles]
            axes[1, 1].bar(x + i*width, values, width, label=db_name)
        
        axes[1, 1].set_xlabel('Percentile')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Percentile Comparison')
        axes[1, 1].set_xticks(x + width / 2)
        axes[1, 1].set_xticklabels([f'P{p}' for p in percentiles])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Latency analysis saved to {output_path}")
    
    def compare_scalability(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: str
    ):
        """Compare scalability across dataset sizes"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        dataset_sizes = []
        metrics_by_db = {}
        
        for test_name, test_results in results.items():
            if 'dataset_size' in test_results:
                dataset_sizes.append(test_results['dataset_size'])
                
                for db_name, metrics in test_results.items():
                    if db_name != 'dataset_size':
                        if db_name not in metrics_by_db:
                            metrics_by_db[db_name] = {
                                'p50': [],
                                'p95': [],
                                'qps': [],
                                'memory': []
                            }
                        metrics_by_db[db_name]['p50'].append(metrics.get('p50', 0))
                        metrics_by_db[db_name]['p95'].append(metrics.get('p95', 0))
                        metrics_by_db[db_name]['qps'].append(metrics.get('qps', 0))
                        metrics_by_db[db_name]['memory'].append(metrics.get('memory_mb', 0))
        
        # P50 Latency vs Size
        for db_name, metrics in metrics_by_db.items():
            axes[0, 0].plot(dataset_sizes, metrics['p50'], 
                          marker='o', label=db_name, linewidth=2)
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('P50 Latency (ms)')
        axes[0, 0].set_title('Latency vs Dataset Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # QPS vs Size
        for db_name, metrics in metrics_by_db.items():
            axes[0, 1].plot(dataset_sizes, metrics['qps'], 
                          marker='o', label=db_name, linewidth=2)
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('QPS')
        axes[0, 1].set_title('Throughput vs Dataset Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log')
        
        # Memory Usage vs Size
        for db_name, metrics in metrics_by_db.items():
            axes[1, 0].plot(dataset_sizes, metrics['memory'], 
                          marker='o', label=db_name, linewidth=2)
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage vs Dataset Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')
        
        # Efficiency Score (QPS / Latency)
        for db_name, metrics in metrics_by_db.items():
            efficiency = np.array(metrics['qps']) / (np.array(metrics['p50']) + 1)
            axes[1, 1].plot(dataset_sizes, efficiency, 
                          marker='o', label=db_name, linewidth=2)
        axes[1, 1].set_xlabel('Dataset Size')
        axes[1, 1].set_ylabel('Efficiency Score (QPS/Latency)')
        axes[1, 1].set_title('Efficiency vs Dataset Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Scalability analysis saved to {output_path}")
    
    def generate_summary_report(
        self,
        all_results: Dict,
        output_path: str
    ):
        """Generate comprehensive summary report"""
        
        report = []
        report.append("="*80)
        report.append("VECTOR DATABASE COMPARISON - SUMMARY REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {pd.Timestamp.now()}")
        report.append("\n")
        
        # Overall winner determination
        report.append("OVERALL PERFORMANCE SUMMARY")
        report.append("-"*80)
        
        scores = {'Milvus': 0, 'Weaviate': 0}
        categories = []
        
        for test_name, results in all_results.items():
            report.append(f"\nTest: {test_name}")
            
            if 'Milvus' in results and 'Weaviate' in results:
                m_lat = results['Milvus'].get('p50_latency_ms', float('inf'))
                w_lat = results['Weaviate'].get('p50_latency_ms', float('inf'))
                
                m_qps = results['Milvus'].get('qps', 0)
                w_qps = results['Weaviate'].get('qps', 0)
                
                # Compare latency (lower is better)
                if m_lat < w_lat:
                    scores['Milvus'] += 1
                    report.append(f"  Latency Winner: Milvus ({m_lat:.2f}ms vs {w_lat:.2f}ms)")
                else:
                    scores['Weaviate'] += 1
                    report.append(f"  Latency Winner: Weaviate ({w_lat:.2f}ms vs {m_lat:.2f}ms)")
                
                # Compare throughput (higher is better)
                if m_qps > w_qps:
                    scores['Milvus'] += 1
                    report.append(f"  Throughput Winner: Milvus ({m_qps:.2f} vs {w_qps:.2f} QPS)")
                else:
                    scores['Weaviate'] += 1
                    report.append(f"  Throughput Winner: Weaviate ({w_qps:.2f} vs {m_qps:.2f} QPS)")
        
        report.append("\n" + "="*80)
        report.append("FINAL SCORE")
        report.append("-"*80)
        report.append(f"Milvus: {scores['Milvus']} wins")
        report.append(f"Weaviate: {scores['Weaviate']} wins")
        
        if scores['Milvus'] > scores['Weaviate']:
            report.append("\nOverall Winner: Milvus")
        elif scores['Weaviate'] > scores['Milvus']:
            report.append("\nOverall Winner: Weaviate")
        else:
            report.append("\nResult: Tie")
        
        report.append("\n" + "="*80)
        report.append("RECOMMENDATIONS")
        report.append("-"*80)
        report.append("\nChoose Milvus if:")
        report.append("  • You need maximum performance at scale (10M+ vectors)")
        report.append("  • GPU acceleration is important")
        report.append("  • You're building a production ML/AI system")
        report.append("  • You need fine-grained control over indexing")
        
        report.append("\nChoose Weaviate if:")
        report.append("  • You need GraphQL API support")
        report.append("  • Hybrid search (vector + keyword) is important")
        report.append("  • You want easier setup and management")
        report.append("  • You need built-in ML model integration")
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {output_path}")