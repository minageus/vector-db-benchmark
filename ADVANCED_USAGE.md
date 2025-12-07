# Advanced Analysis & Utilities Guide

This guide explains how to use the advanced analysis tools and utilities in the vector database comparison project.

## 📊 Available Tools

### 1. **Advanced Analysis** (`run_advanced_analysis.py`)
Comprehensive performance analysis with multiple test scenarios.

### 2. **Concurrent Load Testing** (`run_concurrent_test.py`)
Test database performance under concurrent load.

### 3. **Performance Analyzer** (`analysis/performance_analyzer.py`)
Create detailed visualizations and reports.

### 4. **Recall Calculator** (`utils/recall_calculator.py`)
Measure search accuracy (Recall@K).

### 5. **Concurrent Tester** (`utils/concurrent_tester.py`)
Multi-threaded load testing framework.

---

## 🚀 Quick Start

### Run Interactive Advanced Analysis
```powershell
python run_advanced_analysis.py
```

**Menu Options:**
1. **Latency Distribution Analysis** - Analyze latency patterns with histograms, CDF, box plots
2. **Recall@K Accuracy Test** - Measure search result accuracy compared to ground truth
3. **Scalability Analysis** - Test performance across multiple dataset sizes
4. **Generate Comprehensive Report** - Create a detailed summary report
5. **Run ALL analyses** - Execute all tests in sequence

---

## 📈 Individual Analysis Tools

### 1. Latency Distribution Analysis

Analyzes existing benchmark results to create detailed latency visualizations:
- **Histogram**: Distribution of query latencies
- **Box Plot**: Outlier detection and quartile analysis
- **CDF**: Cumulative distribution function
- **Percentile Comparison**: P50, P75, P90, P95, P99

**Run it:**
```powershell
python run_advanced_analysis.py
# Select option 1
```

**Output:**
- `results/latency_distribution_TIMESTAMP.png` - 4-panel visualization

---

### 2. Recall@K Accuracy Test

Measures how accurately each database returns the true nearest neighbors:

**What it tests:**
- Computes ground truth using brute force search
- Tests both databases at multiple K values (1, 5, 10, 20, 50, 100)
- Calculates Recall@K: intersection of retrieved vs. true neighbors

**Run it:**
```powershell
python run_advanced_analysis.py
# Select option 2
```

**Configuration:**
- Dataset: 10,000 vectors (smaller for brute force computation)
- Dimension: 128
- Queries: 100

**Output:**
- Console: Recall scores for each K value
- `results/recall_comparison_TIMESTAMP.csv` - Detailed results

**Example Output:**
```
K          Milvus          Weaviate        Winner         
----------------------------------------------------------
1          0.9800          0.9500          Milvus         
5          0.9650          0.9400          Milvus         
10         0.9500          0.9200          Milvus         
```

---

### 3. Scalability Analysis

Tests performance across different dataset sizes to understand scaling behavior:

**Dataset Sizes Tested:**
- 10K vectors
- 50K vectors
- 100K vectors
- 500K vectors

**Metrics Collected:**
- P50 latency
- P95 latency
- QPS (queries per second)
- Memory usage

**Run it:**
```powershell
python run_advanced_analysis.py
# Select option 3
```

**Output:**
- `results/scalability_analysis_TIMESTAMP.png` - Multi-panel scaling charts:
  - Latency vs Dataset Size
  - QPS vs Dataset Size
  - Memory Usage vs Dataset Size
  - Efficiency Score (QPS/Latency) vs Dataset Size

---

### 4. Comprehensive Report

Generates a text-based summary report with:
- Overall winner determination
- Category-by-category comparison
- Final scores
- Recommendations based on use case

**Run it:**
```powershell
python run_advanced_analysis.py
# Select option 4
```

**Output:**
- `results/comprehensive_report_TIMESTAMP.txt`

**Example Report:**
```
==============================================================================
VECTOR DATABASE COMPARISON - SUMMARY REPORT
==============================================================================

OVERALL PERFORMANCE SUMMARY
------------------------------------------------------------------------------
Test: k10_nofilter
  Latency Winner: Milvus (2.50ms vs 3.20ms)
  Throughput Winner: Milvus (400.00 vs 312.50 QPS)

==============================================================================
FINAL SCORE
------------------------------------------------------------------------------
Milvus: 8 wins
Weaviate: 4 wins

Overall Winner: Milvus

==============================================================================
RECOMMENDATIONS
------------------------------------------------------------------------------
Choose Milvus if:
  • You need maximum performance at scale (10M+ vectors)
  • GPU acceleration is important
  • You're building a production ML/AI system
  • You need fine-grained control over indexing

Choose Weaviate if:
  • You need GraphQL API support
  • Hybrid search (vector + keyword) is important
  • You want easier setup and management
  • You need built-in ML model integration
```

---

## ⚡ Concurrent Load Testing

Test database performance under realistic concurrent load:

**Run it:**
```powershell
python run_concurrent_test.py
```

**Configuration:**
- **Dataset**: 100,000 vectors
- **Test Duration**: 30 seconds per concurrency level
- **Concurrency Levels**: 1, 5, 10, 25, 50 clients

**What it measures:**
- QPS (Queries Per Second)
- P50, P95, P99 latencies
- Success rate
- Failed requests

**Output:**
- `results/concurrent_test_TIMESTAMP.csv` - Detailed results
- `results/concurrent_test_TIMESTAMP.png` - 4-panel visualization:
  - Throughput vs Concurrency
  - Median Latency vs Concurrency
  - 95th Percentile Latency vs Concurrency
  - Success Rate vs Concurrency

**Example Output:**
```
==============================================================================
Comparison at 10 clients:
==============================================================================
Metric               Milvus               Weaviate            
--------------------------------------------------------------
QPS                  450.23               380.12              
P50 Latency (ms)     22.15                26.32               
P95 Latency (ms)     45.67                52.18               
P99 Latency (ms)     89.23                105.34              
Failed Requests      0                    2                   
```

---

## 🔧 Using Individual Utility Classes

### Recall Calculator

```python
from utils.recall_calculator import RecallCalculator
import numpy as np

# Initialize
recall_calc = RecallCalculator(metric='l2')  # or 'cosine'

# Compute ground truth
queries = np.random.randn(100, 128)
corpus = np.random.randn(10000, 128)
ground_truth = recall_calc.compute_ground_truth(queries, corpus, k=100)

# Calculate recall
retrieved = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # Retrieved IDs
mean_recall, individual_recalls = recall_calc.calculate_recall(
    retrieved, ground_truth, k=10
)

print(f"Recall@10: {mean_recall:.4f}")
```

### Concurrent Tester

```python
from utils.concurrent_tester import ConcurrentTester
import numpy as np

# Initialize
tester = ConcurrentTester(n_clients=10)

# Define search function
def search_func(query):
    # Your database search logic
    return db.search(query, k=10)

# Run load test
queries = np.random.randn(100, 128)
result = tester.run_load_test(
    search_func, 
    queries, 
    duration_seconds=30
)

print(f"QPS: {result.qps:.2f}")
print(f"P50: {result.p50*1000:.2f}ms")
print(f"P95: {result.p95*1000:.2f}ms")
```

### Performance Analyzer

```python
from analysis.performance_analyzer import PerformanceAnalyzer
import numpy as np

analyzer = PerformanceAnalyzer()

# Analyze latencies
latencies_dict = {
    'Milvus': np.array([2.1, 2.3, 2.5, ...]),
    'Weaviate': np.array([3.1, 3.2, 3.4, ...])
}

analyzer.analyze_latency_distribution(
    latencies_dict, 
    'results/my_latency_analysis.png'
)

# Compare scalability
results = {
    '10K': {
        'dataset_size': 10000,
        'Milvus': {'p50': 2.5, 'qps': 400, 'memory_mb': 50},
        'Weaviate': {'p50': 3.2, 'qps': 312, 'memory_mb': 65}
    },
    # ... more sizes
}

analyzer.compare_scalability(results, 'results/scalability.png')

# Generate summary report
analyzer.generate_summary_report(results, 'results/report.txt')
```

---

## 📋 Complete Workflow

### Recommended Testing Sequence:

1. **Initial Benchmark** (Basic comparison)
   ```powershell
   python main.py
   ```

2. **Recall Testing** (Accuracy verification)
   ```powershell
   python run_advanced_analysis.py
   # Select option 2
   ```

3. **Concurrent Load Test** (Real-world performance)
   ```powershell
   python run_concurrent_test.py
   ```

4. **Scalability Analysis** (Growth patterns)
   ```powershell
   python run_advanced_analysis.py
   # Select option 3
   ```

5. **Final Report** (Comprehensive summary)
   ```powershell
   python run_advanced_analysis.py
   # Select option 4
   ```

---

## 📁 Output Files Reference

| File Pattern | Description |
|-------------|-------------|
| `comparison_*.csv` | Basic benchmark results from main.py |
| `latency_comparison_*.png` | Latency bar charts |
| `qps_comparison_*.png` | QPS bar charts |
| `latency_distribution_*.png` | Detailed latency analysis (4 panels) |
| `recall_comparison_*.csv` | Recall@K accuracy results |
| `concurrent_test_*.csv` | Concurrent load test results |
| `concurrent_test_*.png` | Concurrent load visualizations |
| `scalability_analysis_*.png` | Scaling behavior charts |
| `comprehensive_report_*.txt` | Final text summary report |

---

## 🎯 Key Metrics Explained

### Latency Metrics
- **P50 (Median)**: 50% of queries complete faster than this
- **P95**: 95% of queries complete faster than this
- **P99**: 99% of queries complete faster than this

### Throughput Metrics
- **QPS**: Queries Per Second - higher is better
- **Throughput**: Vectors inserted per second during loading

### Accuracy Metrics
- **Recall@K**: % of true top-K neighbors found
  - 1.0 = perfect (found all true neighbors)
  - 0.8 = found 80% of true neighbors

### Resource Metrics
- **Memory Usage**: RAM used during operations
- **Storage Size**: Disk space used by database

---

## 🔍 Troubleshooting

### Issue: "No result files found"
**Solution**: Run `python main.py` first to generate baseline results

### Issue: Out of memory during recall test
**Solution**: Reduce `DATASET_SIZE` in `run_advanced_analysis.py` (line 122)

### Issue: Concurrent test timeouts
**Solution**: Reduce `TEST_DURATION` or `CONCURRENCY_LEVELS` in `run_concurrent_test.py`

### Issue: Ground truth computation too slow
**Solution**: Use smaller dataset or increase timeout. Brute force is O(n²).

---

## 💡 Tips

1. **Start Small**: Begin with smaller datasets for quick iterations
2. **Check Resources**: Monitor CPU/RAM during tests
3. **Save Results**: All results are timestamped - safe to run multiple times
4. **Compare Trends**: Look for patterns across multiple runs
5. **Document Config**: Note your configurations for reproducibility

---

## 📚 Next Steps

- Modify test configurations in the scripts
- Add custom metrics to the analyzers
- Create your own test scenarios
- Extend the framework for additional databases

---

Need help? Check the individual Python files for detailed docstrings and inline comments!
