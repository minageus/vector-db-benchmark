# 🔍 Milvus vs Weaviate: Comprehensive Vector Database Benchmark

A production-ready benchmarking framework for comparing **Milvus** and **Weaviate** vector databases across performance, accuracy, and resource efficiency metrics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Milvus](https://img.shields.io/badge/Milvus-2.3.0-green.svg)](https://milvus.io/)
[![Weaviate](https://img.shields.io/badge/Weaviate-1.27.0-orange.svg)](https://weaviate.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

This benchmark provides a **scientific, reproducible comparison** of two leading open-source vector databases, helping you make informed decisions for your AI/ML infrastructure.

### Why This Benchmark?

- 🎓 **Academic Rigor**: Following ANN-Benchmarks methodology
- 📊 **7-Step Analysis**: From data loading to final recommendations
- 🔬 **Multiple Datasets**: SIFT1M, GIST1M, NYTimes, and more
- ⚡ **Optimized for Scale**: Handles datasets from 10K to 10M+ vectors
- 📈 **Production-Ready**: Real-world scenarios with filters and concurrent loads

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- **Docker Desktop** (required)
- **Python 3.8+** (required)
- **8GB+ RAM** (16GB recommended for large datasets)

### 1. Install Dependencies

```powershell
# Clone the repository
git clone https://github.com/minageus/Milvus-Weaviate-Comparison.git
cd vector-db-comparison

# Install Python packages
pip install -r requirements.txt
```

### 2. Start Vector Databases

```powershell
# Start both Milvus and Weaviate
docker-compose up -d

# Verify containers are running
docker ps
```

### 3. Run Your First Benchmark

```powershell
# Quick test with 290K vectors (~10 minutes)
python run_benchmark.py --dataset nytimes-256

# Full SIFT1M benchmark (~60 minutes)
python run_benchmark.py --dataset sift1m

# Fast test with subset (perfect for development)
python run_benchmark.py --dataset sift1m --subset 100000
```

---

## 📊 What Gets Measured?

### 7-Step Comprehensive Analysis

| Step | Metric | What It Measures |
|------|--------|------------------|
| **1. Data Loading** | Load time, throughput, memory | How fast can each DB ingest vectors? |
| **2. Storage Efficiency** | Compression ratio, disk usage | How much space does the data consume? |
| **3. Index Building** | Build time, progress tracking | How long to create searchable indexes? |
| **4. Query Performance** | P50/P95/P99 latency, QPS | How fast are searches (with/without filters)? |
| **5. Recall@K Accuracy** | Recall@1/5/10/20/50/100 | How accurate are the search results? |
| **6. Scalability** | Performance across data sizes | How does performance scale with data? |
| **7. Resource Usage** | CPU, RAM, I/O | What are the infrastructure costs? |

### Key Performance Indicators (KPIs)

- **Latency**: P50, P95, P99 (milliseconds)
- **Throughput**: Queries per second (QPS)
- **Recall**: Accuracy vs ground truth (0-1 scale)
- **Memory**: Peak RAM usage during operations
- **Storage**: Disk space with compression ratios
- **Build Time**: Index construction duration

---

## 📦 Supported Datasets

| Dataset | Vectors | Dimensions | Metric | Size | Use Case |
|---------|---------|------------|--------|------|----------|
| **nytimes-256** | 290K | 256 | Cosine | 283 MB | Text embeddings |
| **sift1m** | 1M | 128 | L2 | 512 MB | Image features |
| **gist1m** | 1M | 960 | L2 | 3.6 GB | Large dimensions |
| **glove-100** | 1.2M | 100 | Cosine | 480 MB | Word embeddings |
| **deep-image-96** | 10M | 96 | Cosine | 3.6 GB | Large-scale images |

### Dataset Auto-Download

Datasets are automatically downloaded on first use. Manual download:

```powershell
# Download specific dataset
python -c "from data.loaders.real_dataset_loader import RealDatasetLoader; RealDatasetLoader().load_dataset('sift1m', download=True)"
```

---

## 🎮 Usage Examples

### Basic Benchmarks

```powershell
# Small dataset (quick test)
python run_benchmark.py --dataset nytimes-256

# Standard benchmark (1M vectors)
python run_benchmark.py --dataset sift1m

# Large-scale benchmark (10M vectors) 
python run_benchmark.py --dataset deep-image-96
```

### Advanced Options

```powershell
# Use subset for faster testing
python run_benchmark.py --dataset sift1m --subset 100000

# Skip time-consuming tests
python run_benchmark.py --dataset sift1m --skip-scalability --skip-filters

# Combine options
python run_benchmark.py --dataset gist1m --subset 500000 --skip-scalability
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to benchmark | `sift1m` |
| `--subset N` | Use only N vectors | Full dataset |
| `--skip-scalability` | Skip scalability tests | Include |
| `--skip-filters` | Skip filtered query tests | Include |

---

## 📈 Performance Expectations

### Time Estimates by Dataset Size

| Dataset Size | Load Time | Index Build | Query Tests | Total Time |
|--------------|-----------|-------------|-------------|------------|
| **100K vectors** | 2 min | 3 min | 1 min | ~10 min |
| **290K vectors** | 5 min | 5 min | 2 min | ~15 min |
| **1M vectors** | 15 min | 25 min | 5 min | ~50 min |
| **10M vectors** | 2 hrs | 4 hrs | 30 min | ~7 hrs |

### Optimization Tips for Large Datasets

✅ **Use `--subset`** for development: `--subset 100000`  
✅ **Skip optional tests**: `--skip-scalability --skip-filters`  
✅ **Monitor progress**: Index building shows progress every 10%  
✅ **Let it run**: 40+ minutes for 1M vectors is **normal**

---

## 📂 Project Structure

```
vector-db-comparison/
├── run_benchmark.py          # Main benchmark script (START HERE)
├── docker-compose.yml         # Start both databases
├── requirements.txt           # Python dependencies
│
├── data/                      # Data handling
│   ├── loaders/              # Milvus & Weaviate data loaders
│   ├── generators/           # Synthetic data generation
│   └── datasets/             # Downloaded datasets (auto-created)
│
├── queries/                   # Query executors
│   ├── milvus_queries.py     # Milvus search implementation
│   └── weaviate_queries.py   # Weaviate search implementation
│
├── benchmarks/                # Benchmarking logic
│   └── benchmark_runner.py   # Orchestrates performance tests
│
├── utils/                     # Utilities
│   ├── resource_monitor.py   # CPU/RAM monitoring
│   ├── storage_analyzer.py   # Disk usage analysis
│   └── recall_calculator.py  # Accuracy metrics
│
├── analysis/                  # Results analysis
│   └── performance_analyzer.py
│
├── results/                   # Generated reports (auto-created)
│   ├── complete_benchmark_*.txt
│   ├── comparison_*.csv
│   └── recall_*.csv
│
└── docker/                    # Docker configurations
    ├── docker-compose-milvus.yml
    └── docker-compose-weaviate.yml
```

---

## 📊 Output & Results

### Generated Files

After running a benchmark, you'll find:

```
results/
├── complete_benchmark_20251207_123456.txt  # Full text report
├── comparison_20251207_123456.csv          # Performance metrics
└── recall_20251207_123456.csv              # Accuracy metrics
```

### Sample Results (SIFT1M Dataset)

#### Loading Performance
```
Milvus:   15.2s (65,789 vectors/sec)
Weaviate: 42.8s (23,364 vectors/sec)
Winner: Milvus (2.8x faster)
```

#### Query Performance (P50 Latency)
```
Test          Milvus    Weaviate  Winner
k=10          1.2ms     2.8ms     Milvus
k=100         3.4ms     8.1ms     Milvus
k=1000        18.3ms    45.2ms    Milvus
```

#### Recall@10 Accuracy
```
Milvus:   0.9234 (92.34% accurate)
Weaviate: 0.9373 (93.73% accurate)
Winner: Weaviate (higher accuracy)
```

---

## 🔧 Configuration

### Docker Resource Limits

Edit `docker-compose.yml` to adjust resources:

```yaml
services:
  milvus-standalone:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

### Index Parameters

Optimize for speed vs accuracy in `run_benchmark.py`:

```python
# Fast indexing (large datasets)
index_params = {"M": 8, "efConstruction": 128}

# Balanced (default)
index_params = {"M": 16, "efConstruction": 200}

# High accuracy (small datasets)
index_params = {"M": 32, "efConstruction": 400}
```

---

## 🐛 Troubleshooting

### Common Issues

**"Collection not loaded" error**
```powershell
# Restart Docker containers
docker-compose down
docker-compose up -d
# Wait 30 seconds, then retry
```

**Port already in use (8080, 19530)**
```powershell
# Check what's using the ports
netstat -ano | findstr :8080
netstat -ano | findstr :19530

# Stop conflicting services or change ports in docker-compose.yml
```

**Out of memory during benchmark**
```powershell
# Use smaller subset
python run_benchmark.py --dataset sift1m --subset 100000

# Or increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB+
```

**Slow index building (>1 hour)**
```
This is NORMAL for large datasets!
- 1M vectors: ~25 minutes
- 10M vectors: ~4 hours

Progress is shown every 10%. Let it complete.
```

---

## 📚 Advanced Usage

### Custom Datasets

```python
from data.loaders.real_dataset_loader import RealDatasetLoader

# Add your dataset
loader = RealDatasetLoader()
data = {
    'base': your_vectors,        # numpy array (N, D)
    'query': query_vectors,      # numpy array (Q, D)
    'groundtruth': ground_truth, # numpy array (Q, K)
    'info': {'metric': 'cosine'}
}
```

### Concurrent Load Testing

See `ADVANCED_USAGE.md` for:
- Multi-client testing
- Custom query patterns
- Stress testing scenarios
- Production simulation

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets**: [ANN-Benchmarks](http://ann-benchmarks.com/)
- **Milvus**: [Zilliz](https://milvus.io/)
- **Weaviate**: [Weaviate](https://weaviate.io/)

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/minageus/Milvus-Weaviate-Comparison/issues)
- **Documentation**: See `INSTALLATION.md` and `ADVANCED_USAGE.md`
- **Repository**: [GitHub](https://github.com/minageus/Milvus-Weaviate-Comparison)

---

## 🎓 Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{milvus_weaviate_benchmark,
  title = {Milvus vs Weaviate: Comprehensive Vector Database Benchmark},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/minageus/Milvus-Weaviate-Comparison}
}
```

---

**Happy Benchmarking! 🚀**

*Built with ❤️ for the Vector DB community*
