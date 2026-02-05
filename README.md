# Milvus vs Weaviate Benchmark

Benchmarking framework comparing Milvus and Weaviate vector databases on real ANN benchmark datasets. Measures data ingestion throughput, query latency (P50/P95/P99), recall accuracy, and resource usage.

## Installation

```bash
git clone https://github.com/minageus/vector-db-comparison.git
cd vector-db-comparison
pip install -r requirements.txt
docker-compose up -d
```

Requires Python 3.8+, Docker Desktop, and 16 GB RAM.

## Usage

```bash
python run_paper_benchmark.py --dataset sift1m
python run_paper_benchmark.py --dataset sift1m --subset 100000
python run_paper_benchmark.py --dataset gist1m --M 16 --ef 200
```

Key options: `--dataset`, `--subset N`, `--runs N`, `--queries N`, `--M N`, `--ef-construction N`, `--sweep-ef`, `--skip-concurrent`.

## Datasets

| Dataset | Vectors | Dimensions | Metric |
|---------|---------|------------|--------|
| sift1m | 1M | 128 | L2 |
| gist1m | 1M | 960 | L2 |
| glove-100 | 1.2M | 100 | Cosine |
| nytimes-256 | 290K | 256 | Cosine |
| fashion-mnist-784 | 60K | 784 | L2 |

Datasets download automatically on first use.

## Project Structure

```
├── run_paper_benchmark.py     # Main benchmark script
├── generate_paper_charts.py   # Chart generation for paper
├── streamlit_app.py           # Interactive results dashboard
├── docker-compose.yml         # Database containers
├── data/                      # Data loaders
├── queries/                   # Query implementations
├── benchmarks/                # Benchmark orchestration
├── utils/                     # Monitoring and analysis
└── results/                   # Generated reports and charts
```

## License

MIT License

## Acknowledgments

Datasets from [ANN-Benchmarks](http://ann-benchmarks.com/).
