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

## Project Structure

```
├── run_paper_benchmark.py     
├── generate_paper_charts.py   
├── streamlit_app.py           
├── docker-compose.yml         
├── data/                      
├── queries/                   
├── benchmarks/                
├── utils/                     
└── results/                   
```

## Datasets

All datasets sourced from [ANN-Benchmarks](http://ann-benchmarks.com/). The following were benchmarked:

| Dataset | Vectors | Dimensions | Metric | Size |
|---------|---------|------------|--------|------|
| SIFT1M | 1,000,000 | 128 | L2 | ~161 MB |
| GIST1M | 1,000,000 | 960 | L2 | ~3.6 GB |
| GloVe-25 | 1,183,514 | 25 | Angular | ~120 MB |
| GloVe-100 | 1,183,514 | 100 | Angular | ~822 MB |
| GloVe-200 | 1,183,514 | 200 | Angular | ~950 MB |
| GloVe-300 | 1,183,514 | 300 | Angular | — |
| Fashion-MNIST-784 | 60,000 | 784 | L2 | ~217 MB |
| NYTimes-256 | 290,000 | 256 | Angular | ~301 MB |
| Deep-Image-96 | 10,000,000 | 96 | Angular | ~3.8 GB |

Deep-Image-96 was run on 2M and 5M vector subsets. All other datasets were run at full size.

## License

MIT License

## Acknowledgments

Datasets from [ANN-Benchmarks](http://ann-benchmarks.com/).
