import os
import yaml
from data.generators.vector_generator import VectorDataGenerator
from data.loaders.milvus_loader import MilvusLoader
from data.loaders.weaviate_loader import WeaviateLoader
from queries.query_generator import QueryGenerator
from queries.milvus_queries import MilvusQueryExecutor
from queries.weaviate_queries import WeaviateQueryExecutor
from benchmarks.benchmark_runner import BenchmarkRunner

def main():
    print("="*60)
    print("MILVUS vs WEAVIATE BENCHMARK")
    print("="*60)
    
    # Configuration
    DATASET_SIZE = 100000  # Start with small dataset
    DIMENSION = 128
    N_QUERIES = 100
    
    # Step 1: Generate data
    print("\n[1/6] Generating data...")
    generator = VectorDataGenerator()
    ids, vectors, metadata = generator.generate_with_metadata(DATASET_SIZE, DIMENSION)
    
    # Step 2: Load into Milvus
    print("\n[2/6] Loading data into Milvus...")
    milvus_loader = MilvusLoader()
    milvus_loader.connect()
    milvus_loader.create_collection("benchmark_collection", DIMENSION)
    milvus_loader.load_data(ids, vectors, metadata, batch_size=10000)
    milvus_loader.create_index(index_type='HNSW', metric_type='L2')
    milvus_loader.load_collection()
    
    # Step 3: Load into Weaviate
    print("\n[3/6] Loading data into Weaviate...")
    weaviate_loader = WeaviateLoader()
    weaviate_loader.connect()
    weaviate_loader.create_schema(DIMENSION)
    weaviate_loader.load_data(ids, vectors, metadata, batch_size=100)
    
    # Step 4: Generate queries
    print("\n[4/6] Generating queries...")
    query_gen = QueryGenerator()
    query_vectors = query_gen.generate_query_vectors(N_QUERIES, DIMENSION)
    filters = query_gen.generate_filter_conditions(N_QUERIES, selectivity=0.1)
    
    # Step 5: Run benchmarks
    print("\n[5/6] Running benchmarks...")
    
    test_configs = [
        {'name': 'k10_nofilter', 'k': 10, 'use_filters': False},
        {'name': 'k10_filter', 'k': 10, 'use_filters': True, 'filters': filters},
        {'name': 'k100_nofilter', 'k': 100, 'use_filters': False},
        {'name': 'k1000_nofilter', 'k': 1000, 'use_filters': False},
    ]
    
    runner = BenchmarkRunner(output_dir='results')
    
    # Benchmark Milvus
    milvus_executor = MilvusQueryExecutor(milvus_loader.collection)
    runner.run_benchmark('Milvus', milvus_executor, query_vectors, test_configs)
    
    # Benchmark Weaviate
    weaviate_executor = WeaviateQueryExecutor(weaviate_loader.client, 'BenchmarkVector')
    runner.run_benchmark('Weaviate', weaviate_executor, query_vectors, test_configs)
    
    # Step 6: Generate report
    print("\n[6/6] Generating comparison report...")
    comparison_df = runner.generate_comparison_report()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print("\nComparison Summary:")
    print(comparison_df.to_string())

if __name__ == "__main__":
    main()