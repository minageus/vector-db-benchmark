import numpy as np
from typing import List, Tuple, Dict

class QueryGenerator:
    """Generate queries for benchmarking"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_query_vectors(
        self, 
        n_queries: int, 
        dimension: int,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate random query vectors"""
        queries = np.random.randn(n_queries, dimension).astype(np.float32)
        
        if normalize:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / (norms + 1e-8)
        
        return queries
    
    def generate_filter_conditions(
        self, 
        n_queries: int,
        selectivity: float = 0.1
    ) -> List[Dict]:
        """Generate filter conditions for filtered search"""
        filters = []
        
        for _ in range(n_queries):
            filter_type = np.random.choice(['category', 'price', 'rating'])
            
            if filter_type == 'category':
                # Category filter - select specific categories
                n_categories = max(1, int(10 * selectivity))
                categories = np.random.choice(10, n_categories, replace=False).tolist()
                filters.append({'category': {'$in': categories}})
            
            elif filter_type == 'price':
                # Price range filter
                min_price = np.random.uniform(10, 500)
                max_price = min_price + np.random.uniform(100, 500)
                filters.append({'price': {'$gte': min_price, '$lte': max_price}})
            
            else:
                # Rating filter
                min_rating = np.random.uniform(1, 4)
                filters.append({'rating': {'$gte': min_rating}})
        
        return filters