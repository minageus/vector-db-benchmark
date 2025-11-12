import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class RecallCalculator:
    """Calculate recall@k for search accuracy"""
    
    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
    
    def compute_ground_truth(
        self,
        queries: np.ndarray,
        corpus: np.ndarray,
        k: int = 100
    ) -> np.ndarray:
        """Compute ground truth nearest neighbors using brute force"""
        
        print(f"Computing ground truth with {self.metric} metric...")
        
        if self.metric == 'cosine':
            similarities = cosine_similarity(queries, corpus)
            # Higher is better for similarity
            ground_truth = np.argsort(-similarities, axis=1)[:, :k]
        elif self.metric == 'l2':
            distances = euclidean_distances(queries, corpus)
            # Lower is better for distance
            ground_truth = np.argsort(distances, axis=1)[:, :k]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return ground_truth
    
    def calculate_recall(
        self,
        retrieved: List[List[int]],
        ground_truth: np.ndarray,
        k: int = 10
    ) -> Tuple[float, List[float]]:
        """Calculate recall@k"""
        
        recalls = []
        
        for i, retrieved_ids in enumerate(retrieved):
            gt_ids = set(ground_truth[i][:k])
            ret_ids = set(retrieved_ids[:k])
            
            intersection = len(gt_ids.intersection(ret_ids))
            recall = intersection / k if k > 0 else 0
            recalls.append(recall)
        
        mean_recall = np.mean(recalls)
        return mean_recall, recalls