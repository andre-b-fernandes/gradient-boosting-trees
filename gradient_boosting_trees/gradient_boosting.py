"""
    Gradient boosting module
"""
from typing import List
import numpy as np
from tqdm import tqdm
from gradient_boosting_trees.decision_trees.cart import CartRegressionTree


class GradientBoostingRegression:
    def __init__(self, shrinkage: float):
        self._shrinkage = shrinkage
        self._weak_models: List[CartRegressionTree] = []
    
    def fit(self, points: np.ndarray, labels: np.ndarray, n_iterations: int):
        """n_iters equals n_trees"""
        strong_predictions = np.zeros_like(labels)
        
        for _ in tqdm(range(n_iterations)):
            error = strong_predictions - labels
            tree = CartRegressionTree(max_level=3, min_points=10)
            tree.fit(points=points, labels=error)
            self._weak_models.append(tree)
            weak_predictions = tree.predict(points=points)
            strong_predictions += self._shrinkage * weak_predictions
    
    # def predict(self, points: np.array) -> np.array:
    #     return np.array([weak_model.predict(points) for weak_model in self._weak_models]).sum()
