"""
    Gradient boosting module
"""
import numpy as np
from gradient_boosting_trees.decision_trees.cart import CartRegressionTree

# TODO: Read this https://developers.google.com/machine-learning/decision-forests/gradient-boosting and check
def gradient_boosting(points: np.array, labels: np.array, shrinkage=0.01):
    weak_models = []
    strong_predictions = np.zeros_like(labels)
    
    for i in range(20):
        error = strong_predictions - labels
        tree = CartRegressionTree(max_level=5, min_points=10)
        tree.fit(points=points, labels=error)
        weak_models.append(tree)
        weak_predictions = tree.predict(points=points)
        strong_predictions -= shrinkage * weak_predictions
