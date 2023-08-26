"""
    CART REGRESSION TREE MODULE
"""
from typing import Tuple
from gradient_boosting_trees.trees import Node
from gradient_boosting_trees.decision_trees.node.split import find_best_split
import numpy as np


# TODO: Experiment with leaf-wise growth instead of level-wise growth
class CartRegressionTree:
    """"""
    def __init__(self, max_level: int, min_points: int):
        self._max_level = max_level
        self._min_points = min_points
        self._root: Node = None

    def fit(self, points: np.array, labels: np.array) -> None:
        def recursive_cart_tree(points: np.array, labels: np.array, level: int = 0) -> Node:
            if level >= self._max_level or len(points) < self._min_points:
                return Node(threshold=labels.mean())

            feature_idx, threshold_value, lhs, rhs = find_best_split(points=points, labels=labels)
            lhs_points, lhs_labels = lhs
            rhs_points, rhs_labels = rhs

            left = recursive_cart_tree(points=lhs_points, labels=lhs_labels, level=level+1)
            right = recursive_cart_tree(points=rhs_points, labels=rhs_labels, level=level+1)
            return Node(split=(feature_idx, left, right), threshold=threshold_value)

        self._root = recursive_cart_tree(points=points, labels=labels)
    
    def predict(self, points: np.array) -> np.array:
        return np.array([self._root.traverse(point=point).value for point in points])
