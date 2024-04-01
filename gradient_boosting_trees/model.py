"""
    Gradient boosting module
"""
from typing import List
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from gradient_boosting_trees.objectives.squared_error import (
    squared_error,
    squared_error_gradient_hessian,
)
from gradient_boosting_trees.regression.builder import NodeBuilder
from gradient_boosting_trees.regression.tree import RegressionTree


@dataclass
class GBParams:
    shrinkage: float = 0.01  # learning rate


class GBRegressionTrees:
    def __init__(
        self,
        params: GBParams,
        node_builder: NodeBuilder
    ):
        self._params = params
        self._builder = node_builder
        self._weak_models: List[RegressionTree] = []
        self.learning_error = []

    def fit(self, points: np.ndarray, labels: np.ndarray, n_iterations: int):
        self.learning_error = []
        strong_predictions = np.zeros_like(labels)
        self._weak_models = []

        for _ in tqdm(range(n_iterations)):
            error = squared_error(raw_predictions=self.predict(points=points), labels=labels)
            self.learning_error.append(error)

            gradient, hessian = squared_error_gradient_hessian(raw_predictions=strong_predictions, labels=labels)

            self._builder.reset()
            tree = RegressionTree(node_builder=self._builder)
            tree.fit(points=points, labels=gradient)
            self._weak_models.append(tree)
    
            weak_predictions = tree.predict(points=points)
            strong_predictions -= self._params.shrinkage * weak_predictions / hessian

    def predict(self, points: np.array) -> np.array:
        generator = (
            wk_m.predict(points=points) * (self._params.shrinkage)
            for wk_m in self._weak_models
        )
        return np.sum(generator)
