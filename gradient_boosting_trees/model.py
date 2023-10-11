"""
    Gradient boosting module
"""
from typing import List
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from gradient_boosting_trees.objectives.log_loss import log_loss_derivative
from gradient_boosting_trees.objectives.squared_error import squared_error_objective
from gradient_boosting_trees.regression.builder import NodeBuilder
from gradient_boosting_trees.regression.tree import RegressionTree

AVAILABLE_OPTOBJECTIVES = {
    "squared_error": squared_error_objective,
    "log_loss": log_loss_derivative
}

@dataclass
class GBParams:
    shrinkage: float = 0.01 # learning rate


class GBRegressionTrees:
    def __init__(self, params: GBParams, node_builder: NodeBuilder, objective: str = "squared_error"):
        self._params = params
        self._builder = node_builder
        self._weak_models: List[RegressionTree] = []
        try:
            self._objective = AVAILABLE_OPTOBJECTIVES[objective]
        except KeyError:
            raise ValueError(f"Objective {objective} not available. Choose one of {AVAILABLE_OPTOBJECTIVES.keys()}")

    def fit(self, points: np.ndarray, labels: np.ndarray, n_iterations: int):
        """n_iters equals n_trees"""
        strong_predictions = np.zeros_like(labels)

        for _ in tqdm(range(n_iterations)):
            error = self._objective(raw_predictions=strong_predictions, labels=labels)
            self._builder.reset()
            tree = RegressionTree(node_builder=self._builder)
            tree.fit(points=points, labels=error)
            self._weak_models.append(tree)
            weak_predictions = tree.predict(points=points)
            strong_predictions -= self._params.shrinkage * weak_predictions

    def predict(self, points: np.array) -> np.array:
        generator = (wk_m.predict(points=points) * (self._params.shrinkage) for wk_m in self._weak_models)
        return np.sum(generator)
