from typing import Optional
import numpy as np

from gradient_boosting_trees.regression.builder import NodeBuilder
from gradient_boosting_trees.trees import Node


class RegressionTree:
    """A regression tree class which can be fitted and predict on continuous points"""

    def __init__(self, node_builder: NodeBuilder):
        self._builder = node_builder
        self._root: Optional[Node] = None

    def fit(self, points: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the regression tree on the given points and labels.
        """
        self._root = self._builder.build(points=points, labels=labels)

    def predict(self, points: np.array) -> np.array:
        """
        Predict on the given points.

        Arguments:
            points: np.array The points to predict on.
        
        Returns:
            np.array The predicted values.
        """
        return np.array([self._root.traverse(point=point).value for point in points])
