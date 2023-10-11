from typing import Optional
import numpy as np

from gradient_boosting_trees.regression.builder import NodeBuilder
from gradient_boosting_trees.trees import Node


class RegressionTree:
    """A regression tree class which can be fitted and predict on continuous points"""

    def __init__(self, node_builder: NodeBuilder):
        self._builder = node_builder
        self._root: Optional[Node] = None

    def fit(self, points: np.ndarray, labels: np.ndarray):
        """"""
        self._root = self._builder.build(points=points, labels=labels)

    def predict(self, points: np.array) -> np.array:
        """"""
        return np.array([self._root.traverse(point=point).value for point in points])
