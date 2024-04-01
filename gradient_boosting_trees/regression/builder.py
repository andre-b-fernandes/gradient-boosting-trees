from abc import ABC, abstractmethod
import numpy as np

from gradient_boosting_trees.trees import Node


class NodeBuilder(ABC):
    """
    Abstract NodeBuilder class
    """

    def __init__(self, min_points: int) -> None:
        """
        The constructor.

        Arguments:
            min_points: int The number of minimum points a Node must have to be splitted.
        """
        self._min_points = min_points
        self._node_count = 0

    def build(self, points: np.array, labels: np.array) -> Node:
        if self.should_stop(points=points):
            node_id = self._node_count
            self._node_count += 1
            return Node(node_id=node_id, threshold=labels.mean())

        return self.recursive_call(points=points, labels=labels)

    @abstractmethod
    def recursive_call(self, points: np.array, labels: np.array):
        ...

    def should_stop(self, points: np.array) -> bool:
        return len(points) < self._min_points

    def reset(self):
        self._node_count = 0
