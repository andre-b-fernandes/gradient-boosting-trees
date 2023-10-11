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

    def build(self, points: np.array, labels: np.array) -> Node:
        if self.should_stop(points=points):
            return Node(threshold=labels.mean())

        return self.recursive_call(points=points, labels=labels)

    @abstractmethod
    def recursive_call(self, points: np.array, labels: np.array):
        ...

    def should_stop(self, points: np.array) -> bool:
        return len(points) < self._min_points

    @abstractmethod
    def reset(self):
        pass