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
        if self.should_stop(self, points=points):
            return Node(threshold=labels.mean())

        return self.recursive_call(points=points, labels=labels)

    @abstractmethod
    def recursive_call(self, points: np.array, labels: np.array):
        ...
    
    def should_stop(self, points: np.array) -> bool:
        return len(points) < self._min_points


class TreeLevelNodeBuilder(NodeBuilder):
    def __init__(self, min_moints: int, max_level: int) -> None:
        """
            The constructor.

            Arguments:
                min_points: int The number of minimum points a Node must have to be splitted.
                max_level: int The maximum level the tree can get starting from 0
        """
        super().__init__(min_moints)
        self._max_level = max_level
        self._current_level = 0
    
    def should_stop(self, points: np.array) -> bool:
        return super().should_stop(points) or self._current_level > self._max_level

    def recursive_call(self, points: np.array, labels: np.array):
        feature_idx, threshold_value, lhs, rhs = find_best_split(points=points, labels=labels)
        lhs_points, lhs_labels = lhs
        rhs_points, rhs_labels = rhs

        self._current_level += 1
        left = self.build(points=lhs_points, labels=lhs_labels)
        right = self.build(points=rhs_points, labels=rhs_labels)
        return Node(split=(feature_idx, left, right), threshold=threshold_value)
