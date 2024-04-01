import numpy as np

from gradient_boosting_trees.trees import Node
from gradient_boosting_trees.regression.builder import NodeBuilder
from gradient_boosting_trees.regression.cart.split import find_best_split


class TreeLevelNodeBuilder(NodeBuilder):
    """
    Tree level node builder. It will keep exploring both sides of the tree regardless
    of what they are until a maximum level depth has been reached.
    It maintains a counter of the current level and the number of nodes created.
    """

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

    def recursive_call(self, points: np.ndarray, labels: np.ndarray) -> Node:
        """
        The function recursive call on building the tree level-wise. Overriding the
        parent class and finding the best feature split greadily.

        Arguments:
            points: numpy array The current level data points across all features.
            labels: numpy array The labels for the respective points.
        """
        feature_split, lhs, rhs = find_best_split(points=points, labels=labels)
        feature_idx, threshold_value, _ = feature_split
        lhs_points, lhs_labels = lhs
        rhs_points, rhs_labels = rhs

        self._current_level += 1
        left = self.build(points=lhs_points, labels=lhs_labels)
        right = self.build(points=rhs_points, labels=rhs_labels)

        return Node(node_id=self._node_count, split=(feature_idx, left, right), threshold=threshold_value)

    def reset(self):
        super().reset()
        self._current_level = 0


# # FIXME: TODO
# class LeafwiseNodeBuilder(NodeBuilder):
#     def __init__(self, min_points: int, max_leaves: int) -> None:
#         """"""
#         super().__init__(min_points)
#         self._max_leaves = max_leaves
#         self._leaf_count = 0
