from typing import Tuple
import numpy as np


class Node:
    """
    A class which represents a Node belonging to a binary tree.
    Can be a leaf if it has no children, but if it has children it necessarily
    has 2 children, left and right.
    Also contains a threshold, and a feature_idx provided in the split parameter.
    The threshold is the provided threshold meant for split decisions, and the feature_idx
    is the identifier of the feature this node belongs to.
    The threshold can either act as a standard threshold or as a node value when it is a leaf.
    A node also has a node_id which is a unique identifier for the node in the tree.
    """

    def __init__(self, node_id: int, threshold: float, split: Tuple[int, "Node", "Node"] = None):
        """ """
        if split is None:
            split = (None, None, None)

        self._feature_idx, self._left, self._right = split

        has_left = self._left is not None
        has_right = self._right is not None
        if (has_left and not has_right) or (has_right and not has_left):
            raise ValueError("Cannot create node with only one child.")

        self._threshold = threshold
        self.node_id = node_id

    @property
    def value(self) -> float:
        return self._threshold

    @property
    def is_leaf(self) -> bool:
        """
        Wether this node is a tree leaf or not.

        Returns:
            bool: True if this node is a leaf, False otherwise.
        """
        return self._left is None and self._right is None

    def __iter__(self):
        if self.is_leaf:
            yield self
        else:
            yield from self._left
            yield from self._right

    def traverse(self, point: np.array) -> "Node":
        """
        Traverses the tree until it reaches a leaf node.
        Will yield left or right nodes depending on the value of the point.
        If less than the threshold, will yield left node, otherwise right node.

        Args:
            point: A numpy array representing a point in the feature space.

        Returns:
            Node: The next node reached by traversing the tree.
        """
        if self.is_leaf:
            return self

        value = point[self._feature_idx]

        if value < self._threshold:
            return self._left.traverse(point=point)

        return self._right.traverse(point=point)
