from typing import Tuple 
import numpy as np


class Node:
    def __init__(self, threshold: float, split: Tuple[int, "Node", "Node"] = None):
        if split is None:
            split = (None, None, None)

        self._feature_idx, self._left, self._right = split

        has_left = self._left is not None
        has_right = self._right is not None
        if (has_left and not has_right) or (has_right and not has_left):
            raise ValueError("Cannot create node with only one child.")

        self._threshold = threshold
    
    @property
    def value(self) -> float:
        return self._threshold
    
    @property
    def is_leaf(self) -> bool:
        """"""
        return self._left is None and self._right is None
    
    def __iter__(self):
        yield from self._left
        yield from self._right
    
    def traverse(self, point: np.array) -> "Node":
        if self.is_leaf:
            return self

        value = point[self._feature_idx]

        if value < self._threshold:
            return self._left.traverse(point=point)

        return self._right.traverse(point=point)
