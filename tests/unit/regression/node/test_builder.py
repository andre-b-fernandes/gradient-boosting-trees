from typing import Tuple
import pytest
import numpy as np

from gradient_boosting_trees.regression.cart.builder import TreeLevelNodeBuilder
from gradient_boosting_trees.trees import Node


@pytest.mark.parametrize(
    ("min_points, max_level"),
    [
        (5, 10),
        (10, 20),
        (50, 200),
    ],
)
class TestTreeLevelNodeBuilder:
    @staticmethod
    def _build(
        min_points: int, max_level: int, sample: np.ndarray
    ) -> Tuple[Node, TreeLevelNodeBuilder]:
        builder = TreeLevelNodeBuilder(min_moints=min_points, max_level=max_level)
        x, y, labels = sample
        points = np.hstack([x.reshape(len(x), 1), y.reshape(len(y), 1)])
        return builder.build(points=points, labels=labels), builder

    def test_binary_leaf_count(
        self, min_points: int, max_level: int, cosine_sample_2d: np.ndarray
    ):
        node, _ = self._build(
            min_points=min_points, max_level=max_level, sample=cosine_sample_2d
        )
        total_leaves = sum(1 if nd.is_leaf else 0 for nd in node)
        assert total_leaves % 2 == 0

    def test_should_stop(self, min_points: int, max_level: int):
        builder = TreeLevelNodeBuilder(min_moints=min_points, max_level=max_level)
        assert builder.should_stop(points=np.arange(min_points - 1))
        builder._current_level = max_level + 1
        assert builder.should_stop(points=np.arange(min_points))
