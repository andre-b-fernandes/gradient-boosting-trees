from typing import Tuple
import pytest
import numpy as np

from gradient_boosting_trees.regression.node.builder import TreeLevelNodeBuilder
from gradient_boosting_trees.trees import Node

@pytest.mark.parametrize(
    ("min_points, max_level"),
    [
        (5, 10),
        (10, 20),
        (50, 200),
    ]
)
class TestTreeLevelNodeBuilder:
    @staticmethod
    def _build(min_points: int, max_level: int, sample: np.ndarray) -> Tuple[Node, TreeLevelNodeBuilder]:
        builder = TreeLevelNodeBuilder(min_moints=min_points, max_level=max_level)
        x, y, labels = sample
        points = np.hstack([x.reshape(len(x), 1), y.reshape(len(y), 1)])
        return builder.build(points=points, labels=labels), builder

    def test_binary_leaf_count(self, min_points: int, max_level: int, cosine_sample_2d: np.ndarray):
        node, _ = self._build(min_points=min_points, max_level=max_level, sample=cosine_sample_2d)
        total_leaves = sum(1 if nd.is_leaf else 0 for nd in node)
        assert total_leaves % 2 == 0
    
    # TODO: Fix this test
    def test_leaf_length(self, min_points: int, max_level: int, cosine_sample_2d: np.ndarray):
        node, builder = self._build(min_points=min_points, max_level=max_level, sample=cosine_sample_2d)
        n_nodes = sum(1 for _ in node)
        import pdb; pdb.set_trace()
        print(n_nodes)
