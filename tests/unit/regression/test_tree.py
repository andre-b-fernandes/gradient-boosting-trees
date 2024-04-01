from typing import Tuple
from gradient_boosting_trees.regression.cart.builder import TreeLevelNodeBuilder
from gradient_boosting_trees.regression.tree import RegressionTree

import numpy as np


def test_regression_tree_fit_should_set_root(cosine_sample: Tuple[np.ndarray, np.array]):
    x, y = cosine_sample
    builder = TreeLevelNodeBuilder(min_moints=5, max_level=10)
    regression_tree = RegressionTree(node_builder=builder)
    regression_tree.fit(points=x.reshape(len(x), 1), labels=y.reshape(len(y), 1))
    
    assert regression_tree._root is not None


def test_regression_tree_predictions_iterate_over_points(cosine_sample: Tuple[np.ndarray, np.array]):
    x, y = cosine_sample
    builder = TreeLevelNodeBuilder(min_moints=5, max_level=10)
    regression_tree = RegressionTree(node_builder=builder)
    regression_tree.fit(points=x.reshape(len(x), 1), labels=y.reshape(len(y), 1))
    
    predictions = regression_tree.predict(points=x.reshape(len(x), 1))
    assert len(predictions) == len(x)