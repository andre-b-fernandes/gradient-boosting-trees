from gradient_boosting_trees.trees import Node
import numpy as np

import pytest


def test_should_create_node_with_threshold():
    threshold = 0.5
    node = Node(threshold=threshold)

    assert node.value == threshold


@pytest.mark.parametrize(
    "node, is_leaf",
    [
        (
           Node(threshold=0.5, split=(0, Node(threshold=0.5), Node(threshold=0.5))),
           False 
        ),
        (
           Node(threshold=0.5),
           True 
        )
    ]
)
def test_should_check_if_node_is_leaf(node: Node, is_leaf: bool):
    assert node.is_leaf == is_leaf


def test_should_iterate_over_nodes():
    node = Node(threshold=0.5, split=(0, Node(threshold=0.5), Node(threshold=0.5)))
    assert list(node) == [node._left, node._right]


def should_traverse_tree():
    node = Node(threshold=0.5, split=(0, Node(threshold=0.5), Node(threshold=0.5)))
    assert node.traverse(np.array([0.4])) == node._left
    assert node.traverse(np.array([0.6])) == node._right

    node = Node(threshold=0.5, split=(1, Node(threshold=0.5), Node(threshold=0.5)))
    assert node.traverse(np.array([0.4, 0.6])) == node._right
    assert node.traverse(np.array([0.6, 0.4])) == node._left