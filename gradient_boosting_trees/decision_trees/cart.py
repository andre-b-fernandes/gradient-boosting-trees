"""
    CART REGRESSION TREE MODULE
"""
from typing import Tuple
from gradient_boosting_trees.trees import Node
import numpy as np


def find_best_split_feature(feature: np.array, labels: np.array) -> Tuple[int, float, float]:
    """"""
    min_loss = None
    treshold_idx = None
    treshold_value = None

    # assume ordered
    for first_idx, (first, second) in enumerate(zip(feature[:-1], feature[1:])):
        candidate_treshold_value = (first + second) / 2.0 # average between points
        candidate_treshold_idx = first_idx + 1 # or second_idx

        # split area
        lhs = labels[:candidate_treshold_idx]
        rhs = labels[candidate_treshold_idx:]
        
        # split predictions
        pred_lhs = lhs.mean()
        pred_rhs = rhs.mean()
        
        # mse split loss
        lhs_loss = (lhs - pred_lhs)
        rhs_loss = (rhs - pred_rhs)
        total_candidate_loss = np.sqrt(np.sum(lhs_loss) + np.sum(rhs_loss))
        
        # update_min_loss
        if min_loss is None or total_candidate_loss < min_loss:
            min_loss = total_candidate_loss
            treshold_idx = candidate_treshold_idx
            treshold_value = candidate_treshold_value

    return treshold_idx, treshold_value, min_loss

def find_best_split(points: np.array, labels: np.array) -> Tuple[int, float, Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    sorted_idx = points.argsort(axis=0)
    _, n_features = points.shape

    min_feature_loss = None
    best_loss_feature = None
    threshold_idx = None
    threshold_value = None
    
    for feature_idx in range(n_features):
        feature = points[:,feature_idx]
        feature_sorted_idx = sorted_idx[:, feature_idx]
        candidate_idx, candidate_value, candidate_ft_loss = find_best_split_feature(feature=feature[feature_sorted_idx], labels=labels[feature_sorted_idx])
        
        if min_feature_loss is None or candidate_ft_loss <= min_feature_loss:
            best_loss_feature = feature_idx
            threshold_idx = candidate_idx
            threshold_value = candidate_value


    bst_lss_ft_srt_idx = sorted_idx[:, best_loss_feature]
    sorted_points = points[bst_lss_ft_srt_idx]
    sorted_labels = labels[bst_lss_ft_srt_idx]


    lhs_points, lhs_labels = sorted_points[:threshold_idx], sorted_labels[:threshold_idx]
    rhs_points, rhs_labels = sorted_points[threshold_idx:], sorted_labels[threshold_idx:]

    return best_loss_feature, threshold_value, (lhs_points, lhs_labels), (rhs_points, rhs_labels)
    

# TODO: Experiment with leaf-wise growth instead of level-wise growth
class CartRegressionTree:
    """"""
    def __init__(self, max_level: int, min_points: int):
        self._max_level = max_level
        self._min_points = min_points
        self._root: Node = None

    def fit(self, points: np.array, labels: np.array) -> None:
        def recursive_cart_tree(points: np.array, labels: np.array, level: int = 0) -> Node:
            if level >= self._max_level or len(points) < self._min_points:
                return Node(threshold=labels.mean())

            feature_idx, threshold_value, lhs, rhs = find_best_split(points=points, labels=labels)
            lhs_points, lhs_labels = lhs
            rhs_points, rhs_labels = rhs

            left = recursive_cart_tree(points=lhs_points, labels=lhs_labels, level=level+1)
            right = recursive_cart_tree(points=rhs_points, labels=rhs_labels, level=level+1)
            return Node(split=(feature_idx, left, right), threshold=threshold_value)

        self._root = recursive_cart_tree(points=points, labels=labels)
    
    def predict(self, points: np.array) -> np.array:
        return [self._root.traverse(point=point).value for point in points]
