from typing import Tuple
from collections import namedtuple
import numpy as np

FeatureSplit = namedtuple("FeatureSplit", "idx threshold_value min_loss")
HandSide = namedtuple("HandSide", "points labels")


def find_best_split_feature(feature: np.array, labels: np.array) -> FeatureSplit:
    """
    A function which finds the best split point for a feature.
    Performs a gready search for the lowest loss point on the lhs and rhs
    residuals.

    Arguments:
        feature: np.array The feature points (1-D numpy array)
        labels: np.array The label values (1-D numpy array)
    Returns:
        A FeatureSplit tuple (idx, threshold_value, min_loss)
    """
    min_loss = None
    treshold_idx = None
    treshold_value = None

    # assume ordered
    for first_idx, (first, second) in enumerate(zip(feature[:-1], feature[1:])):
        candidate_treshold_value = (first + second) / 2.0  # average between points
        candidate_treshold_idx = first_idx + 1  # or second_idx

        # split area
        lhs = labels[:candidate_treshold_idx]
        rhs = labels[candidate_treshold_idx:]

        # split predictions
        pred_lhs = lhs.mean()
        pred_rhs = rhs.mean()

        # mse split loss
        lhs_loss = (lhs - pred_lhs).sum()
        rhs_loss = (rhs - pred_rhs).sum()
        total_candidate_loss = np.abs(lhs_loss + rhs_loss)

        # update_min_loss
        if min_loss is None or total_candidate_loss < min_loss:
            min_loss = total_candidate_loss
            treshold_idx = candidate_treshold_idx
            treshold_value = candidate_treshold_value

    return FeatureSplit(treshold_idx, treshold_value, min_loss)


def find_best_split(
    points: np.array, labels: np.array
) -> Tuple[FeatureSplit, HandSide, HandSide]:
    """
    A function which finds the best split feature using a grready search for the feature which
    minimizes the residual loss.

    Arguments:
        points: np.array All the features and their data points. point values X features
        labels: np.array Label values with 1-D array
    Returns:
        Tuple[FeatureSplit, HandSide, HandSide] A tuple with the best feature split, the lhs and rhs values
    """
    # sorted idx per feature
    sorted_idx = points.argsort(axis=0)
    _, n_features = points.shape

    min_feature_loss = None
    best_loss_feature = None
    threshold_idx = None
    threshold_value = None

    for feature_idx in range(n_features):
        feature = points[:, feature_idx]
        feature_sorted_idx = sorted_idx[:, feature_idx]
        # use sorted feature to find the best split
        candidate_idx, candidate_value, candidate_ft_loss = find_best_split_feature(
            feature=feature[feature_sorted_idx], labels=labels[feature_sorted_idx]
        )

        if min_feature_loss is None or candidate_ft_loss < min_feature_loss:
            min_feature_loss = candidate_ft_loss
            best_loss_feature = feature_idx
            threshold_idx = candidate_idx
            threshold_value = candidate_value

    # select sorted idx of winner feature
    bst_lss_ft_srt_idx = sorted_idx[:, best_loss_feature]
    # reorg points and labels by the idx of winning sequence
    sorted_points = points[bst_lss_ft_srt_idx]
    sorted_labels = labels[bst_lss_ft_srt_idx]

    # lhs and rhs split by threshold idx of sorted sequence.
    lhs_points, lhs_labels = (
        sorted_points[:threshold_idx],
        sorted_labels[:threshold_idx],
    )
    rhs_points, rhs_labels = (
        sorted_points[threshold_idx:],
        sorted_labels[threshold_idx:],
    )

    # return feature spit, left and right split
    feature_split = FeatureSplit(best_loss_feature, threshold_value, min_feature_loss)
    lhs = HandSide(lhs_points, lhs_labels)
    rhs = HandSide(rhs_points, rhs_labels)

    return feature_split, lhs, rhs
