from typing import Tuple
import numpy as np
import pytest

from gradient_boosting_trees.regression.node.split import find_best_split_feature, find_best_split


@pytest.mark.parametrize(
        "sample, expected", [
            ("exponential_sample", (0, 2, 1.5)),
            ("cosine_sample", (0, 16, 15.5))
        ]
)
def test_find_best_split_feature(sample: np.ndarray, expected: Tuple[float], request):
    sample = request.getfixturevalue(sample)
    feature, labels = sample
    feature_split = find_best_split_feature(feature=feature, labels=labels)
    indx, value, loss = feature_split
    e_loss, e_indx, e_value = expected
    assert pytest.approx(loss, 0.01) == e_loss
    assert indx == e_indx
    assert value == e_value



@pytest.mark.parametrize(
        "sample, expected", [
            ("exponential_sample_2d", (0, 0.605, 0.0)),
            ("cosine_sample_2d", (0, 0.9550000000000001, 0.0))
        ]
)
def test_find_best_split(sample: np.ndarray, expected: Tuple[float], request):
    sample = request.getfixturevalue(sample)
    x, y, labels = sample
    points = np.hstack([x.reshape(len(x), 1), y.reshape(len(y), 1)])
    feature_split, lhs, rhs = find_best_split(points=points, labels=labels)
    assert np.vstack([lhs.points, rhs.points]).shape == points.shape
    assert np.hstack([lhs.labels, rhs.labels]).shape == labels.shape
    assert feature_split == expected
