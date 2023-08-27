from typing import Tuple
import numpy as np
import pytest


@pytest.fixture()
def exponential_sample() -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(100)
    y = np.exp(x)
    return x, y


@pytest.fixture()
def cosine_sample() -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(100)
    y = np.cos(x)
    return x, y


@pytest.fixture()
def exponential_sample_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(20, step=0.01)
    y = np.arange(10, 30, step=0.01)
    return x, y, np.exp(x * y)


@pytest.fixture()
def cosine_sample_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(20, step=0.01)
    y = np.arange(10, 30, step=0.01)
    return x, y, np.cos(x * y)
