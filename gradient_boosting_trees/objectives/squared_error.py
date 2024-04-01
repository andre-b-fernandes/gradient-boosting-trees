import numpy as np

def squared_error(raw_predictions: np.ndarray, labels: np.ndarray) -> float:
    return np.mean((labels - raw_predictions) ** 2)

def squared_error_gradient_hessian(
    raw_predictions: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    # derivative((y - p) ** 2) = 2 * (y - p) where 2 does not matter due to shrinkage
    return 2 * (labels - raw_predictions), np.full_like(raw_predictions, 2)
