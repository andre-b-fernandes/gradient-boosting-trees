import numpy as np

def squared_error_objective(raw_predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # derivative((y - p) ** 2) = 2 * (y - p) where 2 does not matter due to shrinkage
    return labels - raw_predictions
